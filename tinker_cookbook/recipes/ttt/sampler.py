"""Centralized sampler creation for all environments."""
from __future__ import annotations
from abc import ABC, abstractmethod
import os
import threading

import numpy as np

from tinker_cookbook.recipes.ttt.state import (
    InequalitiesState,
    CirclePackingState,
    GpuModeState,
    AleBenchState,
    ErdosState,
    DenoisingState,
    State,
    state_from_dict,
)
from tasks.alphaevolve_ac.best_sequence_utils import _file_lock, _atomic_write_json, _read_json_or_default


SAMPLER_TYPES = {"greedy", "fixed", "puct", "puct_backprop"}  # puct_backprop is alias for puct
INITIAL_EXP_TYPES = {"best_available", "none", "random", "random_no_code"}

# Construction length limits for AC environments
MIN_CONSTRUCTION_LEN = 1000
MAX_CONSTRUCTION_LEN = 100000

# Maximum construction length for Erdos environments
MAX_ERDOS_CONSTRUCTION_LEN = 1000


class StateSampler(ABC):
    """Abstract base class for sampling states."""

    @abstractmethod
    def sample_states(self, num_states: int) -> list[State]:
        """Sample states to start rollouts from."""
        pass

    @abstractmethod
    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        """Update internal storage with new states. Sets parent info automatically."""
        pass

    @abstractmethod
    def flush(self, step: int | None = None):
        """Force save current state to disk."""
        pass
    
    @staticmethod
    def _set_parent_info(child: State, parent: State):
        """Set parent_values and parents on child state from parent."""
        child.parent_values = [parent.value] + parent.parent_values if parent.value is not None else []
        child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents

    @staticmethod
    def _filter_topk_per_parent(states: list[State], parent_states: list[State], k: int) -> tuple[list[State], list[State]]:
        """Keep top-k children (by value) per parent. If k=0, return all."""
        if not states:
            return [], []
        if k == 0:
            return states, parent_states
        # Group by parent id
        parent_to_children: dict[str, list[tuple[State, State]]] = {}
        for child, parent in zip(states, parent_states):
            pid = parent.id
            if pid not in parent_to_children:
                parent_to_children[pid] = []
            parent_to_children[pid].append((child, parent))
        # Keep top-k children per parent (highest value)
        topk_children, topk_parents = [], []
        for children_and_parents in parent_to_children.values():
            sorted_pairs = sorted(children_and_parents, key=lambda x: x[0].value if x[0].value is not None else float('-inf'), reverse=True)
            for child, parent in sorted_pairs[:k]:
                topk_children.append(child)
                topk_parents.append(parent)
        return topk_children, topk_parents


def _sampler_file_for_step(base_path: str, step: int) -> str:
    """Get the sampler file path for a specific step."""
    base_name = base_path.replace(".json", "")
    return f"{base_name}_step_{step:06d}.json"


def create_initial_state(env_type: str, initial_exp_type: str, budget_s: int = 1000) -> State:
    """Create an initial state for a given env type."""
    if initial_exp_type == "best_available":
        if env_type == "ac1":
            from tasks.alphaevolve_ac.sota_alphaevolve2 import height_sequence_1
            construction = list(height_sequence_1)
        elif env_type == "ac2":
            from tasks.alphaevolve_ac2.ae_seq import height_sequence_2
            construction = list(height_sequence_2)
        else:
            construction = []
    elif initial_exp_type == "none":
        construction = []
    elif initial_exp_type == "random":
        if env_type in {"ac1", "ac2"}:
            rng = np.random.default_rng(12345)
            construction = [rng.random()] * rng.integers(1000, 8000)
        elif env_type == "erdos":
            rng = np.random.default_rng()
            n_points = rng.integers(40, 100)
            construction = np.ones(n_points) * 0.5
            perturbation = rng.uniform(-0.4, 0.4, n_points)
            perturbation = perturbation - np.mean(perturbation)
            construction = construction + perturbation
            dx = 2.0 / n_points
            correlation = np.correlate(construction, 1 - construction, mode="full") * dx
            c5_bound = float(np.max(correlation))
            return ErdosState(timestep=-1, code="", value=-c5_bound, c5_bound=c5_bound, construction=list(construction))
        else:
            construction = []
    elif initial_exp_type == "random_no_code":
        # Fixed construction of size 1000, deterministic seed for comparability
        if env_type in {"ac1", "ac2"}:
            rng = np.random.default_rng(42)
            construction = list(rng.random(1000))
        else:
            construction = []
    else:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}")

    # Compute initial value (higher = better)
    if construction:
        if env_type == "ac1":
            from tasks.alphaevolve_ac.verifier_ae import evaluate_sequence
            initial_value = -evaluate_sequence(construction)  # -upper_bound: higher = better
        elif env_type == "ac2":
            from tasks.alphaevolve_ac2.ae_verifier import evaluate_sequence
            initial_value = evaluate_sequence(construction) # Maximize lower bound
    elif env_type == "gpu_mode":
        initial_value = -1_000_000 # Worse than max of initial distribution
    elif env_type in {"ahc039", "ahc058"}:
        initial_value = 0.0
    elif env_type == "erdos":
        initial_value = None  # No initial value for erdos
    else:
        initial_value = 0.0

    timestep = -1

    # Create state (timestep=-1 for initial states)
    if env_type == "ac1":
        from tasks.alphaevolve_ac.prompt import example_ae_program_best_init_and_random_init, example_ae_program
        if initial_exp_type == "none":
            code = "```python\n" + example_ae_program(budget_s) + "\n```"
        elif initial_exp_type == "random_no_code":
            code = ""
        else:
            code = "```python\n" + example_ae_program_best_init_and_random_init(budget_s) + "\n```"
        return InequalitiesState(timestep=timestep, construction=construction, code=code, value=initial_value)
    elif env_type == "ac2":
        from tasks.alphaevolve_ac2.prompt import thetaevolve_initial_program, thetaevolve_initial_program_prev_init
        if initial_exp_type == "none":
            code = "```python\n" + thetaevolve_initial_program + "\n```"
        elif initial_exp_type == "random_no_code":
            code = ""
        else:
            code = "```python\n" + thetaevolve_initial_program_prev_init + "\n```"
        return InequalitiesState(timestep=timestep, construction=construction, code=code, value=initial_value)
    elif env_type == "cp":
        return CirclePackingState(timestep=timestep, construction=None, code="", value=initial_value)
    elif env_type == "mla_decode_nvidia":
        from tasks.gpu_mode.initial_program_mla_decode import INITIAL_CODE, INITIAL_VALUE
        code = INITIAL_CODE
        initial_value = INITIAL_VALUE
        return GpuModeState(timestep=timestep, code=code, value=initial_value)
    elif env_type == "trimul":
        # No initial code or value for trimul
        return GpuModeState(timestep=timestep, code="", value=initial_value)
    elif env_type == "ahc039":
        if initial_exp_type == "best_available":
            from tasks.ale_bench.best_available import AHC039_BEST_CODE, AHC039_BEST_CODE_VALUE
            return AleBenchState(timestep=timestep, code=AHC039_BEST_CODE, value=AHC039_BEST_CODE_VALUE)
        return AleBenchState(timestep=timestep, code="", value=initial_value)
    elif env_type == "ahc058":
        if initial_exp_type == "best_available":
            raise ValueError("AHC058 has no best code available.")
        return AleBenchState(timestep=timestep, code="", value=initial_value)
    elif env_type == "erdos":
        return ErdosState(timestep=timestep, code="", value=initial_value, c5_bound=None, construction=None)
    elif env_type == "denoising":
        from tasks.denoising.task import MAGIC_FUNC
        return DenoisingState(timestep=timestep, code=MAGIC_FUNC, value=0.24, mse=0.2316, poisson=0.0370)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


class GreedySampler(StateSampler):
    """Epsilon-greedy sampler that keeps top-k best states by value."""
    
    def __init__(self, file_path: str, env_type: str = "ac1", budget_s: int = 1000, 
                 initial_exp_type: str = "random", batch_size: int = 1, 
                 resume_step: int | None = None, topk_children: int = 1,
                 epsilon: float = 0.125):
        self.file_path = file_path
        self.env_type = env_type
        self.budget_s = budget_s
        self.initial_exp_type = initial_exp_type
        self.batch_size = batch_size
        self.topk_children = topk_children
        self.epsilon = epsilon  # probability of sampling random state instead of best
        self._top_states: list[State] = []
        self._lock = threading.Lock()
        self._current_step = resume_step if resume_step is not None else 0
        if resume_step is not None:
            self._load(resume_step)

    def _load(self, step: int):
        file_path = _sampler_file_for_step(self.file_path, step)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot resume from step {step}: sampler file not found: {file_path}")
        with _file_lock(f"{file_path}.lock"):
            store = _read_json_or_default(file_path, default=None)
        if store is None:
            raise ValueError(f"Failed to load sampler state from {file_path}")
        self._top_states = [state_from_dict(s) for s in store.get("states", [])]

    def _save(self, step: int):
        if not self._top_states:
            return
        save_path = _sampler_file_for_step(self.file_path, step)
        store = {"step": step, "states": [s.to_dict() for s in self._top_states]}
        with _file_lock(f"{save_path}.lock"):
            _atomic_write_json(save_path, store)

    def sample_states(self, num_states: int) -> list[State]:
        if not self._top_states:
            return [create_initial_state(self.env_type, self.initial_exp_type, self.budget_s) 
                    for _ in range(num_states)]
        # Epsilon-greedy: with prob epsilon, sample random; otherwise sample best
        result = []
        for i in range(num_states):
            if self.epsilon > 0 and np.random.random() < self.epsilon and len(self._top_states) > 1:
                result.append(np.random.choice(self._top_states))
            else:
                result.append(self._top_states[i % len(self._top_states)])
        return result

    def _get_construction_key(self, state: State) -> tuple | str | None:
        if hasattr(state, 'construction') and state.construction:
            return tuple(state.construction)
        if hasattr(state, 'code') and state.code:
            return state.code
        return None

    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        if not states:
            return
        states, parent_states = self._filter_topk_per_parent(states, parent_states, self.topk_children)
        existing = {self._get_construction_key(s) for s in self._top_states}
        existing.discard(None)
        new_states = []
        for child, parent in zip(states, parent_states):
            if isinstance(child, InequalitiesState) and child.construction:
                if not (MIN_CONSTRUCTION_LEN <= len(child.construction) <= MAX_CONSTRUCTION_LEN):
                    continue
            if isinstance(child, ErdosState) and child.construction:
                if len(child.construction) > MAX_ERDOS_CONSTRUCTION_LEN:
                    continue
            key = self._get_construction_key(child)
            if key is not None and key in existing:
                continue
            self._set_parent_info(child, parent)
            new_states.append(child)
            if key is not None:
                existing.add(key)
        if not new_states:
            return
        with self._lock:
            self._top_states.extend(new_states)
            if save:
                self._finalize_and_save(step)

    def _finalize_and_save(self, step: int | None = None):
        self._top_states.sort(key=lambda s: s.value if s.value else 0, reverse=True)
        self._top_states = self._top_states[:self.batch_size]
        if step is not None:
            self._current_step = step
        self._save(self._current_step)

    def flush(self, step: int | None = None):
        with self._lock:
            self._finalize_and_save(step)

    def reload_from_step(self, step: int):
        with self._lock:
            self._top_states = []
            self._current_step = step
            self._load(step)


class FixedSampler(StateSampler):
    """Fixed distribution sampler - always returns same state, never updates."""
    
    def __init__(self, env_type: str = "cp", budget_s: int = 1000, initial_exp_type: str = "none"):
        self.env_type = env_type
        self.budget_s = budget_s
        self.initial_exp_type = initial_exp_type
        self._fixed_state = create_initial_state(env_type, initial_exp_type, budget_s)

    def sample_states(self, num_states: int) -> list[State]:
        return [self._fixed_state] * num_states

    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        pass

    def flush(self, step: int | None = None):
        pass

    def reload_from_step(self, step: int):
        pass


class PUCTSampler(StateSampler):
    """
    PUCT-style sampler with state archive.

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T/G) / (1 + n[i]/G)
    
    where:
      Q(i) = m[i] if n[i]>0 else R(i)  (best reachable value or current reward)
      P(i) = rank-based prior
      scale = max(R) - min(R)
      G = group_size
    """

    def __init__(
        self,
        file_path: str,
        env_type: str = "ac1",
        budget_s: int = 1000,
        initial_exp_type: str = "random",
        max_buffer_size: int = 1000,
        batch_size: int = 1,
        resume_step: int | None = None,
        puct_c: float = 1.0,
        topk_children: int = 2,
        group_size: int = 64,
        **kwargs,
    ):
        self.file_path = file_path
        self.env_type = env_type
        self.budget_s = budget_s
        self.initial_exp_type = initial_exp_type
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.topk_children = topk_children
        self.puct_c = float(puct_c)
        self.group_size = int(group_size)
        # TODO: remove this
        self.group_size = 1
        
        self._states: list[State] = []
        self._initial_states: list[State] = []
        self._last_sampled_states: list[State] = []
        self._last_sampled_indices: list[int] = []
        self._lock = threading.Lock()
        self._current_step = resume_step if resume_step is not None else 0
        
        # PUCT stats
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T: int = 0
        self._last_scale: float = 1.0
        self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
        
        if resume_step is not None:
            self._load(resume_step)
        if not self._states:
            for _ in range(batch_size):
                state = create_initial_state(self.env_type, self.initial_exp_type, self.budget_s)
                self._initial_states.append(state)
                self._states.append(state)
            self._save(self._current_step)

    def _load(self, step: int):
        file_path = _sampler_file_for_step(self.file_path, step)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot resume from step {step}: sampler file not found: {file_path}")
        with _file_lock(f"{file_path}.lock"):
            store = _read_json_or_default(file_path, default=None)
        if store is None:
            raise ValueError(f"Failed to load sampler state from {file_path}")
        self._states = [state_from_dict(s) for s in store.get("states", [])]
        self._initial_states = [state_from_dict(s) for s in store.get("initial_states", [])]
        self._n = store.get("puct_n", {}) or {}
        self._m = store.get("puct_m", {}) or {}
        self._T = int(store.get("puct_T", 0) or 0)

    def _save(self, step: int):
        save_path = _sampler_file_for_step(self.file_path, step)
        store = {
            "step": step,
            "states": [s.to_dict() for s in self._states],
            "initial_states": [s.to_dict() for s in self._initial_states],
            "puct_n": self._n,
            "puct_m": self._m,
            "puct_T": self._T,
        }
        with _file_lock(f"{save_path}.lock"):
            _atomic_write_json(save_path, store)

    def _refresh_random_construction(self, state: State) -> None:
        """Regenerate construction for initial states when initial_exp_type='random'."""
        if self.initial_exp_type != "random" or self.env_type not in {"ac1", "ac2"}:
            return
        rng = np.random.default_rng()
        state.construction = [rng.random()] * rng.integers(1000, 8000)
        if self.env_type == "ac1":
            from tasks.alphaevolve_ac.verifier_ae import evaluate_sequence
            state.value = -evaluate_sequence(state.construction)
        else:
            from tasks.alphaevolve_ac2.ae_verifier import evaluate_sequence
            state.value = evaluate_sequence(state.construction)

    def _get_construction_key(self, state: State) -> tuple | str | None:
        if hasattr(state, 'construction') and state.construction:
            return tuple(state.construction)
        if hasattr(state, 'code') and state.code:
            return state.code
        return None

    def _compute_scale(self, values: np.ndarray, mask: np.ndarray | None = None) -> float:
        if values.size == 0:
            return 1.0
        v = values[mask] if mask is not None else values
        return float(max(np.max(v) - np.min(v), 1e-6)) if v.size > 0 else 1.0

    def _compute_prior(self, values: np.ndarray, scale: float) -> np.ndarray:
        if values.size == 0:
            return np.array([])
        N = len(values)
        ranks = np.argsort(np.argsort(-values))
        weights = (N - ranks).astype(np.float64)
        return weights / weights.sum()

    def _get_lineage(self, state: State) -> set[str]:
        lineage = {state.id}
        for p in (state.parents or []):
            if p.get("id"):
                lineage.add(str(p["id"]))
        return lineage

    def _build_children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self._states:
            for p in (s.parents or []):
                pid = p.get("id")
                if pid:
                    children.setdefault(str(pid), set()).add(s.id)
        return children

    def _get_full_lineage(self, state: State, children_map: dict[str, set[str]]) -> set[str]:
        lineage = self._get_lineage(state)
        queue = [state.id]
        visited = {state.id}
        while queue:
            sid = queue.pop(0)
            for child_id in children_map.get(sid, []):
                if child_id not in visited:
                    visited.add(child_id)
                    lineage.add(child_id)
                    queue.append(child_id)
        return lineage

    def sample_states(self, num_states: int) -> list[State]:
        initial_ids = {s.id for s in self._initial_states}
        candidates = list(self._states)

        if not candidates:
            picked = [create_initial_state(self.env_type, self.initial_exp_type, self.budget_s)
                      for _ in range(num_states)]
            self._last_sampled_states = picked
            self._last_sampled_indices = []
            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
            return picked

        vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
        scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
        self._last_scale = scale
        P = self._compute_prior(vals, scale)
        G = self.group_size
        sqrtT = np.sqrt(1.0 + self._T / G)

        scores = []
        for i, s in enumerate(candidates):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, vals[i])
            Q = m if n > 0 else vals[i]
            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n / G)
            score = Q + bonus
            scores.append((score, vals[i], s, n, Q, P[i], bonus))

        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if num_states > 1:
            children_map = self._build_children_map()
            picked, top_scores, blocked_ids = [], [], set()
            for entry in scores:
                s = entry[2]
                if s.id in blocked_ids:
                    continue
                picked.append(s)
                top_scores.append(entry)
                blocked_ids.update(self._get_full_lineage(s, children_map))
                if len(picked) >= num_states:
                    break
        else:
            top_scores = scores[:num_states]
            picked = [t[2] for t in top_scores]

        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
        self._last_sampled_states = picked
        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]

        for s in picked:
            if s.id in initial_ids:
                self._refresh_random_construction(s)

        return picked

    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        if not states:
            return
        assert len(states) == len(parent_states)

        # Update PUCT stats for ALL states
        parent_max: dict[str, float] = {}
        parent_obj: dict[str, State] = {}
        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue
            pid = parent.id
            parent_obj[pid] = parent
            parent_max[pid] = max(parent_max.get(pid, float("-inf")), float(child.value))

        for pid, y in parent_max.items():
            self._m[pid] = max(self._m.get(pid, y), y)
            parent = parent_obj[pid]
            anc_ids = [pid] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
            for aid in anc_ids:
                self._n[aid] = self._n.get(aid, 0) + 1
            self._T += 1

        if not states:
            return

        # Apply topk filter and dedup
        states, parent_states = self._filter_topk_per_parent(states, parent_states, self.topk_children)
        existing = {self._get_construction_key(s) for s in self._states}
        existing.discard(None)
        
        new_states = []
        for child, parent in zip(states, parent_states):
            if child.value is None:
                continue
            if isinstance(child, InequalitiesState) and child.construction:
                if not (MIN_CONSTRUCTION_LEN <= len(child.construction) <= MAX_CONSTRUCTION_LEN):
                    continue
            if isinstance(child, ErdosState) and child.construction:
                if len(child.construction) > MAX_ERDOS_CONSTRUCTION_LEN:
                    continue
            key = self._get_construction_key(child)
            if key is not None and key in existing:
                continue
            self._set_parent_info(child, parent)
            new_states.append(child)
            if key is not None:
                existing.add(key)

        if not new_states:
            return
        with self._lock:
            self._states.extend(new_states)
            if save:
                self._finalize_and_save(step)

    def _finalize_and_save(self, step: int | None = None):
        if len(self._states) > self.max_buffer_size:
            actual_values = [s.value if s.value is not None else float('-inf') for s in self._states]
            by_actual = list(np.argsort(actual_values)[::-1])
            initial_ids = {s.id for s in self._initial_states}
            initial_indices = {i for i, s in enumerate(self._states) if s.id in initial_ids}
            keep = set(initial_indices)
            for i in by_actual:
                if len(keep) >= self.max_buffer_size:
                    break
                keep.add(i)
            self._states = [self._states[i] for i in sorted(keep)]
        if step is not None:
            self._current_step = step
        self._save(self._current_step)

    def flush(self, step: int | None = None):
        with self._lock:
            if self.topk_children > 0:
                by_parent: dict[str, list[State]] = {}
                no_parent: list[State] = []
                for s in self._states:
                    pid = s.parents[0]["id"] if s.parents else None
                    if pid:
                        by_parent.setdefault(pid, []).append(s)
                    else:
                        no_parent.append(s)
                filtered = []
                for children in by_parent.values():
                    children.sort(key=lambda x: x.value if x.value is not None else float('-inf'), reverse=True)
                    filtered.extend(children[:self.topk_children])
                self._states = no_parent + filtered
            self._finalize_and_save(step)

    def record_failed_rollout(self, parent: State):
        anc_ids = [parent.id] + [str(p["id"]) for p in (parent.parents or []) if p.get("id")]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + 1
        self._T += 1

    def reload_from_step(self, step: int):
        with self._lock:
            self._states = []
            self._initial_states = []
            self._current_step = step
            self._load(step)
            if not self._states:
                for _ in range(self.batch_size):
                    state = create_initial_state(self.env_type, self.initial_exp_type, self.budget_s)
                    self._initial_states.append(state)
                    self._states.append(state)

    def get_sample_stats(self) -> dict:
        def _stats(values, prefix):
            arr = np.array([v for v in values if v is not None])
            if len(arr) == 0:
                return {}
            return {
                f"{prefix}/mean": float(np.mean(arr)),
                f"{prefix}/std": float(np.std(arr)),
                f"{prefix}/min": float(np.min(arr)),
                f"{prefix}/max": float(np.max(arr)),
            }
        buffer_values = [s.value for s in self._states]
        buffer_timesteps = [s.timestep for s in self._states]
        buffer_constr_lens = [len(s.construction) if hasattr(s, 'construction') and s.construction else 0 for s in self._states]
        sampled_values = [s.value for s in self._last_sampled_states]
        sampled_timesteps = [s.timestep for s in self._last_sampled_states]
        sampled_constr_lens = [len(s.construction) if hasattr(s, 'construction') and s.construction else 0 for s in self._last_sampled_states]
        stats = {
            "puct/buffer_size": len(self._states),
            "puct/sampled_size": len(self._last_sampled_states),
            "puct/T": self._T,
            "puct/scale_last": float(self._last_scale),
        }
        stats.update(_stats(buffer_values, "puct/buffer_value"))
        stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
        stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
        stats.update(_stats(sampled_values, "puct/sampled_value"))
        stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
        stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
        return stats

    def get_sample_table(self) -> tuple[list[str], list[tuple]]:
        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
        rows = []
        if not self._last_sampled_states:
            return columns, rows
        indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
        stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
            parent_val = state.parent_values[0] if state.parent_values else None
            constr = getattr(state, 'construction', None)
            constr_len = len(constr) if constr is not None else 0
            obs_len = len(state.observation) if state.observation else 0
            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
        return columns, rows


def create_sampler(
    sampler_type: str,
    log_path: str,
    env_type: str = "ac1",
    budget_s: int = 1000,
    initial_exp_type: str = "random",
    batch_size: int = 1,
    resume_step: int | None = None,
    epsilon: float = 0.0,
    **kwargs,
) -> StateSampler:
    """Factory function to create samplers by type."""
    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Supported: {SAMPLER_TYPES}")
    if initial_exp_type not in INITIAL_EXP_TYPES:
        raise ValueError(f"Unknown initial_exp_type: {initial_exp_type}. Supported: {INITIAL_EXP_TYPES}")
    
    if sampler_type == "greedy":
        if not log_path:
            raise ValueError(f"log_path is required when using sampler_type={sampler_type}")
        sampler_path = os.path.join(log_path, "experience_sampler.json")
        return GreedySampler(sampler_path, env_type=env_type, budget_s=budget_s, 
                             initial_exp_type=initial_exp_type, batch_size=batch_size, 
                             resume_step=resume_step, epsilon=epsilon,
                             )
    elif sampler_type == "fixed":
        return FixedSampler(env_type=env_type, budget_s=budget_s, initial_exp_type=initial_exp_type)
    elif sampler_type in ("puct", "puct_backprop"):
        if not log_path:
            raise ValueError(f"log_path is required when using sampler_type={sampler_type}")
        sampler_path = os.path.join(log_path, f"{sampler_type}_sampler.json")
        return PUCTSampler(sampler_path, env_type=env_type, budget_s=budget_s, 
                          initial_exp_type=initial_exp_type, batch_size=batch_size, 
                          resume_step=resume_step, **kwargs)

    raise ValueError(f"Unknown sampler_type: {sampler_type}")


def get_or_create_sampler_with_default(
    sampler_type: str,
    log_path: str,
    env_type: str,
    budget_s: int = 1000,
    initial_exp_type: str = "random",
    batch_size: int = 1,
    resume_step: int | None = None,
    **kwargs,
) -> StateSampler:
    """Get sampler. Initial experience is created automatically if needed."""
    return create_sampler(sampler_type, log_path, env_type=env_type, budget_s=budget_s, 
                          initial_exp_type=initial_exp_type, batch_size=batch_size,
                          resume_step=resume_step, **kwargs)
