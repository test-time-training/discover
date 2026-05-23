"""Centralized sampler creation for all environments."""
from __future__ import annotations
from abc import ABC, abstractmethod
import os
import threading

import numpy as np

from ttt_discover.tinker_utils.state import State, state_from_dict
from ttt_discover.tinker_utils.best_sequence_utils import _file_lock, _atomic_write_json, _read_json_or_default


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


def create_initial_state(env_type: type, problem_type: str) -> State:
    """Create initial state by delegating to the env type. Custom envs implement create_initial_state on their class."""
    name = getattr(env_type, "env_name", env_type.__name__)
    print(f"Creating initial state for {name}")
    return env_type.create_initial_state(problem_type)


class PUCTSampler(StateSampler):
    """
    PUCT-style sampler with state archive.

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])
    
    where:
      Q(i) = m[i] if n[i]>0 else R(i)  (best reachable value or current reward)
      P(i) = rank-based prior
      scale = max(R) - min(R)
    """

    def __init__(
        self,
        file_path: str,
        env_type: type,
        problem_type: str = "",
        max_buffer_size: int = 1000,
        batch_size: int = 1,
        resume_step: int | None = None,
        puct_c: float = 1.0,
        topk_children: int = 2,
    ):
        self.file_path = file_path
        self.env_type = env_type
        self.problem_type = problem_type
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.topk_children = topk_children
        self.puct_c = float(puct_c)
        
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
                state = create_initial_state(self.env_type, self.problem_type)
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
        state_cls = self.env_type.state_type
        self._states = [state_from_dict(s, state_type=state_cls) for s in store.get("states", [])]
        self._initial_states = [state_from_dict(s, state_type=state_cls) for s in store.get("initial_states", [])]
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
        """Regenerate construction for initial states when env expects random construction (e.g. AC)."""
        # For ac
        if not getattr(self.env_type, "construction_length_limits", None):
            return
        rng = np.random.default_rng()
        state.construction = [rng.random()] * rng.integers(1000, 8000)
        if self.problem_type == "ac1":
            from ttt_discover.tinker_utils.ac_helpers import evaluate_sequence_ac1
            state.value = -evaluate_sequence_ac1(state.construction)
        else:
            from ttt_discover.tinker_utils.ac_helpers import evaluate_sequence_ac2
            state.value = evaluate_sequence_ac2(state.construction)

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
            picked = [
                create_initial_state(self.env_type, self.problem_type)
                for _ in range(num_states)
            ]
            self._last_sampled_states = picked
            self._last_sampled_indices = []
            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
            return picked

        vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
        scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
        self._last_scale = scale
        P = self._compute_prior(vals, scale)
        sqrtT = np.sqrt(1.0 + self._T)

        scores = []
        for i, s in enumerate(candidates):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, vals[i])
            Q = m if n > 0 else vals[i]
            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
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
            limits = getattr(self.env_type, "construction_length_limits", None)
            if limits and child.construction:
                lo, hi = limits
                if not (lo <= len(child.construction) <= hi):
                    continue
            max_len = getattr(self.env_type, "max_construction_len", None)
            if max_len is not None and child.construction and len(child.construction) > max_len:
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
                    state = create_initial_state(self.env_type, self.problem_type)
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


class SimpleStateSampler(StateSampler):
    """Always returns the same initial state. No buffer, no PUCT. For standard RL training."""

    def __init__(self, initial_state: State):
        self._initial_state = initial_state

    def sample_states(self, num_states: int) -> list[State]:
        return [self._initial_state] * num_states

    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
        pass  # no-op

    def flush(self, step: int | None = None):
        pass  # no-op


def create_sampler(
    log_path: str,
    env_type: type,
    problem_type: str = "",
    batch_size: int = 1,
    resume_step: int | None = None,
) -> StateSampler:
    """Factory function to create samplers. Pass the env type (from config.env_type)."""
    if not log_path:
        raise ValueError("log_path is required when using PUCT sampler")
    sampler_path = os.path.join(log_path, "puct_sampler.json")
    return PUCTSampler(
        file_path=sampler_path,
        env_type=env_type,
        problem_type=problem_type,
        batch_size=batch_size,
        resume_step=resume_step,
    )


def get_or_create_sampler_with_default(
    log_path: str,
    env_type: type,
    problem_type: str = "",
    batch_size: int = 1,
    resume_step: int | None = None,
) -> StateSampler:
    """Get sampler. Initial experience is created via env_type.create_initial_state."""
    return create_sampler(
        log_path=log_path,
        env_type=env_type,
        problem_type=problem_type,
        batch_size=batch_size,
        resume_step=resume_step,
    )
