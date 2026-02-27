import asyncio
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, TypeVar

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tinker_cookbook.recipes.ttt.state import State
from tinker_cookbook.recipes.ttt.sampler import StateSampler, get_or_create_sampler_with_default, SAMPLER_TYPES, INITIAL_EXP_TYPES


@dataclass
class DatasetConfig:
    """General configuration for dataset and environment creation."""
    dataset_path: str
    dataset_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    problem_idx: str
    seed: int = 0
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    dataset_timeout: int = 300
    sampler_type: str = "greedy"
    initial_exp_type: str = "random"
    log_path: str = ""
    adv_estimator: str | None = None
    timeout: float = 1000.0
    convo_prefix: Any = None
    # AC-specific optional fields
    sweep_hyperparams: bool = False
    max_hyperparam_combos: int = 16
    budget_s: int = 1000
    # GPU mode specific optional field
    gpu_mode_score_scale: float = 3000.0


class SingleProblemDataset(RLDataset):
    def __init__(
        self,
        config: DatasetConfig,
        renderer: renderers.Renderer,
        sampler: StateSampler,
        split: Literal["train", "test"] = "train",
    ):
        self.config = config
        self.split = split
        self.batch_size = config.batch_size
        self.group_size = config.group_size if split == "train" else 1
        self.renderer = renderer
        self.problem_idx = config.problem_idx
        self.sampler = sampler

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def __len__(self) -> int:
        return 1

    def _make_env_group_builder(
        self, initial_state: State, group_size: int
    ) -> ProblemGroupBuilder:
        """Create an environment group builder using the appropriate env class based on dataset_name."""
        env_class = self._get_env_class(self.config.dataset_name)
        
        return ProblemGroupBuilder(
            env_thunk=partial(
                env_class,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                config=self.config,
            ),
            num_envs=group_size,
        )

    def _get_env_class(self, dataset_name: str) -> type[ProblemEnv]:
        """Get the appropriate environment class based on dataset name."""
        match dataset_name:
            case "cp":
                from tinker_cookbook.recipes.ttt.env_cp import CirclePackingEnv
                return CirclePackingEnv
            case "ac1" | "ac2":
                from tinker_cookbook.recipes.ttt.env_ac import AutoCorrInequalityEnv
                return AutoCorrInequalityEnv
            case "ale_bench":
                from tinker_cookbook.recipes.ttt.env_ale_bench import AleBenchEnv
                return AleBenchEnv
            case "denoising":
                from tinker_cookbook.recipes.ttt.env_denoising import DenoisingEnv
                return DenoisingEnv
            case "erdos":
                from tinker_cookbook.recipes.ttt.env_erdos import ErdosMinOverlapEnv
                return ErdosMinOverlapEnv
            case "trimul" | "mla_decode_nvidia":
                from tinker_cookbook.recipes.ttt.env_gpu_mode import GpuModeEnv
                return GpuModeEnv
            case "tictactoe":
                from tinker_cookbook.recipes.ttt.env_tictactoe import TictactoeEnv
                return TictactoeEnv
            case _:
                raise ValueError(f"Unknown dataset name: {dataset_name}")


@chz.chz
class SingleProblemDatasetBuilder(RLDatasetBuilder):
    config: DatasetConfig

    async def __call__(self) -> tuple[SingleProblemDataset, SingleProblemDataset]:
        if self.config.problem_idx is None:
            raise ValueError("problem_idx is required")
        if not self.config.log_path:
            raise ValueError("log_path is required for dataset")
        
        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.config.renderer_name, tokenizer=tokenizer)
        
        # Get sampler - this may need to be environment-specific
        sampler = self._get_sampler()
        
        datasets = [
            SingleProblemDataset(
                config=self.config,
                renderer=renderer,
                sampler=sampler,
                split=split,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])

    def _get_sampler(self) -> StateSampler:
        """Get the appropriate sampler based on dataset name."""
        # For CP, we need to parse problem_idx to get n
        if self.config.dataset_name == "cp":
            from tinker_cookbook.recipes.ttt.env_cp import parse_cp_problem_idx
            n, _ = parse_cp_problem_idx(self.config.problem_idx)
            if n not in (26, 32):
                raise ValueError(f"Invalid problem_idx: {self.config.problem_idx}. Must be 26, 32, 26_improvement, 32_improvement, 26_improvement_v2, or 32_improvement_v2")
            return get_or_create_sampler_with_default(
                self.config.sampler_type, 
                self.config.log_path, 
                self.config.dataset_name, 
                initial_exp_type=self.config.initial_exp_type, 
                n=n, 
                batch_size=self.config.batch_size, 
                group_size=self.config.group_size,
            )
        elif self.config.dataset_name == "ale_bench":
            # For ale_bench, use problem_idx (ahc039 or ahc058) as env_type
            if self.config.problem_idx not in ("ahc039", "ahc058"):
                raise ValueError(f"Invalid problem_idx for ale_bench: {self.config.problem_idx}. Must be 'ahc039' or 'ahc058'")
            return get_or_create_sampler_with_default(
                self.config.sampler_type, 
                self.config.log_path, 
                self.config.problem_idx,  # Use problem_idx as env_type for ale_bench
                initial_exp_type=self.config.initial_exp_type, 
                n=None, 
                batch_size=self.config.batch_size, 
                group_size=self.config.group_size,
            )
        else:
            # For other environments, use a default n or let the sampler handle it
            return get_or_create_sampler_with_default(
                self.config.sampler_type, 
                self.config.log_path, 
                self.config.dataset_name, 
                initial_exp_type=self.config.initial_exp_type, 
                n=None, 
                batch_size=self.config.batch_size, 
                group_size=self.config.group_size,
            )


# Valid dataset names
VALID_DATASET_NAMES = {
    "cp", "ac1", "ac2", "ale_bench", "denoising", "erdos", "trimul", "mla_decode_nvidia",
    "tictactoe",
}


def get_single_problem_dataset_builder(
    config: DatasetConfig,
    **kwargs,
) -> RLDatasetBuilder:
    """
    Unified function to get a single problem dataset builder.
    Args:
        config: General dataset configuration object
        **kwargs: Additional environment-specific arguments (unused for general builder)
    Returns:
        The appropriate dataset builder instance
    """
    if config.dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name}. Available: {sorted(VALID_DATASET_NAMES)}"
        )
    
    if not config.log_path:
        raise ValueError("log_path is required for dataset")

    builder = SingleProblemDatasetBuilder(
        config=config,
    )
    return builder