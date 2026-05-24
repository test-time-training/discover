"""
Entry point for KernelBench RL training via TTT-Discover.

Usage:
    python -m examples.kernelbench.run

Or import and call run_kernelbench() directly.
"""

import asyncio
import logging
import os

import ttt_discover.tinker_utils.misc_utils as misc_utils
from ttt_discover.rl.train import Config, main
from ttt_discover.tinker_utils.dataset_builder import DatasetConfig, MultiProblemDatasetBuilder

from kernelbench_tinker.envs.kernelbench_client import get_problem_ids
from kernelbench_tinker.modal.evaluator import (
    ModalEvaluatorConfig,
    ModalKernelEvaluator,
    set_modal_evaluator,
)

from examples.kernelbench.env import KernelBenchDiscoverEnv

logger = logging.getLogger(__name__)


def run_kernelbench(
    # Problem selection
    level: int = 1,
    start_problem: int | None = None,
    end_problem: int | None = None,
    backend: str = "triton",
    dataset_src: str = "huggingface",
    prompt_option: str = "one_shot",
    prompt_precision: str | None = None,
    prompt_include_hardware: bool = False,
    prompt_gpu_name: str | None = None,
    # Modal / eval
    gpu_type: str = "A100",
    modal_timeout: int = 120,
    measure_performance: bool = True,
    num_correct_trials: int = 5,
    # Model
    model_name: str = "openai/gpt-oss-20b",
    lora_rank: int = 32,
    # Training
    group_size: int = 8,
    groups_per_batch: int = 4,
    learning_rate: float = 4e-5,
    num_epochs: int = 1,
    temperature: float = 1.0,
    kl_penalty_coef: float = 0.0,
    phase1_max_tokens: int = 16000,
    save_every: int = 2,
    # Dataset
    shuffle: bool = True,
    seed: int = 0,
    # Logging
    experiment_name: str = "kernelbench-rl",
    wandb_project: str | None = "kernelbench-discover",
):
    asyncio.run(_run_impl(**locals()))


async def _run_impl(
    level, start_problem, end_problem, backend, dataset_src, prompt_option,
    prompt_precision, prompt_include_hardware, prompt_gpu_name,
    gpu_type, modal_timeout, measure_performance, num_correct_trials,
    model_name, lora_rank, group_size, groups_per_batch, learning_rate,
    num_epochs, temperature, kl_penalty_coef, phase1_max_tokens, save_every,
    shuffle, seed, experiment_name, wandb_project,
):
    # Set up Modal evaluator
    modal_config = ModalEvaluatorConfig(
        enabled=True, gpu_type=gpu_type, timeout=modal_timeout
    )
    set_modal_evaluator(ModalKernelEvaluator(modal_config))
    logger.info("Modal evaluator configured: GPU=%s timeout=%ds", gpu_type, modal_timeout)

    # Build problem list
    problem_ids = get_problem_ids(
        level, start=start_problem, end=end_problem, dataset_src=dataset_src
    )
    problem_types = [f"{level}:{pid}" for pid in problem_ids]
    logger.info("Training on %d problems (level %d)", len(problem_types), level)

    # Log path
    log_path = f"./tinker_log/{experiment_name}"
    misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(log_path, "train.log"),
        filemode="a",
        force=True,
    )

    # Dataset config
    dataset_config = DatasetConfig(
        env_type=KernelBenchDiscoverEnv,
        problem_type=problem_types[0],  # placeholder; MultiProblemDataset overrides per problem
        batch_size=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name="gpt_oss_high_reasoning",
        eval_timeout=modal_timeout,
        log_path=log_path,
        backend=backend,
        measure_performance=measure_performance,
        num_correct_trials=num_correct_trials,
        prompt_option=prompt_option,
        prompt_precision=prompt_precision,
        prompt_include_hardware=prompt_include_hardware,
        prompt_gpu_name=prompt_gpu_name or (gpu_type if prompt_include_hardware else None),
        dataset_src=dataset_src,
    )

    dataset_builder = MultiProblemDatasetBuilder(
        config=dataset_config,
        problem_types=problem_types,
        shuffle=shuffle,
        seed=seed,
    )

    rl_config = Config(
        env_type=KernelBenchDiscoverEnv,
        problem_type=problem_types[0],  # placeholder
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=model_name,
        lora_rank=lora_rank,
        temperature=temperature,
        wandb_project=wandb_project,
        wandb_name=experiment_name,
        log_path=log_path,
        load_checkpoint_path=None,
        kl_penalty_coef=kl_penalty_coef,
        num_substeps=1,
        save_every=save_every,
        num_epochs=num_epochs,
        loss_fn="importance_sampling",
        adv_estimator="mean_baseline",
        adv_estimator_beta=2.0,
        remove_constant_reward_groups=True,
        phase1_max_tokens=phase1_max_tokens,
        local_model_path=None,
    )

    await main(rl_config)


if __name__ == "__main__":
    run_kernelbench()
