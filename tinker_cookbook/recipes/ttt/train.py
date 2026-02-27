import asyncio
import logging
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
import os
from tinker_cookbook.recipes.ttt.dataset_builder import get_single_problem_dataset_builder
from typing import Any
logger = logging.getLogger(__name__)

# Import DatasetConfig from dataset_builder
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "openai/gpt-oss-120b"
    lora_rank: int = 32
    renderer_name: str | None = "gpt_oss_high_reasoning"
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "cp"  # Options: cp, ac1, ac2
    dataset_path: str = "ttt-ttt9/xh-gptoss-v6-2-rule-apply-single-turn-prior-with-val-pub"
    seed: int = 0  # Random seed for data shuffling
    problem_idx: str = "1818057f"
    test_num_rollouts: int = 1
    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    num_epochs: int = 50
    max_tokens: int = 26000
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    adv_estimator: str="entropic"
    adv_estimator_beta: float = 2.0
    gpu_mode_score_scale: float = 3000.0  # scale reciprocal reward
    
    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "tinker-cookbook"
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 3

    # Checkpointing
    save_every: int = 5

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    
    # AC-specific
    sweep_hyperparams: bool = False
    max_hyperparam_combos: int = 16
    num_cpus_per_task: int = 1
    eval_timeout: int = 1000
    dataset_timeout: int = 1000
    sampler_type: str = "greedy"
    initial_exp_type: str = "random"  # "best_available", "none", "random"

    # BoN mode
    eval_BoN: bool = False
    
    debug_skip_llm: bool = False
    
    dynamic_max_tokens: bool = True
    
    # Two-phase sampling: if model exhausts phase1 tokens without stop, force completion
    two_phase_sampling: bool = False
    phase1_max_tokens: int = 26000
    
    # Local model path (avoids HuggingFace API rate limits)
    local_model_path: str | None = None


def get_dataset_builder(
    dataset_path: str,
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    problem_idx: str,
    seed: int = 0,
    sweep_hyperparams: bool = False,
    max_hyperparam_combos: int = 16,
    num_cpus_per_task: int = 1,
    eval_timeout: int = 300,
    dataset_timeout: int = 300,
    sampler_type: str = "greedy",
    initial_exp_type: str = "random",
    log_path: str = "",
    adv_estimator: str | None = None,
    gpu_mode_score_scale: float = 3000.0,
) -> RLDatasetBuilder:
    # Create general config object
    config = DatasetConfig(
        dataset_path=dataset_path,
        dataset_name=env,
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
        problem_idx=problem_idx,
        seed=seed,
        num_cpus_per_task=num_cpus_per_task,
        eval_timeout=eval_timeout,
        dataset_timeout=dataset_timeout,
        sampler_type=sampler_type,
        initial_exp_type=initial_exp_type,
        log_path=log_path,
        adv_estimator=adv_estimator if env != "ale_bench" else None,
        sweep_hyperparams=sweep_hyperparams,
        max_hyperparam_combos=max_hyperparam_combos,
        gpu_mode_score_scale=gpu_mode_score_scale,
    )
    
    # Use the general single problem dataset builder for all environments
    return get_single_problem_dataset_builder(config)


def generate_run_name(
    env: str,
    problem_idx: str,
    model_name: str,
    lora_rank: int,
    learning_rate: float,
    group_size: int,
    groups_per_batch: int,
    loss_fn: str,
    seed: int,
) -> str:
    """Generate a run name from configuration parameters."""
    model_name_safe = model_name.replace("/", "-")
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    return f"{env}-{problem_idx}-{model_name_safe}-{lora_rank}rank-{learning_rate}lr-{group_size}group-{groups_per_batch}batch-{loss_fn}-seed{seed}-{timestamp}"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Ray is needed to dispatch jobs across cpus
    if cli_config.env in ("cp", "ac1", "ac2", "ale_bench", "erdos", "tictactoe"):
        import ray
        if not ray.is_initialized():
            ray.init()
        else:
            if cli_config.env != "ale_bench": 
                ray.init("auto")

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    run_name = generate_run_name(
        env=cli_config.env,
        problem_idx=cli_config.problem_idx,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        learning_rate=cli_config.learning_rate,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        loss_fn=cli_config.loss_fn,
        seed=cli_config.seed,
    )
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"./logs/{run_name}"
    log_file = os.path.join(log_path, "train.log")

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    # Create full config
    config = Config(
        env=cli_config.env,
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            dataset_path=cli_config.dataset_path,
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.local_model_path or cli_config.model_name,  # for tokenizer only
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            problem_idx=cli_config.problem_idx,
            sweep_hyperparams=cli_config.sweep_hyperparams,
            max_hyperparam_combos=cli_config.max_hyperparam_combos,
            num_cpus_per_task=cli_config.num_cpus_per_task,
            eval_timeout=cli_config.eval_timeout,
            dataset_timeout=cli_config.dataset_timeout,
            sampler_type=cli_config.sampler_type,
            initial_exp_type=cli_config.initial_exp_type,
            log_path=log_path,
            adv_estimator=cli_config.adv_estimator,
            gpu_mode_score_scale=cli_config.gpu_mode_score_scale,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        dynamic_max_tokens=cli_config.dynamic_max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        num_epochs=cli_config.num_epochs,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        test_num_rollouts=cli_config.test_num_rollouts,
        eval_BoN=cli_config.eval_BoN,
        adv_estimator=cli_config.adv_estimator,
        adv_estimator_beta=cli_config.adv_estimator_beta,
        debug_skip_llm=cli_config.debug_skip_llm,
        remove_constant_reward_groups=True,
        two_phase_sampling=cli_config.two_phase_sampling,
        phase1_max_tokens=cli_config.phase1_max_tokens,
        local_model_path=cli_config.local_model_path,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", force=True)
    logger.info("Logging to %s", log_file)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
