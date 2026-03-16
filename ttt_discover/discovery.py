import asyncio
import logging
import os
from typing import Literal

import chz
from ttt_discover.tinker_utils.dataset_builder import Environment
from ttt_discover.environments.utils.cpu_scheduler import CpuScheduler
import ttt_discover.tinker_utils.misc_utils as misc_utils
from ttt_discover.rl.train import Config, main
from ttt_discover.tinker_utils.dataset_builder import DatasetConfig, get_single_problem_dataset_builder

logger = logging.getLogger(__name__)


@chz.chz
class DiscoverConfig:
    """Simple config for discovery with RL training."""

    # Model config
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = "qwen3"
    save_every: int = 2

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 4e-5
    num_epochs: int = 50
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000  # Two-phase sampling: total prompt + thinking token budget

    # Infrastructure config
    vllm_url: str = "http://localhost:8000/v1"
    checkpoint_dir: str = "./checkpoints"
    training_device: str = "cuda:1"

    # Misc config
    experiment_name: str | None = None
    wandb_project: str | None = "tinker-cookbook"

    # Environment-specific
    env_type: str = Environment
    problem_type: str = "26"
    num_cpus_per_task: int = 0
    eval_timeout: int = 1000


def init_ray(num_cpus_per_task: int, env_type: str):
    import ray

    if not ray.is_initialized():
        ray.init()
    else:
        if env_type.__name__ != "AhcEnv":
            ray.init("auto")

    try:
        # Try to get existing actor by name
        _scheduler = ray.get_actor("cpu_scheduler")
        print("Found existing cpu_scheduler actor.")
    except ValueError:
        # If not found, create a new one
        print("Creating new cpu_scheduler actor.")
        _scheduler = CpuScheduler.options(
            name="cpu_scheduler",
            lifetime="detached",
        ).remote(
            num_cpus_per_task=num_cpus_per_task,
            num_persistent_workers=0,
        )


async def discover_impl(config: DiscoverConfig):
    """Convert discover config to full config and run training."""

    # Ray is needed to dispatch jobs across cpus
    if config.num_cpus_per_task > 0:
        init_ray(config.num_cpus_per_task, config.env_type)

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())

    renderer_name = config.renderer_name

    # create log path if it doesn't exist
    log_path = f"./tinker_log/{config.experiment_name}"
    log_file = os.path.join(log_path, "train.log")

    # Resolve env_name -> env type and build dataset
    dataset_config = DatasetConfig(
        env_type=config.env_type,
        problem_type=config.problem_type,
        batch_size=config.groups_per_batch,
        group_size=config.group_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        num_cpus_per_task=config.num_cpus_per_task,
        eval_timeout=config.eval_timeout,
        log_path=log_path,
    )
    dataset_builder = get_single_problem_dataset_builder(dataset_config)

    rl_config = Config(
        env_type=dataset_config.env_type,
        problem_type=config.problem_type,
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=config.experiment_name,
        log_path=log_path,
        load_checkpoint_path=None,
        kl_penalty_coef=config.kl_penalty_coef,
        num_substeps=1,
        save_every=config.save_every,
        num_epochs=config.num_epochs,
        loss_fn="importance_sampling",
        adv_estimator="entropic_adaptive_beta",
        adv_estimator_beta=2.0, # Unused with entropic_adaptive_beta
        remove_constant_reward_groups=True,
        phase1_max_tokens=config.phase1_max_tokens,
        local_model_path=None,
        vllm_url=config.vllm_url,
        checkpoint_dir=config.checkpoint_dir,
        training_device=config.training_device,
    )

    misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", force=True)
    logger.info("Logging to %s", log_file)

    # Run training
    await main(rl_config)
    
def discover(config: DiscoverConfig):
    asyncio.run(discover_impl(config))
