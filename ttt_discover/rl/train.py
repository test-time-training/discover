"""
Implements RL on general MDPs
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence, cast

import chz
import numpy as np
import wandb
import math
import ttt_discover.tinker_compat as tinker
import torch
from ttt_discover.tinker_compat.types import LossFnType
from ttt_discover.tinker_utils.misc_utils import get_last_checkpoint, save_checkpoint_async
from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
from ttt_discover.rl.data_processing import (
    assemble_training_data,
    remove_constant_reward_groups,
)
from ttt_discover.rl.metric_util import compute_trajectory_metrics
from ttt_discover.rl.rollouts import do_group_rollout
from ttt_discover.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from ttt_discover.tinker_utils.misc_utils import Tokenizer
from ttt_discover.tinker_utils import ml_log
from ttt_discover.tinker_utils.misc_utils import safezip, split_list, timed, all_same
from ttt_discover.tinker_utils.trace import scope, get_scope_context
from ttt_discover.tinker_utils.ml_log import WandbLogger


logger = logging.getLogger(__name__)


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by logp_base - logp_current - avg_kl,
    where avg_kl is the average of logp_base - logp_current (which is -KL[current, base])
    """
    # Compute logprobs at all data items
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    # compute the logprob differences, zeroed out when the mask == 0
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    logprob_diffs = [
        (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
        for base_logprobs, sampled_logprobs, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
        [mask.sum() for mask in float_masks]
    )
    for i, datum in enumerate(data_D):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )
    return {"kl_policy_base": float(avg_logp_diff)}


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup], adv_estimator: str, adv_estimator_beta: float, adv_estimator_mu: float = 5/1.503163635, adv_estimator_sigma: float = 0.000001) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        if adv_estimator == "mean_baseline":
            advantages_G = rewards_G - rewards_G.mean()
        elif adv_estimator == "entropic":
            beta = adv_estimator_beta
            s_safe = rewards_G - rewards_G.max(dim=-1, keepdim=True)[0]
            e = torch.exp(beta * s_safe)
            k = e.shape[0]
            if k == 1:
                Z = e 
            else:
                Z = (e.sum() - e) / (k - 1)
            w = e / (Z + 1e-12)
            advantages_G = w - 1.0
        elif adv_estimator == "entropic_adaptive_beta":
            delta = np.log(2)
            beta_max = 1e6
            iters = 60
            eps = 1e-12

            r = rewards_G.float()
            k = r.shape[0]

            if k < 2:
                beta = r.new_tensor(0.0)
            else:
                logK = math.log(k)

                def kl_hat(beta_scalar: float) -> float:
                    # q_beta over samples: q ∝ exp(beta * r), KL(q||uniform)
                    b = r.new_tensor(beta_scalar)
                    logits = b * (r - r.max(dim=0, keepdim=True).values)      # stable
                    logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                    q = torch.exp(logq)
                    kl = (q * (logq + logK)).sum(dim=0)   
                    return float(kl.mean().item())    

                lo, hi = 0.0, 1.0
                if kl_hat(hi) < delta:
                    while hi < beta_max and kl_hat(hi) < delta:
                        hi *= 2.0
                    if kl_hat(hi) < delta:
                        beta = r.new_tensor(hi)  # best effort
                    else:
                        beta = None
                else:
                    beta = None

                if beta is None:
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if kl_hat(mid) < delta:
                            lo = mid
                        else:
                            hi = mid
                    beta = r.new_tensor(hi)

            # LOO entropic advantages using solved beta
            e = torch.exp(beta * (r - r.max(dim=0, keepdim=True).values))

            if k == 1:
                Z = e
            else:
                Z = (e.sum(dim=0, keepdim=True) - e) / (k - 1)

            w = e / (Z + eps)
            advantages_G = w - 1.0
        else:
            raise ValueError(f"Invalid advantage estimator: {adv_estimator}")
        advantages_P.append(advantages_G)

    return advantages_P


@scope
async def enqueue_optim_step(
    training_client: tinker.TrainingClient,
    learning_rate: float,
) -> tinker.APIFuture[tinker.OptimStepResponse]:
    """Enqueue an optimizer step and return the future"""
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)
    return optim_step_future


@scope
async def consume_optim_step(
    optim_step_future: tinker.APIFuture[tinker.OptimStepResponse],
) -> tinker.OptimStepResponse:
    """Apply the accumulated gradients to update the model weights and return the result"""
    return await optim_step_future.result_async()


@scope
def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


@scope
async def enqueue_forward_backward(
    training_client: tinker.TrainingClient,
    batch_d: List[tinker.Datum],
    loss_fn: LossFnType,
) -> tinker.APIFuture[tinker.ForwardBackwardOutput]:
    """Enqueue a forward-backward pass for a minibatch of data and return the future"""
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(remove_mask, batch_d)), loss_fn=loss_fn
    )
    return fwd_bwd_future


@scope
async def consume_forward_backward(
    fwd_bwd_future: tinker.APIFuture[tinker.ForwardBackwardOutput],
) -> List[torch.Tensor]:
    """Consume the result of a forward-backward pass and return the training logprobs"""
    fwd_bwd_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D: list[torch.Tensor] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)

    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D


@scope
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []

    if len(batches_md) == 0:
        return training_logprobs_D

    enqueued_futures: (
        tuple[
            tinker.APIFuture[tinker.ForwardBackwardOutput],
            tinker.APIFuture[tinker.OptimStepResponse],
        ]
        | None
    ) = (
        await enqueue_forward_backward(training_client, batches_md[0], loss_fn),
        await enqueue_optim_step(training_client, learning_rate),
    )

    for i in range(len(batches_md)):
        assert enqueued_futures is not None

        fwd_bwd_future, optim_step_future = enqueued_futures
        enqueued_futures = None

        # Enqueue the next forward-backward pass and optimizer step before consuming the current result
        if i != len(batches_md) - 1:
            assert enqueued_futures is None
            enqueued_futures = (
                await enqueue_forward_backward(training_client, batches_md[i + 1], loss_fn),
                await enqueue_optim_step(training_client, learning_rate),
            )

        training_logprobs = await consume_forward_backward(fwd_bwd_future)
        training_logprobs_D.extend(training_logprobs)

        await consume_optim_step(optim_step_future)

    assert enqueued_futures is None

    return training_logprobs_D


@chz.chz
class Config:
    env_type: type  # Environment type (from registry or passed directly for custom envs)
    problem_type: str
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    num_epochs: int = 1
    temperature: float = 1.0  # Changing sampling temperature is not generally recommended; does not currently play well with KL penalty
    lora_rank: int = 32
    adv_estimator: str="entropic_adaptive_beta"
    adv_estimator_beta: float = 2.0

    kl_penalty_coef: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: LossFnType = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    save_every: int = 20  # 0 = disabled
    load_checkpoint_path: str | None = None

    # Two-phase sampling: phase1_max_tokens for token completion
    phase1_max_tokens: int = 26000

    # Local model path (avoids HuggingFace API rate limits)
    local_model_path: str | None = None

    # Infrastructure config (passed to ServiceClient)
    vllm_url: str = "http://localhost:8000/v1"
    checkpoint_dir: str = "./checkpoints"
    training_device: str = "cuda:1"


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    step_idx=-1,
    model_name: str = "",
    phase1_max_tokens: int = 27000,
) -> TrajectoryGroup | None:
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    
    policy = TwoPhaseTokenCompleter(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        phase1_max_tokens=phase1_max_tokens,
        temperature=temperature,
    )

    trajectory_group = await do_group_rollout(env_group_builder, policy, step_idx)

    # Remove if all trajectories have the same reward
    if do_remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
        return None
    else:
        return trajectory_group


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    log_path: str | None = None,
    train_step: int | None = None,
    adv_estimator: str="mean_baseline",
    adv_estimator_beta: float = 2.0,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P, adv_estimator, adv_estimator_beta=adv_estimator_beta)
        if advantages_P:
            flat_adv = torch.cat(advantages_P)
            metrics.update(
                {
                    "advantage/mean": flat_adv.mean().item(),
                    "advantage/min": flat_adv.min().item(),
                    "advantage/max": flat_adv.max().item(),
                }
            )
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        log_path=cfg.log_path,
        train_step=i_batch,
        adv_estimator=cfg.adv_estimator,
        adv_estimator_beta=cfg.adv_estimator_beta,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    sampling_client, full_batch_metrics = await save_checkpoint_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        cfg.log_path,
        cfg.save_every,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""
    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")

    # Initial sampling client
    print("Get sampling client...")
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    for i_batch in range(start_batch, end_batch):
        train_table = None
        test_table = None
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Make sure we are clearing out existing entrees
        # in case of resuming from previous checkpoints
        from ttt_discover.tinker_utils.best_sequence_utils import get_best_bound_path, clear_step_entry
        best_seq_path = get_best_bound_path(cfg.log_path)
        clear_step_entry(best_seq_path, i_batch)

        # Get batch and sample trajectories
        print("Load dataset batch...")
        dataset_batch_idx = i_batch % num_batches_per_epoch
        env_group_builders_P = dataset.get_batch(dataset_batch_idx)

        # Log sampler stats if available (PER sampler)
        print("Log sampler stats...")
        sampler_table_columns, sampler_table_data = None, None
        if hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'get_sample_stats'):
            sampler_stats = dataset.sampler.get_sample_stats()
            metrics.update(sampler_stats)
            if hasattr(dataset.sampler, 'get_sample_table'):
                sampler_table_columns, sampler_table_data = dataset.sampler.get_sample_table()


        print("Sampling...")
        with timed("sampling", metrics):
            # Note: do_remove_constant_reward_groups=False here because we remove
            # constant reward groups after all rollouts are collected (below)
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=False,
                            step_idx=i_batch,
                            model_name=cfg.local_model_path or cfg.model_name,
                            phase1_max_tokens=cfg.phase1_max_tokens,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )

        if hasattr(dataset, 'flush'):
            dataset.flush(step=i_batch + 1)

        if cfg.remove_constant_reward_groups:
            trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

        # Train step
        print("Training...")
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )

        if 'table' in train_step_metrics:
            table_data = train_step_metrics.pop('table')
            # Compute actual advantages using the configured estimator
            advantages_P = compute_advantages(trajectory_groups_P, cfg.adv_estimator, cfg.adv_estimator_beta)
            flat_advantages = [adv.item() for adv_G in advantages_P for adv in adv_G]
            table_data = [(*row, flat_advantages[i]) for i, row in enumerate(table_data)]
            train_table = {
                f"gen&score_train_{i_batch}":
                    wandb.Table(
                        columns=[
                            "Prompt", "Gen Sequence", "Reward", "Correctness", "Gen Sequence PostProc", "Message", "Initial Raw Score", "Advantage"
                        ],
                        data=table_data
                    )
            }
        
        if len(ml_logger.loggers) >= 2:
            if train_table is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics(train_table, step=i_batch)
            if test_table is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics(test_table, step=i_batch)
            if sampler_table_data is not None and isinstance(ml_logger.loggers[2], WandbLogger):
                ml_logger.loggers[2].log_metrics({
                    f"sampler_states_{i_batch}": wandb.Table(
                        columns=sampler_table_columns,
                        data=sampler_table_data
                    )
                }, step=i_batch)

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    if cfg.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    resume_info = get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    print("Create training client...")
    service_client = tinker.ServiceClient(
        base_url=None,
        vllm_url=cfg.vllm_url,
        checkpoint_dir=cfg.checkpoint_dir,
        training_device=cfg.training_device,
    )
    print("Training client created!")
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    # Get tokenizer (use local path if provided, otherwise from training client)
    if cfg.local_model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.local_model_path, use_fast=True)
    else:
        tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    print("Create dataset...")
    dataset = await cfg.dataset_builder()
    print("Dataset created!")
    
    # If resuming from step > 0, reload sampler from the correct checkpoint step
    if resume_info and start_batch > 0 and hasattr(dataset, 'sampler') and hasattr(dataset.sampler, 'reload_from_step'):
        logger.info(f"Reloading sampler state from step {start_batch}")
        dataset.sampler.reload_from_step(start_batch)
    
    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")
    num_batches_total = num_batches_per_epoch * cfg.num_epochs
    logger.info(
        f"Will train for {cfg.num_epochs} epoch(s) x {num_batches_per_epoch} batches = {num_batches_total} steps"
    )

    # Training loop
    print("Training loop...")
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches_total,
        num_batches=num_batches_total,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches_total:
        _ = await save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches_total},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
