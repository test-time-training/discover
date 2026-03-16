"""
TrainingClient: drop-in replacement for tinker.TrainingClient.

Uses HuggingFace TRL's GRPOTrainer for LoRA fine-tuning via PEFT.
GRPOTrainer handles loss computation (GRPO/IS/PPO), optimizer management,
gradient accumulation, and mixed-precision — all battle-tested by HuggingFace.

Two usage modes:
  1. Backward-compatible: forward_backward_async() + optim_step_async()
     (called by the existing train.py loop)
  2. Full loop: train_full() delegates to GRPOTrainer.train()
     (for future refactor where GRPOTrainer drives the outer loop)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ttt_discover.tinker_compat.types import (
    APIFuture,
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LossFnType,
    OptimStepResponse,
    SaveResult,
    TensorData,
)

if TYPE_CHECKING:
    from trl import GRPOTrainer
    from ttt_discover.tinker_compat.sampling import SamplingClient

logger = logging.getLogger(__name__)


def _datum_batch_to_grpo_inputs(
    data: list[Datum],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a Datum list to the tensor format used by TRL's GRPOTrainer.

    GRPOTrainer's compute_loss expects:
      - prompt_ids            (B, prompt_len)
      - completion_ids        (B, completion_len)
      - prompt_mask           (B, prompt_len)        — 1 for real tokens
      - completion_mask       (B, completion_len)    — 1 for completion, 0 for pad
      - advantages            (B,)                   — scalar per sample
      - old_per_token_logps   (B, completion_len)    — log-probs from sampling policy

    Our Datum has a right-shifted model_input and left-shifted targets.
    We reconstruct (prompt | completion) by splitting on mask==0/1.
    """
    prompt_ids_list: list[torch.Tensor] = []
    completion_ids_list: list[torch.Tensor] = []
    completion_mask_list: list[torch.Tensor] = []
    advantages_list: list[float] = []
    old_logps_list: list[torch.Tensor] = []

    for datum in data:
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().long()
        advantages_t = datum.loss_fn_inputs["advantages"].to_torch().float()
        old_logps_t = datum.loss_fn_inputs["logprobs"].to_torch().float()
        mask_field = datum.loss_fn_inputs.get("mask")
        comp_mask = (
            mask_field.to_torch().float()
            if mask_field is not None
            else torch.ones_like(advantages_t)
        )

        # Split target tokens into prompt (mask==0) and completion (mask==1)
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        comp_mask_vals: list[float] = []
        old_lp_vals: list[float] = []

        for tid, m, lp in zip(
            target_tokens.tolist(), comp_mask.tolist(), old_logps_t.tolist()
        ):
            if m > 0.5:
                completion_tokens.append(int(tid))
                comp_mask_vals.append(1.0)
                old_lp_vals.append(float(lp))
            else:
                prompt_tokens.append(int(tid))

        # Scalar advantage: masked mean over completion positions
        masked_adv = (advantages_t * comp_mask).sum() / comp_mask.sum().clamp(min=1.0)

        prompt_ids_list.append(torch.tensor(prompt_tokens, dtype=torch.long))
        completion_ids_list.append(torch.tensor(completion_tokens, dtype=torch.long))
        completion_mask_list.append(torch.tensor(comp_mask_vals, dtype=torch.float))
        advantages_list.append(float(masked_adv.item()))
        old_logps_list.append(torch.tensor(old_lp_vals, dtype=torch.float))

    def _pad_long(tensors: list[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
        max_len = max(t.shape[0] for t in tensors)
        return torch.stack(
            [F.pad(t.long(), (0, max_len - t.shape[0]), value=pad_val) for t in tensors]
        ).to(device)

    def _pad_float(tensors: list[torch.Tensor], pad_val: float = 0.0) -> torch.Tensor:
        max_len = max(t.shape[0] for t in tensors)
        return torch.stack(
            [F.pad(t.float(), (0, max_len - t.shape[0]), value=pad_val) for t in tensors]
        ).to(device)

    prompt_ids = _pad_long(prompt_ids_list)
    completion_ids = _pad_long(completion_ids_list)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "prompt_mask": (prompt_ids != 0).long(),
        "completion_mask": _pad_float(completion_mask_list),
        "advantages": torch.tensor(advantages_list, dtype=torch.float, device=device),
        "old_per_token_logps": _pad_float(old_logps_list),
    }


class TrainingClient:
    """Drop-in replacement for tinker.TrainingClient, backed by TRL's GRPOTrainer.

    Uses GRPOTrainer for:
    - GRPO/IS/PPO loss computation (battle-tested by HuggingFace)
    - Optimizer and LR management via Accelerate
    - Mixed precision (bf16) training
    - PEFT LoRA handling

    The existing train.py loop calls forward_backward_async + optim_step_async.
    These delegate to GRPOTrainer's compute_loss and optimizer.
    """

    def __init__(
        self,
        trainer: "GRPOTrainer",
        tokenizer,
        checkpoint_dir: str,
        vllm_url: str = "http://localhost:8000/v1",
        model_name: str = "",
        _lora_adapter_counter: int = 0,
    ):
        self.trainer = trainer
        self.model = trainer.model
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vllm_url = vllm_url
        self.model_name = model_name
        self._lora_adapter_counter = _lora_adapter_counter

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Backward-compatible API (called by existing train.py loop)
    # ------------------------------------------------------------------

    async def forward_backward_async(
        self,
        batch: list[Datum],
        loss_fn: LossFnType = "importance_sampling",
    ) -> APIFuture[ForwardBackwardOutput]:
        """Compute loss via TRL's GRPOTrainer.compute_loss and accumulate grads."""
        inputs = _datum_batch_to_grpo_inputs(batch, self.device)

        # GRPOTrainer.compute_loss handles bf16 autocast via Accelerate
        loss = self.trainer.compute_loss(self.model, inputs)

        if hasattr(self.trainer, "accelerator"):
            self.trainer.accelerator.backward(loss)
        else:
            loss.backward()

        # Compute per-datum logprobs for logging/metrics
        with torch.no_grad():
            full_ids = torch.cat(
                [inputs["prompt_ids"], inputs["completion_ids"]], dim=1
            )
            full_mask = torch.cat(
                [inputs["prompt_mask"], inputs["completion_mask"].long()], dim=1
            )
            out = self.model(input_ids=full_ids, attention_mask=full_mask)
            # Logits shifted: position i predicts token i+1
            prompt_len = inputs["prompt_ids"].shape[1]
            logits_comp = out.logits[:, prompt_len - 1 : -1]
            log_probs = F.log_softmax(logits_comp, dim=-1)
            new_logps = log_probs.gather(
                dim=-1,
                index=inputs["completion_ids"].unsqueeze(-1),
            ).squeeze(-1)  # (B, completion_len)

        comp_mask = inputs["completion_mask"]
        loss_fn_outputs = []
        for i in range(len(batch)):
            lp = new_logps[i][comp_mask[i].bool()].detach().cpu()
            loss_fn_outputs.append({"logprobs": TensorData.from_torch(lp)})

        return APIFuture.from_value(
            ForwardBackwardOutput(
                loss_fn_outputs=loss_fn_outputs,
                metrics={"loss": loss.item()},
            )
        )

    async def optim_step_async(
        self, adam_params: AdamParams
    ) -> APIFuture[OptimStepResponse]:
        """Apply accumulated gradients via the trainer's optimizer."""
        optimizer = self.trainer.optimizer
        for pg in optimizer.param_groups:
            pg["lr"] = adam_params.learning_rate
        optimizer.step()
        optimizer.zero_grad()
        return APIFuture.from_value(OptimStepResponse())

    # ------------------------------------------------------------------
    # Full-loop API (for future use when GRPOTrainer drives the outer loop)
    # ------------------------------------------------------------------

    def train_full(self) -> None:
        """Run GRPOTrainer.train() — handles the complete training loop."""
        self.trainer.train()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    async def save_state_async(self, name: str) -> APIFuture[SaveResult]:
        """Save LoRA weights + optimizer state for full resumption."""
        save_dir = self.checkpoint_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # TRL's save_model saves the PEFT adapter in HF format (vLLM-compatible)
        lora_dir = save_dir / "lora"
        self.trainer.save_model(str(lora_dir))

        torch.save(
            self.trainer.optimizer.state_dict(),
            str(save_dir / "optimizer.pt"),
        )

        return APIFuture.from_value(SaveResult(path=str(save_dir)))

    async def save_weights_for_sampler_async(self, name: str) -> APIFuture[SaveResult]:
        """Save LoRA adapter weights only (no optimizer) for vLLM to load."""
        save_dir = self.checkpoint_dir / f"{name}_sampler"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(save_dir))
        return APIFuture.from_value(SaveResult(path=str(save_dir)))

    # ------------------------------------------------------------------
    # Sampling client creation (LoRA hot-swap via vLLM)
    # ------------------------------------------------------------------

    async def save_weights_and_get_sampling_client_async(self) -> "SamplingClient":
        """Save current LoRA weights and return a SamplingClient using them."""
        from ttt_discover.tinker_compat.sampling import SamplingClient

        self._lora_adapter_counter += 1
        adapter_name = f"lora_step_{self._lora_adapter_counter}"
        save_dir = self.checkpoint_dir / "live_adapter"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(save_dir))

        await SamplingClient.load_lora_adapter(
            self.vllm_url,
            adapter_name=adapter_name,
            adapter_path=str(save_dir),
        )

        return SamplingClient(
            base_url=self.vllm_url,
            model_name=self.model_name,
            lora_name=adapter_name,
        )

    def create_sampling_client(self, path: str) -> "SamplingClient":
        """Create a SamplingClient from a saved checkpoint path."""
        from ttt_discover.tinker_compat.sampling import SamplingClient
        import asyncio

        self._lora_adapter_counter += 1
        adapter_name = f"lora_ckpt_{self._lora_adapter_counter}"

        coro = SamplingClient.load_lora_adapter(
            self.vllm_url,
            adapter_name=adapter_name,
            adapter_path=path,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

        return SamplingClient(
            base_url=self.vllm_url,
            model_name=self.model_name,
            lora_name=adapter_name,
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def get_tokenizer(self):
        return self.tokenizer
