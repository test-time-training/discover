"""
ServiceClient: drop-in replacement for tinker.ServiceClient.

Factory for creating TrainingClient (backed by TRL GRPOTrainer + PEFT) and
SamplingClient (backed by vLLM).  GRPOTrainer handles model loading, LoRA setup,
optimizer creation, and all training mechanics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ttt_discover.tinker_compat.sampling import SamplingClient
from ttt_discover.tinker_compat.training import TrainingClient

logger = logging.getLogger(__name__)

# Default LoRA target modules for decoder-only transformers
# (Qwen3, Llama-3, Mistral, Gemma, etc.)
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class ServiceClient:
    """Drop-in replacement for tinker.ServiceClient.

    Creates TrainingClient via TRL's GRPOTrainer (handles LoRA, optimizer,
    and loss computation) and SamplingClient via vLLM's HTTP API.
    """

    def __init__(
        self,
        base_url: str | None = None,
        vllm_url: str = "http://localhost:8000/v1",
        checkpoint_dir: str = "./checkpoints",
        training_device: str = "cuda:1",
    ):
        # base_url accepted for API compat with tinker but unused
        self.vllm_url = vllm_url
        self.checkpoint_dir = checkpoint_dir
        self.training_device = training_device
        self._model_name: str = ""

    # ------------------------------------------------------------------
    # Training client creation
    # ------------------------------------------------------------------

    async def create_lora_training_client_async(
        self,
        model_name: str,
        rank: int = 32,
    ) -> TrainingClient:
        """Load base model, apply LoRA via GRPOTrainer, return TrainingClient."""
        from peft import LoraConfig, TaskType
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        self._model_name = model_name
        logger.info(f"Creating GRPOTrainer for model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            target_modules=DEFAULT_LORA_TARGET_MODULES,
            bias="none",
        )

        # GRPOConfig: minimal settings — train.py overrides lr via optim_step_async.
        # We disable GRPOTrainer's own generation/reward pipeline since train.py
        # drives generation via TwoPhaseTokenCompleter.
        grpo_config = GRPOConfig(
            output_dir=self.checkpoint_dir,
            bf16=True,
            num_generations=1,         # we don't use trainer.train()
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=4e-5,        # overridden per step by optim_step_async
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to=[],              # we handle logging via our own wandb setup
            logging_steps=9_999_999,
            save_steps=9_999_999,
        )

        dummy_dataset = _make_dummy_dataset()
        trainer = GRPOTrainer(
            model=model_name,
            args=grpo_config,
            peft_config=lora_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
        )

        logger.info("GRPOTrainer ready.")
        trainer.model.print_trainable_parameters()

        return TrainingClient(
            trainer=trainer,
            tokenizer=tokenizer,
            checkpoint_dir=self.checkpoint_dir,
            vllm_url=self.vllm_url,
            model_name=model_name,
        )

    async def create_training_client_from_state_async(
        self,
        state_path: str,
    ) -> TrainingClient:
        """Load base model + LoRA from checkpoint, fresh optimizer."""
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        state_dir = Path(state_path)
        lora_dir = state_dir / "lora"

        peft_cfg = PeftConfig.from_pretrained(str(lora_dir))
        base_model_name = peft_cfg.base_model_name_or_path
        self._model_name = base_model_name

        logger.info(f"Loading checkpoint from {state_path} (base: {base_model_name})")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        grpo_config = GRPOConfig(
            output_dir=self.checkpoint_dir,
            bf16=True,
            num_generations=1,
            per_device_train_batch_size=1,
            learning_rate=4e-5,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to=[],
            logging_steps=9_999_999,
            save_steps=9_999_999,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="bfloat16",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(lora_dir), is_trainable=True)

        dummy_dataset = _make_dummy_dataset()
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dummy_dataset,
            processing_class=tokenizer,
        )

        return TrainingClient(
            trainer=trainer,
            tokenizer=tokenizer,
            checkpoint_dir=self.checkpoint_dir,
            vllm_url=self.vllm_url,
            model_name=base_model_name,
        )

    async def create_training_client_from_state_with_optimizer_async(
        self,
        state_path: str,
    ) -> TrainingClient:
        """Load base model + LoRA + optimizer state for full resumption."""
        import torch

        client = await self.create_training_client_from_state_async(state_path)

        optimizer_path = Path(state_path) / "optimizer.pt"
        if optimizer_path.exists():
            logger.info(f"Restoring optimizer state from {optimizer_path}")
            state_dict = torch.load(str(optimizer_path), map_location="cpu")
            client.trainer.optimizer.load_state_dict(state_dict)
        else:
            logger.warning(
                f"No optimizer state at {optimizer_path}; using fresh optimizer"
            )

        return client

    # ------------------------------------------------------------------
    # Sampling client creation
    # ------------------------------------------------------------------

    def create_sampling_client(self, base_model: str | None = None) -> SamplingClient:
        """Create a SamplingClient pointing at the vLLM base model (no LoRA)."""
        return SamplingClient(
            base_url=self.vllm_url,
            model_name=base_model or self._model_name,
            lora_name=None,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dummy_dataset():
    """Return a minimal dataset so GRPOTrainer constructor succeeds.

    train.py drives the loop via forward_backward_async, so this dataset is
    never actually iterated — it just satisfies the constructor requirement.
    """
    try:
        from datasets import Dataset
        return Dataset.from_dict({"prompt": ["dummy"]})
    except ImportError:
        return [{"prompt": "dummy"}]
