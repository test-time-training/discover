"""
SamplingClient: drop-in replacement for tinker.SamplingClient.

Uses vLLM's OpenAI-compatible API running on localhost for inference.
Supports LoRA adapter selection per-request and logprob computation.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from ttt_discover.tinker_compat.types import (
    ModelInput,
    SampleResult,
    SampleSequence,
    SamplingParams,
)

logger = logging.getLogger(__name__)


class SamplingClient:
    """Drop-in replacement for tinker.SamplingClient backed by vLLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "",
        lora_name: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.lora_name = lora_name

    def _effective_model(self) -> str:
        """Return the model name to use in API requests.

        If a LoRA adapter is loaded, use its name; otherwise the base model.
        """
        return self.lora_name or self.model_name

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> SampleResult:
        """Sample from the model, returning tokens + logprobs.

        Maps to vLLM's POST /v1/completions endpoint.
        """
        tokens = prompt.to_flat_tokens()

        # Build stop sequences — vLLM expects strings for stop.
        # Integer stop tokens are sent via stop_token_ids.
        stop_strings: list[str] = []
        stop_token_ids: list[int] = []
        for s in sampling_params.stop:
            if isinstance(s, int):
                stop_token_ids.append(s)
            else:
                stop_strings.append(s)

        payload: dict[str, Any] = {
            "model": self._effective_model(),
            "prompt": tokens,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "n": num_samples,
            "logprobs": True,  # vLLM: return per-token logprobs
            "echo": False,
        }
        if stop_strings:
            payload["stop"] = stop_strings
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids

        data = await self._post("/completions", payload)

        sequences: list[SampleSequence] = []
        for choice in data.get("choices", []):
            # vLLM returns token IDs in the logprobs structure
            lp_info = choice.get("logprobs", {})
            out_tokens = lp_info.get("tokens", [])
            out_token_ids = lp_info.get("token_ids", [])
            out_logprobs = lp_info.get("token_logprobs", [])

            # If token_ids aren't returned separately, fall back to text decode
            if not out_token_ids and out_tokens:
                # vLLM should return token_ids; this is a fallback
                out_token_ids = out_tokens

            # Ensure logprobs are floats (None at position 0 means prompt token)
            clean_logprobs = [
                float(lp) if lp is not None else 0.0 for lp in out_logprobs
            ]

            sequences.append(
                SampleSequence(tokens=out_token_ids, logprobs=clean_logprobs)
            )

        return SampleResult(sequences=sequences)

    # ------------------------------------------------------------------
    # Logprob computation (for KL penalty)
    # ------------------------------------------------------------------

    async def compute_logprobs_async(
        self, sequence_input: ModelInput
    ) -> list[float]:
        """Compute per-token logprobs for an existing sequence.

        Uses vLLM's prompt_logprobs feature: send the full sequence as the
        prompt with max_tokens=0 and prompt_logprobs=1. vLLM returns the
        log-probability of each prompt token given its prefix.
        """
        tokens = sequence_input.to_flat_tokens()

        payload: dict[str, Any] = {
            "model": self._effective_model(),
            "prompt": tokens,
            "max_tokens": 1,  # need at least 1 for vLLM to return prompt_logprobs
            "temperature": 0.0,
            "logprobs": True,
            "echo": True,  # include prompt tokens in output
        }

        data = await self._post("/completions", payload)

        choice = data.get("choices", [{}])[0]
        lp_info = choice.get("logprobs", {})
        all_logprobs = lp_info.get("token_logprobs", [])

        # echo=True returns logprobs for prompt + generated tokens.
        # First token has no logprob (None); trim to match input length.
        result = [float(lp) if lp is not None else 0.0 for lp in all_logprobs]

        # Trim to original sequence length (drop the 1 generated token)
        return result[: len(tokens)]

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    @staticmethod
    async def load_lora_adapter(
        vllm_url: str,
        adapter_name: str,
        adapter_path: str,
    ) -> None:
        """Load or reload a LoRA adapter in vLLM.

        Uses vLLM's POST /v1/load_lora_adapter API.
        """
        url = vllm_url.rstrip("/").replace("/v1", "") + "/v1/load_lora_adapter"
        payload = {
            "lora_name": adapter_name,
            "lora_path": adapter_path,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.info(f"Loaded LoRA adapter '{adapter_name}' from {adapter_path}")
                else:
                    text = await resp.text()
                    logger.warning(
                        f"Failed to load LoRA adapter (status {resp.status}): {text}"
                    )

    @staticmethod
    async def unload_lora_adapter(
        vllm_url: str,
        adapter_name: str,
    ) -> None:
        """Unload a LoRA adapter from vLLM."""
        url = vllm_url.rstrip("/").replace("/v1", "") + "/v1/unload_lora_adapter"
        payload = {"lora_name": adapter_name}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.info(f"Unloaded LoRA adapter '{adapter_name}'")
                else:
                    text = await resp.text()
                    logger.warning(
                        f"Failed to unload LoRA adapter (status {resp.status}): {text}"
                    )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    async def _post(self, endpoint: str, payload: dict) -> dict:
        """Send a POST request to the vLLM server."""
        url = f"{self.base_url}{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"vLLM request failed (status {resp.status}): {text}"
                    )
                return await resp.json()
