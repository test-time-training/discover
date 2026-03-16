"""
Drop-in replacements for tinker data types.

Provides ModelInput, EncodedTextChunk, Datum, TensorData, SamplingParams,
AdamParams, APIFuture, and result types that match tinker's API surface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar

import torch

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Loss function type
# ---------------------------------------------------------------------------

LossFnType: TypeAlias = Literal["importance_sampling", "ppo"]

# ---------------------------------------------------------------------------
# Model input chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedTextChunk:
    """A chunk of pre-tokenized text."""

    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


# ModelInputChunk is just EncodedTextChunk for now (tinker also supports
# image chunks but TTT-Discover only uses text).
ModelInputChunk: TypeAlias = EncodedTextChunk


@dataclass(frozen=True)
class ModelInput:
    """Immutable sequence of token chunks, matching tinker.ModelInput."""

    chunks: list[ModelInputChunk] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)

    def append_int(self, token: int) -> "ModelInput":
        """Return a *new* ModelInput with one token appended."""
        if self.chunks and isinstance(self.chunks[-1], EncodedTextChunk):
            new_last = EncodedTextChunk(tokens=self.chunks[-1].tokens + [token])
            return ModelInput(chunks=list(self.chunks[:-1]) + [new_last])
        return ModelInput(chunks=list(self.chunks) + [EncodedTextChunk(tokens=[token])])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def to_flat_tokens(self) -> list[int]:
        """Flatten all chunks into a single token list."""
        tokens: list[int] = []
        for chunk in self.chunks:
            tokens.extend(chunk.tokens)
        return tokens


# ---------------------------------------------------------------------------
# Tensor wrapper
# ---------------------------------------------------------------------------


class TensorData:
    """Thin wrapper around a torch.Tensor, matching tinker.TensorData."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @staticmethod
    def from_torch(tensor: torch.Tensor) -> "TensorData":
        return TensorData(tensor)

    def to_torch(self) -> torch.Tensor:
        return self._tensor

    @property
    def data(self) -> torch.Tensor:
        return self._tensor


# ---------------------------------------------------------------------------
# Training datum
# ---------------------------------------------------------------------------


@dataclass
class Datum:
    """A single training example, matching tinker.Datum."""

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


# ---------------------------------------------------------------------------
# Sampling config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingParams:
    """Parameters for LLM sampling, matching tinker.SamplingParams."""

    stop: list[str] | list[int] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdamParams:
    """Adam optimizer parameters, matching tinker.AdamParams."""

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Async future
# ---------------------------------------------------------------------------


class APIFuture(Generic[T]):
    """Wraps an asyncio-awaitable value, matching tinker.APIFuture[T]."""

    def __init__(self, value: T):
        self._value = value

    async def result_async(self) -> T:
        return self._value

    @classmethod
    def from_value(cls, value: T) -> "APIFuture[T]":
        return cls(value)


# ---------------------------------------------------------------------------
# Sampling result types
# ---------------------------------------------------------------------------


@dataclass
class SampleSequence:
    """One sampled sequence from the model."""

    tokens: list[int]
    logprobs: list[float] | None = None


@dataclass
class SampleResult:
    """Result from SamplingClient.sample_async()."""

    sequences: list[SampleSequence]


# ---------------------------------------------------------------------------
# Forward-backward result types
# ---------------------------------------------------------------------------


@dataclass
class ForwardBackwardOutput:
    """Result from TrainingClient.forward_backward_async()."""

    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimStepResponse:
    """Result from TrainingClient.optim_step_async()."""

    pass


# ---------------------------------------------------------------------------
# Checkpoint save result
# ---------------------------------------------------------------------------


@dataclass
class SaveResult:
    """Result from save_state_async / save_weights_for_sampler_async."""

    path: str
