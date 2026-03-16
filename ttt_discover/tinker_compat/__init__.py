"""
tinker_compat: Drop-in replacement for the tinker package.

Import this module as ``import ttt_discover.tinker_compat as tinker`` to
replace all ``import tinker`` occurrences without changing any other code.

Public API mirrors tinker exactly:
  - ModelInput, ModelInputChunk, EncodedTextChunk
  - Datum, TensorData
  - SamplingParams, AdamParams
  - LossFnType
  - APIFuture
  - SampleResult, SampleSequence
  - ForwardBackwardOutput, OptimStepResponse, SaveResult
  - SamplingClient
  - TrainingClient
  - ServiceClient
"""

from ttt_discover.tinker_compat.types import (
    AdamParams,
    APIFuture,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LossFnType,
    ModelInput,
    ModelInputChunk,
    OptimStepResponse,
    SampleResult,
    SampleSequence,
    SamplingParams,
    SaveResult,
    TensorData,
)
from ttt_discover.tinker_compat.sampling import SamplingClient
from ttt_discover.tinker_compat.training import TrainingClient
from ttt_discover.tinker_compat.service import ServiceClient

__all__ = [
    "AdamParams",
    "APIFuture",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LossFnType",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepResponse",
    "SampleResult",
    "SampleSequence",
    "SamplingClient",
    "SamplingParams",
    "SaveResult",
    "ServiceClient",
    "TensorData",
    "TrainingClient",
]
