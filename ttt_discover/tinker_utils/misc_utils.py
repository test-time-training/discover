"""
Small utilities requiring only basic python libraries.
"""

import os
import logging
import time
from contextlib import contextmanager
import shutil
from typing import Any, Sequence, TypeVar, cast, Literal

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))


def dict_mean(list_of_dicts: list[dict[str, float | int | str]]) -> dict[str, float | str]:
    key2values = {}
    
    for d in list_of_dicts:
        for k, v in d.items():
            key2values.setdefault(k, []).append(v)
    
    result = {}
    for k, values in key2values.items():
        # Find first non-None value to determine type
        first_val = next((v for v in values if v is not None), None)
        if first_val is None:
            continue  # All values are None, skip
        if isinstance(first_val, str):
            result[k] = first_val
        else:
            arr = [v for v in values if isinstance(v, (int, float))]
            if arr:
                result[k] = float(np.mean(arr))
                result[f"{k}/min"] = float(np.min(arr))
                result[f"{k}/max"] = float(np.max(arr))
    
    return result


def all_same(xs: list[Any]) -> bool:
    return all(x == xs[0] for x in xs)


def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    """
    Split a sequence into a list of lists, where the sizes are as equal as possible,
    and the long and short lists are as uniformly distributed as possible.

    Args:
        lst: The sequence to split
        num_splits: Number of sublists to create

    Returns:
        A list of sublists with sizes differing by at most 1

    Raises:
        ValueError: If num_splits > len(lst) or num_splits <= 0

    Examples:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2, 3], [4, 5]]
        >>> split_list([1, 2, 3, 4, 5], 3)
        [[1, 2], [3, 4], [5]]
    """
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_splits > len(lst):
        raise ValueError(f"Cannot split list of length {len(lst)} into {num_splits} parts")

    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]


LogdirBehavior = Literal["delete", "resume", "ask", "raise"]


def check_log_dir(log_dir: str, behavior_if_exists: LogdirBehavior):
    """
    Call this at the beginning of CLI entrypoint to training scripts. This handles
    cases that occur if we're trying to log to a directory that already exists.
    The user might want to resume, overwrite, or delete it.

    Args:
        log_dir: The directory to check.
        behavior_if_exists: What to do if the log directory already exists.

        "ask": Ask user if they want to delete the log directory.
        "resume": Continue to the training loop, which means we'll try to resume from the last checkpoint.
        "delete": Delete the log directory and start logging there.
        "raise": Raise an error if the log directory already exists.

    Returns:
        None
    """
    if os.path.exists(log_dir):
        if behavior_if_exists == "delete":
            logger.info(
                f"Log directory {log_dir} already exists. Will delete it and start logging there."
            )
            shutil.rmtree(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"Log directory {log_dir} already exists. What do you want to do? [delete, resume, exit]: "
                )
                if user_input == "delete":
                    shutil.rmtree(log_dir)
                    return
                elif user_input == "resume":
                    return
                elif user_input == "exit":
                    exit(0)
                else:
                    logger.warning(
                        f"Invalid input: {user_input}. Please enter 'delete', 'resume', or 'exit'."
                    )
        elif behavior_if_exists == "resume":
            return
        elif behavior_if_exists == "raise":
            raise ValueError(f"Log directory {log_dir} already exists. Will not delete it.")
        else:
            raise AssertionError(f"Invalid behavior_if_exists: {behavior_if_exists}")
    else:
        logger.info(
            f"Log directory {log_dir} does not exist. Will create it and start logging there."
        )


from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    import os
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # If it's a local path, use it directly
    if os.path.isdir(model_name):
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)

    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "thinkingmachineslabinc/meta-llama-3-tokenizer"

    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)



import asyncio
import json
import logging
import os
from typing import Any, Literal

import ttt_discover.tinker_compat as tinker

from ttt_discover.tinker_utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


@scope
def load_checkpoints_file(log_dir: str) -> list[dict[str, Any]]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return read_jsonl(checkpoint_path)


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """
    Get the last checkpoint from the checkpoints.jsonl file in the specified log directory.

    Args:
        log_dir: The directory to check.
        required_key: The key to check for in the checkpoint.
            We might save partial checkpoints (e.g. sampler) in the same file,
            so we need to filter to the rows that have a fully-resumable checkpoint.

    Returns:
        The last checkpoint, or None if no checkpoint is found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{required_key}' in {log_dir}"
        )
        logger.info(f"Using last checkpoint: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.info(f"No checkpoints found with key {required_key} in {log_dir}")
        return None


@scope
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths
