"""
KernelBench environment for TTT-Discover.

Implements discover's Environment interface for training across KernelBench problems.
Each problem is identified by problem_type="level:problem_id" (e.g. "1:3").
"""

import asyncio
import logging

from ttt_discover import Environment, State
from ttt_discover.tinker_utils.dataset_builder import VerifyResult

from kernelbench_tinker.envs.kernelbench_client import (
    evaluate_kernel_async,
    get_prompt_for_problem,
)
from kernelbench_tinker.training.reward import RewardConfig, compute_reward

logger = logging.getLogger(__name__)

TASK_INSTRUCTION = """You are an expert GPU kernel developer. Optimize the given PyTorch model \
by writing an efficient custom CUDA/Triton kernel.

Return your solution as a Python class named `ModelNew` that implements the same interface \
as the reference `Model` class.

Wrap your final implementation in a single code block:
```python
# your complete implementation
class ModelNew(nn.Module):
    ...
```"""


def _parse_problem_type(problem_type: str) -> tuple[int, int]:
    """Parse "level:problem_id" string into (level, problem_id)."""
    level_str, pid_str = problem_type.split(":")
    return int(level_str), int(pid_str)


class KernelBenchDiscoverEnv(Environment):
    """Discover Environment for KernelBench kernel optimization."""

    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(timestep=-1, construction=None, code="", value=0.0)

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def _should_keep_code_separators(self) -> bool:
        return False

    def check_format(self, parsed_code: str) -> bool:
        return bool(parsed_code and parsed_code.strip() and "class ModelNew" in parsed_code)

    def get_question(self) -> str:
        level, problem_id = _parse_problem_type(self.problem_type)
        prompt = get_prompt_for_problem(
            level=level,
            problem_id=problem_id,
            backend=self.config.backend,
            option=self.config.prompt_option,
            dataset_src=self.config.dataset_src,
            precision=self.config.prompt_precision,
            include_hardware=self.config.prompt_include_hardware,
            gpu_name=self.config.prompt_gpu_name,
        )
        return f"{TASK_INSTRUCTION}\n\n{prompt}"

    def _run_verification(
        self,
        generation: str,
        problem_type: str,
        log_path: str,
        state: State,
    ) -> VerifyResult:
        level, problem_id = _parse_problem_type(problem_type)
        try:
            result = asyncio.run(
                evaluate_kernel_async(
                    level=level,
                    problem_id=problem_id,
                    backend=self.config.backend,
                    kernel_code=generation,
                    dataset_src=self.config.dataset_src,
                    num_correct_trials=self.config.num_correct_trials,
                    measure_performance=self.config.measure_performance,
                    timeout=float(self.eval_timeout),
                )
            )
        except Exception as e:
            logger.warning("KernelBench eval error: %s", e)
            return VerifyResult(
                reward=0.0,
                msg=f"eval error: {e}",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )

        correctness = float(result["correctness"])
        speedup = result.get("speedup") or 0.0
        reward = compute_reward(result, RewardConfig())
        msg = result.get("error_message") or f"speedup={speedup:.3f}x correct={correctness}"

        return VerifyResult(
            reward=reward,
            msg=msg,
            correctness=correctness,
            raw_score=speedup if correctness > 0 else 0.0,
            result_construction=None,
            stdout=result.get("error_message", ""),
        )
