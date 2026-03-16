from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Sequence
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import logging
import re

import chz
import ttt_discover.tinker_compat as tinker
from ttt_discover.tinker_utils import renderers, logtree
from ttt_discover.rl.types import (
    ProblemEnv, ProblemGroupBuilder, EnvGroupBuilder, RLDataset, RLDatasetBuilder,
    Env, Action, StepResult
)
from ttt_discover.tinker_utils.misc_utils import get_tokenizer
from ttt_discover.tinker_utils.state import State
from ttt_discover.tinker_utils.sampler import StateSampler, get_or_create_sampler_with_default

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """General configuration for dataset and environment creation.

    Provide env_type (for custom envs, pass the class).
    After get_single_problem_dataset_builder(), env_type is set on config for internal use.
    """
    problem_type: str
    env_type: type
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    num_cpus_per_task: int = 1
    eval_timeout: int = 300
    log_path: str = ""
    timeout: float = 8000.0 # Timeout for async grading, not sandbox timeout
    convo_prefix: Any = None
    gpu_mode_score_scale: float = 3000.0


class SingleProblemDataset(RLDataset):
    def __init__(
        self,
        config: DatasetConfig,
        renderer: renderers.Renderer,
        sampler: StateSampler,
    ):
        self.config = config
        self.batch_size = config.batch_size
        self.group_size = config.group_size
        self.renderer = renderer
        self.problem_type = config.problem_type
        self.sampler = sampler

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        states = self.sampler.sample_states(self.batch_size)
        return [self._make_env_group_builder(state, self.group_size) for state in states]

    def flush(self, step: int | None = None):
        """Flush sampler state to disk. Call after batch completes."""
        self.sampler.flush(step)

    def __len__(self) -> int:
        return 1

    def _make_env_group_builder(
        self, initial_state: State, group_size: int
    ) -> ProblemGroupBuilder:
        """Create an environment group builder using the env type from config."""
        env_type = self.config.env_type
        if env_type is None:
            raise ValueError("config.env_type must be set")
        logging_name = getattr(env_type, "env_name", env_type.__name__)
        return ProblemGroupBuilder(
            env_thunk=partial(
                env_type,
                self.renderer,
                initial_state=initial_state,
                sampler=self.sampler,
                config=self.config,
            ),
            num_envs=group_size,
            logging_name=logging_name,
        )


@chz.chz
class SingleProblemDatasetBuilder(RLDatasetBuilder):
    config: DatasetConfig

    async def __call__(self) -> SingleProblemDataset:
        if self.config.problem_type is None:
            raise ValueError("problem_type is required")
        if not self.config.log_path:
            raise ValueError("log_path is required for dataset")
        
        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.config.renderer_name, tokenizer=tokenizer)
        
        # Get sampler - this may need to be environment-specific
        sampler = self._get_sampler()
        
        dataset = SingleProblemDataset(
            config=self.config,
            renderer=renderer,
            sampler=sampler,
        )
        return dataset

    def _get_sampler(self) -> StateSampler:
        """Get the appropriate sampler; env_type is already set on config."""
        return get_or_create_sampler_with_default(
            log_path=self.config.log_path,
            env_type=self.config.env_type,
            batch_size=self.config.batch_size,
            problem_type=self.config.problem_type,
        )


def get_single_problem_dataset_builder(
    config: DatasetConfig,
    **kwargs,
) -> RLDatasetBuilder:
    """
    Unified function to get a single problem dataset builder.
    Custom envs: pass env_type=YourEnv in DatasetConfig; no registry needed.
    """
    if not config.log_path:
        raise ValueError("log_path is required for dataset")

    return SingleProblemDatasetBuilder(config=config)


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True, keep_separators=True):
    """Extract the last code block from input text.
    
    Args:
        input_text: Text to parse
        codeblock_seps: List of language identifiers to look for
        last_response_strict: If True, return empty string for invalid code; otherwise return original text
        keep_separators: If True, return code with ```language wrapper; if False, return code only
    """
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        
        # Check if content is empty
        if not code_content or code_content.strip() == '':
            if last_response_strict:
                return ''
            else:
                return input_text
        
        if keep_separators:
            return f'```{language}\n{code_content}\n```'
        else:
            return code_content
    else:
        if last_response_strict:
            return ''
        else:
            return input_text


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)

# Shared ThreadPoolExecutor for all environments
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)


class Environment(ProblemEnv):

    state_type: State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        """Create an initial state for rollouts. Override in subclasses that need a different initial state."""
        return State(timestep=-1, construction=None, code="", value=0.0)

    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler,
        config,
    ):
        super().__init__(renderer, convo_prefix=config.convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.config = config
        self.timeout = config.timeout
        self.num_cpus_per_task = config.num_cpus_per_task
        self.eval_timeout = config.eval_timeout
        self.log_path = config.log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.problem_type = config.problem_type
    
    @abstractmethod
    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state.
        
        Returns:
            Formatted prompt string
        """
        pass

    def is_maximize(self) -> bool:
        return True
    
    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> State:
        """Create the next state from the current step.
        
        Args:
            step_idx: Current step index
            parsed_code: Parsed code from response
            outs: Output dictionary from _verify_code
            
        Returns:
            New State object
        """
        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=outs.raw_score if self.is_maximize() else -outs.raw_score, # higher = better
            observation=outs.stdout,
        )
    
    def _build_metrics(
        self,
        outs: VerifyResult,
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        """Build metrics dictionary for StepResult.
        
        Args:
            outs: VerifyResult from _run_verification
            correct_format: Whether the code format was valid
            message: Parsed message from renderer
            parsed_code: Parsed code string
            
        Returns:
            Metrics dictionary
        """
        correctness = outs.correctness
        return {
            "format": correct_format,
            "reward": outs.reward,
            "correctness": correctness,
            "raw_score": outs.raw_score if correctness > 0 else None,
            "initial_raw_score": self.initial_state.value,
            "msg": outs.msg,
            "prompt": self.get_question(),
            "response": message['content'],
            "parsed_code": parsed_code,
        }
    
    def _get_code_languages(self) -> list[str]:
        """Return list of code block languages to parse. Override if needed."""
        return ["python"]
    
    def _should_keep_code_separators(self) -> bool:
        """Whether to keep ```language separators in parsed code. Override if needed."""
        return True
    
    def check_format(self, parsed_code: str) -> bool:
        """Check if parsed code has valid format."""
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True
    
    async def check_answer(self, parsed_code: str, step: int) -> VerifyResult:
        """Check answer asynchronously with timeout."""
        if not self.check_format(parsed_code):
            return VerifyResult(
                reward=0.0,
                msg="Invalid code",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )
        
        return await self._safe_grade(parsed_code, step)

    def _run_verification(
        self,
        generation: str,
        problem_type: str,
        log_path: str,
        state: State,
    ) -> VerifyResult:

        task = self.reward_function(problem_type=problem_type, log_dir=log_path, eval_timeout=self.eval_timeout, num_cpus_per_task=self.num_cpus_per_task)
        out = task.get_reward(generation, state=state)

        return VerifyResult(
            reward=out["reward"],
            msg=out["msg"],
            correctness=out["correctness"],
            raw_score=out["raw_score"],
            result_construction=out.get("result_construction", None),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
        )
    
    async def _safe_grade(self, given_answer: str, step: int) -> VerifyResult:
        """Async grader: runs _verify_code in a background thread with asyncio timeout."""
        loop = asyncio.get_running_loop()
        start_time = time.time()
        
        try:
            out = await asyncio.wait_for(
                loop.run_in_executor(
                    SAFE_GRADE_EXECUTOR,
                    partial[VerifyResult](
                        self._run_verification,
                        given_answer,
                        self.problem_type,
                        self.log_path,
                        self.state,
                    )
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"Timeout grading: took {elapsed:.1f}s, limit was {self.timeout:.1f}s")
            return VerifyResult(
                reward=0.0, 
                msg="Timeout grading", 
                correctness=0.0, 
                raw_score=0.0, 
                result_construction=None, 
                stdout=""
            )
        except Exception as e:
            import traceback
            error_msg = f"Error grading: {e}\n{traceback.format_exc()}"
            logger.warning(f"Exception while grading: {e}")
            return VerifyResult(
                reward=0.0,
                msg=f"Error grading: {error_msg}",
                correctness=0.0,
                raw_score=0.0,
                result_construction=None,
                stdout="",
            )
        
        return out

    async def step(self, action: Action, step_idx: int) -> StepResult:
        """Process a step: parse response, verify code, compute reward, update state."""
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        
        # Parse code based on environment-specific settings
        languages = self._get_code_languages()
        keep_separators = self._should_keep_code_separators()
        parsed_code = last_codeblock_postprocess(
            response,
            codeblock_seps=languages,
            keep_separators=keep_separators
        )

        correct_format = float(parse_success) and float(self.check_format(parsed_code))
        
        # Verify code
        outs = await self.check_answer(parsed_code, step_idx)
        reward = outs.reward
        correctness = outs.correctness
        raw_score = outs.raw_score
        msg = outs.msg
        
        # Logging
        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Reward: {reward:.4f}, Correctness: {correctness:.4f}, Raw Score: {raw_score:.4f}, Msg: {msg}"
        )
        
        # Build metrics
        metrics = self._build_metrics(outs, correct_format, message, parsed_code)
        
        # Create step result
        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )
        
        # Update sampler with new state if we have valid result
        if correctness > 0:
            try:
                next_state = self._create_next_state(step_idx, parsed_code, outs)
                self.sampler.update_states([next_state], [self.initial_state], save=False)
            except Exception as e:
                logger.warning(f"Failed to create next state: {e}")
                if hasattr(self.sampler, 'record_failed_rollout'):
                    self.sampler.record_failed_rollout(self.initial_state)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)
        
        return step_result
    
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for TTT environments.")