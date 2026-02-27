from types import SimpleNamespace
from typing import Any

from tinker_cookbook.utils import logtree
from tasks.tictactoe.task import TictactoeTask
from tasks.tictactoe.prompt import SYSTEM_PROMPT
from tinker_cookbook.recipes.ttt.state import TictactoeState
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 1,
        "rew_type": "neg_linear",
        "fail_score": -100.0,
        "eval_timeout": 600,
        "worst_perf_log": -10000,
        "n_item": 200,
    }
}


def verify_tictactoe(
    generation: str,
    step: int,
    num_cpus_per_task: int = 1,
    eval_timeout: int = 600,
    log_path: str = "",
    state: TictactoeState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    task = TictactoeTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)

    coefficients = None
    actual_mse = None
    if out["correctness"] > 0 and "result_construction" in out and out["result_construction"] is not None:
        result = out["result_construction"]
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            coefficients = list(result[0]) if hasattr(result[0], '__iter__') else None
            actual_mse = float(result[1])

    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": -actual_mse if actual_mse is not None else None,  # higher = better
        "mse": actual_mse,
        "coefficients": coefficients,
        "stdout": out.get("stdout", ""),
    }


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


class TictactoeEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_improvement_prompt(self, state: TictactoeState) -> str:
        has_code = state.code and state.code.strip()

        # Value context
        if state.parent_values and state.value is not None:
            before_mse = -state.parent_values[0]
            after_mse = -state.value
            value_ctx = f"\nMSE before and after running the code above (lower is better): {before_mse:.6f} -> {after_mse:.6f}"
        elif state.value is not None:
            value_ctx = f"\nCurrent MSE (lower is better): {-state.value:.6f}"
        elif state.mse is not None:
            value_ctx = f"\nCurrent MSE (lower is better): {state.mse:.6f}"
        else:
            value_ctx = ""

        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "...(truncated)\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))

        coefficients_section = ""
        if hasattr(state, 'coefficients') and state.coefficients is not None:
            coefficients_section = "\nYou can access the previous best coefficients via the `initial_coefficients` global variable.\n"

        if has_code:
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing the evaluation polynomial.{value_ctx}

Reason about how you could further improve these coefficients.
Try something meaningfully different from the above approach -- different optimization strategies, different coefficient initialization, or different mathematical reasoning about tic-tac-toe structure.
Unless you make a meaningful improvement, you will not be rewarded.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to find optimal polynomial coefficients for the tic-tac-toe evaluation function.
"""

        return f"""{prompt}
{coefficients_section}{code_section}"""

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 1,
        eval_timeout: int = 600,
        log_path: str = "",
        state: TictactoeState = None,
        **kwargs,
    ) -> dict[str, Any]:
        return verify_tictactoe(generation, step, num_cpus_per_task, eval_timeout, log_path, state)

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "mse": None,
            "coefficients": None,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "mse": None,
            "coefficients": None,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        performance = outs.get("performance")
        if _is_entropic_adv(self.adv_estimator):
            mse = -performance if performance is not None else float('inf')
            return 1.0 / (1e-8 + mse) if (correctness > 0 and mse >= 0) else 0.0
        else:
            return outs["score"]

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> TictactoeState:
        performance = outs.get("performance")
        if performance is None:
            return None
        parent_state = self.initial_state
        parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
        return TictactoeState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,  # -mse, higher = better
            coefficients=outs.get("coefficients"),
            mse=outs.get("mse"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        score = outs["score"]
        correctness = outs["correctness"]
        mse = outs.get("mse")
        return {
            "format": correct_format,
            "score": score,
            "correctness": correctness,
            "correct": correctness,
            "mse": mse,
            "performance": mse,
            "performance/best": mse if correctness > 0 else None,
            "initial_performance": -self.initial_state.value if self.initial_state.value is not None else None,
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }
