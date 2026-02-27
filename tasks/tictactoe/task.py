import inspect
import numpy as np

from tasks.base_reward_task import BaseRewardTask
from tasks.tictactoe.verifier import (
    evaluate_tictactoe_polynomial,
    verify_tictactoe_coefficients,
    _check_winner,
    _minimax,
    _enumerate_states_and_minimax,
    _get_monomial_subsets,
    _evaluate_polynomial,
    LINES,
)


class TictactoeTask(BaseRewardTask):

    def __init__(self, config, log_path=""):
        super().__init__(config, log_path)

    def get_function_name(self) -> str:
        return "run"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        imports = "import numpy as np\nfrom itertools import combinations"

        # Inject all verifier functions so they're available in the sandbox
        sources = []
        for obj in [LINES]:
            sources.append(f"LINES = {obj!r}")
        for func in [
            _check_winner,
            _minimax,
            _enumerate_states_and_minimax,
            _get_monomial_subsets,
            _evaluate_polynomial,
            verify_tictactoe_coefficients,
            evaluate_tictactoe_polynomial,
        ]:
            sources.append(inspect.getsource(func))

        base = imports + "\n\n" + "\n\n".join(sources) + "\n\n"

        # Inject previous coefficients if available
        if state is not None and hasattr(state, "coefficients") and state.coefficients is not None:
            base += f"initial_coefficients = {state.coefficients!r}\n\n"

        return base + generation

    def get_reward(self, result) -> float:
        coefficients, claimed_mse = result
        # Independently recompute MSE (never trust LLM's claimed value)
        mse = evaluate_tictactoe_polynomial(coefficients)
        return float(mse)

    def verify(self, result, *, step, **kwargs) -> bool:
        try:
            coefficients, claimed_mse = result
            mse = evaluate_tictactoe_polynomial(coefficients)
            if not np.isfinite(mse) or mse < 0:
                return False
            return True
        except Exception:
            return False
