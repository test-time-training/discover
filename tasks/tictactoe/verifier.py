import numpy as np
from itertools import combinations


# Board layout:
#   0 | 1 | 2
#   ---------
#   3 | 4 | 5
#   ---------
#   6 | 7 | 8
#
# Cell values: X = 1, empty = 0, O = -1

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),              # diags
]


def _check_winner(board):
    """Return 1 if X wins, -1 if O wins, 0 if no winner."""
    for a, b, c in LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0


def _minimax(board, is_x_turn):
    """Compute minimax value from X's perspective."""
    winner = _check_winner(board)
    if winner != 0:
        return winner
    empty = [i for i in range(9) if board[i] == 0]
    if not empty:
        return 0  # draw
    if is_x_turn:
        best = -2
        for i in empty:
            board[i] = 1
            val = _minimax(board, False)
            board[i] = 0
            best = max(best, val)
        return best
    else:
        best = 2
        for i in empty:
            board[i] = -1
            val = _minimax(board, True)
            board[i] = 0
            best = min(best, val)
        return best


def _enumerate_states_and_minimax():
    """Enumerate all reachable tic-tac-toe board states with minimax values.

    Returns:
        boards: list of 9-element tuples (each cell in {-1, 0, 1})
        values: list of minimax values from X's perspective (+1, 0, -1)
    """
    seen = {}  # board_tuple -> minimax_value

    def recurse(board, is_x_turn):
        key = tuple(board)
        if key in seen:
            return
        seen[key] = _minimax(list(board), is_x_turn)
        if _check_winner(board) != 0:
            return
        empty = [i for i in range(9) if board[i] == 0]
        if not empty:
            return
        piece = 1 if is_x_turn else -1
        for i in empty:
            board[i] = piece
            recurse(board, not is_x_turn)
            board[i] = 0

    recurse([0] * 9, True)
    boards = list(seen.keys())
    values = [seen[b] for b in boards]
    return boards, values


def _get_monomial_subsets():
    """Return all subsets of {0,...,8} with |S| <= 3 (multilinear monomials).

    Returns:
        List of 130 tuples. Ordering:
          [0]:       ()           -- constant term
          [1]-[9]:   (0,), (1,), ..., (8,)  -- degree 1
          [10]-[45]: (0,1), (0,2), ..., (7,8) -- degree 2
          [46]-[129]: (0,1,2), (0,1,3), ..., (6,7,8) -- degree 3
    """
    subsets = [()]
    for k in range(1, 4):
        for combo in combinations(range(9), k):
            subsets.append(combo)
    return subsets


def _evaluate_polynomial(coefficients, board, subsets):
    """Evaluate multilinear polynomial at a board position.

    Args:
        coefficients: list of 130 floats
        board: tuple/list of 9 values in {-1, 0, 1}
        subsets: list of 130 subsets from _get_monomial_subsets()

    Returns:
        float: polynomial value
    """
    result = 0.0
    for coeff, subset in zip(coefficients, subsets):
        if coeff == 0.0:
            continue
        monomial = 1.0
        for idx in subset:
            monomial *= board[idx]
        result += coeff * monomial
    return result


def verify_tictactoe_coefficients(coefficients):
    """Validate polynomial coefficients and compute MSE against minimax values.

    Args:
        coefficients: list/array of exactly 130 floats

    Returns:
        float: Mean Squared Error over all reachable tic-tac-toe states

    Raises:
        ValueError: on invalid input
    """
    if not isinstance(coefficients, (list, tuple, np.ndarray)):
        raise ValueError(f"coefficients must be list/tuple/ndarray, got {type(coefficients)}")

    coefficients = [float(c) for c in coefficients]
    if len(coefficients) != 130:
        raise ValueError(f"Expected 130 coefficients, got {len(coefficients)}")

    for i, c in enumerate(coefficients):
        if not np.isfinite(c):
            raise ValueError(f"Coefficient {i} is not finite: {c}")

    subsets = _get_monomial_subsets()
    boards, minimax_values = _enumerate_states_and_minimax()

    mse = 0.0
    for board, true_value in zip(boards, minimax_values):
        pred = _evaluate_polynomial(coefficients, board, subsets)
        mse += (pred - true_value) ** 2
    mse /= len(boards)

    return float(mse)


def evaluate_tictactoe_polynomial(coefficients):
    """Entry point: validate coefficients and return MSE against minimax ground truth."""
    return verify_tictactoe_coefficients(coefficients)
