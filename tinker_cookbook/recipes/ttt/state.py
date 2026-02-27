from __future__ import annotations
from abc import ABC, abstractmethod
import uuid

import numpy as np

from tinker_cookbook.rl.types import Action, StepResult


def to_json_serializable(obj):
    """Convert numpy arrays and other non-JSON-serializable types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


class State(ABC):
    id: str  # unique identifier for this state
    timestep: int  # the training step this state was first visited at
    value: float  # Expected value of starting from this state (higher = better)
    parent_values: list[float]  # list of ancestor values (most recent first) for terminal value estimation
    parents: list[dict]  # list of parent refs [{"id": ..., "timestep": ...}, ...] (most recent first)
    observation: str  # stdout/logs from the code that created this state

    def __init__(self, timestep: int, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        self.id = id if id is not None else str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.parent_values = parent_values if parent_values is not None else []
        self.parents = parents if parents is not None else []
        self.observation = observation

    def estimate_value(self, experiences_from_state: list) -> float:
        """Estimate state value from experiences starting from this state."""
        assert len(experiences_from_state) > 0
        rewards = [exp.step_result.reward for exp in experiences_from_state]
        self.value = sum(rewards) / len(rewards)
        return self.value

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize state to dict."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> State:
        """Deserialize state from dict."""
        pass


class InequalitiesState(State):
    construction: list[float]  # the step function construction
    code: str  # the code that generated the construction

    def __init__(self, timestep: int, construction: list[float], code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.construction = to_json_serializable(construction)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "InequalitiesState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> InequalitiesState:
        return cls(
            timestep=d["timestep"],
            construction=d["construction"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


def _to_tuple_of_tuples(obj):
    """Convert nested list [[x,y,r],...] to tuple of tuples for hashability."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        return tuple(tuple(c) for c in obj)
    return obj


class CirclePackingState(State):
    """State for circle packing - holds code and construction (circles)."""
    construction: tuple  # tuple of tuples, each as (x, y, r) - hashable
    code: str  # the code that generated the result

    def __init__(self, timestep: int, construction, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.construction = _to_tuple_of_tuples(construction)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "CirclePackingState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction) if self.construction is not None else None,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> CirclePackingState:
        return cls(
            timestep=d["timestep"],
            construction=_to_tuple_of_tuples(d.get("construction")),
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


class GpuModeState(State):
    """State for gpu mode - holds code."""
    code: str  # the code that generated the result

    def __init__(self, timestep: int, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "GpuModeState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> GpuModeState:
        return cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


class AleBenchState(State):
    """State for ALE Bench - holds code."""
    code: str  # the code that generated the result

    def __init__(self, timestep: int, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None):
        super().__init__(timestep, value, parent_values, parents, id)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "AleBenchState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> AleBenchState:
        return cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
        )


class ErdosState(State):
    """State for Erdos min overlap problem - holds code and construction (h_values)."""
    code: str
    c5_bound: float
    construction: list[float]

    def __init__(self, timestep: int, code: str, value: float = None, c5_bound: float = None, construction: list[float] = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code
        self.c5_bound = c5_bound
        self.construction = to_json_serializable(construction) if construction is not None else None

    def to_dict(self) -> dict:
        return {
            "type": "ErdosState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "code": self.code,
            "c5_bound": self.c5_bound,
            "construction": to_json_serializable(self.construction) if self.construction is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ErdosState":
        return cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            c5_bound=d.get("c5_bound"),
            construction=d.get("construction") or d.get("h_values"),  # backward compat
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


class DenoisingState(State):
    code: str
    mse: float
    poisson: float

    def __init__(self, timestep: int, code: str, value: float = None, mse: float = None, poisson: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code
        self.mse = mse
        self.poisson = poisson

    def to_dict(self) -> dict:
        return {
            "type": "DenoisingState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "code": self.code,
            "mse": self.mse,
            "poisson": self.poisson,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DenoisingState":
        return cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            mse=d.get("mse"),
            poisson=d.get("poisson"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


class TictactoeState(State):
    """State for tic-tac-toe evaluation polynomial optimization."""
    code: str
    coefficients: list[float]
    mse: float

    def __init__(self, timestep: int, code: str, value: float = None,
                 coefficients: list[float] = None, mse: float = None,
                 parent_values: list[float] = None, parents: list[dict] = None,
                 id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code
        self.coefficients = to_json_serializable(coefficients) if coefficients is not None else None
        self.mse = mse

    def to_dict(self) -> dict:
        return {
            "type": "TictactoeState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "code": self.code,
            "coefficients": to_json_serializable(self.coefficients) if self.coefficients is not None else None,
            "mse": self.mse,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TictactoeState":
        return cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            coefficients=d.get("coefficients"),
            mse=d.get("mse"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )


# Registry for state types
STATE_REGISTRY = {
    "InequalitiesState": InequalitiesState,
    "CirclePackingState": CirclePackingState,
    "GpuModeState": GpuModeState,
    "AleBenchState": AleBenchState,
    "ErdosState": ErdosState,
    "DenoisingState": DenoisingState,
    "TictactoeState": TictactoeState,
}


def state_from_dict(d: dict | None) -> State | None:
    """Deserialize any state type from dict."""
    if d is None:
        return None
    state_type = d.get("type")
    if state_type not in STATE_REGISTRY:
        raise ValueError(f"Unknown state type: {state_type}")
    return STATE_REGISTRY[state_type].from_dict(d)


class Experience:
    """A single transition from prev_state to next_state via action.
    
    We sample experiences instead of states so we can use:
    - next_state: as the initial state for the new rollout
    - step_result.reward: to show in the prompt (the actual metric achieved)
    - prev_state: to maintain the tree of dependencies
    - is_initial: True for seed experiences (prev_state=None), False for rollout results
    """
    prev_state: State | None
    action: Action | None
    step_result: StepResult
    next_state: State
    is_initial: bool

    def __init__(self, prev_state: State | None, action: Action | None, step_result: StepResult, next_state: State, is_initial: bool = False):
        self.prev_state = prev_state
        self.action = action
        self.step_result = step_result
        self.next_state = next_state
        self.is_initial = is_initial

    def to_dict(self) -> dict:
        """Serialize experience to dict for storage."""
        return {
            "prev_state": self.prev_state.to_dict() if self.prev_state else None,
            "action": to_json_serializable(self.action) if self.action is not None else None,
            "step_result": {
                "reward": to_json_serializable(self.step_result.reward),
                "episode_done": self.step_result.episode_done,
                "metrics": to_json_serializable(self.step_result.metrics),
            },
            "next_state": self.next_state.to_dict(),
            "is_initial": self.is_initial,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> Experience:
        """Deserialize experience from dict."""
        step_result = StepResult(
            reward=d["step_result"]["reward"],
            episode_done=d["step_result"]["episode_done"],
            next_observation=None,  # Not serialized
            next_stop_condition=None,  # Not serialized
            metrics=d["step_result"].get("metrics", {}),
        )
        return cls(
            prev_state=state_from_dict(d["prev_state"]),
            action=d.get("action"),  # May be None if not saved
            step_result=step_result,
            next_state=state_from_dict(d["next_state"]),
            is_initial=d.get("is_initial", False),
        )