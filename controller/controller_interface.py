"""
Common controller interface for host-side Crazyflie controllers.

This is the union of the public and private API of:

- ``ControllerPID`` (controller/controller_pid.py)
- ``AugmentedCascadeController`` (controller/controller_augmented_pid.py)

Every method from either controller is declared here with its original
signature, so each concrete controller can be made to implement the full
interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from .attitude_controller import AttitudeController
from .controller_types import (
    ATTITUDE_RATE,
    Attitude,
    Control,
    SensorData,
    Setpoint,
    State,
)
from .position_controller import PositionController


class ControllerInterface(ABC):
    """Abstract base for cascaded PID-style controllers."""

    # ------------------------------------------------------------------
    # Shared state (present in both controllers)
    # ------------------------------------------------------------------
    attitude_controller: AttitudeController
    position_controller: PositionController
    attitude_desired: Attitude
    rate_desired: Attitude
    actuator_thrust: float

    # ------------------------------------------------------------------
    # State from ControllerPID
    # ------------------------------------------------------------------
    attitude_update_dt: float
    cmd_thrust: float
    cmd_roll: float
    cmd_pitch: float
    cmd_yaw: float
    r_roll: float
    r_pitch: float
    r_yaw: float

    # ------------------------------------------------------------------
    # State from AugmentedCascadeController
    # ------------------------------------------------------------------
    loop_hz: float
    dt: float
    outer_div: int
    outer_dt: float
    tick: int
    control: Control
    aug_x: Any
    aug_y: Any
    aug_z: Any
    _last_diag: Dict[str, float]

    # ------------------------------------------------------------------
    # Construction / initialisation
    # ------------------------------------------------------------------
    @abstractmethod
    def __init__(self, loop_hz: float = ATTITUDE_RATE):
        """
        Create the controller.

        ``ControllerPID`` takes no arguments (implicitly ``ATTITUDE_RATE``);
        ``AugmentedCascadeController`` takes ``loop_hz``.
        """

    @abstractmethod
    def init(self) -> None:
        """Initialise sub-controllers (from ControllerPID)."""

    @abstractmethod
    def test(self) -> bool:
        """Return True if the controller is initialised (from ControllerPID)."""

    @abstractmethod
    def reset(self, state: State) -> None:
        """
        Reset PIDs and internal state to the given vehicle state
        (from AugmentedCascadeController).
        """

    # ------------------------------------------------------------------
    # Control update
    # ------------------------------------------------------------------
    @abstractmethod
    def step(
        self, setpoint: Setpoint, sensors: SensorData, state: State, sim_time: float
    ) -> Tuple[Control, Dict[str, float]]:
        """
        Run one controller update and return the control output plus
        diagnostics (from AugmentedCascadeController).
        """

    # ------------------------------------------------------------------
    # Logging / diagnostics
    # ------------------------------------------------------------------
    def get_logging_data(self, state: State, setpoint: Setpoint) -> Dict[str, float]:
        """
        Return {name: value} pairs for the host to log and print.

        Controllers with extra values should extend this via super().
        """
        return {
            "position_x": state.position.x,
            "setpoint_x": setpoint.position.x,
            "position_y": state.position.y,
            "setpoint_y": setpoint.position.y,
            "position_z": state.position.z,
            "setpoint_z": setpoint.position.z,
            "roll": self.control.roll,
            "pitch": self.control.pitch,
            "yaw": self.control.yaw,
        }
