"""
Main PID controller implementation for Crazyflie

Based on controller_pid.c from Crazyflie firmware
Copyright (C) 2011-2012 Bitcraze AB
"""

from typing import Optional, Dict, Tuple

from .controller_interface import ControllerInterface

from .attitude_controller import AttitudeController
from .position_controller import PositionController
from .controller_types import (
    ATTITUDE_RATE,
    POSITION_RATE,
    Attitude,
    Control,
    ControlMode,
    Quaternion,
    SensorData,
    Setpoint,
    StabMode,
    State,
    cap_angle,
    degrees,
    quat2rpy,
    radians,
    rate_do_execute,
)


class ControllerPID(ControllerInterface):
    """Main PID controller for Crazyflie"""

    def __init__(self, loop_hz: float):
        """Initialize PID controller"""
        self.attitude_desired = Attitude()
        self.rate_desired = Attitude()
        self.actuator_thrust = 0.0

        # Logging variables
        self.cmd_thrust = 0.0
        self.cmd_roll = 0.0
        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0
        self.r_roll = 0.0
        self.r_pitch = 0.0
        self.r_yaw = 0.0

        # Update rate
        self.attitude_update_dt = 1.0 / ATTITUDE_RATE

        # Sub-controllers
        self.attitude_controller = AttitudeController(self.attitude_update_dt)
        self.position_controller = PositionController()


        # Stabilizer step
        self.stabilizer_step = 1

        # Persistent control for logging
        self.control = Control()

    def init(self):
        """Initialize the controller and sub-controllers"""
        self.attitude_controller = AttitudeController(self.attitude_update_dt)
        self.position_controller = PositionController()
        self.stabilizer_step = 1

    def test(self) -> bool:
        """
        Test if controller is properly initialized

        Returns:
            True if initialized
        """
        if self.attitude_controller is None or self.position_controller is None:
            return False
        return self.attitude_controller.test()

    def reset(self, state: State):
        self.attitude_controller = AttitudeController(self.attitude_update_dt)
        self.position_controller = PositionController()
        self.stabilizer_step = 1

    def step(
        self,
        setpoint: Setpoint,
        sensors: SensorData,
        state: State,
        sim_time: float
    ) -> Tuple[Control, Dict[str, float]]:
        """
        Main PID controller update

        Args:
            setpoint: Desired setpoint
            sensors: Sensor data
            state: Current state estimate
            sim_time: Current simulation time
        """

        self.control.control_mode = ControlMode.CONTROL_MODE_LEGACY

        # Update at ATTITUDE_RATE
        if rate_do_execute(ATTITUDE_RATE, self.stabilizer_step):
            # Rate-controlled YAW is moving YAW angle setpoint
            if setpoint.mode.yaw == StabMode.MODE_VELOCITY:
                self.attitude_desired.yaw = cap_angle(
                    self.attitude_desired.yaw
                    + setpoint.attitude_rate.yaw * self.attitude_update_dt
                )

                yaw_max_delta = self.attitude_controller.get_yaw_max_delta()
                if yaw_max_delta != 0.0:
                    delta = cap_angle(self.attitude_desired.yaw - state.attitude.yaw)
                    # Keep the yaw setpoint within +/- yawMaxDelta from current yaw
                    if delta > yaw_max_delta:
                        self.attitude_desired.yaw = state.attitude.yaw + yaw_max_delta
                    elif delta < -yaw_max_delta:
                        self.attitude_desired.yaw = state.attitude.yaw - yaw_max_delta

            elif setpoint.mode.yaw == StabMode.MODE_ABS:
                self.attitude_desired.yaw = setpoint.attitude.yaw

            elif setpoint.mode.quat == StabMode.MODE_ABS:
                # Convert quaternion to RPY
                setpoint_quat = Quaternion(
                    w=setpoint.attitude_quaternion.w,
                    x=setpoint.attitude_quaternion.x,
                    y=setpoint.attitude_quaternion.y,
                    z=setpoint.attitude_quaternion.z,
                )
                rpy = quat2rpy(setpoint_quat)
                self.attitude_desired.yaw = degrees(rpy.z)

            self.attitude_desired.yaw = cap_angle(self.attitude_desired.yaw)

        # Update position controller at POSITION_RATE
        if rate_do_execute(POSITION_RATE, self.stabilizer_step):
            thrust, attitude_desired = self.position_controller.position_controller(
                setpoint, state
            )
            self.actuator_thrust = thrust
            # Update attitude desired from position controller
            self.attitude_desired.roll = attitude_desired.roll
            self.attitude_desired.pitch = attitude_desired.pitch
            # Note: yaw is handled separately above

        # Update attitude controller at ATTITUDE_RATE
        if rate_do_execute(ATTITUDE_RATE, self.stabilizer_step):
            # Switch between manual and automatic position control
            if setpoint.mode.z == StabMode.MODE_DISABLE:
                self.actuator_thrust = setpoint.thrust

            if (
                setpoint.mode.x == StabMode.MODE_DISABLE
                or setpoint.mode.y == StabMode.MODE_DISABLE
            ):
                self.attitude_desired.roll = setpoint.attitude.roll
                self.attitude_desired.pitch = setpoint.attitude.pitch

            # Correct attitude PID
            (
                self.rate_desired.roll,
                self.rate_desired.pitch,
                self.rate_desired.yaw,
            ) = self.attitude_controller.correct_attitude_pid(
                state.attitude.roll,
                state.attitude.pitch,
                state.attitude.yaw,
                self.attitude_desired.roll,
                self.attitude_desired.pitch,
                self.attitude_desired.yaw,
            )

            # For roll and pitch, if velocity mode, overwrite rateDesired with setpoint
            # value. Also reset the PID to avoid error buildup
            if setpoint.mode.roll == StabMode.MODE_VELOCITY:
                self.rate_desired.roll = setpoint.attitude_rate.roll
                self.attitude_controller.reset_roll_attitude_pid(state.attitude.roll)

            if setpoint.mode.pitch == StabMode.MODE_VELOCITY:
                self.rate_desired.pitch = setpoint.attitude_rate.pitch
                self.attitude_controller.reset_pitch_attitude_pid(state.attitude.pitch)

            # Correct rate PID (note: gyro.y is negated)
            self.attitude_controller.correct_rate_pid(
                sensors.gyro.x,
                -sensors.gyro.y,
                sensors.gyro.z,
                self.rate_desired.roll,
                self.rate_desired.pitch,
                self.rate_desired.yaw,
            )

            # Get actuator output
            roll_out, pitch_out, yaw_out = (
                self.attitude_controller.get_actuator_output()
            )

            self.control.roll = roll_out
            self.control.pitch = pitch_out
            self.control.yaw = -yaw_out  # Negate yaw output

            # Update logging variables
            self.cmd_thrust = self.control.thrust
            self.cmd_roll = self.control.roll
            self.cmd_pitch = self.control.pitch
            self.cmd_yaw = self.control.yaw
            self.r_roll = radians(sensors.gyro.x)
            self.r_pitch = -radians(sensors.gyro.y)
            self.r_yaw = radians(sensors.gyro.z)

        # Set thrust
        self.control.thrust = self.actuator_thrust

        # Handle zero thrust case (safety)
        if self.control.thrust == 0:
            self.control.thrust = 0
            self.control.roll = 0
            self.control.pitch = 0
            self.control.yaw = 0

            self.cmd_thrust = self.control.thrust
            self.cmd_roll = self.control.roll
            self.cmd_pitch = self.control.pitch
            self.cmd_yaw = self.control.yaw

            # Reset all PIDs
            self.attitude_controller.reset_all_pid(
                state.attitude.roll, state.attitude.pitch, state.attitude.yaw
            )
            self.position_controller.reset_all_pid(
                state.position.x, state.position.y, state.position.z
            )

            # Reset the calculated YAW angle for rate control
            self.attitude_desired.yaw = state.attitude.yaw

        self.stabilizer_step += 1
        return self.control, dict()



# C-style function interfaces for compatibility
_controller_pid_instance: Optional[ControllerPID] = None


def controller_pid_init():
    """Initialize controller (C-style function)"""
    global _controller_pid_instance
    _controller_pid_instance = ControllerPID()
    _controller_pid_instance.init()


def controller_pid_test() -> bool:
    """Test if controller is initialized (C-style function)"""
    global _controller_pid_instance
    if _controller_pid_instance is None:
        return False
    return _controller_pid_instance.test()


def controller_pid(
    setpoint: Setpoint,
    sensors: SensorData,
    state: State,
    stabilizer_step: int,
):
    """Run controller (C-style function)"""
    global _controller_pid_instance
    if _controller_pid_instance is not None:
        control, _ = _controller_pid_instance.step(
            setpoint, sensors, state, stabilizer_step
        )
        return control
