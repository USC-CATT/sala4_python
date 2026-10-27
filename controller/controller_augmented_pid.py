
"""
Host-side augmented cascaded PID with raw PWM motor output.

Baseline control comes from the existing cascaded PID structure. The
augmentation uses a velocity-only nominal model for each translational axis:

    vhat_dot = u

with measured velocity error:

    e_v = v - vhat

Its filtered derivative is approximated using:

    H(s) = s / (1 + mu s)

and the least-squares correction v* is computed from:

    min ||B v + e_f||

For the scalar velocity model with B = 1, this reduces to v* = -e_f.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from controller.controller_interface import ControllerInterface

from controller.attitude_controller import AttitudeController
from controller.controller_types import (
    POSITION_RATE,
    AccData,
    Attitude,
    AttitudeRate,
    Axis3f,
    Control,
    ControlMode,
    GyroData,
    Position,
    SensorData,
    Setpoint,
    SetpointMode,
    StabMode,
    State,
    Velocity,
    MotorThrust,
    cap_angle,
)
from controller.pid import constrain
from controller.position_controller import PositionController
from telemetry import Telemetry
from motorRaw import MotorRaw

# Augmentation tuning.
# XY augmentation is enabled by default for wind rejection.
# Z augmentation is disabled by default since vertical thrust correction is
# usually more sensitive on hardware.
AUG_ENABLE_XY = True
AUG_ENABLE_Z = True
AUG_START_TIME = 2.0  # Time after which augmentation starts applying (but the internal state is initialized from the measurements before that)
AUG_RAMP_TIME = 2.0  # Time over which the augmentation output is ramped up to its full value after AUG_START_TIME
# B: 0.05 T2: 0.15 T3: 0.01 T4: 0.35  T5: 1.0
# B: 0.1  T2: 0.25 T3: 0.05 T4: 0.65 T5: 1.0
AUG_MU_XY = 0.05
AUG_MU_Z = 0.05

# Safety limits for augmented outputs, used to prevent excessive correction
AUG_V_LIMIT_Z = 0.3  # change back to 0.3
AUG_V_LIMIT_XY = 0.3  # change this back to 0.3
AUG_MAX_ATTITUDE_DELTA_DEG = 10  # Max roll/pitch angle delta corresponding to the XY augmentation output, used to prevent excessive attitude correction from the augmentation
# max thrust is 65535
AUG_MAX_THRUST_DELTA = 4000.0  # Max thrust delta corresponding to the Z augmentation output, used to prevent excessive thrust correction from the augmentation
AUG_DISABLE_IF_TILT_OVER_DEG = 35.0
AUG_DISABLE_IF_SPEED_OVER_MPS = 2.0
GRAVITY = 9.81

UINT16_MAX = 65535


class DisturbanceAugmentor1D:
    """1D augmentor that estimates disturbance from velocity error only."""

    def __init__(
        self,
        dt: float,
        mu: float,
        v_limit: float,
        start_time: float,
        ramp_time: float,
        enabled: bool,
    ):
        self.dt = float(dt)
        self.mu = float(mu)
        self.v_limit = float(v_limit)
        self.start_time = float(start_time)
        self.ramp_time = float(ramp_time)
        self.enabled = bool(enabled)

        self.vhat = 0.0
        self.e_prev = 0.0
        self.ef = 0.0
        self.initialized = False
        self.last_v = 0.0

    def reset(self, vel: float = 0.0) -> None:
        self.vhat = float(vel)
        self.e_prev = 0.0
        self.ef = 0.0
        self.initialized = False
        self.last_v = 0.0

    def _align_to_measurement(self, vel: float) -> None:
        self.reset(vel=vel)

    def _least_squares_correction(self) -> float:
        # For scalar B = 1, argmin ||B v + e_f|| gives v* = -e_f.
        return -self.ef

    def compute(self, sim_time: float, vel: float, u_nom: float) -> float:
        # u_nom is the cascaded PID's nominal acceleration command for this axis.
        if not self.enabled:
            self._align_to_measurement(vel)
            return 0.0

        if sim_time < self.start_time:
            self._align_to_measurement(vel)
            return 0.0

        dt = max(self.dt, 1e-6)
        mu = max(self.mu, 1e-6)

        # Velocity-only nominal model: vhat_dot = u_nom.
        self.vhat += dt * u_nom
        e_vel = vel - self.vhat

        if self.initialized:
            e_dot = (e_vel - self.e_prev) / dt
        else:
            e_dot = 0.0

        alpha = 1.0 - math.exp(-dt / mu)
        self.ef += alpha * (e_dot - self.ef)

        self.e_prev = e_vel
        self.initialized = True

        v_est = self._least_squares_correction()
        if self.ramp_time > 0.0:
            ramp = constrain((sim_time - self.start_time) / self.ramp_time, 0.0, 1.0)
        else:
            ramp = 1.0
        self.last_v = constrain(v_est * ramp, -self.v_limit, self.v_limit)
        return self.last_v


class AugmentedCascadeController(ControllerInterface):
    """Baseline cascaded PID with additive nominal-model correction."""

    def __init__(self, loop_hz: float):
        self.loop_hz = float(loop_hz)
        self.dt = 1.0 / self.loop_hz
        self.outer_div = max(1, int(round(self.loop_hz / POSITION_RATE)))
        self.outer_dt = self.outer_div / self.loop_hz
        self.tick = 0

        self.attitude_controller = AttitudeController(update_dt=self.dt)
        self.position_controller = PositionController()

        self.aug_x = DisturbanceAugmentor1D(
            dt=self.outer_dt,
            mu=AUG_MU_XY,
            v_limit=AUG_V_LIMIT_XY,
            start_time=AUG_START_TIME,
            ramp_time=AUG_RAMP_TIME,
            enabled=AUG_ENABLE_XY,
        )
        self.aug_y = DisturbanceAugmentor1D(
            dt=self.outer_dt,
            mu=AUG_MU_XY,
            v_limit=AUG_V_LIMIT_XY,
            start_time=AUG_START_TIME,
            ramp_time=AUG_RAMP_TIME,
            enabled=AUG_ENABLE_XY,
        )
        self.aug_z = DisturbanceAugmentor1D(
            dt=self.outer_dt,
            mu=AUG_MU_Z,
            v_limit=AUG_V_LIMIT_Z,
            start_time=AUG_START_TIME,
            ramp_time=AUG_RAMP_TIME,
            enabled=AUG_ENABLE_Z,
        )

        self.control = Control()
        self.attitude_desired = Attitude()
        self.rate_desired = Attitude()
        self.actuator_thrust = 0.0
        self._last_diag: Dict[str, float] = {
            "v_aug_x": 0.0,
            "v_aug_y": 0.0,
            "v_aug_z": 0.0,
        }
    def init(self):
        pass
    def test(self):
        if self.attitude_controller is None or self.position_controller is None:
            return False
        return self.attitude_controller.test()

    @staticmethod
    def _nominal_xy_input(
        roll_deg: float, pitch_deg: float, yaw_deg: float
    ) -> Tuple[float, float]:
        roll_rad = math.radians(roll_deg)
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        ax = GRAVITY * (roll_rad * math.sin(yaw_rad) + pitch_rad * math.cos(yaw_rad))
        ay = GRAVITY * (-roll_rad * math.cos(yaw_rad) + pitch_rad * math.sin(yaw_rad))
        return ax, ay

    @staticmethod
    def _attitude_delta_from_xy(
        v_aug_x: float, v_aug_y: float, yaw_deg: float
    ) -> Tuple[float, float]:
        yaw_rad = math.radians(yaw_deg)
        d_roll_rad = (
            v_aug_x * math.sin(yaw_rad) - v_aug_y * math.cos(yaw_rad)
        ) / GRAVITY
        d_pitch_rad = (
            v_aug_x * math.cos(yaw_rad) + v_aug_y * math.sin(yaw_rad)
        ) / GRAVITY
        d_roll_deg = math.degrees(d_roll_rad)
        d_pitch_deg = math.degrees(d_pitch_rad)
        d_roll_deg = constrain(
            d_roll_deg, -AUG_MAX_ATTITUDE_DELTA_DEG, AUG_MAX_ATTITUDE_DELTA_DEG
        )
        d_pitch_deg = constrain(
            d_pitch_deg, -AUG_MAX_ATTITUDE_DELTA_DEG, AUG_MAX_ATTITUDE_DELTA_DEG
        )
        return d_roll_deg, d_pitch_deg

    def reset(self, state: State) -> None:
        self.tick = 0
        self.aug_x.reset(vel=state.velocity.x)
        self.aug_y.reset(vel=state.velocity.y)
        self.aug_z.reset(vel=state.velocity.z)
        self.attitude_controller.reset_all_pid(
            state.attitude.roll, state.attitude.pitch, state.attitude.yaw
        )
        self.position_controller.reset_all_pid(
            state.position.x, state.position.y, state.position.z
        )
        self.attitude_desired.roll = state.attitude.roll
        self.attitude_desired.pitch = state.attitude.pitch
        self.attitude_desired.yaw = state.attitude.yaw
        self.rate_desired = Attitude()
        self.actuator_thrust = 0.0
        self._last_diag["v_aug_x"] = 0.0
        self._last_diag["v_aug_y"] = 0.0
        self._last_diag["v_aug_z"] = 0.0

    def step(
        self, setpoint: Setpoint, sensors: SensorData, state: State, sim_time: float
    ) -> Tuple[Control, Dict[str, float]]:
        self.tick += 1
        update_outer = (self.tick == 1) or ((self.tick % self.outer_div) == 0)

        if setpoint.mode.yaw == StabMode.MODE_VELOCITY:
            self.attitude_desired.yaw = cap_angle(
                self.attitude_desired.yaw + setpoint.attitude_rate.yaw * self.dt
            )
        elif setpoint.mode.yaw == StabMode.MODE_ABS:
            self.attitude_desired.yaw = setpoint.attitude.yaw
        self.attitude_desired.yaw = cap_angle(self.attitude_desired.yaw)

        if update_outer:
            thrust_nom, attitude_nom = self.position_controller.position_controller(
                setpoint, state
            )
            self.actuator_thrust = thrust_nom

            tilt_deg = max(abs(state.attitude.roll), abs(state.attitude.pitch))
            horiz_speed = math.hypot(state.velocity.x, state.velocity.y)
            apply_xy = (
                setpoint.mode.x == StabMode.MODE_ABS
                and setpoint.mode.y == StabMode.MODE_ABS
                and tilt_deg < AUG_DISABLE_IF_TILT_OVER_DEG
                and horiz_speed < AUG_DISABLE_IF_SPEED_OVER_MPS
            )

            if apply_xy:
                u_nom_x, u_nom_y = self._nominal_xy_input(
                    attitude_nom.roll, attitude_nom.pitch, self.attitude_desired.yaw
                )
                v_aug_x = self.aug_x.compute(
                    sim_time=sim_time,
                    vel=state.velocity.x,
                    u_nom=u_nom_x,
                )
                v_aug_y = self.aug_y.compute(
                    sim_time=sim_time,
                    vel=state.velocity.y,
                    u_nom=u_nom_y,
                )
            else:
                self.aug_x.reset(vel=state.velocity.x)
                self.aug_y.reset(vel=state.velocity.y)
                v_aug_x = 0.0
                v_aug_y = 0.0

            d_roll, d_pitch = self._attitude_delta_from_xy(
                v_aug_x, v_aug_y, self.attitude_desired.yaw
            )
            self.attitude_desired.roll = constrain(
                attitude_nom.roll + d_roll,
                -self.position_controller.r_limit,
                self.position_controller.r_limit,
            )
            self.attitude_desired.pitch = constrain(
                attitude_nom.pitch + d_pitch,
                -self.position_controller.p_limit,
                self.position_controller.p_limit,
            )

            apply_z = setpoint.mode.z == StabMode.MODE_ABS and AUG_ENABLE_Z
            if apply_z:
                u_nom_z = (
                    thrust_nom - self.position_controller.thrust_base
                ) / self.position_controller.thrust_scale
                v_aug_z = self.aug_z.compute(
                    sim_time=sim_time,
                    vel=state.velocity.z,
                    u_nom=u_nom_z,
                )
                thrust_delta = constrain(
                    v_aug_z * self.position_controller.thrust_scale,
                    -AUG_MAX_THRUST_DELTA,
                    AUG_MAX_THRUST_DELTA,
                )
                self.actuator_thrust = constrain(
                    thrust_nom + thrust_delta, 0.0, UINT16_MAX
                )
            else:
                self.aug_z.reset(vel=state.velocity.z)
                v_aug_z = 0.0

            self._last_diag["v_aug_x"] = v_aug_x
            self._last_diag["v_aug_y"] = v_aug_y
            self._last_diag["v_aug_z"] = v_aug_z

        if setpoint.mode.z == StabMode.MODE_DISABLE:
            self.actuator_thrust = setpoint.thrust

        if (
            setpoint.mode.x == StabMode.MODE_DISABLE
            or setpoint.mode.y == StabMode.MODE_DISABLE
        ):
            self.attitude_desired.roll = setpoint.attitude.roll
            self.attitude_desired.pitch = setpoint.attitude.pitch

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

        if setpoint.mode.roll == StabMode.MODE_VELOCITY:
            self.rate_desired.roll = setpoint.attitude_rate.roll
            self.attitude_controller.reset_roll_attitude_pid(state.attitude.roll)

        if setpoint.mode.pitch == StabMode.MODE_VELOCITY:
            self.rate_desired.pitch = setpoint.attitude_rate.pitch
            self.attitude_controller.reset_pitch_attitude_pid(state.attitude.pitch)

        self.attitude_controller.correct_rate_pid(
            sensors.gyro.x,
            -sensors.gyro.y,
            sensors.gyro.z,
            self.rate_desired.roll,
            self.rate_desired.pitch,
            self.rate_desired.yaw,
        )
        roll_out, pitch_out, yaw_out = self.attitude_controller.get_actuator_output()

        self.control.control_mode = ControlMode.CONTROL_MODE_LEGACY
        self.control.roll = roll_out
        self.control.pitch = pitch_out
        self.control.yaw = -yaw_out
        self.control.thrust = self.actuator_thrust

        if self.control.thrust == 0:
            self.control.roll = 0
            self.control.pitch = 0
            self.control.yaw = 0
            self.reset(state)

        return self.control, dict(self._last_diag)

    def get_logging_data(self, state: State, setpoint: Setpoint) -> Dict[str, float]:
        return {**super().get_logging_data(state, setpoint), **self._last_diag}
