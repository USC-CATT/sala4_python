#!/usr/bin/env python3
"""Lightweight live telemetry display for host-side Crazyflie controllers."""

from multiprocessing import Process, Queue, Event
from queue import Empty, Full

import math
from controller.controller_types import (
    AccData,
    Attitude,
    AttitudeRate,
    Axis3f,
    Control,
    GyroData,
    Position,
    SensorData,
    Setpoint,
    SetpointMode,
    StabMode,
    State,
    Velocity,
    MotorThrust,
)
import time
from collections import deque
from typing import Optional

try:
    import matplotlib

    try:
        matplotlib.use("QtAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
    import matplotlib.pyplot as plt
except Exception as exc:
    matplotlib = None
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

BOX_SIZE_M = 2.0
UPDATE_HZ = 15.0
HISTORY_SECONDS = 90.0
STALE_TIMEOUT_S = 0.75

BG_COLOR = "#0f172a"
PANEL_BG = "#1e293b"
FG_COLOR = "#e2e8f0"
MUTED_COLOR = "#94a3b8"
GRID_COLOR = "#334155"
TRAIL_COLOR = "#38bdf8"
DRONE_COLOR = "#f43f5e"
SETPOINT_COLOR = "#4ade80"
ACCENT_COLOR = "#facc15"


class _NullDisplay:
    def __init__(self, reason: str):
        print(f"[LiveDisplay] Disabled: {reason}")

    def log_callback(self, _timestamp, data, _logconf):
        return

    def update_state(self, **_kwargs):
        return

    def update_position(self, *_args, **_kwargs):
        return

    def update_velocity(self, *_args, **_kwargs):
        return

    def update_attitude(self, *_args, **_kwargs):
        return

    def update_gyro(self, *_args, **_kwargs):
        return

    def update_battery(self, *_args, **_kwargs):
        return

    def update_setpoint(self, *_args, **_kwargs):
        return

    def close(self):
        return


def Telemetry(*args, **kwargs):
    if _MATPLOTLIB_IMPORT_ERROR is not None:
        return _NullDisplay(str(_MATPLOTLIB_IMPORT_ERROR))
    return _TelemetryDisplay(*args, **kwargs)


def _run_child(queue, stop_event, box_size_m, update_hz, history_s):
    renderer = _TelemetryRenderer(queue, stop_event, box_size_m, update_hz, history_s)
    renderer.run()


class _TelemetryRenderer:
    def __init__(self, queue, stop_event, box_size_m, update_hz, history_s):
        self._queue = queue
        self._stop = stop_event
        self._half = box_size_m / 2.0
        self._period = 1.0 / max(update_hz, 1.0)
        max_points = max(100, int(history_s * 50))

        self._position = None
        self._velocity = None
        self._attitude = None
        self._gyro = None
        self._setpoint = None
        self._thrust = None
        self._vbat = None
        self._last_update = None
        self._trail_x = deque(maxlen=max_points)
        self._trail_y = deque(maxlen=max_points)
        self._trail_z = deque(maxlen=max_points)

        self._fig = None

    def run(self):
        self._build_figure()
        if self._fig is None:
            self._stop.wait()
            return
        plt.show(block=False)
        while not self._stop.wait(self._period):
            self._drain()
            try:
                self._render()
            except Exception:
                pass
        try:
            plt.close(self._fig)
        except Exception:
            pass

    def _drain(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if item is None:
                self._stop.set()
                return
            self._apply(item)

    def _apply(self, data):
        if data.get("position") is not None:
            p = data["position"]
            self._position = p
            self._trail_x.append(p.x)
            self._trail_y.append(p.y)
            self._trail_z.append(p.z)
        if data.get("velocity") is not None:
            self._velocity = data["velocity"]
        if data.get("attitude") is not None:
            self._attitude = data["attitude"]
        if data.get("gyro") is not None:
            self._gyro = data["gyro"]
        if data.get("thrust") is not None:
            self._thrust = data["thrust"]
        if data.get("battery") is not None:
            self._vbat = data["battery"]
        if data.get("setpoint") is not None:
            self._setpoint = data["setpoint"]
        self._last_update = time.monotonic()

    def _build_figure(self):
        try:
            plt.ion()
            fig = plt.figure(figsize=(12.5, 6.5), facecolor=BG_COLOR)
            fig.canvas.manager.set_window_title("Crazyflie Telemetry")
            ax_3d = fig.add_axes([0.03, 0.08, 0.55, 0.84], projection="3d", facecolor=PANEL_BG)
            ax_text = fig.add_axes([0.62, 0.34, 0.35, 0.58], facecolor=PANEL_BG)
            ax_bat = fig.add_axes([0.62, 0.12, 0.35, 0.10], facecolor=PANEL_BG)
        except Exception as exc:
            print(f"[LiveDisplay] Could not open a window ({exc}); running headless.")
            self._fig = None
            return

        self._fig = fig
        self._ax_3d = ax_3d
        self._ax_text = ax_text
        self._ax_bat = ax_bat

        h = self._half
        top = 2.0 * h
        edges = [
            ((-h, -h, 0.0), (h, -h, 0.0)), ((-h, h, 0.0), (h, h, 0.0)),
            ((-h, -h, top), (h, -h, top)), ((-h, h, top), (h, h, top)),
            ((-h, -h, 0.0), (-h, h, 0.0)), ((h, -h, 0.0), (h, h, 0.0)),
            ((-h, -h, top), (-h, h, top)), ((h, -h, top), (h, h, top)),
            ((-h, -h, 0.0), (-h, -h, top)), ((h, -h, 0.0), (h, -h, top)),
            ((-h, h, 0.0), (-h, h, top)), ((h, h, 0.0), (h, h, top)),
        ]
        for start, end in edges:
            ax_3d.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=GRID_COLOR, linewidth=1.0,
            )
        ax_3d.plot([0, 0], [0, 0], [0.0, top], color=GRID_COLOR, linewidth=0.6, alpha=0.5)
        ax_3d.plot([0, 0], [-h, h], [0.0, 0.0], color=GRID_COLOR, linewidth=0.6, alpha=0.5)
        ax_3d.plot([-h, h], [0, 0], [0.0, 0.0], color=GRID_COLOR, linewidth=0.6, alpha=0.5)

        (self._trail_line,) = ax_3d.plot([], [], [], color=TRAIL_COLOR, linewidth=1.8, alpha=0.9)
        (self._drone_point,) = ax_3d.plot([], [], [], marker="o", color=DRONE_COLOR, markersize=9)
        (self._setpoint_point,) = ax_3d.plot([], [], [], marker="X", color=SETPOINT_COLOR, markersize=9)

        ax_3d.set_xlim(-h, h)
        ax_3d.set_ylim(-h, h)
        ax_3d.set_zlim(0.0, top)
        ax_3d.set_box_aspect((1, 1, 1))
        ax_3d.set_xlabel("X (m)", color=MUTED_COLOR, labelpad=0)
        ax_3d.set_ylabel("Y (m)", color=MUTED_COLOR, labelpad=0)
        ax_3d.set_zlabel("Z (m)", color=MUTED_COLOR, labelpad=0)
        ax_3d.tick_params(colors=MUTED_COLOR, labelsize=8)
        ax_3d.set_title(f"Flight Box  {2 * self._half:.0f} x {2 * self._half:.0f} x {2 * self._half:.0f} m",
                        color=FG_COLOR, fontsize=12, pad=12)
        try:
            ax_3d.xaxis.set_pane_color(BG_COLOR)
            ax_3d.yaxis.set_pane_color(BG_COLOR)
            ax_3d.zaxis.set_pane_color(BG_COLOR)
        except Exception:
            pass

        ax_text.set_xticks([])
        ax_text.set_yticks([])
        for spine in ax_text.spines.values():
            spine.set_color(GRID_COLOR)
        self._text_artist = ax_text.text(
            0.04, 0.97, "", transform=ax_text.transAxes, family="monospace",
            fontsize=11, color=FG_COLOR, verticalalignment="top",
        )

        ax_bat.set_xlim(3.0, 4.3)
        ax_bat.set_ylim(0.0, 1.0)
        ax_bat.set_yticks([])
        ax_bat.tick_params(colors=MUTED_COLOR, labelsize=8)
        ax_bat.set_xlabel("Battery (V)", color=MUTED_COLOR, fontsize=9)
        for spine in ("top", "right", "left"):
            ax_bat.spines[spine].set_visible(False)
        self._bat_artist = ax_bat.barh([0.5], [0.0], height=0.55, color=ACCENT_COLOR)[0]
        self._bat_label = ax_bat.text(
            0.98, 0.5, "", transform=ax_bat.transAxes, family="monospace",
            fontsize=11, color=FG_COLOR, verticalalignment="center", horizontalalignment="right",
        )

    def _render(self):
        position = self._position
        velocity = self._velocity
        attitude = self._attitude
        gyro = self._gyro
        setpoint = self._setpoint
        vbat = self._vbat
        last_update = self._last_update
        trail = (list(self._trail_x), list(self._trail_y), list(self._trail_z))

        stale = last_update is None or (time.monotonic() - last_update) > STALE_TIMEOUT_S

        self._trail_line.set_data(trail[0], trail[1])
        self._trail_line.set_3d_properties(trail[2])
        if position is not None and not stale:
            self._drone_point.set_data([position.x], [position.y])
            self._drone_point.set_3d_properties([position.z])
        else:
            self._drone_point.set_data([], [])
            self._drone_point.set_3d_properties([])
        if setpoint is not None:
            self._setpoint_point.set_data([setpoint.position.x], [setpoint.position.y])
            self._setpoint_point.set_3d_properties([setpoint.position.z])
        else:
            self._setpoint_point.set_data([], [])
            self._setpoint_point.set_3d_properties([])

        self._text_artist.set_text(self._format_panel(position, velocity, attitude, gyro, setpoint, stale))

        if vbat is not None:
            self._bat_artist.set_width(max(0.0, min(vbat, 4.3) - 3.0))
            self._bat_artist.set_color(self._battery_color(vbat))
            self._bat_label.set_text(f"{vbat:.2f} V")
        else:
            self._bat_label.set_text("---")

        self._fig.canvas.draw()
        tk_canvas = getattr(self._fig.canvas, "_tkcanvas", None)
        if tk_canvas is not None:
            tk_canvas.update()
        else:
            self._fig.canvas.flush_events()

    def _format_panel(self, position, velocity, attitude, gyro, setpoint, stale):
        def g(obj, name):
            if obj is None:
                return None
            return getattr(obj, name, None)

        px, py, pz = g(position, "x"), g(position, "y"), g(position, "z")
        vx, vy, vz = g(velocity, "x"), g(velocity, "y"), g(velocity, "z")
        roll = g(attitude, "roll"); pitch = g(attitude, "pitch"); yaw = g(attitude, "yaw")
        gx, gy, gz = g(gyro, "x"), g(gyro, "y"), g(gyro, "z")

        sp = None
        if setpoint is not None:
            sp = getattr(setpoint, "position", None)
        sx, sy, sz = g(sp, "x"), g(sp, "y"), g(sp, "z")

        rows = [
            "  POSITION (m)        VELOCITY (m/s)",
            f"  x  {self._fmt(px)}   vx  {self._fmt(vx)}",
            f"  y  {self._fmt(py)}   vy  {self._fmt(vy)}",
            f"  z  {self._fmt(pz)}   vz  {self._fmt(vz)}",
            "",
            "  ATTITUDE (deg)      GYRO (deg/s)",
            f"  roll  {self._fmt(roll)}   gx  {self._fmt(gx)}",
            f"  pitch {self._fmt(pitch)}   gy  {self._fmt(gy)}",
            f"  yaw   {self._fmt(yaw)}   gz  {self._fmt(gz)}",
            "",
            "  SETPOINT (m)",
            f"  x  {self._fmt(sx)}    y  {self._fmt(sy)}    z  {self._fmt(sz)}",
        ]
        if stale:
            rows.append("")
            rows.append("  NO DATA")
        return "\n".join(rows)

    @staticmethod
    def _fmt(value):
        if value is None:
            return "  ---  "
        return f"{value:+7.2f}"

    @staticmethod
    def _battery_color(vbat):
        if vbat >= 3.9:
            return "#4ade80"
        if vbat >= 3.6:
            return "#facc15"
        return "#f87171"


class _TelemetryDisplay:
    def __init__(self, box_size_m=BOX_SIZE_M, update_hz=UPDATE_HZ, history_s=HISTORY_SECONDS):
        self._queue = Queue()
        self._stop = Event()
        self._process = Process(target=_run_child, args=(self._queue, self._stop, box_size_m, update_hz, history_s), daemon=True)
        self._process.start()

    def update_state(self, position=None, velocity=None, attitude=None, thrust=None,
                     gyro=None, battery=None, setpoint=None):
        try:
            self._queue.put_nowait({
                "position": position,
                "velocity": velocity,
                "attitude": attitude,
                "thrust": thrust,
                "gyro": gyro,
                "battery": battery,
                "setpoint": setpoint,
            })
        except Full:
            pass

    def update_position(self, x, y, z):
        self.update_state(position=Position(x, y, z))

    def update_velocity(self, vx, vy, vz):
        self.update_state(velocity=Velocity(vx, vy, vz))

    def update_attitude(self, roll, pitch, yaw):
        self.update_state(attitude=Attitude(roll, pitch, yaw))

    def update_gyro(self, gx, gy, gz):
        self.update_state(gyro=GyroData(gx, gy, gz))

    def update_battery(self, vbat):
        self.update_state(battery=vbat)

    def update_setpoint(self, x, y, z):
        self.update_state(setpoint=Setpoint(Position(x, y, z)))

    @staticmethod
    def triple(data, key_x, key_y, key_z):
        return (data[key_x], data[key_y], data[key_z])

    def close(self):
        try:
            self._queue.put(None)
        except Exception:
            pass
        self._stop.set()
        if self._process.is_alive():
            self._process.join(timeout=2.0)


def _demo():

    display = Telemetry()
    try:
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            display.update_state(
                position=Position(0.8 * math.cos(0.5 * t), 0.8 * math.sin(0.5 * t), 0.8 + 0.1 * math.sin(t)),
                velocity=Velocity(-0.4 * math.sin(0.5 * t), 0.4 * math.cos(0.5 * t), 0.1 * math.cos(t)),
                attitude=Attitude(5.0 * math.sin(0.5 * t), -5.0 * math.cos(0.5 * t), t * 5.0 % 360.0),
                gyro=GyroData(10.0 * math.cos(0.5 * t), 10.0 * math.sin(0.5 * t), 2.0),
                battery=4.2 - 0.0001 * t,
                setpoint=Setpoint(Position(0.8, 0.0, 0.85)),
            )
            time.sleep(0.04)
    except KeyboardInterrupt:
        pass
    finally:
        display.close()


if __name__ == "__main__":
    _demo()
