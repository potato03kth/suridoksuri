#!/usr/bin/env python3
"""FW 오프보드 setpoint 대안 프로브 (SITL 전용, 비행코드 무수정).

SITL-7 S3 근본원인 규명용. `run_scenario.py` 하니스와 무관하게 **최소 비행**만
만들어서, `/mavros/setpoint_raw/local`(PositionTarget)의 type_mask 조합별로
**고정익이 실제로 선회하는지**만 측정한다.

시퀀스
  MAVROS 연결 → 프리플라이트 우회 파라미터 → ARM → CommandTOL(MC 이륙)
  → MC OFFBOARD 유지 → MAV_CMD_DO_VTOL_TRANSITION(param1=4, FW)
  → FW 확인 후 페이즈별 setpoint 스트리밍

페이즈(기본):
  pos_vel   위치+속도 (type_mask: accel/yaw/yaw_rate 무시)   ← A안 핵심
  vel_only  속도만    (type_mask: pos/accel/yaw/yaw_rate 무시) ← B안

판정: 각 페이즈 시작 시점 대비 **동/북 변위비**와 지상코스 변화.
목표는 전부 정동(EAST) — 정북으로 계속 날면 "선회 안 함".

사용:
  python3 tools/sitl/fw_offboard_probe.py --out /tmp/probe

주의: PX4/MAVROS는 이 스크립트가 직접 띄우지 않는다. 별도로 먼저 기동할 것.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from mavros_msgs.srv import CommandBool, CommandLong, CommandTOL, SetMode

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)

# PositionTarget type_mask 비트
IGN_PX, IGN_PY, IGN_PZ = 1, 2, 4
IGN_VX, IGN_VY, IGN_VZ = 8, 16, 32
IGN_AFX, IGN_AFY, IGN_AFZ = 64, 128, 256
FORCE, IGN_YAW, IGN_YAW_RATE = 512, 1024, 2048

MASK_POS_ONLY = (IGN_VX | IGN_VY | IGN_VZ | IGN_AFX | IGN_AFY | IGN_AFZ
                 | IGN_YAW | IGN_YAW_RATE)
MASK_POS_VEL = (IGN_AFX | IGN_AFY | IGN_AFZ | IGN_YAW | IGN_YAW_RATE)
MASK_VEL_ONLY = (IGN_PX | IGN_PY | IGN_PZ | IGN_AFX | IGN_AFY | IGN_AFZ
                 | IGN_YAW | IGN_YAW_RATE)

BYPASS = {"CBRK_SUPPLY_CHK": 894281, "NAV_DLL_ACT": 0}


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def set_bypass_params(log):
    for name, val in BYPASS.items():
        ok = False
        for _ in range(40):
            rc, out = sh(["ros2", "param", "set", "/mavros/param",
                          name, str(val)], timeout=25)
            rc2, out2 = sh(["ros2", "param", "get", "/mavros/param", name],
                           timeout=20)
            if f"value is: {val}" in out2:
                ok = True
                break
            time.sleep(3.0)
        log(f"param {name}={val} -> {'ok' if ok else 'FAIL'}")


class Probe(Node):
    def __init__(self, args):
        super().__init__("fw_offboard_probe")
        self.args = args
        self.state = State()
        self.ext = ExtendedState()
        self.pose = None          # ENU
        self.rows = []
        self.phase = "init"
        self.sp_mode = "hold"     # hold | pos_only | pos_vel | vel_only
        self.hold_enu = (0.0, 0.0, 50.0)
        self.t_phase0 = None

        self.create_subscription(State, "/mavros/state", self._cb_state, QOS)
        self.create_subscription(ExtendedState, "/mavros/extended_state",
                                 self._cb_ext, QOS)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose",
                                 self._cb_pose, QOS)
        self.raw_pub = self.create_publisher(
            PositionTarget, "/mavros/setpoint_raw/local", QOS)
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.cmd_cli = self.create_client(CommandLong, "/mavros/cmd/command")
        self.tol_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.create_timer(0.05, self._stream)     # 20 Hz setpoint
        self.create_timer(0.20, self._record)

    # ── 콜백 ──────────────────────────────────────────────────────────
    def _cb_state(self, m):
        self.state = m

    def _cb_ext(self, m):
        self.ext = m

    def _cb_pose(self, m):
        self.pose = m

    def enu(self):
        p = self.pose.pose.position
        return p.x, p.y, p.z          # E, N, Up

    # ── setpoint 스트리밍 ─────────────────────────────────────────────
    def _stream(self):
        if self.pose is None:
            return
        e, n, u = self.enu()
        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # =1, MAVROS가 ENU→NED

        if self.sp_mode == "hold":
            m.type_mask = MASK_POS_ONLY
            m.position.x, m.position.y, m.position.z = self.hold_enu
        elif self.sp_mode in ("pos_only", "pos_vel"):
            # 정동(EAST) 목표: 현재 위치에서 동쪽으로 args.lead m 앞
            tx = self.tgt_enu[0]
            ty = self.tgt_enu[1]
            m.type_mask = (MASK_POS_ONLY if self.sp_mode == "pos_only"
                           else MASK_POS_VEL)
            m.position.x, m.position.y, m.position.z = tx, ty, self.hold_enu[2]
            if self.sp_mode == "pos_vel":
                # 경로 접선 = 정동(ENU +x)
                m.velocity.x = float(self.args.speed)
                m.velocity.y = 0.0
                m.velocity.z = 0.0
        elif self.sp_mode == "vel_only":
            m.type_mask = MASK_VEL_ONLY
            m.velocity.x = float(self.args.speed)
            m.velocity.y = 0.0
            m.velocity.z = 0.0
        else:
            return
        self.raw_pub.publish(m)

    def _record(self):
        if self.pose is None:
            return
        e, n, u = self.enu()
        self.rows.append({
            "t": time.time(), "phase": self.phase, "sp_mode": self.sp_mode,
            "east": round(e, 2), "north": round(n, 2), "up": round(u, 2),
            "mode": self.state.mode, "armed": self.state.armed,
            "vtol_state": self.ext.vtol_state,
        })

    # ── 헬퍼 ──────────────────────────────────────────────────────────
    def log(self, s):
        self.get_logger().info(s)

    def spin(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, cli, req, name):
        if not cli.wait_for_service(timeout_sec=15.0):
            self.log(f"{name}: 서비스 없음")
            return None
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 15 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        r = fut.result()
        self.log(f"{name} -> {r}")
        return r

    def wait_until(self, pred, timeout, label):
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                self.log(f"{label}: 도달 ({time.time()-t0:.1f}s)")
                return True
        self.log(f"{label}: TIMEOUT ({timeout}s)")
        return False

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        return self.call(self.mode_cli, req, f"set_mode({mode})")

    # ── 본 시퀀스 ─────────────────────────────────────────────────────
    def run(self):
        self.phase = "wait_conn"
        self.wait_until(lambda: self.state.connected, 120, "MAVROS 연결")
        self.wait_until(lambda: self.pose is not None, 60, "local_position")

        self.phase = "arm"
        req = CommandBool.Request()
        req.value = True
        self.call(self.arm_cli, req, "ARM")
        self.wait_until(lambda: self.state.armed, 30, "armed")

        self.phase = "takeoff"
        req = CommandTOL.Request()
        req.altitude = float(self.args.alt)   # AMSL 근사(SITL 지면 ~0)
        req.min_pitch = 0.0
        # ⚠️ MAV_CMD_NAV_TAKEOFF 관례상 "현재 위치 사용"은 NaN. 0.0/0.0 은 실제
        #    좌표(널 아일랜드)로 해석돼 이륙이 취소된다 (작업 H, 커밋 000f478).
        req.latitude = float("nan")
        req.longitude = float("nan")
        self.call(self.tol_cli, req, "CommandTOL")
        self.wait_until(lambda: self.enu()[2] >= self.args.alt - 5.0,
                        180, f"고도 {self.args.alt}m")

        e, n, u = self.enu()
        self.hold_enu = (e, n, float(self.args.alt))
        self.phase = "mc_offboard"
        self.sp_mode = "hold"
        self.spin(2.0)
        self.set_mode("OFFBOARD")
        self.wait_until(lambda: self.state.mode == "OFFBOARD", 30, "OFFBOARD")
        self.spin(5.0)

        self.phase = "transition"
        req = CommandLong.Request()
        req.command = 3000            # MAV_CMD_DO_VTOL_TRANSITION
        req.param1 = 4.0              # MAV_VTOL_STATE_FW
        self.call(self.cmd_cli, req, "DO_VTOL_TRANSITION(FW)")
        self.wait_until(lambda: self.ext.vtol_state == 4, 90, "FW 천이")
        self.spin(3.0)

        for name in self.args.phases.split(","):
            name = name.strip()
            if not name:
                continue
            e, n, u = self.enu()
            # 정동 목표: 현재 지점에서 동쪽 lead m (북 성분 동일 유지)
            self.tgt_enu = (e + float(self.args.lead), n)
            self.log(f"=== PHASE {name}: 시작지점 E={e:.1f} N={n:.1f} "
                     f"목표 E={self.tgt_enu[0]:.1f} N={self.tgt_enu[1]:.1f} "
                     f"(정동 {self.args.lead}m) ===")
            self.phase = name
            self.sp_mode = name
            p0 = (e, n)
            self.spin(float(self.args.dur))
            e1, n1, _ = self.enu()
            de, dn = e1 - p0[0], n1 - p0[1]
            crs = math.degrees(math.atan2(de, dn)) % 360.0
            self.log(f"=== PHASE {name} 결과: dE={de:+.1f}m dN={dn:+.1f}m "
                     f"평균지상코스={crs:.1f}° (정동=90°, 정북=0°) "
                     f"mode={self.state.mode} vtol={self.ext.vtol_state} ===")

        self.phase = "done"
        self.sp_mode = "hold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fw_probe")
    ap.add_argument("--alt", type=float, default=50.0)
    ap.add_argument("--dur", type=float, default=70.0)
    ap.add_argument("--lead", type=float, default=800.0)
    ap.add_argument("--speed", type=float, default=18.0)
    ap.add_argument("--phases", default="pos_vel,vel_only,pos_only")
    args = ap.parse_args()

    rclpy.init()
    node = Probe(args)
    set_bypass_params(node.get_logger().info)
    try:
        node.run()
    finally:
        path = args.out + ".csv"
        with open(path, "w", newline="") as f:
            if node.rows:
                w = csv.DictWriter(f, fieldnames=list(node.rows[0].keys()))
                w.writeheader()
                w.writerows(node.rows)
        node.get_logger().info(f"기록: {path} ({len(node.rows)} 행)")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main() or 0)
