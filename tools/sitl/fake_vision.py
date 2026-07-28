#!/usr/bin/env python3
"""F2 SITL 검증용 합성 vision 발행기 — `vision/ros/shim_node.py` 의 와이어 계약만 흉내낸다.

SITL 에는 카메라도 shim 도 없다. 그래서 핸드오프 §2 의 계약(토픽·프레임·QoS·level 형식)을
그대로 재현하는 최소 발행기를 만들어 `offboard_node` 의 F2 서브상태를 자극한다.

🔴 계약 재현 항목
  - `/vision/landing_setpoint`  PoseStamped, frame_id="map", **ENU**(x=E, y=N, z=up)
  - `/vision/target_status`     DiagnosticArray, level 은 **1바이트 bytes**(shim 과 동일)
  - QoS BEST_EFFORT / KEEP_LAST / depth=1
  - 발행 주파수 기본 4.4Hz (실측값. 10Hz 아님)
  - 하트비트(`vision/link`) 1Hz

스케줄은 node.log 의 정규식이 처음 나타난 시각(t0)을 기준으로 잡는다.

## 사용법

    # HOLD → VISION_SEARCH 진입을 로그로 감지한 뒤부터 타겟을 계속 발행
    python3 tools/sitl/fake_vision.py \
        --enu-x 6.0 --enu-y 302.0 --enu-z 0.2 \
        --watch-log <run>/node.log --on-match 'VISION_SEARCH' \
        --landhint-at 40.0

    # 장애주입 (인수인계 §6-1)
    --emit-count 1      단발 오탐 1프레임 (#5 — 탐색이 중단되면 안 된다)
    --veto-at 5         vision 거부권 지속 (#2)
    --linkdead-at 10    생산자 사망 = vision/link ERROR (#1)
    --exit-at 10        shim 자체 사망 = status 까지 침묵 (#4)

⚠️ 발행 주파수 기본 4.4Hz 는 **실측값**이다(median 0.2207s / p95 0.310s).
`--rate 10` 으로 올려 시험하면 래치·stale 판정이 실기체보다 후하게 나온다.

이 파일은 2026-07-29 F2 검증에서 폐기 클론(`/root/f2tools/`)에만 있던 것을
저장소로 들여온 것이다 — 로컬 전용 도구가 재현을 망친 이력이 반복돼서다.
"""
import argparse
import math
import re
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

OK, WARN, ERROR = 0, 1, 2


def kv(k, v):
    m = KeyValue()
    m.key = str(k)
    m.value = str(v)
    return m


def st(name, level, message, values=()):
    s = DiagnosticStatus()
    s.name = name
    s.level = bytes([int(level)])      # 🔴 shim_node.py:221 과 같은 형식
    s.message = message
    s.hardware_id = "fake_vision"
    s.values = list(values)
    return s


class FakeVision(Node):
    def __init__(self, a):
        super().__init__("fake_vision")
        self.a = a
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_sp = self.create_publisher(PoseStamped, "/vision/landing_setpoint", qos)
        self.pub_st = self.create_publisher(DiagnosticArray, "/vision/target_status", qos)
        self.t_start = time.monotonic()
        self.t0 = None                 # 트리거 성립 시각
        self.emitted = 0
        self.log_path = Path(a.watch_log) if a.watch_log else None
        self.pat = re.compile(a.on_match) if a.on_match else None
        self._log_pos = 0
        self.create_timer(1.0 / a.rate, self.tick)
        self.create_timer(1.0, self.heartbeat)
        self.get_logger().info(
            f"fake_vision 시작 mode={a.mode} target ENU=({a.enu_x},{a.enu_y},{a.enu_z}) "
            f"rate={a.rate}Hz watch={a.watch_log} match={a.on_match!r}")

    # ── 트리거 ────────────────────────────────────────────────
    def trigger_check(self):
        if self.t0 is not None or self.pat is None:
            if self.pat is None and self.t0 is None:
                self.t0 = self.t_start
            return
        if not self.log_path or not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r", errors="replace") as f:
                f.seek(self._log_pos)
                chunk = f.read()
                self._log_pos = f.tell()
        except OSError:
            return
        if chunk and self.pat.search(chunk):
            self.t0 = time.monotonic()
            self.get_logger().warn(f"트리거 성립 t={self.t0 - self.t_start:.1f}s")

    def rel(self):
        return None if self.t0 is None else time.monotonic() - self.t0

    # ── 상태 발행 ─────────────────────────────────────────────
    def heartbeat(self):
        self.trigger_check()
        r = self.rel()
        a = self.a
        if r is not None and a.exit_at >= 0 and r >= a.exit_at:
            self.get_logger().error(f"exit_at {a.exit_at}s 도달 → 프로세스 종료(shim 사망 모사)")
            raise SystemExit(0)
        link_dead = (r is not None and a.linkdead_at >= 0 and r >= a.linkdead_at)
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [st("vision/link", ERROR if link_dead else OK,
                         "producer_dead" if link_dead else "alive",
                         [kv("heartbeat_period_s", "1.0")])]
        self.pub_st.publish(msg)

    def emitting(self):
        r = self.rel()
        a = self.a
        if r is None or r < a.emit_from:
            return False
        if a.linkdead_at >= 0 and r >= a.linkdead_at:
            return False
        if a.emit_count >= 0 and self.emitted >= a.emit_count:
            return False
        return True

    def tick(self):
        self.trigger_check()
        r = self.rel()
        a = self.a
        if r is None:
            return
        emit = self.emitting()
        veto = (a.veto_at >= 0 and r >= a.veto_at)
        land_hint = (a.landhint_at >= 0 and r >= a.landhint_at)

        now = self.get_clock().now().to_msg()
        if emit:
            p = PoseStamped()
            p.header.stamp = now
            p.header.frame_id = "map"
            # 🔴 ENU. 흔들림(jitter)을 주어 산포 필터가 실제로 도는지도 본다.
            j = a.jitter * math.sin(2.0 * math.pi * 0.7 * r)
            p.pose.position.x = float(a.enu_x + j)
            p.pose.position.y = float(a.enu_y + j)
            p.pose.position.z = float(a.enu_z)
            # 기수방위 유지 = 현재 yaw. 여기서는 yaw_enu=0(동쪽) 대신 북(π/2)으로 둔다.
            yaw_enu = math.pi / 2.0
            p.pose.orientation.z = math.sin(yaw_enu / 2.0)
            p.pose.orientation.w = math.cos(yaw_enu / 2.0)
            self.pub_sp.publish(p)
            self.emitted += 1

        msg = DiagnosticArray()
        msg.header.stamp = now
        state = "HOLD" if veto else ("TERMINAL" if land_hint else "LOCK")
        hint = "hold" if veto else ("land" if land_hint else "center")
        msg.status = [
            st("vision/target", OK if emit else WARN,
               "ok" if emit else "no_target"),
            st("vision/state", WARN if veto else OK, f"state={state}",
               [kv("state", state), kv("command_hint", hint),
                kv("command_is_advisory", "true")]),
            st("vision/setpoint", OK if emit else WARN,
               "ok" if emit else "no_target"),
        ]
        self.pub_st.publish(msg)
        if self.emitted and self.emitted % 22 == 0:
            self.get_logger().info(
                f"t={r:.1f}s emitted={self.emitted} veto={veto} hint={land_hint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="steady")
    ap.add_argument("--enu-x", type=float, default=0.0)
    ap.add_argument("--enu-y", type=float, default=0.0)
    ap.add_argument("--enu-z", type=float, default=0.0)
    ap.add_argument("--rate", type=float, default=4.4)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--watch-log", default=None)
    ap.add_argument("--on-match", default=None)
    ap.add_argument("--emit-from", type=float, default=0.0)
    ap.add_argument("--emit-count", type=int, default=-1)
    ap.add_argument("--veto-at", type=float, default=-1.0)
    ap.add_argument("--landhint-at", type=float, default=-1.0)
    ap.add_argument("--linkdead-at", type=float, default=-1.0)
    ap.add_argument("--exit-at", type=float, default=-1.0)
    a = ap.parse_args()
    rclpy.init()
    n = FakeVision(a)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
