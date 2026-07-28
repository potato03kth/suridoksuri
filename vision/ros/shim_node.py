"""컨테이너(`fc`, `ros:humble`) 안에서 도는 **얇은 ROS2 shim 노드** — `vision/ros/shim_core.py`의 어댑터.

실행(컨테이너 안):

    docker exec fc bash -lc '
      source /opt/ros/humble/setup.bash
      PYTHONPATH=/drone_ws/src/suridoksuri python3 -m vision.ros.shim_node
    '

확인:

    ros2 topic echo /vision/target_pose
    ros2 topic echo /vision/target_status

## 🔴 이 파일이 저장소에서 rclpy를 import하는 유일한 vision 파일이다

랩탑 `.venv`에도 RPi `picam-venv`에도 rclpy가 없다. 그래서 **`vision/ros/__init__.py`가
이 파일을 import하지 않고**, 변환 로직은 전부 rclpy를 모르는 `shim_core.py`에 있다.
이 파일은 "계획(dataclass) → 실제 msg 필드 대입"과 소켓 수명주기만 담당한다 —
**판단이 여기 있으면 랩탑에서 테스트할 수 없다.** 이 얇음 자체가 회귀테스트 대상이다
(`tests/test_shim_core.py`가 이 파일을 **소스 원문으로 읽어** 금지 패턴을 검사한다).

## 🔴 `msg.frame`은 `plan.frame`에서만 온다

`mavros_msgs/msg/LandingTarget`의 `frame` 상수는 실제 `MAV_FRAME` enum과 어긋나 있다
(`LOCAL_NED`가 msg에선 2, 실제로는 1). 정답인 정수 리터럴 12(`BODY_FRD`)는
`shim_core.MAV_FRAME_BODY_FRD` 하나에만 있고 이 파일은 그것을 **경유해서만** 쓴다.
`LandingTarget.LOCAL_NED` 같은 이름이 이 파일에 등장하는 순간 회귀테스트가 red가 된다.

## 왜 executor를 안 돌리는가

이 노드는 구독도 타이머도 서비스도 없다 — **발행만 한다.** rclpy 퍼블리셔는 executor 없이도
동작하므로 `rclpy.spin`을 띄우지 않고 메인 스레드가 곧장 소켓 루프를 돈다. 스레드가 하나뿐이라
종료 경로도 하나다(SIGTERM → `stop_event` → 루프 탈출 → `destroy_node`).

## 재접속은 shim의 책임이다

vision이 **서버**, shim이 **클라이언트**다(`utils/target_sink.py` 참조 — 문서 §8 권고와
반대 방향이고, 바뀌는 것은 "재접속 책임의 소재"뿐이다). 그래서 이 파일이 재접속 루프를 갖는다.
끊긴 동안에도 `heartbeat_period_s`마다 `vision/link` status를 `ERROR`로 내보낸다 —
**"생산자 사망"과 "shim 사망"을 가르는 유일한 신호**이기 때문이다(pose 침묵은 둘 다 같다).

## SIGTERM만 믿는다

이 저장소의 원격 프로세스는 비대화형 SSH 백그라운드 자식이라 bash가 SIGINT를 `SIG_IGN`으로
만들어 버리는 것이 `/proc/PID/status`로 실측 확인됐다(`tools/h264_stream.py` 1A).
`tools/h264_stream.py::_install_sigterm_handler`와 **동일 패턴**(핸들러는 이벤트만 세팅).
"""
from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from vision.ros.shim_core import (
    DEFAULT_HARDWARE_ID,
    DEFAULT_HOST,
    DEFAULT_LANDING_TARGET_TOPIC,
    DEFAULT_PORT,
    DEFAULT_POSE_FRAME_ID,
    DEFAULT_POSE_TOPIC,
    DEFAULT_STALE_WARN_S,
    DEFAULT_STATUS_TOPIC,
    DEFAULT_UNKNOWN_COVARIANCE,
    ShimConfig,
    ShimRouter,
)

#: `fc_ros`의 `_MAVROS_QOS`와 같은 BEST_EFFORT 계열(§9 F1). depth만 1인 이유는 이게 제어용
#: 스트림이라 **최신성 > 완전성**이기 때문 — 밀린 pose를 뒤늦게 받아봐야 유도엔 해롭다
#: (`utils/target_sink.py`의 drop-oldest 큐와 같은 철학). 호환성에 영향을 주는 것은
#: reliability/durability이지 depth가 아니다.
_DEFAULT_QOS_DEPTH = 1
#: 소켓 recv 타임아웃 = stop_event 확인 주기. `utils/target_sink.py::_POLL_S`와 같은 값·같은 이유.
_POLL_S = 0.2
_DEFAULT_RECONNECT_DELAY_S = 1.0
_DEFAULT_HEARTBEAT_PERIOD_S = 1.0
#: 한 줄의 상한. 정상 레코드는 ~600B다. 상한이 없으면 깨진 생산자가 개행 없이 계속 보낼 때
#: 메모리를 무한히 먹는다.
_MAX_LINE_BYTES = 1 << 20


class _LineReader:
    """소켓 → 줄 단위. **`makefile("r").readline()`을 쓰지 않는다.**

    소켓에 타임아웃을 걸어야 `stop_event`를 주기적으로 확인할 수 있는데, 버퍼드 텍스트
    스트림의 `readline()`은 타임아웃 예외가 나면 이미 읽은 부분 바이트의 행방이 구현
    의존이다. 직접 버퍼를 들고 있으면 타임아웃이 나도 **부분 줄이 그대로 보존**된다.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._buf = b""

    def read_lines(self) -> tuple:
        """`(lines, eof)`. 타임아웃이면 `([], False)` — 정상이며 EOF가 아니다."""
        try:
            chunk = self._sock.recv(65536)
        except socket.timeout:
            return [], False
        except OSError:
            return [], True
        if not chunk:
            return [], True  # 🔴 EOF = 생산자 사망
        self._buf += chunk
        if len(self._buf) > _MAX_LINE_BYTES:
            self._buf = b""
            return [], False
        lines = []
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            lines.append(raw.decode("utf-8", errors="replace"))
        return lines, False


class VisionShimNode(Node):
    """계획(dataclass) → 실제 ROS 메시지. **여기에 판단을 넣지 마라**(파일 docstring 참조)."""

    def __init__(self, cfg: ShimConfig, qos_depth: int):
        super().__init__("vision_shim")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )
        self.cfg = cfg
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, cfg.pose_topic, qos
        )
        self._status_pub = self.create_publisher(DiagnosticArray, cfg.status_topic, qos)
        self._lt_pub = None
        if cfg.publish_landing_target:
            # mavros_msgs는 이 노드가 LandingTarget 피벗을 켤 때만 필요하다 — 지연 import로
            # 두면 mavros 없는 최소 ROS 환경에서도 기본 경로가 돈다.
            from mavros_msgs.msg import LandingTarget

            self._landing_target_cls = LandingTarget
            self._lt_pub = self.create_publisher(
                LandingTarget, cfg.landing_target_topic, qos
            )
        self.published_poses = 0
        self.published_statuses = 0
        self.published_landing_targets = 0

    # ---------- 발행 ----------

    def emit(self, output) -> None:
        stamp = Time(nanoseconds=int(output.stamp_ns)).to_msg()
        if output.statuses:
            arr = DiagnosticArray()
            arr.header.stamp = stamp
            arr.header.frame_id = self.cfg.pose_frame_id
            for plan in output.statuses:
                st = DiagnosticStatus()
                # ⚠️ `DiagnosticStatus.level`은 msg 상 `byte` 타입이라 rclpy가 **1바이트
                # `bytes`** 를 요구한다(int를 넣으면 AssertionError). 실기체 컨테이너에서
                # 실측 확인한 사항이다.
                st.level = bytes([int(plan.level)])
                st.name = plan.name
                st.message = plan.message
                st.hardware_id = plan.hardware_id
                st.values = [KeyValue(key=k, value=v) for k, v in plan.values]
                arr.status.append(st)
            self._status_pub.publish(arr)
            self.published_statuses += 1
        if output.pose is not None:
            plan = output.pose
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = plan.frame_id
            msg.pose.pose.position.x = float(plan.position[0])
            msg.pose.pose.position.y = float(plan.position[1])
            msg.pose.pose.position.z = float(plan.position[2])
            msg.pose.pose.orientation.x = float(plan.orientation[0])
            msg.pose.pose.orientation.y = float(plan.orientation[1])
            msg.pose.pose.orientation.z = float(plan.orientation[2])
            msg.pose.pose.orientation.w = float(plan.orientation[3])
            msg.pose.covariance = [float(v) for v in plan.covariance]
            self._pose_pub.publish(msg)
            self.published_poses += 1
        if output.landing_target is not None and self._lt_pub is not None:
            plan = output.landing_target
            lt = self._landing_target_cls()
            lt.header.stamp = stamp
            lt.header.frame_id = plan.frame_id
            lt.target_num = int(plan.target_num)
            # 🔴 정수 리터럴 경로. msg 상수 금지 — 파일 docstring 참조.
            lt.frame = int(plan.frame)
            lt.angle = [float(plan.angle[0]), float(plan.angle[1])]
            lt.distance = float(plan.distance)
            lt.size = [float(plan.size[0]), float(plan.size[1])]
            lt.pose.position.x = float(plan.position[0])
            lt.pose.position.y = float(plan.position[1])
            lt.pose.position.z = float(plan.position[2])
            lt.pose.orientation.x = float(plan.orientation[0])
            lt.pose.orientation.y = float(plan.orientation[1])
            lt.pose.orientation.z = float(plan.orientation[2])
            lt.pose.orientation.w = float(plan.orientation[3])
            lt.type = int(plan.type)
            self._lt_pub.publish(lt)
            self.published_landing_targets += 1


def _install_sigterm_handler(stop_event: threading.Event) -> None:
    """`tools/h264_stream.py::_install_sigterm_handler`와 동일 패턴 — 이벤트만 세팅한다."""

    def _handler(signum, frame):  # noqa: ARG001 - signal handler 표준 시그니처
        stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        pass


def run(node: VisionShimNode, router: ShimRouter, args, stop_event: threading.Event) -> int:
    """접속 → 수신 → 발행 → (EOF면) 재접속. `stop_event`가 서면 빠져나온다."""
    endpoint = f"{args.host}:{args.port}"
    log = node.get_logger()
    last_heartbeat = 0.0

    while not stop_event.is_set():
        sock = None
        try:
            sock = socket.create_connection((args.host, args.port), timeout=_POLL_S)
        except OSError as e:
            now = time.monotonic()
            if now - last_heartbeat >= args.heartbeat_period_s:
                last_heartbeat = now
                for out in router.on_disconnected_heartbeat(
                    recv_wall_ns=time.time_ns(), endpoint=endpoint
                ):
                    node.emit(out)
                log.warn(f"생산자 접속 대기 중 ({endpoint}): {e}")
            stop_event.wait(args.reconnect_delay_s)
            continue

        sock.settimeout(_POLL_S)
        log.info(f"생산자 접속됨 ({endpoint})")
        for out in router.on_connected(recv_wall_ns=time.time_ns(), endpoint=endpoint):
            node.emit(out)
        reader = _LineReader(sock)

        while not stop_event.is_set():
            lines, eof = reader.read_lines()
            for line in lines:
                if not line.strip():
                    continue
                for out in router.push(
                    line,
                    recv_monotonic_ns=time.monotonic_ns(),
                    recv_wall_ns=time.time_ns(),
                ):
                    node.emit(out)
            if eof:
                # 🔴 §5.4 — EOF는 "생산자 사망"이다. pose는 침묵하고 status만 ERROR로 나간다.
                log.error("생산자 EOF — pose 발행 중단, /vision/target_status ERROR 발행")
                for out in router.on_producer_gone(
                    recv_wall_ns=time.time_ns(), reason="eof"
                ):
                    node.emit(out)
                break

        try:
            sock.close()
        except OSError:
            pass
        if args.exit_on_eof:
            return 0
        stop_event.wait(args.reconnect_delay_s)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vision JSON Lines → ROS2 (컨테이너 안 shim 노드)"
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    p.add_argument("--status-topic", default=DEFAULT_STATUS_TOPIC)
    p.add_argument("--landing-target-topic", default=DEFAULT_LANDING_TARGET_TOPIC)
    p.add_argument(
        "--publish-landing-target",
        action="store_true",
        help="🔴 기본 꺼짐 — px4_config.yaml의 listen_lt가 false라 구독자가 없다(FC 결정 D2).",
    )
    p.add_argument("--pose-frame-id", default=DEFAULT_POSE_FRAME_ID)
    p.add_argument("--stale-warn-s", type=float, default=DEFAULT_STALE_WARN_S)
    p.add_argument("--unknown-covariance", type=float, default=DEFAULT_UNKNOWN_COVARIANCE)
    p.add_argument("--hardware-id", default=DEFAULT_HARDWARE_ID)
    p.add_argument("--qos-depth", type=int, default=_DEFAULT_QOS_DEPTH)
    p.add_argument("--reconnect-delay-s", type=float, default=_DEFAULT_RECONNECT_DELAY_S)
    p.add_argument("--heartbeat-period-s", type=float, default=_DEFAULT_HEARTBEAT_PERIOD_S)
    p.add_argument(
        "--exit-on-eof",
        action="store_true",
        help="EOF에서 재접속하지 않고 종료(페일세이프 실증·검증용).",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ShimConfig(
        pose_topic=args.pose_topic,
        status_topic=args.status_topic,
        landing_target_topic=args.landing_target_topic,
        pose_frame_id=args.pose_frame_id,
        publish_landing_target=args.publish_landing_target,
        stale_warn_s=args.stale_warn_s,
        unknown_covariance=args.unknown_covariance,
        hardware_id=args.hardware_id,
    )
    stop_event = threading.Event()
    _install_sigterm_handler(stop_event)

    rclpy.init()
    node = VisionShimNode(cfg, args.qos_depth)
    router = ShimRouter(cfg)
    try:
        rc = run(node, router, args, stop_event)
    finally:
        node.get_logger().info(
            f"종료 — target={router.targets} state_hint={router.state_hints} "
            f"rejected={router.rejected} unpaired={router.unpaired_targets} "
            f"pose_pub={node.published_poses} status_pub={node.published_statuses} "
            f"lt_pub={node.published_landing_targets}"
        )
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
