"""
`TargetSink` 포트(`docs/vision_plan.md` §7.2) + localhost TCP 소켓 구현 (작업 V1).

`docs/vision_fc_interface.md` §8 권고 R4의 **호스트 쪽 절반**이다. 컨테이너 안의 shim 노드
(Phase 2, 이 파일 범위 밖)가 클라이언트로 접속해 JSON Lines를 읽어 ROS2 토픽으로 재발행한다.

```
[호스트 / picam-venv / Py3.12 / ROS 없음]        [fc 컨테이너 / Humble / Py3.10]
 vision main.py                                   vision_bridge_node  ← Phase 2
   └ LandingStateMachine ─┐                          │
   └ TargetEstimate ──────┤                          │
                          │                          │
                  SocketTargetSink ── TCP ─────────▶ (클라이언트로 접속)
                  (이 파일)         127.0.0.1:8091
```

## 왜 TCP인가 (Unix domain socket이 아니라)

**결정적 근거는 컨테이너 설정 2줄이다**(§2.1, 실측 확인된 사실):

| 사실 | TCP에 대한 함의 | UDS에 대한 함의 |
|---|---|---|
| `NetworkMode=host` | 컨테이너가 호스트 네트워크 네임스페이스를 **그대로 공유** → `127.0.0.1`이 문자 그대로 같은 루프백. **추가 설정 0** | 무관 |
| 마운트가 `/home/suri/drone_ws:/drone_ws` **하나뿐** | 무관 | 소켓 파일이 컨테이너 마운트 네임스페이스에 보여야 하므로 **새 bind mount를 추가하거나** 소켓을 `drone_ws`(=git 워킹트리) 안에 두어야 한다 |

즉 TCP는 **지금 있는 컨테이너 설정 그대로 동작**하지만, UDS는 컨테이너 실행 설정 변경(=FC 도메인
+ 배포 절차 변경)을 요구한다. vision 도메인이 혼자 완결할 수 없는 의존을 만드는 셈이라 기각했다.
UDS의 이점(파일 권한 기반 접근제어, 포트 충돌 없음)은 `127.0.0.1` 바인드로 대부분 대체된다.

⚠️ **바인드 주소는 `127.0.0.1`이지 `0.0.0.0`이 아니다.** `utils/stream.py`(MJPEG)는 랩탑
브라우저에서 봐야 해서 `0.0.0.0`을 기본값으로 쓰지만, 이쪽은 `NetworkMode=host`라 `0.0.0.0`으로
열면 **비행 중 기체의 유도 스트림이 주변 네트워크에 그대로 노출된다.** 소비자가 같은 네트워크
네임스페이스 안에 있으므로 루프백으로 충분하다.

## 왜 vision이 서버인가 (세션 지시) — 그리고 문서 권고와 갈리는 지점

`docs/vision_fc_interface.md` §8 권고 4번은 **shim을 서버, vision을 클라이언트**로 두자고 했다
("FC 스택이 장수명, vision이 재시작 대상"). 이 파일은 세션 지시에 따라 **반대로**(vision=서버)
구현했다. 페일세이프 관점에서 실제로 무엇이 달라지는지 정직하게 적어 둔다:

- ✅ **§5.4의 핵심 성질은 그대로 유지된다.** "프로세스가 죽으면 커널이 소켓을 닫아 소비자가
  **EOF를 즉시** 받는다"는 방향 무관이다 — vision(서버)이 죽으면 accept된 연결이 전부 닫히고
  shim(클라이언트)은 `recv()`에서 즉시 `b""`를 받는다. DDS 토픽 침묵을 기다릴 필요가 없다.
  (`tests/test_target_sink.py`가 실제 프로세스가 아닌 sink 종료로 이 성질을 검증한다.)
- 🔀 **바뀌는 것은 재접속 책임의 소재뿐이다.** vision=서버면 **shim이 재접속 루프를 갖는다**
  (Phase 2 작업). vision=클라이언트면 반대였다. 어느 쪽이든 한쪽은 재시도해야 한다.
- ⚠️ vision=서버의 실제 비용: 프로세스 재시작 시 `TIME_WAIT` 때문에 같은 포트 재바인드가
  거부될 수 있다 → `SO_REUSEADDR`로 해소했다(아래 `start()`).

## 비차단 계약 — 최우선 요구

**소비자가 없어도, 끊겨도, 느려도 vision 파이프라인은 절대 블로킹되거나 죽지 않는다.**
검출이 착륙 유도보다 먼저 죽으면 안 되기 때문이다. 구현:

- `publish()`는 **bounded queue에 넣기만 한다** — 소켓 I/O를 한 줄도 하지 않는다.
  큐가 가득 차면 **가장 오래된 것부터 버린다**(`utils/blackbox.py::_DropOldestQueueHandler`와
  `utils/stream.py::_DropOldestFrameQueue`가 세운 패턴 그대로 재사용, 새 패턴 발명 금지).
  제어용 스트림에서는 **최신성 > 완전성**이다 — 오래된 추정치는 어차피 소비자가 stale로 버린다.
  버린 개수는 `dropped` 카운터로 관측 가능하고, `seq`가 와이어에 실리므로 **소비자도 유실을
  알아챌 수 있다**(seq에 구멍이 난다).
- 인코딩·전송·accept는 전부 **별도 스레드**에서 일어난다.
- 느린 소비자는 `send_timeout_s`로 잘라낸다 — 타임아웃이 나면 그 클라이언트를 **끊는다**
  (`sendall`이 중간에 끊기면 그 연결의 바이트 스트림은 이미 깨져서 살릴 수 없다).
- `start()` 전/`stop()` 후 `publish()`는 조용한 no-op(`utils/stream.py::push_frame` 전례).

## SIGTERM

이 저장소의 원격 프로세스는 **SIGINT가 안 먹는다** — 비대화형 SSH 백그라운드 자식은 bash가
SIGINT를 `SIG_IGN`으로 만들어버리는 것이 `/proc/PID/status`의 SigIgn 비트로 실측 확인됐고
nohup/setsid로도 회피되지 않았다(`tools/h264_stream.py` 1A). 그래서 graceful shutdown을
보장하는 유일한 신호는 **SIGTERM**이다. `install_signal_handlers()`가 `h264_stream.py::
_install_sigterm_handler`와 동일한 패턴(핸들러는 이벤트만 세팅)을 따른다.
"""
from __future__ import annotations

import abc
import queue
import signal
import socket
import threading
import time
from typing import Iterable, Optional

from vision.core.state_machine import Decision
from vision.core.target import TargetEstimate
from vision.core.wire import (
    FailsafeContract,
    build_invalid_target_record,
    build_state_hint_record,
    build_target_record,
    encode_line,
)

# 기본값 — 근거는 `vision/CLAUDE.md`의 "target_sink 소켓 기본값" 절에 기록한다(매직넘버 금지, §7.3).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8091
DEFAULT_MAX_QUEUE = 8
DEFAULT_SEND_TIMEOUT_S = 0.5
_POLL_S = 0.2  # accept/send 루프가 stop_event를 확인하는 주기


def sample_clocks() -> tuple[int, int]:
    """`(monotonic_ns, wall_ns)` — **같은 시점**의 두 클록(§4.6).

    코어(`core/wire.py`)는 wall-clock을 쓰지 않는 순수 로직이라(§7.5 결정론) 실제 클록 샘플링은
    어댑터인 이 파일의 책임이다. 두 호출 사이의 간격이 그대로 `clock_offset_ns` 오차가 되므로
    사이에 아무것도 넣지 않는다.
    """
    return time.monotonic_ns(), time.time_ns()


class TargetSink(abc.ABC):
    """`docs/vision_plan.md` §7.2가 이름으로만 지정해 뒀던 포트. 코어는 이 인터페이스만 알고
    전송수단(소켓/파일/ROS)은 모른다."""

    @abc.abstractmethod
    def publish(self, record: dict) -> None:
        """레코드 1건 발행. **절대 블로킹하거나 예외를 던지지 않는다.**"""

    def close(self) -> None:
        """정리. 기본은 no-op."""


class NullSink(TargetSink):
    """아무데도 안 보내는 sink. `--target-sink` 미지정 시 기본값으로 쓰라고 만든 것 —
    호출자가 `if sink is not None` 분기를 곳곳에 두지 않아도 되게 한다."""

    def __init__(self) -> None:
        self.published = 0

    def publish(self, record: dict) -> None:
        self.published += 1


class _DropOldestQueue:
    """가득 차면 가장 오래된 것을 버리고 새 것을 넣는다 — **절대 블로킹하지 않는다.**

    `utils/stream.py::_DropOldestFrameQueue`와 동일 설계(evict-then-insert를 락으로 원자화 —
    여러 producer 스레드가 동시에 put해도 `queue.Full` 경쟁이 나지 않는다). 새 패턴을 발명하지
    않는다는 이 저장소의 원칙 그대로다.

    버린 개수를 `dropped`로 센다 — `stream.py` 쪽엔 없는 것으로, 관측성(§7.4)을 위해 추가했다.
    """

    def __init__(self, maxsize: int):
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._put_lock = threading.Lock()
        self.dropped = 0

    def put(self, item) -> None:
        with self._put_lock:
            while True:
                try:
                    self._q.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        self._q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass

    def get(self, timeout: Optional[float] = None):
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()


class SocketTargetSink(TargetSink):
    """localhost TCP **서버**. 접속한 모든 클라이언트에게 JSON Lines를 브로드캐스트한다.

    사용법:
        sink = SocketTargetSink()
        sink.start()
        sink.install_signal_handlers()      # 메인 스레드에서만 (선택)
        ...
        sink.publish_target(estimate)       # 파이프라인 루프 안에서, 매 프레임(비차단)
        sink.publish_state_hint(decision, frame_id=...)
        ...
        sink.stop()

    확인:  nc 127.0.0.1 8091
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_queue: int = DEFAULT_MAX_QUEUE,
        send_timeout_s: float = DEFAULT_SEND_TIMEOUT_S,
        contract: Optional[FailsafeContract] = None,
    ):
        self.host = host
        self.port = port
        self.send_timeout_s = send_timeout_s
        self.contract = contract or FailsafeContract()

        self._queue = _DropOldestQueue(max_queue)
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._started = False

        self._seq = 0
        self._seq_lock = threading.Lock()

        # 관측성 카운터(§7.4)
        self.sent = 0
        self.total_connections = 0
        self.dropped_clients = 0

    # ---------- 수명주기 ----------

    def start(self) -> None:
        """리스너 소켓을 열고 accept/send 스레드를 띄운다. 이미 시작돼 있으면 no-op(idempotent).

        ⚠️ **bind 실패는 여기서 예외로 올라간다** — 포트 충돌은 기동 시점의 설정 오류이고,
        조용히 삼키면 "발행되는 줄 알았는데 아무데도 안 가는" 상태로 비행하게 된다. 비차단
        계약이 지키는 것은 **`publish()`가 절대 안 죽는다**는 것이지 기동 실패 은폐가 아니다.
        """
        if self._started:
            return
        self._stop_event.clear()

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 프로세스 재시작 시 TIME_WAIT로 재바인드가 거부되는 것을 막는다(vision=서버 구조의 비용).
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(4)
        # accept 루프가 stop_event를 주기적으로 확인할 수 있게 — 시그널 핸들러에서 소켓을
        # 닫는(스레드 교차 close) 대신 폴링으로 푼다.
        listener.settimeout(_POLL_S)
        self._listener = listener
        self.port = listener.getsockname()[1]  # port=0(임시 포트) 요청 시 실제 포트로 갱신

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="target-sink-accept"
        )
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="target-sink-send"
        )
        self._started = True
        self._accept_thread.start()
        self._send_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """이벤트를 세팅하고 스레드가 정리될 때까지 기다린다. 여러 번 불러도 안전(idempotent).

        스레드가 자기 소켓을 자기가 닫는다 — 클라이언트 소켓이 닫히는 순간 소비자는 **EOF**를
        받는다(§5.4의 "침묵보다 엄밀한" 사망 감지).
        """
        self._stop_event.set()
        for t in (self._accept_thread, self._send_thread):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
        # 스레드가 시간 내에 안 끝났을 경우를 대비한 안전망(정상 경로에선 이미 닫혀 있다).
        self._close_listener()
        self._close_all_clients()
        self._accept_thread = None
        self._send_thread = None
        self._started = False

    def close(self) -> None:
        self.stop()

    def install_signal_handlers(self) -> None:
        """SIGTERM(+가능하면 SIGINT)을 받으면 종료 이벤트를 세팅한다.

        `tools/h264_stream.py::_install_sigterm_handler`와 **동일한 패턴** — 핸들러는 이벤트만
        세팅하고 실제 정리는 스레드가 한다(핸들러 안에서 join/close 같은 무거운 일을 하지 않는다).
        SIGINT는 "있으면 좋은" 보너스일 뿐 이 계약이 SIGINT에 의존하지 않는다(파일 docstring 참조).

        ⚠️ `signal.signal`은 **메인 스레드에서만** 호출 가능하다. 워커 스레드에서 부르면
        `ValueError`가 나므로 조용히 넘어간다(라이브러리로 embed된 경우를 위한 방어).
        """

        def _handler(signum, frame):  # noqa: ARG001 - signal handler 표준 시그니처
            self._stop_event.set()

        try:
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            return
        try:
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            pass

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    @property
    def dropped(self) -> int:
        """큐 오버플로로 버린 레코드 수."""
        return self._queue.dropped

    @property
    def client_count(self) -> int:
        """현재 접속해 있는 소비자 수.

        ⚠️ **락이 필수다** — 이 목록은 accept 스레드가 추가하고 send 스레드가(죽은 소비자를
        정리하며) 제거한다. 파이프라인 스레드가 화면 오버레이(`utils/visualize.py::
        draw_sink_status`)를 그리려고 매 프레임 읽는 값이라 스레드 교차 접근이 상시 일어난다.
        """
        with self._clients_lock:
            return len(self._clients)

    @property
    def last_seq(self) -> int:
        """마지막으로 발급된 `seq`(= 지금까지 조립한 레코드 수). 0이면 아직 한 건도 없다.

        화면 오버레이가 "발행이 실제로 돌고 있는가"를 보여주는 값이다 — 프레임이 넘어가는데
        이 숫자가 멈춰 있으면 발행이 죽은 것이다. `next_seq()`와 같은 락을 쓴다.
        """
        with self._seq_lock:
            return self._seq

    # ---------- 파이프라인 스레드에서 호출 (비차단) ----------

    def publish(self, record: dict) -> None:
        """레코드 1건을 큐에 넣는다. **소켓 I/O 없음 — 절대 블로킹하지 않는다.**

        `start()` 전이거나 `stop()` 후면 조용히 무시한다(`utils/stream.py::push_frame` 전례).
        """
        if not self._started or self._stop_event.is_set():
            return
        self._queue.put(record)

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def publish_target(self, estimate: TargetEstimate) -> dict:
        """`TargetEstimate` → 레코드 조립 + 발행. 조립된 레코드를 돌려준다(로깅/테스트용)."""
        mono, wall = sample_clocks()
        record = build_target_record(
            estimate,
            seq=self.next_seq(),
            stamp_monotonic_ns=mono,
            stamp_wall_ns=wall,
            contract=self.contract,
        )
        self.publish(record)
        return record

    def publish_invalid(self, *, frame_id: int, reason: str) -> dict:
        """검출 없음/거절을 `valid=false`로 알린다 — **발행을 멈추지 않는 것이 계약**(§5.4)."""
        mono, wall = sample_clocks()
        record = build_invalid_target_record(
            seq=self.next_seq(),
            frame_id=frame_id,
            reason=reason,
            stamp_monotonic_ns=mono,
            stamp_wall_ns=wall,
            contract=self.contract,
        )
        self.publish(record)
        return record

    def publish_state_hint(self, decision: Decision, *, frame_id: int) -> dict:
        """상태머신 `Decision` 발행. `command`는 `command_hint`로 나간다(`core/wire.py` 참조)."""
        mono, wall = sample_clocks()
        record = build_state_hint_record(
            decision,
            seq=self.next_seq(),
            frame_id=frame_id,
            stamp_monotonic_ns=mono,
            stamp_wall_ns=wall,
        )
        self.publish(record)
        return record

    # ---------- accept 스레드 ----------

    def _accept_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                listener = self._listener
                if listener is None:
                    break
                try:
                    conn, _addr = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break  # 리스너가 닫혔다 — 종료 경로
                # 10Hz 제어 스트림에서 Nagle 지연(최대 수십 ms)은 그대로 유도 지연이다.
                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                conn.settimeout(self.send_timeout_s)
                with self._clients_lock:
                    self._clients.append(conn)
                    self.total_connections += 1
        finally:
            self._close_listener()

    # ---------- send 스레드 ----------

    def _send_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    record = self._queue.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                payload = encode_line(record).encode("utf-8")
                self._broadcast(payload)
        finally:
            # 여기서 클라이언트 소켓을 닫는 것이 소비자에게 **EOF**를 주는 경로다(§5.4).
            self._close_all_clients()

    def _broadcast(self, payload: bytes) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return  # 소비자가 없어도 아무 일도 일어나지 않는다 — 조용히 버린다
        dead = []
        for conn in clients:
            try:
                conn.sendall(payload)
            except (socket.timeout, OSError):
                # 느리거나(타임아웃) 끊긴 소비자. 타임아웃이면 이미 부분 전송돼 그 연결의
                # 바이트 스트림이 깨졌을 수 있으므로 살리지 않고 끊는다 — 소비자는 재접속하면
                # 되고, 제어 스트림에서는 최신성이 완전성보다 중요하다.
                dead.append(conn)
        if dead:
            with self._clients_lock:
                for conn in dead:
                    if conn in self._clients:
                        self._clients.remove(conn)
                        self.dropped_clients += 1
            for conn in dead:
                _safe_close(conn)
        self.sent += len(clients) - len(dead)

    # ---------- 정리 ----------

    def _close_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            _safe_close(listener)

    def _close_all_clients(self) -> None:
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for conn in clients:
            _safe_close(conn)


def _safe_close(sock: socket.socket) -> None:
    try:
        sock.close()
    except OSError:
        pass


def install_sigterm_handler(sinks: Iterable[TargetSink]) -> None:
    """여러 sink를 한꺼번에 종료시키는 SIGTERM 핸들러(편의 함수)."""
    targets = list(sinks)

    def _handler(signum, frame):  # noqa: ARG001 - signal handler 표준 시그니처
        for s in targets:
            if isinstance(s, SocketTargetSink):
                s._stop_event.set()

    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        pass
