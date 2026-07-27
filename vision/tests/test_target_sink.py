"""`vision/utils/target_sink.py` — 실제 소켓으로 검증한다(몽키패치 없음).

`utils/stream.py`의 테스트 관례를 따른다: 진짜 서버를 띄우고, 진짜 클라이언트로 접속해, 진짜
바이트를 받아 파싱한다.

**최우선 요구는 "소비자가 없어도/끊겨도/느려도 파이프라인이 절대 블로킹되거나 죽지 않는다"** 이므로
그 세 경우를 각각 실측 시간으로 확인한다.

⚠️ bounded queue의 드롭 의미론은 `_DropOldestQueue`를 **직접** 검증한다(소비 스레드가 동시에
큐를 비우는 상태에서 "몇 개 남았나"를 단언하면 근본적으로 플래키해진다 — 실제로
`tests/test_blackbox.py::test_bounded_queue_drops_oldest_under_burst`가 그 함정에 빠져 있다.
자세한 진단은 이 파일 맨 아래 주석 참조). 그 대신 end-to-end에서는 **"최신 레코드는 절대
유실되지 않는다"** 는, 소비 스레드가 돌아도 항상 참인 성질을 확인한다.
"""
import json
import os
import queue
import signal
import socket
import threading
import time

import pytest

from vision.core.state_machine import Decision, LandingState
from vision.core.target import TargetEstimate
from vision.utils.target_sink import (
    NullSink,
    SocketTargetSink,
    _DropOldestQueue,
    sample_clocks,
)


def _estimate(frame_id: int = 1) -> TargetEstimate:
    return TargetEstimate(
        position=(1.0, -2.0, 12.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        confidence=1.0,
        target_type="aruco_23",
        frame_id=frame_id,
        timestamp=1769000000.0,
    )


def _decision() -> Decision:
    return Decision(
        state=LandingState.PRECISION_SERVO, command="descend", reason="servoing"
    )


def _wait_for(pred, timeout=3.0, interval=0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _wait_while_publishing(sink, pred, timeout=3.0) -> bool:
    """`pred`가 참이 될 때까지 계속 발행하며 기다린다.

    끊긴 TCP 소켓은 **쓰기를 시도해야** 죽은 걸 알 수 있다(첫 write는 보통 성공하고 그 다음
    write에서 EPIPE가 난다). 발행을 멈추고 기다리면 서버는 영원히 눈치채지 못한다."""
    deadline = time.monotonic() + timeout
    i = 0
    while time.monotonic() < deadline:
        if pred():
            return True
        sink.publish_invalid(frame_id=i, reason="probe")
        i += 1
        time.sleep(0.02)
    return pred()


def _close(conn, f=None) -> None:
    """소켓을 **진짜로** 닫는다.

    ⚠️ `socket.makefile()`이 만든 파일 객체는 내부적으로 소켓 참조를 하나 더 들고 있어서
    (`_io_refs`), 파일을 안 닫고 `conn.close()`만 하면 **FIN이 나가지 않는다** — 상대는 연결이
    살아 있다고 본다. 이 함정 때문에 실제로 이 파일의 끊김 테스트가 처음에 실패했다."""
    if f is not None:
        try:
            f.close()
        except OSError:
            pass
    conn.close()


@pytest.fixture
def sink():
    s = SocketTargetSink(port=0, max_queue=8)  # port=0 → OS가 빈 포트를 준다
    s.start()
    yield s
    s.stop()


def _connect(s: SocketTargetSink, timeout=3.0) -> socket.socket:
    conn = socket.create_connection(("127.0.0.1", s.port), timeout=timeout)
    conn.settimeout(timeout)
    assert _wait_for(lambda: s.client_count >= 1), "서버가 연결을 accept하지 않았다"
    return conn


# ── 기본 발행/수신 ────────────────────────────────────────────────────────────

def test_real_client_receives_real_json_lines(sink):
    conn = _connect(sink)
    try:
        f = conn.makefile("r", encoding="utf-8")
        sink.publish_target(_estimate(frame_id=7))
        rec = json.loads(f.readline())
        assert rec["type"] == "target"
        assert rec["frame_id"] == 7
        assert rec["position_frd"] == pytest.approx([2.0, 1.0, 12.0])
        assert rec["valid"] is True
    finally:
        conn.close()


def test_state_hint_and_target_share_one_increasing_seq(sink):
    """`seq`에 구멍이 나면 소비자가 유실을 알아챌 수 있다 — 단조증가가 계약이다."""
    conn = _connect(sink)
    try:
        f = conn.makefile("r", encoding="utf-8")
        sink.publish_target(_estimate())
        sink.publish_state_hint(_decision(), frame_id=1)
        sink.publish_invalid(frame_id=2, reason="no_detection")
        seqs, types = [], []
        for _ in range(3):
            rec = json.loads(f.readline())
            seqs.append(rec["seq"])
            types.append(rec["type"])
        assert seqs == sorted(seqs) and len(set(seqs)) == 3
        assert types == ["target", "state_hint", "target"]
    finally:
        conn.close()


def test_invalid_records_keep_flowing_when_detection_is_lost(sink):
    """§5.4 — 검출 상실 시 침묵하지 않는다(침묵은 "죽음"으로 예약)."""
    conn = _connect(sink)
    try:
        f = conn.makefile("r", encoding="utf-8")
        for i in range(3):
            sink.publish_invalid(frame_id=i, reason="no_detection")
        for i in range(3):
            rec = json.loads(f.readline())
            assert rec["valid"] is False and rec["reason"] == "no_detection"
    finally:
        conn.close()


def test_two_clients_both_receive_the_broadcast(sink):
    a, b = _connect(sink), None
    try:
        b = socket.create_connection(("127.0.0.1", sink.port), timeout=3.0)
        b.settimeout(3.0)
        assert _wait_for(lambda: sink.client_count >= 2)
        sink.publish_target(_estimate(frame_id=99))
        for conn in (a, b):
            assert json.loads(conn.makefile("r", encoding="utf-8").readline())["frame_id"] == 99
    finally:
        a.close()
        if b is not None:
            b.close()


# ── 🔴 비차단 계약: 소비자 없음 / 끊김 / 느림 ─────────────────────────────────

def test_publish_never_blocks_when_no_consumer_is_connected(sink):
    """최우선 요구 — 소비자가 아예 없어도 파이프라인이 멈추지 않는다."""
    assert sink.client_count == 0
    worst = 0.0
    for i in range(2000):
        t0 = time.monotonic()
        sink.publish_target(_estimate(frame_id=i))
        worst = max(worst, time.monotonic() - t0)
    assert worst < 0.15, f"소비자 없는데 단일 publish()가 최대 {worst:.3f}s — 블로킹 의심"


def test_publish_never_raises_when_no_consumer_is_connected(sink):
    """검출이 착륙 유도보다 먼저 죽으면 안 된다 — 예외도 못 나간다."""
    for i in range(50):
        sink.publish_target(_estimate(frame_id=i))
        sink.publish_state_hint(_decision(), frame_id=i)
        sink.publish_invalid(frame_id=i, reason="none")


def test_publish_never_blocks_with_a_slow_consumer_that_never_reads(sink):
    """느린 소비자(접속만 하고 한 바이트도 안 읽음)가 붙어 있어도 파이프라인은 논-블로킹.

    ⚠️ 소켓 버퍼를 **실제로 가득 채워야** 이 테스트에 이빨이 생긴다. 루프백 소켓 버퍼는
    자동튜닝으로 수 MB까지 커져서, 소량(수천 건)만 밀어 넣으면 `publish()`가 소켓 I/O를 직접
    하도록 망가뜨려도 버퍼에 다 들어가 버려 **테스트가 통과해 버린다**(파괴검증 D15에서 실제로
    관측됨). 그래서 ① 클라이언트 수신버퍼를 최소로 줄이고 ② 버퍼 용량을 확실히 넘는 양을 민다.
    """
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    try:
        conn.connect(("127.0.0.1", sink.port))
        conn.settimeout(3.0)
        assert _wait_for(lambda: sink.client_count >= 1)

        # ⚠️ **총 시간이 아니라 단일 호출 최대 지연을 본다.** 총 시간으로 재면 이빨이 없다:
        # publish()가 소켓 I/O를 직접 하도록 망가뜨려도, 버퍼가 찼을 때 한 번
        # `send_timeout_s`(0.5s)만큼 멈춘 뒤 그 클라이언트를 죽은 것으로 끊어버리므로 이후
        # 호출은 다시 빨라져 **총합에 0.5s 스톨이 묻힌다**(파괴검증 D15에서 실제로 관측됨).
        # 10Hz 제어 루프에서 0.5s 스톨은 5프레임 결손이라 반드시 잡아야 한다.
        n = 20000                      # 약 12MB — 어떤 자동튜닝 버퍼도 넘어선다
        worst = 0.0
        for i in range(n):
            t0 = time.monotonic()
            sink.publish_target(_estimate(frame_id=i))
            worst = max(worst, time.monotonic() - t0)
        assert worst < 0.15, (
            f"단일 publish()가 최대 {worst:.3f}s 걸렸다 — 파이프라인이 소비자에 물렸다"
        )
    finally:
        conn.close()


def test_pipeline_survives_consumer_disconnect(sink):
    """소비자가 사라져도 발행 측은 죽지 않고, 서버가 죽은 클라이언트를 정리한다."""
    conn = _connect(sink)
    f = conn.makefile("r", encoding="utf-8")
    sink.publish_target(_estimate(frame_id=1))
    assert json.loads(f.readline())["frame_id"] == 1

    _close(conn, f)
    # 끊긴 소켓으로 계속 쏟아부어도 예외가 안 나야 하고, 서버는 죽은 클라이언트를 정리해야 한다.
    assert _wait_while_publishing(sink, lambda: sink.client_count == 0), \
        "끊긴 클라이언트가 정리되지 않았다"
    assert sink.dropped_clients >= 1


def test_consumer_can_reconnect_after_disconnect(sink):
    """소비자(Phase 2 shim)가 재시작해도 다시 붙어서 받을 수 있어야 한다."""
    first = _connect(sink)
    f1 = first.makefile("r", encoding="utf-8")
    sink.publish_target(_estimate(frame_id=1))
    assert json.loads(f1.readline())["frame_id"] == 1
    _close(first, f1)
    assert _wait_while_publishing(sink, lambda: sink.client_count == 0)

    second = _connect(sink)
    f2 = second.makefile("r", encoding="utf-8")
    try:
        sink.publish_target(_estimate(frame_id=2))
        # 앞서 probe로 흘려보낸 무효 레코드가 섞여 있을 수 있으니 target 레코드까지 읽는다.
        rec = json.loads(f2.readline())
        while rec["type"] != "target":
            rec = json.loads(f2.readline())
        assert rec["frame_id"] == 2
        assert sink.total_connections == 2
    finally:
        _close(second, f2)


def test_latest_record_is_never_lost_under_a_burst(sink):
    """드롭이 일어나더라도 **최신 레코드는 반드시 도착한다**(drop-oldest의 핵심 성질).
    소비 스레드가 동시에 도는 상태에서도 항상 참이라 플래키하지 않다."""
    conn = _connect(sink)
    try:
        f = conn.makefile("r", encoding="utf-8")
        for i in range(300):
            sink.publish_target(_estimate(frame_id=i))
        last_seq = sink.publish_target(_estimate(frame_id=999))["seq"]

        deadline = time.monotonic() + 5.0
        seen = None
        while time.monotonic() < deadline:
            rec = json.loads(f.readline())
            if rec["seq"] == last_seq:
                seen = rec
                break
        assert seen is not None, "최신 레코드가 유실됐다 — drop-oldest가 아니라 drop-newest다"
        assert seen["frame_id"] == 999
    finally:
        conn.close()


# ── bounded queue 드롭 의미론 (결정론적 — 소비 스레드 없음) ────────────────────

def test_bounded_queue_drops_oldest_and_keeps_newest():
    q = _DropOldestQueue(3)
    for i in range(10):
        q.put(i)
    assert q.qsize() == 3
    assert [q.get(timeout=0.1) for _ in range(3)] == [7, 8, 9]


def test_bounded_queue_counts_what_it_dropped():
    q = _DropOldestQueue(2)
    for i in range(10):
        q.put(i)
    assert q.dropped == 8  # 10건 넣고 2건만 남음


def test_bounded_queue_put_never_blocks_when_full():
    q = _DropOldestQueue(1)
    t0 = time.monotonic()
    for i in range(20000):
        q.put(i)
    assert time.monotonic() - t0 < 2.0
    assert q.get(timeout=0.1) == 19999


def test_bounded_queue_is_safe_under_concurrent_producers():
    """evict-then-insert를 락으로 원자화하지 않으면 `queue.Full`이 터진다
    (`utils/stream.py`가 이 락을 둔 이유)."""
    q = _DropOldestQueue(2)
    errors = []

    def worker():
        try:
            for i in range(3000):
                q.put(i)
        except Exception as e:      # noqa: BLE001 - 어떤 예외든 실패로 본다
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"동시 producer에서 예외 발생: {errors}"


def test_sink_exposes_its_drop_counter(sink):
    assert sink.dropped == 0
    assert isinstance(sink.dropped, int)


# ── 수명주기 ──────────────────────────────────────────────────────────────────

def test_publish_before_start_is_a_silent_noop():
    s = SocketTargetSink(port=0)
    s.publish_target(_estimate())       # start() 전 — 예외 없이 조용히 무시
    s.publish_state_hint(_decision(), frame_id=1)
    assert s.dropped == 0


def test_publish_after_stop_is_a_silent_noop(sink):
    sink.stop()
    sink.publish_target(_estimate())
    assert sink.stopping is True


def test_start_and_stop_are_idempotent():
    s = SocketTargetSink(port=0)
    s.start()
    port = s.port
    s.start()                            # 두 번째 start는 no-op — 포트가 안 바뀐다
    assert s.port == port
    s.stop()
    s.stop()                             # 두 번째 stop도 안전


def test_ephemeral_port_is_resolved_after_start():
    s = SocketTargetSink(port=0)
    s.start()
    try:
        assert s.port != 0
        socket.create_connection(("127.0.0.1", s.port), timeout=2.0).close()
    finally:
        s.stop()


def test_binds_loopback_only_not_all_interfaces():
    """🔴 `NetworkMode=host`라 0.0.0.0으로 열면 비행 중 유도 스트림이 주변 네트워크에 노출된다."""
    s = SocketTargetSink(port=0)
    assert s.host == "127.0.0.1"
    s.start()
    try:
        assert s._listener.getsockname()[0] == "127.0.0.1"
    finally:
        s.stop()


def test_port_conflict_surfaces_as_an_error_not_silence():
    """기동 실패를 삼키면 "발행되는 줄 알았는데 아무데도 안 가는" 상태로 비행하게 된다."""
    first = SocketTargetSink(port=0)
    first.start()
    try:
        second = SocketTargetSink(port=first.port)
        with pytest.raises(OSError):
            second.start()
    finally:
        first.stop()


# ── 소비자의 사망 감지: EOF (§5.4) ────────────────────────────────────────────

def test_consumer_gets_eof_when_the_sink_shuts_down(sink):
    """🔴 §5.4 — "프로세스가 죽으면 커널이 소켓을 닫아 소비자가 **EOF를 즉시** 받는다".
    이 성질이 소켓 전송을 고른 이유 자체다(DDS 토픽 침묵보다 엄밀).

    vision=서버 구조에서도 이 성질이 유지되는지가 핵심 확인 대상이다(문서 §8 권고는
    vision=클라이언트였다 — `utils/target_sink.py` docstring 참조)."""
    conn = _connect(sink)
    try:
        f = conn.makefile("r", encoding="utf-8")
        sink.publish_target(_estimate())
        assert json.loads(f.readline())["type"] == "target"

        sink.stop()
        assert f.readline() == "", "sink 종료 후 소비자가 EOF를 못 받았다"
    finally:
        conn.close()


# ── SIGTERM ───────────────────────────────────────────────────────────────────

def test_sigterm_triggers_graceful_shutdown_and_consumer_eof():
    """🔴 이 저장소의 원격 프로세스는 SIGINT가 안 먹는다(비대화형 SSH 자식에서 SIG_IGN) —
    graceful shutdown을 보장하는 유일한 신호가 SIGTERM이다(`tools/h264_stream.py` 1A 실측).

    **진짜 SIGTERM을 자기 프로세스에 보낸다.**"""
    original_term = signal.getsignal(signal.SIGTERM)
    original_int = signal.getsignal(signal.SIGINT)
    s = SocketTargetSink(port=0)
    s.start()
    conn = None
    try:
        s.install_signal_handlers()
        conn = _connect(s)
        f = conn.makefile("r", encoding="utf-8")
        s.publish_target(_estimate())
        assert json.loads(f.readline())["type"] == "target"

        os.kill(os.getpid(), signal.SIGTERM)

        assert _wait_for(lambda: s.stopping), "SIGTERM 후에도 종료 이벤트가 세팅되지 않았다"
        assert _wait_for(lambda: s.client_count == 0, timeout=3.0), "스레드가 정리되지 않았다"
        assert f.readline() == "", "SIGTERM 종료 후 소비자가 EOF를 못 받았다"
    finally:
        if conn is not None:
            conn.close()
        s.stop()
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)


def test_install_signal_handlers_does_not_raise_off_the_main_thread():
    """워커 스레드에서 embed된 경우 `signal.signal`이 ValueError를 던진다 — 삼켜야 한다."""
    s = SocketTargetSink(port=0)
    errors = []

    def worker():
        try:
            s.install_signal_handlers()
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []


# ── 잡동사니 ──────────────────────────────────────────────────────────────────

def test_null_sink_swallows_everything():
    s = NullSink()
    s.publish({"any": "thing"})
    s.close()
    assert s.published == 1


def test_sample_clocks_returns_two_distinct_clock_readings():
    """§4.6 — 벽시계와 단조시계를 같은 시점에 뽑는다. 둘은 기준이 다르므로 값이 크게 다르다."""
    mono, wall = sample_clocks()
    assert isinstance(mono, int) and isinstance(wall, int)
    assert wall > 1_700_000_000_000_000_000     # 2023년 이후 벽시계(ns)
    assert mono < wall                          # 단조시계는 부팅 기준이라 훨씬 작다


def test_contract_is_carried_into_published_records():
    """sink에 준 계약이 실제 레코드에 반영되는가(하드코딩 아님)."""
    from vision.core.wire import FailsafeContract
    s = SocketTargetSink(port=0, contract=FailsafeContract(unverified_calib_floor_agl_m=7.5))
    rec = s.publish_target(_estimate())      # start() 전이어도 레코드는 조립돼 반환된다
    assert rec["closed_loop_floor_agl_m"] == 7.5


# ──────────────────────────────────────────────────────────────────────────────
# 참고: `tests/test_blackbox.py::test_bounded_queue_drops_oldest_under_burst` 플래키 진단
# ──────────────────────────────────────────────────────────────────────────────
# 그 테스트는 `BlackBoxLogger(max_queue=5)`에 50건을 넣고 파일에 남은 레코드가 `<= 5`라고 단언
# 한다. 그런데 `BlackBoxLogger.__init__`이 이미 `QueueListener`를 **start()** 해 둔 상태라
# 넣는 동안 리스너 스레드가 **동시에 큐를 비워 파일로 흘려보낸다** — 즉 파일에는 5건보다 훨씬
# 많이 남는 게 정상이고, 리스너가 그 짧은 구간에 한 번도 스케줄되지 않아야만 통과한다.
# (세션 지시가 보고한 "단독 10회에 9회 실패"와 일치한다.)
#
# 코드 버그가 아니라 **테스트 설계 버그**다. drop-oldest가 보장하는 것은 "큐에 최대 N건"이지
# "싱크에 최대 N건"이 아니다. 최소 수정은 첫 단언을 지우고 둘째 단언(최신 프레임 불변)만 남기는
# 것: `assert records[-1]["frame_id"] == 49`. 이 파일은 같은 함정을 피하려고 드롭 의미론을
# 소비자 없는 `_DropOldestQueue`에서 결정론적으로 검증하고, end-to-end에서는 소비 스레드가
# 돌아도 항상 참인 "최신은 안 잃는다"만 단언한다.
# (기존 테스트 수정은 이번 세션 범위 밖이라 손대지 않았다.)
