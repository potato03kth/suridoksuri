"""작업 G 순수 함수 테스트 — 폴더 넘버링·최신 ulog 선택.

실행: cd tools/flight_logs && pytest test_flight_logs.py
(pymavlink 불필요 — 순수 함수만 import)
"""

import pytest

from pull_ulog import (
    LOG_DATA_CHUNK,
    LogEntry,
    coverage,
    download_log,
    merge_intervals,
    missing_ranges,
    next_flight_dirname,
    pick_latest_log,
    ulog_filename,
)

D = "2026-07-06"


class TestNextFlightDirname:
    def test_empty_root(self):
        assert next_flight_dirname([], D) == "2026-07-06_flight01"

    def test_increments_todays_max(self):
        existing = ["2026-07-06_flight01", "2026-07-06_flight02"]
        assert next_flight_dirname(existing, D) == "2026-07-06_flight03"

    def test_ignores_other_dates(self):
        existing = ["2026-07-05_flight07", "2026-07-04_flight99"]
        assert next_flight_dirname(existing, D) == "2026-07-06_flight01"

    def test_ignores_unrelated_entries(self):
        existing = ["README.md", "2026-07-06_flightXX", "2026-07-06_flight2_backup"]
        assert next_flight_dirname(existing, D) == "2026-07-06_flight01"

    def test_gap_does_not_refill(self):
        # 01이 지워져도 재사용하지 않는다 — 최대값 + 1
        existing = ["2026-07-06_flight02", "2026-07-06_flight05"]
        assert next_flight_dirname(existing, D) == "2026-07-06_flight06"

    def test_two_digit_padding_and_beyond(self):
        assert next_flight_dirname(["2026-07-06_flight09"], D) == "2026-07-06_flight10"
        assert next_flight_dirname(["2026-07-06_flight99"], D) == "2026-07-06_flight100"


class TestPickLatestLog:
    def test_empty_returns_none(self):
        assert pick_latest_log([]) is None

    def test_all_timed_uses_time_utc(self):
        entries = [
            LogEntry(1, 1000, 10),
            LogEntry(2, 3000, 10),  # 최신 시각
            LogEntry(3, 2000, 10),  # id는 크지만 시각이 앞섬
        ]
        assert pick_latest_log(entries).log_id == 2

    def test_timed_tie_breaks_by_id(self):
        entries = [LogEntry(1, 1000, 10), LogEntry(2, 1000, 10)]
        assert pick_latest_log(entries).log_id == 2

    def test_zero_time_falls_back_to_id(self):
        # GPS 락 없이 기록된 로그(time_utc=0)가 섞이면 id 기준 — 최대 id가 최신
        entries = [
            LogEntry(1, 5000, 10),  # 시각 있음 (과거 비행)
            LogEntry(2, 0, 10),     # 최신이지만 GPS 시각 없음
        ]
        assert pick_latest_log(entries).log_id == 2

    def test_single_entry(self):
        assert pick_latest_log([LogEntry(7, 0, 10)]).log_id == 7


class TestUlogFilename:
    def test_with_utc_time(self):
        import calendar
        ts = calendar.timegm((2026, 7, 6, 12, 34, 56, 0, 0, 0))
        assert ulog_filename(LogEntry(3, ts, 10)) == "log_3_2026-07-06-12-34-56.ulg"

    def test_without_time(self):
        assert ulog_filename(LogEntry(3, 0, 10)) == "log_3.ulg"


class TestMergeIntervals:
    def test_empty(self):
        assert merge_intervals([]) == []

    def test_single(self):
        assert merge_intervals([(0, 90)]) == [(0, 90)]

    def test_disjoint_sorted(self):
        assert merge_intervals([(0, 90), (200, 300)]) == [(0, 90), (200, 300)]

    def test_adjacent_merges(self):
        # 인접 구간(끝==시작)은 하나로 — 순서대로 도착한 청크들
        assert merge_intervals([(0, 90), (90, 180), (180, 270)]) == [(0, 270)]

    def test_overlap_merges(self):
        assert merge_intervals([(0, 100), (50, 120)]) == [(0, 120)]

    def test_unsorted_input(self):
        # 뒤섞여 도착한 청크도 정렬·병합
        assert merge_intervals([(180, 270), (0, 90), (90, 180)]) == [(0, 270)]

    def test_duplicate_chunks(self):
        # 재요청으로 같은 청크가 중복 도착해도 안전
        assert merge_intervals([(0, 90), (0, 90), (90, 180)]) == [(0, 180)]


class TestMissingRanges:
    def test_nothing_received(self):
        assert missing_ranges([], 300) == [(0, 300)]

    def test_fully_received(self):
        assert missing_ranges([(0, 300)], 300) == []

    def test_one_gap_in_middle(self):
        assert missing_ranges([(0, 90), (180, 300)], 300) == [(90, 180)]

    def test_gap_at_start(self):
        assert missing_ranges([(90, 300)], 300) == [(0, 90)]

    def test_gap_at_end(self):
        assert missing_ranges([(0, 270)], 300) == [(270, 300)]

    def test_multiple_gaps(self):
        assert missing_ranges([(90, 180), (270, 360)], 400) == [
            (0, 90), (180, 270), (360, 400)
        ]

    def test_received_beyond_size_is_clamped(self):
        # 마지막 청크가 size를 넘겨 도착해도 [0,size)만 본다
        assert missing_ranges([(0, 320)], 300) == []


class TestCoverage:
    def test_exact(self):
        assert coverage([(0, 300)], 300) == 300

    def test_partial(self):
        assert coverage([(0, 90), (180, 270)], 300) == 180

    def test_overlap_not_double_counted(self):
        assert coverage([(0, 100), (50, 120)], 300) == 120

    def test_clamped_to_size(self):
        # size를 넘는 부분은 세지 않는다
        assert coverage([(0, 500)], 300) == 300

    def test_empty(self):
        assert coverage([], 300) == 0


# ---------------------------------------------------------------------------
# download_log 재조립 검증 — 가짜 링크(pymavlink 불필요).
# serial(무손실·순차) / 손실 링크 / 죽은 링크를 시뮬레이션한다.
# 핵심 요구: 실기체 serial(무손실 in-order)에서 반드시 바이트 동일하게 완결.
# ---------------------------------------------------------------------------

class _FakeMsg:
    def __init__(self, log_id, ofs, data):
        self.id = log_id
        self.ofs = ofs
        self.count = len(data)
        self.data = data  # bytes


class _FakeMav:
    def __init__(self, link):
        self._link = link

    def log_request_data_send(self, sysid, comp, log_id, ofs, count):
        self._link._on_request(log_id, ofs, count)

    def log_request_end_send(self, sysid, comp):
        self._link.ended = True


class _FakeLink:
    """LOG_REQUEST_DATA에 LOG_DATA로 응답하는 가짜 링크.

    reorder=True  : 각 응답 버스트를 역순으로 전달(UDP 재정렬 모사).
    drop(ofs)->bool: 첫 요청 때 해당 청크를 한 번 누락(재요청 시 전달) — 손실 모사.
    dead=True     : 어떤 요청에도 데이터를 주지 않음 — 죽은 링크.
    """

    def __init__(self, source, log_id=42, reorder=False, drop=None, dead=False):
        self.source = source
        self.size = len(source)
        self.log_id = log_id
        self.reorder = reorder
        self.drop = drop or (lambda ofs: False)
        self.dead = dead
        self.target_system = 1
        self.target_component = 1
        self.port = None  # 비소켓 → _grow_udp_rcvbuf 는 무영향
        self.mav = _FakeMav(self)
        self.queue = []
        self.ended = False
        self._dropped_once = set()

    def _on_request(self, log_id, ofs, count):
        if self.dead:
            return
        end = min(ofs + count, self.size)
        chunks = []
        o = ofs
        while o < end:
            n = min(LOG_DATA_CHUNK, end - o)
            if self.drop(o) and o not in self._dropped_once:
                self._dropped_once.add(o)  # 이번엔 누락, 다음 요청 때 전달
            else:
                chunks.append(_FakeMsg(log_id, o, self.source[o:o + n]))
            o += n
        if self.reorder:
            chunks.reverse()
        self.queue.extend(chunks)

    def recv_match(self, type=None, blocking=True, timeout=None):
        return self.queue.pop(0) if self.queue else None


def _src(n):
    return bytes(((i * 37 + 11) & 0xFF) for i in range(n))


def _read(p):
    with open(p, "rb") as f:
        return f.read()


class TestDownloadLogFakeLink:
    def test_serial_lossless_byte_identical(self, tmp_path):
        # 실기체 serial 모사: 무손실·순차. 반드시 바이트 동일 완결.
        src = _src(300 * 1024)  # 여러 윈도우에 걸침
        link = _FakeLink(src)
        out = tmp_path / "log.ulg"
        download_log(link, LogEntry(42, 0, len(src)), str(out),
                     window=64 * 1024, idle=0.01)
        assert _read(out) == src
        assert link.ended is True

    def test_out_of_order_delivery_byte_identical(self, tmp_path):
        # 버스트가 역순으로 도착해도 offset 기록이라 안전.
        src = _src(200 * 1024)
        link = _FakeLink(src, reorder=True)
        out = tmp_path / "log.ulg"
        download_log(link, LogEntry(42, 0, len(src)), str(out),
                     window=64 * 1024, idle=0.01)
        assert _read(out) == src

    def test_lossy_link_recovers_byte_identical(self, tmp_path):
        # 3청크마다 1개를 첫 요청에서 누락 → 누락 구간만 재요청해 수렴.
        src = _src(150 * 1024)
        link = _FakeLink(src, drop=lambda ofs: (ofs // LOG_DATA_CHUNK) % 3 == 1)
        out = tmp_path / "log.ulg"
        download_log(link, LogEntry(42, 0, len(src)), str(out),
                     window=32 * 1024, idle=0.01, max_no_progress=50)
        assert _read(out) == src

    def test_size_exact_multiple_of_chunk(self, tmp_path):
        # size가 90의 정확한 배수여도(마지막 청크 count==90) 정상 종료.
        src = _src(90 * 500)
        link = _FakeLink(src)
        out = tmp_path / "log.ulg"
        download_log(link, LogEntry(42, 0, len(src)), str(out),
                     window=90 * 100, idle=0.01)
        assert _read(out) == src

    def test_dead_link_raises_and_removes_partial(self, tmp_path):
        # 무한 hang 금지: 진행이 없으면 예외 + 부분 파일 제거(조용한 실패 방지).
        src = _src(50 * 1024)
        link = _FakeLink(src, dead=True)
        out = tmp_path / "log.ulg"
        with pytest.raises(Exception):
            download_log(link, LogEntry(42, 0, len(src)), str(out),
                         window=16 * 1024, idle=0.01, max_no_progress=3)
        assert not out.exists()
