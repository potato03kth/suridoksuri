"""작업 G 순수 함수 테스트 — 폴더 넘버링·최신 ulog 선택·비행로그 진단.

실행: cd tools/flight_logs && pytest test_flight_logs.py
(pymavlink/pyulog 불필요 — 순수 함수만 import)
"""

import math

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
from analyze_flight import (
    arming_state_name,
    classify_mode_transition,
    detect_rate_onset,
    first_sustained_nonzero,
    motor_label,
    nav_state_name,
    parse_transition_alt,
    quat_to_euler_deg,
)
from collect_new_logs import (
    catchall_dirname,
    classify_dirname,
    collected_ulog_ids,
    is_dated_dirname,
    notes_skeleton,
    parse_ulog_list,
    plan_dir_sync,
    run as collect_run,
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
# analyze_flight.py 순수 함수 테스트
# ---------------------------------------------------------------------------

class TestQuatToEulerDeg:
    def test_identity_quat_is_level(self):
        assert quat_to_euler_deg(1, 0, 0, 0) == (0.0, 0.0, 0.0)

    def test_90deg_roll(self):
        # roll +90도 쿼터니언: (cos45, sin45, 0, 0)
        c = math.cos(math.radians(45)); s = math.sin(math.radians(45))
        roll, pitch, yaw = quat_to_euler_deg(c, s, 0, 0)
        assert roll == pytest.approx(90.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_180deg_roll_wraps(self):
        # roll 180도는 +180/-180 경계 — atan2 특성상 부호 어느 쪽이든 허용
        roll, _, _ = quat_to_euler_deg(0, 1, 0, 0)
        assert abs(roll) == pytest.approx(180.0, abs=1e-6)


class TestMotorLabel:
    def test_front_right(self):
        assert motor_label(1.0, 1.0) == "전우"

    def test_rear_left(self):
        assert motor_label(-1.0, -1.0) == "후좌"

    def test_front_left(self):
        assert motor_label(1.0, -1.0) == "전좌"

    def test_rear_right(self):
        assert motor_label(-1.0, 1.0) == "후우"

    def test_center_axis_labeled_mid(self):
        # 육각/동축 등 PX 또는 PY가 0인 배치 — 예외 없이 "중"으로 라벨
        assert motor_label(0.0, 1.0) == "중우"
        assert motor_label(1.0, 0.0) == "전중"


class TestClassifyModeTransition:
    def test_failsafe_wins_even_if_intention_changed(self):
        assert classify_mode_transition(True, True) == "FAILSAFE_FORCED"

    def test_intentional_change(self):
        assert classify_mode_transition(True, False) == "INTENTIONAL_CHANGE"

    def test_unclassified_when_neither(self):
        assert classify_mode_transition(False, False) == "AUTO_RECOVERY_OR_UNCLASSIFIED"


class TestNavArmingStateNames:
    def test_known_nav_states(self):
        assert nav_state_name(17) == "AUTO_TAKEOFF"
        assert nav_state_name(4) == "AUTO_LOITER"
        assert nav_state_name(2) == "POSCTL"

    def test_unknown_nav_state_falls_back(self):
        assert nav_state_name(99) == "UNKNOWN(99)"

    def test_arming_states(self):
        assert arming_state_name(2) == "ARMED"
        assert arming_state_name(1) == "STANDBY"


class TestDetectRateOnset:
    def test_no_onset_when_flat(self):
        times = [i * 0.1 for i in range(50)]
        rates = [0.5] * 50
        assert detect_rate_onset(times, rates) is None

    def test_detects_sustained_jump_after_baseline(self):
        # 0~1.5s는 잡음(±1deg/s), 이후 계속 30deg/s로 튐
        times = [i * 0.05 for i in range(80)]  # 0~3.95s
        rates = [1.0 if t <= 1.5 else 30.0 for t in times]
        onset = detect_rate_onset(times, rates, baseline_end_s=1.5, consec=3)
        assert onset is not None
        onset_t, onset_v, thresh = onset
        assert onset_t == pytest.approx(1.55, abs=1e-6)  # baseline 직후 첫 샘플
        assert onset_v == 30.0

    def test_single_spike_does_not_trigger(self):
        # consec=3 요구 — 단발 스파이크 1개는 무시
        times = [i * 0.1 for i in range(30)]
        rates = [1.0] * 30
        rates[20] = 50.0
        assert detect_rate_onset(times, rates, baseline_end_s=1.0, consec=3) is None

    def test_insufficient_baseline_returns_none(self):
        times = [0.1, 0.2]
        rates = [1.0, 1.0]
        assert detect_rate_onset(times, rates, baseline_end_s=1.5) is None


class TestFirstSustainedNonzero:
    def test_transient_blip_ignored(self):
        # arm 직후(skip_before 이전) 블립은 애초에 안 봄
        times = [0.1, 0.3, 0.6, 1.2, 1.4]
        vals = [0.05, 0.05, 0.05, 0.0, 0.0]
        assert first_sustained_nonzero(times, vals, skip_before=1.0) is None

    def test_sustained_after_skip_detected(self):
        times = [0.5, 1.2, 1.4, 1.6, 1.8]
        vals = [0.05, 0.0, 0.2, 0.3, 0.4]
        onset = first_sustained_nonzero(times, vals, skip_before=1.0, consec=3)
        assert onset == (1.4, pytest.approx(0.2))

    def test_none_when_never_sustained(self):
        times = [1.1, 1.2, 1.3, 1.4]
        vals = [0.2, 0.0, 0.2, 0.0]
        assert first_sustained_nonzero(times, vals, skip_before=1.0, consec=2) is None


class TestParseTransitionAlt:
    def test_extracts_value(self):
        text = "- **비행 조건:** (launch 인자: vehicle_type:=mc transition_alt:=5.0 waypoints:=[...])"
        assert parse_transition_alt(text) == 5.0

    def test_none_when_absent(self):
        assert parse_transition_alt("아무 내용 없음") is None

    def test_none_for_empty_input(self):
        assert parse_transition_alt(None) is None
        assert parse_transition_alt("") is None


# ---------------------------------------------------------------------------
# collect_new_logs.py 순수 함수 테스트
# ---------------------------------------------------------------------------

class TestIsDatedDirname:
    def test_dated(self):
        assert is_dated_dirname("2026-07-20_flight02")
        assert is_dated_dirname("2026-07-20_manual")

    def test_not_dated(self):
        assert not is_dated_dirname("README.md")
        assert not is_dated_dirname("rosbag")


class TestClassifyDirname:
    def test_flight_folder(self):
        assert classify_dirname("2026-07-20_flight02") == "flight"
        assert classify_dirname("2026-07-20_flight100") == "flight"

    def test_catchall_folders(self):
        assert classify_dirname("2026-07-20_manual") == "catchall"
        assert classify_dirname("2026-07-18_unlogged") == "catchall"


class TestPlanDirSync:
    def test_new_flight_folder_gets_full_copy(self):
        plan = plan_dir_sync(["2026-07-21_flight01"], [])
        assert plan == [("2026-07-21_flight01", "full")]

    def test_existing_flight_folder_skipped(self):
        # 이미 로컬에 있는 flightNN 폴더는 절대 건드리지 않는다
        plan = plan_dir_sync(["2026-07-21_flight01"], ["2026-07-21_flight01"])
        assert plan == []

    def test_catchall_always_merged_even_if_exists(self):
        plan = plan_dir_sync(["2026-07-21_manual"], ["2026-07-21_manual"])
        assert plan == [("2026-07-21_manual", "merge")]

    def test_ignores_undated_entries(self):
        plan = plan_dir_sync(["README.md", "2026-07-21_flight01"], [])
        assert plan == [("2026-07-21_flight01", "full")]

    def test_sorted_output(self):
        plan = plan_dir_sync(["2026-07-21_flight02", "2026-07-21_flight01"], [])
        assert [d for d, _ in plan] == ["2026-07-21_flight01", "2026-07-21_flight02"]


class TestParseUlogList:
    def test_parses_table_rows(self):
        text = (
            "   id  UTC 시각              크기\n"
            "   25  2026-07-20 10:37:04     608,862\n"
            "   26  2026-07-20 10:37:50   1,823,644\n"
        )
        entries = parse_ulog_list(text)
        assert entries == [
            (25, "2026-07-20 10:37:04", 608862),
            (26, "2026-07-20 10:37:50", 1823644),
        ]

    def test_gps_time_missing_becomes_none(self):
        text = "   13  (GPS 시각 없음)          3,300,000\n"
        entries = parse_ulog_list(text)
        assert entries == [(13, None, 3300000)]

    def test_ignores_header_and_blank_lines(self):
        text = "   id  UTC 시각              크기\n\n"
        assert parse_ulog_list(text) == []

    def test_empty_input(self):
        assert parse_ulog_list("") == []


class TestCollectedUlogIds:
    def test_extracts_ids_from_various_filenames(self, tmp_path):
        (tmp_path / "2026-07-20_flight02").mkdir()
        (tmp_path / "2026-07-20_flight02" / "log_19_2026-07-20-10-31-12.ulg").write_text("x")
        (tmp_path / "2026-07-18_unlogged").mkdir()
        (tmp_path / "2026-07-18_unlogged" / "log_0_2026-07-18-03-06-52.ulg").write_text("x")
        (tmp_path / "2026-07-06_flight01").mkdir()
        (tmp_path / "2026-07-06_flight01" / "log_13.ulg").write_text("x")  # 시각 없는 케이스
        assert collected_ulog_ids(str(tmp_path)) == {19, 0, 13}

    def test_empty_root(self, tmp_path):
        assert collected_ulog_ids(str(tmp_path)) == set()

    def test_ignores_non_ulog_files(self, tmp_path):
        (tmp_path / "2026-07-20_flight01").mkdir()
        (tmp_path / "2026-07-20_flight01" / "launch.log").write_text("x")
        assert collected_ulog_ids(str(tmp_path)) == set()


class TestRunTimeoutResilience:
    def test_hard_timeout_does_not_raise(self):
        # 실사용 중 SSH 순간끊김 하나로 전체 스크립트가 uncaught 예외로 죽던 버그의 회귀 테스트.
        r = collect_run(["sleep", "2"], timeout=0.1)
        assert r.returncode != 0
        assert "타임아웃" in r.stderr

    def test_normal_command_still_works(self):
        r = collect_run(["echo", "hi"], timeout=5)
        assert r.returncode == 0
        assert r.stdout.strip() == "hi"


class TestCatchallDirname:
    def test_format(self):
        assert catchall_dirname("2026-07-21") == "2026-07-21_manual"


class TestNotesSkeleton:
    def test_contains_required_fields(self):
        text = notes_skeleton("2026-07-21_manual")
        assert "# 2026-07-21_manual" in text
        assert "- **비행 조건:**" in text
        assert "- **관찰:**" in text
        assert "- **결론:**" in text

    def test_never_prefills_observation_or_conclusion(self):
        # 해석은 사람 몫 — 관찰/결론 줄은 항상 빈 채로 끝나야 한다
        text = notes_skeleton("2026-07-21_manual")
        obs_line = next(l for l in text.splitlines() if l.startswith("- **관찰:**"))
        concl_line = next(l for l in text.splitlines() if l.startswith("- **결론:**"))
        assert obs_line == "- **관찰:**"
        assert concl_line == "- **결론:**"


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
