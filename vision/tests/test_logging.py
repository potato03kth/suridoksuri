import json
import logging
from vision.utils.logging import setup_dual_sink_logger, log_provenance_header, get_git_commit_hash


def test_dual_sink_has_console_and_file_handlers(tmp_path):
    logger = setup_dual_sink_logger("test_dual_sink_a", str(tmp_path))
    assert len(logger.handlers) == 2


def test_file_sink_receives_message(tmp_path):
    logger = setup_dual_sink_logger("test_dual_sink_b", str(tmp_path))
    logger.info("hello file sink")
    for h in logger.handlers:
        h.flush()

    log_file = tmp_path / "test_dual_sink_b.log"
    assert "hello file sink" in log_file.read_text()


def test_console_level_does_not_suppress_file_level(tmp_path):
    logger = setup_dual_sink_logger("test_dual_sink_c", str(tmp_path), console_level=logging.WARNING)
    logger.debug("debug goes to file only")
    for h in logger.handlers:
        h.flush()

    log_file = tmp_path / "test_dual_sink_c.log"
    assert "debug goes to file only" in log_file.read_text()


def test_repeated_setup_does_not_duplicate_handlers(tmp_path):
    setup_dual_sink_logger("test_dual_sink_d", str(tmp_path))
    logger = setup_dual_sink_logger("test_dual_sink_d", str(tmp_path))
    assert len(logger.handlers) == 2


def test_provenance_header_has_git_hash_and_config(tmp_path):
    logger = setup_dual_sink_logger("test_dual_sink_e", str(tmp_path))
    log_provenance_header(logger, {"preset": "vertiport_coarse"}, calibration_id="calib_v1")
    for h in logger.handlers:
        h.flush()

    line = next(l for l in (tmp_path / "test_dual_sink_e.log").read_text().splitlines() if "provenance" in l)
    header = json.loads(line.split("provenance ", 1)[1])
    assert header["config"] == {"preset": "vertiport_coarse"}
    assert header["calibration_id"] == "calib_v1"
    assert "git_commit" in header


def test_get_git_commit_hash_returns_nonempty_string():
    assert isinstance(get_git_commit_hash(), str)
    assert len(get_git_commit_hash()) > 0
