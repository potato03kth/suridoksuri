import json
import logging
import logging.handlers
import subprocess
from pathlib import Path


def get_git_commit_hash() -> str:
    """git 저장소 밖이거나 git이 없으면 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup_dual_sink_logger(
    name: str,
    log_dir: str,
    console_level: int = logging.INFO,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    이중 싱크 구조적 로깅 (vision_plan.md §7.4): 터미널(사람용, 레벨) + 파일(.log, 크기 로테이션).
    파일 싱크는 항상 DEBUG까지 남긴다 — 터미널 레벨은 사람이 보는 소음만 줄인다.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / f"{name}.log", maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    return logger


def log_provenance_header(logger: logging.Logger, config: dict, calibration_id: str | None = None) -> None:
    """세션 시작 시 1회 기록: 해석된 config 전체 + git 커밋 해시 + 캘리브레이션 id (§7.3 provenance echo)."""
    header = {
        "git_commit": get_git_commit_hash(),
        "calibration_id": calibration_id,
        "config": config,
    }
    logger.info("provenance %s", json.dumps(header, ensure_ascii=False))
