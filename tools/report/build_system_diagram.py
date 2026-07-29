#!/usr/bin/env python3
"""보고서용 온보드 시스템 구성도 생성기 (RPi 5 ↔ Pixhawk 6C, 지상장비 포함 전체 계통).

`build_org_chart.py` 와 같은 방식 — 좌표를 1x 픽셀 공간에 적고 --scale 로 확대 렌더한다.

이 그림이 말하려는 것 세 가지:

  1. Pixhawk 는 네트워크가 아니라 **파일**(`/dev/ttyACM0`, char 166:0)로 붙는다.
     USB CDC-ACM 클래스 → 커널 `cdc_acm` 드라이버 → 가상 시리얼 노드.
  2. 컨테이너 경계를 넘는 통로는 **딱 3개**다 (`docker inspect fc` 실측, 2026-07-30):
         NetworkMode = "host"                        ← ①
         Devices     = /dev/ttyACM0 rwm              ← ②
         Binds       = /home/suri/drone_ws:/drone_ws ← ③
     컨테이너 전용 가상 네트워크(브리지·vLAN)는 존재하지 않는다.
  3. **영상은 컨테이너를 지나지 않고, 지상국 텔레메트리는 RPi 를 지나지 않는다.**
     둘 다 자주 잘못 그려지는 지점이다.

배선 근거(전부 실측):
  lsusb                 3185:0038 Auterion PX4 FMU v6C.x, cdc_acm, Full-Speed 12 Mbps
  udevadm               MAJOR=166 MINOR=0, by-id usb-Auterion_PX4_FMU_v6C.x_0-if00
  CRITICAL_PARAMS.md    MAV_0_CONFIG=101(TELEM1) / MAV_1,2_CONFIG=0
  px4_v6c_patch_build   crsf_rc → UART /dev/ttyS1(TELEM3), RC_CRSF_PRT_CFG=103
  fc_pusher_thrust_audit CA_ROTOR0..3 AZ=-1(리프트), CA_ROTOR4 AX=+1(푸셔, AUX5)

사용:
    python tools/report/build_system_diagram.py [출력경로] [--scale N]

기본 출력: docs/figures/system_diagram.png (3x) + system_diagram_1x.png (대조용)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── 폰트 ────────────────────────────────────────────────────────────────────
FONT_DIR = "/usr/share/fonts/truetype/nanum"
F_REG = f"{FONT_DIR}/NanumGothic.ttf"
F_BLD = f"{FONT_DIR}/NanumGothicBold.ttf"
F_MONO = f"{FONT_DIR}/NanumGothicCoding.ttf"

# ── 색 (사용자 손그림 팔레트를 그대로 따른다) ────────────────────────────────
PURPLE = (122, 59, 176)   # 물리 장비 경계 (RPi5 / Pixhawk)
ORANGE = (232, 116, 22)   # 도커 컨테이너 경계
BLUE = (37, 122, 214)     # 우리가 작성한 프로세스
RED = (206, 43, 43)       # 외부 미들웨어 (mavros)
GREEN = (28, 138, 92)     # 커널 / 하드웨어 링크
TEAL = (18, 132, 150)     # 기체 탑재 주변 하드웨어
GRAY = (112, 112, 112)
MUTE = (155, 155, 155)    # 미구현 · 미배선
INK = (28, 28, 28)
BG = (255, 255, 255)
WASH_ORANGE = (253, 245, 236)
WASH_GRAY = (244, 245, 247)
WASH_GREEN = (238, 248, 243)
WASH_TEAL = (238, 247, 249)
WASH_BLUE = (240, 246, 253)

CANVAS = (1790, 1200)


# ── 렌더 헬퍼 ───────────────────────────────────────────────────────────────
class Pen:
    """1x 좌표를 받아 SCALE 배로 그린다."""

    def __init__(self, draw: ImageDraw.ImageDraw, scale: float):
        self.d = draw
        self.s = scale
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, path: str, px: float) -> ImageFont.FreeTypeFont:
        key = (path, round(px * self.s))
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(path, round(px * self.s))
        return self._fonts[key]

    def fill(self, xy, color, radius=8):
        self.d.rounded_rectangle(
            [v * self.s for v in xy], radius=radius * self.s, fill=color
        )

    def box(self, xy, color, width=2.0, radius=8, fill=None):
        x0, y0, x1, y1 = (v * self.s for v in xy)
        self.d.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius * self.s,
            outline=color,
            width=max(1, round(width * self.s)),
            fill=fill,
        )

    def dbox(self, xy, color, width=2.0, dash=7, gap=5):
        """점선 사각형 — 격리된 벽이 아니라 '구멍 뚫린 경계'를 뜻한다."""
        x0, y0, x1, y1 = xy
        segs = []
        for a, b in _dashes(x0, x1, dash, gap):
            segs += [(a, y0, b, y0), (a, y1, b, y1)]
        for a, b in _dashes(y0, y1, dash, gap):
            segs += [(x0, a, x0, b), (x1, a, x1, b)]
        for sx0, sy0, sx1, sy1 in segs:
            self.d.line(
                (sx0 * self.s, sy0 * self.s, sx1 * self.s, sy1 * self.s),
                fill=color,
                width=max(1, round(width * self.s)),
            )

    def bullet(self, x, y, color, tri=False, r=3.0):
        """목록 글머리 — 나눔고딕에 ▸/▪ 글리프가 없어 직접 그린다."""
        if tri:
            pts = [(x, y - r), (x + r * 1.6, y), (x, y + r)]
            self.d.polygon([(a * self.s, b * self.s) for a, b in pts], fill=color)
        else:
            self.d.rectangle(
                ((x - r) * self.s, (y - r) * self.s, (x + r) * self.s, (y + r) * self.s),
                fill=color,
            )

    def text(self, xy, s, color=INK, px=14, bold=False, mono=False, anchor="la"):
        path = F_MONO if mono else (F_BLD if bold else F_REG)
        self.d.text(
            (xy[0] * self.s, xy[1] * self.s),
            s,
            font=self.font(path, px),
            fill=color,
            anchor=anchor,
        )

    def lines(self, s, x, y, color=INK, px=14, lead=17, bold=False, mono=False):
        for i, ln in enumerate(s):
            self.text((x, y + i * lead), ln, color, px, bold, mono)

    def arrow(self, pts, color, width=2.0, head=7.0, double=False, dashed=False):
        """직교 폴리라인 + 화살촉. dashed=True 는 무선 링크."""
        if dashed:
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                for a, b in _dashes(0.0, 1.0, 0.045, 0.03):
                    self.d.line(
                        (
                            (x0 + (x1 - x0) * a) * self.s,
                            (y0 + (y1 - y0) * a) * self.s,
                            (x0 + (x1 - x0) * b) * self.s,
                            (y0 + (y1 - y0) * b) * self.s,
                        ),
                        fill=color,
                        width=max(1, round(width * self.s)),
                    )
        else:
            self.d.line(
                [(x * self.s, y * self.s) for x, y in pts],
                fill=color,
                width=max(1, round(width * self.s)),
                joint="curve",
            )
        self._head(pts[-2], pts[-1], color, head)
        if double:
            self._head(pts[1], pts[0], color, head)

    def _head(self, frm, to, color, head):
        (x0, y0), (x1, y1) = frm, to
        dx, dy = x1 - x0, y1 - y0
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        a = (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55)
        b = (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)
        self.d.polygon(
            [(p[0] * self.s, p[1] * self.s) for p in ((x1, y1), a, b)], fill=color
        )


def _dashes(a: float, b: float, dash: float, gap: float):
    out, t = [], a
    while t < b:
        out.append((t, min(t + dash, b)))
        t += dash + gap
    return out


# ── 기하 (1x 픽셀) ──────────────────────────────────────────────────────────
CAM_HW = (38, 88, 320, 176)
GROUND = (820, 88, 1750, 176)

RPI = (38, 240, 712, 1172)
CAM0 = (58, 292, 320, 348)
LIBCAM = (58, 364, 320, 430)
VISION = (58, 452, 692, 646)
V_SRC = (76, 508, 268, 578)
V_PIPE = (282, 508, 474, 578)
V_SM = (488, 508, 674, 578)
SINK = (58, 668, 252, 742)
MJPEG = (266, 668, 460, 742)
REC = (474, 668, 692, 742)
CONT = (52, 774, 706, 1062)
SHIM = (70, 820, 322, 900)
FCROS = (70, 916, 322, 1044)
MAVROS = (410, 820, 690, 906)
ROSBAG = (410, 926, 690, 990)
KERN = (52, 1080, 706, 1160)

FCU = (820, 240, 1266, 748)
PERIPH = (820, 782, 1266, 1172)

LEGEND = (1310, 240, 1750, 704)
NOTES = (1310, 738, 1750, 1172)


def draw(pen: Pen) -> None:
    # ── 제목 ────────────────────────────────────────────────────────────────
    pen.text((36, 22), "온보드 컴퓨팅 시스템 구성", INK, 26, bold=True)
    pen.text(
        (36, 58),
        "Raspberry Pi 5  ↔  Pixhawk 6C  —  컨테이너 경계를 넘는 통로는 ①②③ 세 개뿐이다"
        "   ·   실선 = 유선,  점선 = 무선",
        GRAY,
        13,
    )

    # ── 기체 외부: 카메라 모듈 ───────────────────────────────────────────────
    pen.fill(CAM_HW, WASH_TEAL)
    pen.box(CAM_HW, TEAL, 2.0, 8)
    pen.text((58, 100), "카메라 모듈", TEAL, 15, bold=True)
    pen.lines(
        [
            "IMX708 클론 CAM109 (서드파티, 정품 CM3 아님)",
            "기체 하방 장착 · 오토포커스",
        ],
        58,
        126,
        GRAY,
        10.5,
        lead=16,
    )

    # ── 지상 장비 ───────────────────────────────────────────────────────────
    pen.fill(GROUND, WASH_TEAL)
    pen.box(GROUND, TEAL, 2.0, 8)
    pen.text((780, 98), "지상 장비", TEAL, 15, bold=True)
    pen.lines(
        ["조종기 — ExpressLRS 송신기", "수동 이착륙 · 비상 조종권 회수"],
        780,
        124,
        GRAY,
        10.5,
        lead=16,
    )
    pen.lines(
        ["지상국 노트북 — QGroundControl", "MAVLink 텔레메트리 · ulog 회수"],
        1090,
        124,
        GRAY,
        10.5,
        lead=16,
    )
    pen.lines(
        ["영상 다운링크 수신 (RTL8812AU)", "미장착 — 목표 하드웨어"],
        1400,
        124,
        MUTE,
        10.5,
        lead=16,
    )

    # ── 호스트 ──────────────────────────────────────────────────────────────
    pen.box(RPI, PURPLE, 2.6, 10)
    pen.text(
        (58, 254), "Raspberry Pi 5  ·  Ubuntu 24.04 (호스트)", PURPLE, 16, bold=True
    )

    pen.box(CAM0, GREEN, 2.0, 7)
    pen.text((76, 302), "CAM0 커넥터", GREEN, 13, bold=True)
    pen.text((76, 324), "MIPI CSI-2 4-lane (RPi5 CAM/DISP 0)", GRAY, 10.5)

    pen.box(LIBCAM, GREEN, 2.0, 7)
    pen.text((76, 374), "libcamera / picamera2", GREEN, 13, bold=True)
    pen.lines(
        ["로컬 소스빌드 (ipa_rpi_pisp.so)", "/dev/video19 · /dev/media0~4"],
        76,
        396,
        GRAY,
        10.5,
        lead=15,
    )

    # vision 파이프라인
    pen.fill(VISION, WASH_BLUE, 8)
    pen.box(VISION, BLUE, 2.4, 8)
    pen.text((76, 464), "vision/main.py", BLUE, 15, bold=True, mono=True)
    pen.text(
        (76, 488), "picam-venv · Python 3.12 · ROS 없음 · cv2 4.13", GRAY, 10.5
    )

    pen.box(V_SRC, BLUE, 1.8, 6, fill=BG)
    pen.text((90, 518), "FrameSource", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["Live (picamera2)", "Dir · Bag (결정론적 재생)"], 90, 538, GRAY, 10, lead=14
    )

    pen.box(V_PIPE, BLUE, 1.8, 6, fill=BG)
    pen.text((296, 518), "Pipeline", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["registry → 모듈 체인", "preset yaml 로 시나리오 전환"],
        296,
        538,
        GRAY,
        10,
        lead=14,
    )

    pen.box(V_SM, BLUE, 1.8, 6, fill=BG)
    pen.text((502, 518), "state_machine · target", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["탐색 → 추적 → 확정", "TargetEstimate (body FLU 상대 pose)"],
        502,
        538,
        GRAY,
        10,
        lead=14,
    )

    pen.lines(
        [
            "모듈  denoise · illumination · color · edge · morphology · background · aruco ·"
            " tracker · fusion · prior_score",
            "         distress_box/mat (빠께스) · vertiport_ring/v/field (초록구역)",
            "프리셋  vertiport_coarse/fine · distress_coarse/fine · low_light · video ·"
            " single_frame",
        ],
        76,
        592,
        GRAY,
        10,
        lead=17,
    )

    # 출력 3갈래
    pen.box(SINK, BLUE, 2.2, 7)
    pen.text((74, 678), "utils/target_sink.py", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["TCP 서버 127.0.0.1:8091", "JSON Lines · 비차단 drop-oldest"],
        74,
        700,
        GRAY,
        10,
        lead=15,
    )

    pen.box(MJPEG, BLUE, 2.2, 7)
    pen.text((282, 678), "utils/stream.py", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["MJPEG 0.0.0.0:8080", "디버그 뷰 (지상 브라우저)"], 282, 700, GRAY, 10, lead=15
    )

    pen.box(REC, BLUE, 2.2, 7)
    pen.text((490, 678), "blackbox · h264_stream", BLUE, 12, bold=True, mono=True)
    pen.lines(
        ["JSONL 로그 + H.264 녹화/다운링크", "영상은 컨테이너를 지나지 않는다"],
        490,
        700,
        GRAY,
        10,
        lead=15,
    )

    # ── 컨테이너 ────────────────────────────────────────────────────────────
    pen.fill(CONT, WASH_ORANGE, 10)
    pen.dbox(CONT, ORANGE, 2.4)
    pen.text((72, 786), "Docker 컨테이너  fc", ORANGE, 15, bold=True)
    pen.text((72, 808), "ros:humble  (Ubuntu 22.04 · Python 3.10)", GRAY, 10.5)

    pen.box(SHIM, BLUE, 2.2, 7, fill=BG)
    pen.text((86, 832), "vision/ros/shim_node.py", BLUE, 12, bold=True, mono=True)
    pen.lines(
        [
            "shim_core — TCP 클라이언트 · 재접속",
            "body FLU → ENU 절대 setpoint 변환",
            "/mavros/local_position/pose 구독",
        ],
        86,
        854,
        GRAY,
        10,
        lead=15,
    )

    pen.box(FCROS, BLUE, 2.2, 7, fill=BG)
    pen.text((86, 928), "offboard_node", BLUE, 12, bold=True, mono=True)
    pen.text((86, 948), "telemetry_node · mission_node", BLUE, 12, mono=True)
    pen.lines(
        [
            "fc_ros — 비행 상태기계",
            "fc_bridge — 경로계획 · L1 유도 ·",
            "                 speed_profile · precision_land",
        ],
        86,
        972,
        GRAY,
        10,
        lead=15,
    )
    pen.text((86, 1022), "정밀착륙 서브상태 — 미구현 (F2)", MUTE, 10, bold=True)

    pen.box(MAVROS, RED, 2.2, 7, fill=BG)
    pen.text((426, 832), "mavros_node", RED, 15, bold=True, mono=True)
    pen.lines(
        [
            "ROS 2 토픽·서비스  ↔  MAVLink v2 변환",
            "fcu_url:=/dev/ttyACM0:57600",
            "/mavros/{state,altitude,battery,imu,...}",
        ],
        426,
        858,
        GRAY,
        10,
        lead=15,
    )

    pen.box(ROSBAG, GRAY, 1.8, 7, fill=BG)
    pen.text((426, 936), "ros2 bag  ·  launch.log", INK, 12, bold=True, mono=True)
    pen.text((426, 960), "③ /drone_ws 바인드 마운트에 기록", GRAY, 10)

    # ── 커널 ────────────────────────────────────────────────────────────────
    pen.fill(KERN, WASH_GREEN, 8)
    pen.box(KERN, GREEN, 2.2, 8)
    pen.text(
        (72, 1092), "Linux 커널  —  호스트와 컨테이너가 공유한다", GREEN, 14, bold=True
    )
    pen.text(
        (72, 1116),
        "cdc_acm 드라이버  →  /dev/ttyACM0   (문자 디바이스 char 166:0, root:dialout)",
        INK,
        11.5,
        mono=True,
    )
    pen.text(
        (72, 1138),
        "by-id: usb-Auterion_PX4_FMU_v6C.x_0-if00        네트워크 네임스페이스도 이 커널 것 하나뿐",
        GRAY,
        10,
    )

    # ── Pixhawk ─────────────────────────────────────────────────────────────
    pen.box(FCU, PURPLE, 2.6, 10)
    pen.text((844, 258), "Pixhawk 6C", PURPLE, 20, bold=True)
    pen.lines(
        [
            "PX4 커스텀 펌웨어 — F-17/F-4 패치 + crsf_rc 드라이버",
            "Build Jul 28 2026 20:27:22   ·   MAV_TYPE 22 (VTOL)",
        ],
        844,
        290,
        GRAY,
        11,
        lead=17,
    )

    pen.text((844, 336), "포트 배치 (실측 파라미터)", INK, 12, bold=True)
    ports = [
        ("USB (CDC-ACM)", "컴패니언 링크 → MAVROS", "② 이 선", GREEN),
        ("TELEM1", "MAV_0_CONFIG=101 → 텔레메트리 라디오", "", TEAL),
        ("TELEM3 /dev/ttyS1", "RC_CRSF_PRT_CFG=103 → ELRS 수신기", "", TEAL),
        ("GPS1", "GPS · 컴퍼스 모듈", "", TEAL),
        ("MAIN / AUX PWM", "ESC · 제어면 서보", "", TEAL),
    ]
    y = 362
    for name, desc, tag, col in ports:
        pen.bullet(850, y + 6, col, tri=True)
        pen.text((866, y), name, INK, 11, bold=True, mono=True)
        pen.text((866, y + 17), desc, GRAY, 10.5)
        if tag:
            pen.text((1210, y), tag, ORANGE, 11, bold=True)
        y += 40

    pen.text((844, 572), "온보드 기능", INK, 12, bold=True)
    pen.lines(
        [
            "· EKF2 자세 · 위치 추정 (GPS + IMU + 자기계 + 기압계)",
            "· VTOL 상태기계 — MC ↔ FW 천이",
            "· 제어 얼로케이터  CA_ROTOR0..3 (리프트) · CA_ROTOR4 (푸셔)",
            "· 로그 기록 → SD 카드 (ulog)",
            "· 안전장치 — RC 손실 / 데이터링크 손실 / 배터리 페일세이프",
        ],
        844,
        598,
        GRAY,
        10.5,
        lead=19,
    )
    pen.text(
        (844, 706),
        "AGL 소스: 라이다 미배선 → 이륙지점 기준 평탄지 가정",
        MUTE,
        10.5,
        bold=True,
    )

    # ── 기체 탑재 주변 하드웨어 ──────────────────────────────────────────────
    pen.fill(PERIPH, WASH_TEAL, 8)
    pen.box(PERIPH, TEAL, 2.0, 8)
    pen.text((844, 794), "기체 탑재 주변 하드웨어", TEAL, 14, bold=True)

    hw = [
        ("ExpressLRS 수신기", "CRSF 프로토콜 → TELEM3", TEAL),
        ("텔레메트리 라디오", "지상국 직결 — RPi·도커 무관", TEAL),
        ("GPS · 컴퍼스 모듈", "CAL_MAG0 캘리브레이션 적용", TEAL),
        ("피토관", "대기속도 IAS — 천이 판정", TEAL),
        ("리프트 로터 ×4", "CA_ROTOR0..3, AZ = -1.0 (수직추력)", TEAL),
        ("푸셔 모터 ×1", "CA_ROTOR4, AX = +1.0, AUX 5번 출력", TEAL),
        ("제어면 서보", "에일러론 · 엘리베이터 · 러더", TEAL),
        ("배터리 · 전원모듈 · BEC", "RPi5 는 비-PD 급전 (PSU_MAX_CURRENT 완화)", TEAL),
    ]
    y = 822
    for name, desc, col in hw:
        pen.bullet(850, y + 5, col)
        pen.text((868, y - 1), name, INK, 11, bold=True)
        pen.text((868, y + 16), desc, GRAY, 10)
        y += 40

    pen.text((850, 1146), "라이다 (AGL) — 미배선", MUTE, 10.5, bold=True)

    # ── 연결선 ──────────────────────────────────────────────────────────────
    cx = 189  # 좌측 컬럼 중심
    pen.arrow([(cx, CAM_HW[3]), (cx, CAM0[1])], TEAL, 2.2)
    pen.text((cx + 12, 194), "MIPI CSI-2 리본 케이블", GRAY, 10.5)
    pen.arrow([(cx, CAM0[3]), (cx, LIBCAM[1])], GREEN, 2.0)
    pen.arrow([(cx, LIBCAM[3]), (cx, VISION[1])], GREEN, 2.0)
    pen.text((cx + 12, 434), "프레임 버퍼", GRAY, 10)

    # vision 내부 파이프라인
    pen.arrow([(V_SRC[2], 543), (V_PIPE[0], 543)], BLUE, 1.8, head=5.5)
    pen.arrow([(V_PIPE[2], 543), (V_SM[0], 543)], BLUE, 1.8, head=5.5)

    # vision → 출력 3갈래
    for bx in (SINK, MJPEG, REC):
        mid = (bx[0] + bx[2]) / 2
        pen.arrow([(mid, VISION[3]), (mid, bx[1])], BLUE, 1.8, head=5.5)

    # ① 소켓 — 컨테이너 경계를 넘는다
    pen.arrow([(155, SINK[3]), (155, SHIM[1])], BLUE, 2.4)
    pen.text((167, 748), "①", ORANGE, 15, bold=True)
    pen.text(
        (185, 750),
        "--network host  →  컨테이너의 127.0.0.1 이 곧 호스트의 127.0.0.1",
        ORANGE,
        11,
    )

    # shim → fc_ros
    pen.arrow([(196, SHIM[3]), (196, FCROS[1])], BLUE, 2.0)
    pen.text((208, 902), "/vision/landing_setpoint", GRAY, 10)

    # fc_ros ↔ mavros
    pen.arrow(
        [(FCROS[2], 950), (366, 950), (366, 878), (MAVROS[0], 878)],
        RED,
        2.0,
        double=True,
    )
    pen.text((338, 906), "ROS 2 DDS", GRAY, 10, )
    pen.text((338, 922), "토픽 · 서비스", GRAY, 10)

    # mavros → 커널 (②)
    pen.arrow([(676, MAVROS[3]), (676, KERN[1])], GREEN, 2.4, double=True)
    pen.text((410, 1002), "②", ORANGE, 15, bold=True)
    pen.lines(
        [
            "--device /dev/ttyACM0 — 커널 노드 패스스루.",
            "복제·포워딩이 아니라 같은 객체를 가리키는",
            "두 번째 이름이다.",
        ],
        430,
        1004,
        ORANGE,
        10,
        lead=15,
    )
    pen.arrow([(648, 1019), (666, 1019)], ORANGE, 1.6, head=5)

    # 커널 → Pixhawk (USB) — 회랑(712~820)에 라벨을 세로로 쌓는다
    pen.arrow([(KERN[2], 1120), (766, 1120), (766, 400), (FCU[0], 400)], PURPLE, 2.6,
              double=True)
    pen.lines(
        ["USB-C 케이블", "Full-Speed", "12 Mbps", "CDC-ACM 위에", "MAVLink v2"],
        718,
        300,
        PURPLE,
        9.5,
        lead=14,
    )

    # 무선 링크 (지상 ↔ 기체)
    pen.arrow([(940, GROUND[3]), (940, FCU[1])], TEAL, 2.0, dashed=True, double=True)
    pen.text((952, 190), "RC 2.4 GHz", TEAL, 11, bold=True)
    pen.text((952, 208), "ExpressLRS 조종 링크", GRAY, 10)

    pen.arrow([(1220, GROUND[3]), (1220, FCU[1])], TEAL, 2.0, dashed=True, double=True)
    pen.text((1208, 190), "MAVLink 텔레메트리", TEAL, 11, bold=True, anchor="ra")
    pen.text((1208, 208), "RPi·도커를 지나지 않는다", GRAY, 10, anchor="ra")

    # Pixhawk ↔ 주변 하드웨어
    pen.arrow([(1043, FCU[3]), (1043, PERIPH[1])], TEAL, 2.2, double=True)
    pen.text((1055, 752), "UART · PWM · I2C/CAN 결선", GRAY, 10.5)

    # ── 범례: 3개의 통로 ────────────────────────────────────────────────────
    pen.fill(LEGEND, WASH_GRAY, 8)
    pen.box(LEGEND, GRAY, 1.8, 8)
    pen.text((1332, 260), "컨테이너 경계를 넘는 통로 = 3개뿐", INK, 15, bold=True)
    pen.text((1332, 286), "docker inspect fc 실측 (2026-07-30)", GRAY, 10.5)

    items = [
        (
            "①",
            "--network host",
            [
                "네트워크 네임스페이스를 아예 분리하지 않는다.",
                "컨테이너 전용 가상 네트워크(브리지·vLAN)는",
                "존재하지 않는다 → 127.0.0.1 이 그대로 통한다.",
            ],
        ),
        (
            "②",
            "--device /dev/ttyACM0",
            [
                "커널 디바이스 노드를 같은 major:minor 로 노출",
                "+ cgroup rwm 허용. Pixhawk 는 네트워크 포트가",
                "아니라 파일이다. 두 프로세스가 동시에 열면",
                "MAVLink 스트림이 반씩 찢어진다.",
            ],
        ),
        (
            "③",
            "-v ~/drone_ws:/drone_ws",
            [
                "소스 바인드 마운트. 호스트에서 git pull,",
                "컨테이너 안에서 colcon build 하는 이유.",
            ],
        ),
    ]
    y = 318
    for mark, code, body in items:
        pen.text((1334, y), mark, ORANGE, 15, bold=True)
        pen.text((1358, y + 1), code, INK, 12, bold=True, mono=True)
        pen.lines(body, 1358, y + 24, GRAY, 10.5, lead=16)
        y += 24 + 16 * len(body) + 16

    pen.text((1332, 596), "데이터 흐름 한 줄 요약", INK, 12, bold=True)
    pen.lines(
        [
            "카메라 → libcamera → vision/main.py → ① 소켓 8091",
            "→ shim_node → offboard_node → mavros → ② /dev/ttyACM0",
            "→ USB CDC-ACM → PX4 → 얼로케이터 → ESC · 서보",
        ],
        1332,
        620,
        GRAY,
        10.5,
        lead=16,
    )

    # ── 노트 ────────────────────────────────────────────────────────────────
    pen.fill(NOTES, WASH_GRAY, 8)
    pen.box(NOTES, GRAY, 1.8, 8)
    pen.text((1332, 758), "이 그림에서 꼭 읽어야 할 것", INK, 15, bold=True)
    pen.lines(
        [
            "· 영상은 컨테이너를 지나지 않는다. 카메라·MJPEG·H.264",
            "   전부 호스트에서 끝난다.",
            "· 지상국 텔레메트리도 RPi 를 지나지 않는다 — 픽스호크에",
            "   직결된 별도 라디오다. RPi 가 죽어도 지상국은 살아 있다.",
            "· 두 도메인(vision · fc)은 코드 수준 import 도, colcon·pip",
            "   빌드 의존도 없다. 만나는 곳은 ① 소켓 하나뿐이다.",
            "· 환경이 갈린 이유: libcamera 스택은 Ubuntu 24.04/Py3.12,",
            "   ROS 2 Humble 은 Ubuntu 22.04/Py3.10 을 요구한다.",
        ],
        1332,
        788,
        GRAY,
        10.5,
        lead=17,
    )

    pen.text((1332, 950), "현재 미구현 · 미배선 (회색 표기)", INK, 13, bold=True)
    pen.lines(
        [
            "· offboard_node 정밀착륙 서브상태 (F2) — 토픽만 나오고",
            "   받는 쪽이 아직 없다.",
            "· 라이다 AGL 미배선 → 이륙지점 기준 평탄지 가정.",
            "· RTL8812AU 영상 다운링크 미장착 (목표 하드웨어).",
            "· 카메라 모듈은 2026-07-29 물리 고장 — 소프트웨어 스택은",
            "   유효하고 Pi 쪽 결백은 실측으로 확정됐다.",
        ],
        1332,
        978,
        MUTE,
        10.5,
        lead=17,
    )

    pen.text(
        (1332, 1136),
        "※ fcu_url 의 :57600 은 CDC-ACM 에선 무의미하다 — 보드레이트는 실제",
        GRAY,
        10,
    )
    pen.text(
        (1332, 1152),
        "   UART 에만 있는 개념이고, 여기선 USB 속도로 전송된다.",
        GRAY,
        10,
    )


def build(scale: float) -> Image.Image:
    img = Image.new("RGB", (round(CANVAS[0] * scale), round(CANVAS[1] * scale)), BG)
    draw(Pen(ImageDraw.Draw(img), scale))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="docs/figures/system_diagram.png")
    ap.add_argument("--scale", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = build(args.scale)
    img.save(out)
    print(f"{out}  {img.width}x{img.height}")

    one = out.with_name(out.stem + "_1x" + out.suffix)
    img1 = build(1.0)
    img1.save(one)
    print(f"{one}  {img1.width}x{img1.height}")


if __name__ == "__main__":
    main()
