#!/usr/bin/env python3
"""보고서용 온보드 시스템 구성도 생성기 (RPi 5 ↔ Pixhawk 6C).

`build_org_chart.py` 와 같은 방식 — 좌표를 1x 픽셀 공간에 적고 --scale 로 확대 렌더한다.

이 그림이 말하려는 것은 **컨테이너 경계를 넘는 통로가 딱 3개뿐**이라는 것이다.
`docker inspect fc` 실측(2026-07-30):

    NetworkMode = "host"
    Devices     = [{"PathOnHost":"/dev/ttyACM0","PathInContainer":"/dev/ttyACM0",
                    "CgroupPermissions":"rwm"}]
    Binds       = ["/home/suri/drone_ws:/drone_ws"]

즉 컨테이너 전용 가상 네트워크(브리지/vLAN)는 존재하지 않는다. Pixhawk 는 네트워크가
아니라 USB CDC-ACM 문자 디바이스(`/dev/ttyACM0`, char 166:0)로 붙고, 그 노드를
`--device` 로 컨테이너에 그대로 노출한 것이다 — 포워딩이 아니라 같은 커널 객체다.

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
GRAY = (112, 112, 112)
INK = (28, 28, 28)
BG = (255, 255, 255)
WASH_ORANGE = (253, 244, 235)
WASH_GRAY = (243, 244, 246)
WASH_GREEN = (238, 248, 243)

CANVAS = (1300, 848)


# ── 렌더 헬퍼 ───────────────────────────────────────────────────────────────
class Pen:
    """1x 좌표를 받아 SCALE 배로 그린다."""

    def __init__(self, draw: ImageDraw.ImageDraw, scale: float):
        self.d = draw
        self.s = scale
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, path: str, px: int) -> ImageFont.FreeTypeFont:
        key = (path, px)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(path, round(px * self.s))
        return self._fonts[key]

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
        """점선 사각형 (컨테이너 경계처럼 '뚫려 있는' 경계를 표현)."""
        x0, y0, x1, y1 = xy
        segs = []
        for x in _dashes(x0, x1, dash, gap):
            segs.append((x[0], y0, x[1], y0))
            segs.append((x[0], y1, x[1], y1))
        for y in _dashes(y0, y1, dash, gap):
            segs.append((x0, y[0], x0, y[1]))
            segs.append((x1, y[0], x1, y[1]))
        for sx0, sy0, sx1, sy1 in segs:
            self.d.line(
                (sx0 * self.s, sy0 * self.s, sx1 * self.s, sy1 * self.s),
                fill=color,
                width=max(1, round(width * self.s)),
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

    def lines(self, s, x, y, color=INK, px=14, lead=17, bold=False, mono=False,
              anchor="la"):
        for i, ln in enumerate(s):
            self.text((x, y + i * lead), ln, color, px, bold, mono, anchor)

    def arrow(self, pts, color, width=2.0, head=7.0, double=False):
        """직교 폴리라인 + 화살촉."""
        sp = [(x * self.s, y * self.s) for x, y in pts]
        self.d.line(sp, fill=color, width=max(1, round(width * self.s)), joint="curve")
        self._head(pts[-2], pts[-1], color, head)
        if double:
            self._head(pts[1], pts[0], color, head)

    def _head(self, frm, to, color, head):
        (x0, y0), (x1, y1) = frm, to
        dx, dy = x1 - x0, y1 - y0
        n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        tip = (x1, y1)
        a = (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55)
        b = (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)
        self.d.polygon(
            [(p[0] * self.s, p[1] * self.s) for p in (tip, a, b)], fill=color
        )


def _dashes(a: float, b: float, dash: float, gap: float):
    out, t = [], a
    while t < b:
        out.append((t, min(t + dash, b)))
        t += dash + gap
    return out


# ── 기하 (1x 픽셀) ──────────────────────────────────────────────────────────
HOST = (30, 74, 800, 812)
CAM = (62, 116, 268, 166)
VISION = (62, 196, 372, 288)
SINK = (62, 306, 372, 366)
CONT = (48, 396, 782, 700)
SHIM = (74, 446, 322, 520)
OFFB = (74, 548, 322, 632)
MAVROS = (466, 452, 756, 536)
DWS = (466, 560, 692, 622)
KERN = (48, 722, 782, 796)
WHY = (400, 196, 780, 366)
LEGEND = (940, 74, 1272, 430)
FCU = (940, 470, 1272, 812)


def draw(pen: Pen) -> None:
    # ── 제목 ────────────────────────────────────────────────────────────────
    pen.text((30, 24), "온보드 컴퓨팅 시스템 구성", INK, 24, bold=True)
    pen.text(
        (30, 52),
        "Raspberry Pi 5 ↔ Pixhawk 6C  —  컨테이너 경계를 넘는 통로는 ①②③ 세 개뿐이다",
        GRAY,
        13,
    )

    # ── 호스트 ──────────────────────────────────────────────────────────────
    pen.box(HOST, PURPLE, 2.6, 10)
    pen.text((48, 86), "Raspberry Pi 5  ·  Ubuntu 24.04 (호스트)", PURPLE, 16, bold=True)

    pen.box(CAM, GREEN, 2.0, 7)
    pen.text((76, 126), "CSI 카메라", GREEN, 14, bold=True)
    pen.text((76, 145), "libcamera / picamera2", GRAY, 11)

    pen.box(VISION, BLUE, 2.2, 7)
    pen.text((76, 208), "vision/main.py", BLUE, 15, bold=True, mono=True)
    pen.lines(
        [
            "picam-venv · Python 3.12 · ROS 없음",
            "ArUco 검출 → 기체기준 상대 pose (body FLU)",
        ],
        76,
        233,
        GRAY,
        11,
        lead=16,
    )

    pen.box(SINK, BLUE, 2.2, 7)
    pen.text((76, 316), "utils/target_sink.py", BLUE, 13, bold=True, mono=True)
    pen.text((76, 339), "TCP 서버  127.0.0.1:8091  ·  JSON Lines", GRAY, 11)

    # ── 왜 두 환경으로 갈라졌나 ─────────────────────────────────────────────
    pen.d.rounded_rectangle(
        [v * pen.s for v in WHY], radius=8 * pen.s, fill=WASH_GRAY
    )
    pen.box(WHY, GRAY, 1.6, 8)
    pen.text((418, 210), "왜 두 환경으로 갈라졌나", INK, 13, bold=True)
    pen.lines(
        [
            "· 카메라 스택(libcamera·picamera2)은 호스트의",
            "   Ubuntu 24.04 / Python 3.12 에서만 동작한다.",
            "· ROS 2 Humble 은 Ubuntu 22.04 / Python 3.10 을",
            "   요구한다 → 컨테이너로 분리.",
            "· 두 도메인은 코드 수준 import 도, colcon·pip 빌드",
            "   의존도 없다. 만나는 곳은 ① 소켓 하나뿐이다.",
        ],
        418,
        238,
        GRAY,
        10.5,
        lead=19,
    )

    # ── 컨테이너 (경계는 점선 = 격리가 아니라 구멍 뚫린 경계) ────────────────
    pen.d.rounded_rectangle(
        [v * pen.s for v in CONT],
        radius=10 * pen.s,
        fill=WASH_ORANGE,
    )
    pen.dbox(CONT, ORANGE, 2.4)
    pen.text((70, 408), "Docker 컨테이너  fc", ORANGE, 15, bold=True)
    pen.text((70, 429), "ros:humble  (Ubuntu 22.04 · Python 3.10)", GRAY, 11)

    pen.box(SHIM, BLUE, 2.2, 7, fill=BG)
    pen.text((88, 458), "vision/ros/shim_node.py", BLUE, 13, bold=True, mono=True)
    pen.lines(
        ["TCP 클라이언트 · 재접속 담당", "body FLU → ENU 절대 setpoint 변환"],
        88,
        481,
        GRAY,
        11,
        lead=16,
    )

    pen.box(OFFB, BLUE, 2.2, 7, fill=BG)
    pen.text((88, 560), "offboard_node", BLUE, 13, bold=True, mono=True)
    pen.text((88, 581), "telemetry_node  ·  mission_node", BLUE, 13, mono=True)
    pen.text((88, 606), "fc_ros — 상태기계 · L1 유도", GRAY, 11)

    pen.box(MAVROS, RED, 2.2, 7, fill=BG)
    pen.text((484, 464), "mavros_node", RED, 15, bold=True, mono=True)
    pen.lines(
        [
            "ROS 2 토픽 ↔ MAVLink 변환기",
            "fcu_url:=/dev/ttyACM0:57600",
        ],
        484,
        490,
        GRAY,
        11,
        lead=16,
    )

    pen.box(DWS, GRAY, 1.8, 7, fill=BG)
    pen.text((484, 572), "/drone_ws", INK, 13, bold=True, mono=True)
    pen.text((484, 595), "③ 호스트 ~/drone_ws 바인드 마운트", GRAY, 11)

    # ── 커널 (호스트·컨테이너 공유) ──────────────────────────────────────────
    pen.d.rounded_rectangle(
        [v * pen.s for v in KERN], radius=8 * pen.s, fill=WASH_GREEN
    )
    pen.box(KERN, GREEN, 2.2, 8)
    pen.text(
        (70, 732), "Linux 커널  —  호스트와 컨테이너가 공유한다", GREEN, 14, bold=True
    )
    pen.text(
        (70, 756),
        "cdc_acm 드라이버  →  /dev/ttyACM0   (문자 디바이스 char 166:0, root:dialout)",
        INK,
        12,
        mono=True,
    )
    pen.text(
        (70, 776),
        "by-id: usb-Auterion_PX4_FMU_v6C.x_0-if00        네트워크 네임스페이스도 이 커널 것 하나뿐",
        GRAY,
        10.5,
    )

    # ── Pixhawk ─────────────────────────────────────────────────────────────
    pen.box(FCU, PURPLE, 2.6, 10)
    pen.text((964, 504), "Pixhawk 6C", PURPLE, 20, bold=True)
    pen.text((964, 532), "PX4 커스텀 펌웨어", GRAY, 12)
    pen.lines(
        [
            "USB VID:PID  3185:0038  (Auterion PX4 FMU v6C.x)",
            "USB 디스크립터가 CDC-ACM 클래스를 선언한다",
            "  · If 0  Communications  (제어)",
            "  · If 1  CDC Data        (MAVLink 바이트스트림)",
            "",
            "EKF2 자세·위치 추정  ·  모터/서보 PWM 출력",
            "RC 수신기  ·  GPS  ·  IMU",
        ],
        964,
        572,
        INK,
        11.5,
        lead=19,
    )

    # ── 연결선 ──────────────────────────────────────────────────────────────
    cx = 165  # 좌측 컬럼 중심
    pen.arrow([(cx, CAM[3]), (cx, VISION[1])], GREEN, 2.0)
    pen.text((cx + 10, 174), "CSI-2 리본", GRAY, 10.5)

    pen.arrow([(cx, VISION[3]), (cx, SINK[1])], BLUE, 2.0)

    # ① 소켓 — 컨테이너 경계를 넘는다
    pen.arrow([(cx, SINK[3]), (cx, SHIM[1])], BLUE, 2.4)
    pen.text((cx + 12, 372), "①", ORANGE, 15, bold=True)
    pen.text(
        (cx + 30, 373),
        "--network host  →  컨테이너의 127.0.0.1 이 곧 호스트의 127.0.0.1",
        ORANGE,
        11,
    )

    # shim → offboard (ROS 2)
    pen.arrow([(cx + 33, SHIM[3]), (cx + 33, OFFB[1])], BLUE, 2.0)
    pen.text((cx + 45, 526), "ROS 2 토픽  /vision/landing_setpoint", GRAY, 10.5)

    # offboard ↔ mavros (ROS 2 DDS)
    pen.arrow(
        [(OFFB[2], 578), (394, 578), (394, 494), (MAVROS[0], 494)],
        RED,
        2.0,
        double=True,
    )
    pen.text((398, 543), "ROS 2 DDS", GRAY, 10.5)
    pen.text((398, 559), "토픽 · 서비스", GRAY, 10.5)

    # mavros → 커널 (② --device)
    pen.arrow([(726, MAVROS[3]), (726, KERN[1])], GREEN, 2.4, double=True)
    pen.text((464, 640), "②", ORANGE, 15, bold=True)
    pen.lines(
        [
            "--device /dev/ttyACM0",
            "커널 노드 패스스루 — 복제·포워딩이",
            "아니라 같은 객체를 가리키는 두 번째 이름",
        ],
        484,
        642,
        ORANGE,
        10.5,
        lead=15,
    )
    pen.arrow([(700, 658), (716, 658)], ORANGE, 1.6, head=5)

    # 커널 → Pixhawk (USB)
    pen.arrow([(KERN[2], 759), (FCU[0], 759)], PURPLE, 2.6, double=True)
    pen.text((812, 688), "USB-C 케이블", PURPLE, 12, bold=True)
    pen.lines(
        ["USB Full-Speed 12 Mbps", "CDC-ACM 가상 시리얼", "위에 MAVLink v2"],
        812,
        710,
        GRAY,
        10,
        lead=14,
    )

    # ── 범례 ────────────────────────────────────────────────────────────────
    pen.d.rounded_rectangle(
        [v * pen.s for v in LEGEND], radius=8 * pen.s, fill=WASH_GRAY
    )
    pen.box(LEGEND, GRAY, 1.8, 8)
    pen.text((960, 96), "컨테이너 경계를 넘는 통로 = 3개뿐", INK, 14, bold=True)
    pen.text((960, 120), "docker inspect fc 실측 (2026-07-30)", GRAY, 10.5)

    items = [
        (
            "①",
            "--network host",
            [
                "네트워크 네임스페이스를 아예 분리하지",
                "않는다. 컨테이너 전용 가상 네트워크",
                "(브리지·vLAN)는 존재하지 않는다.",
            ],
        ),
        (
            "②",
            "--device /dev/ttyACM0",
            [
                "커널 디바이스 노드를 같은 major:minor",
                "로 노출 + cgroup rwm 허용.",
                "Pixhawk 는 네트워크가 아니라 파일이다.",
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
    y = 152
    for mark, code, body in items:
        pen.text((962, y), mark, ORANGE, 15, bold=True)
        pen.text((986, y + 1), code, INK, 12, bold=True, mono=True)
        pen.lines(body, 986, y + 24, GRAY, 10.5, lead=15)
        y += 24 + 15 * len(body) + 14

    pen.text(
        (962, 400),
        "※ fcu_url 의 :57600 은 CDC-ACM 에선 무의미 (USB 속도로 전송)",
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
