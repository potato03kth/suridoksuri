#!/usr/bin/env python3
"""보고서용 팀 조직도(역할 분담표) 이미지 생성기.

원본 스크린샷(1061x645)의 선·박스·텍스트 위치를 픽셀 단위로 실측해 재현한다.
좌표는 전부 원본 1x 픽셀 공간으로 적고, 출력 시 SCALE 을 곱해 고해상도로 뽑는다.

사용:
    python tools/report/build_org_chart.py [출력경로] [--scale N]

기본 출력: docs/figures/org_chart.png (3x, 3183x1935) + org_chart_1x.png (검증용)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── 폰트 ────────────────────────────────────────────────────────────────────
# 원본은 맑은 고딕 계열. 리눅스에서 자간·글자폭이 가장 근접한 것이 나눔고딕 17px
# (실측 잉크폭 오차 합계 9px / 9개 문자열 — Noto CJK·나눔바른고딕보다 정확).
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_PX = 17

FG = (0, 0, 0)
BG = (255, 255, 255)

# ── 기하 (원본 1x 픽셀 실측값) ───────────────────────────────────────────────
CANVAS = (1061, 645)

TOP_Y0, TOP_DIV, TOP_Y1 = 26, 69.5, 218  # 최상단 박스 3개
R1_Y0, R1_DIV, R1_Y1 = 276, 321, 380  # 1단: 파트명 / 책임자
R2_Y0, R2_DIV, R2_Y1 = 439, 483.5, 631  # 2단: 역할 및 담당 업무
BUS_Y = 247  # 소속 박스에서 내려와 5갈래로 갈라지는 가로 버스

TOP_BOXES = [(12, 160), (233, 381), (455, 781)]
COLS = [(12, 160), (233, 381), (455, 603), (677, 825), (898, 1046)]

LINE_W = 1.4  # 1x 기준 선 두께 (원본은 1~2px 안티에일리어싱)

# ── 내용 ────────────────────────────────────────────────────────────────────
TOP_CELLS = [
    ("소속", ["연세대학교"]),
    ("참여 연구자", ["팀장 김승원 등 2명"]),
    (
        "역할 및 담당 업무",
        [
            "·시스템통합: 체계종합",
            "·지상통제: 비행 경로 지정 및 모니터링",
            "·기체설계: 기체 구조 설계 및 정비",
            "·임무제어: 자율 비행 및 AI 소프트웨어",
            "·조종사: 수동 이착륙 및 비상 조종",
        ],
    ),
]

R1_CELLS = [
    ("시스템 통합", "책임자 김태훈"),
    ("기체 설계", "책임자 김승원"),
    ("임무 제어", "책임자 김태훈"),
    ("지상통제", "책임자 김승효"),
    ("조종사", "책임자 김승원"),
]

R2_HEAD = "역할 및 담당 업무"
# 줄바꿈은 원본 그대로 하드코딩한다 (자동 워드랩이 아니라 실측 줄나눔 재현).
R2_BODIES = [
    [". 대회 규정집 분석", "기반의 개발 일정", "관리, 시스템 통", "합(HW-SW) 조율"],
    [
        ". 대회 규정에 맞춘",
        "무인항공기 구조",
        "설계(CAD), 하드",
        "웨어(FC, 모터, 배",
        "터리 등) 조립",
        "및 기체 정비",
    ],
    [
        ". Flight Controller",
        "(FC) 파라미터 튜",
        "닝 및 컴퓨터 비전",
        "(AI) 기반 물체 인",
        "식·임무 자동화",
        "알고리즘 개발",
    ],
    [
        ". 지상 통제 소프트",
        "웨어(GCS)를 활용",
        "한 실시간 비행 경",
        "로(Waypoint) 설정",
        "및 기체 상태(텔레",
        "메트리) 모니터링",
    ],
    [
        ". 기체 수동 이착륙",
        "제어 및 비상 상황",
        "시 비상 착륙/수동",
        "전환을 통한 안전",
        "확보 (세이프티 비",
        "행)",
    ],
]

PITCH_TOP = 21.0  # 최상단 불릿 목록 줄간격 (실측)
PITCH_R2 = 21.4  # 2단 본문 줄간격 (실측)
BODY_PAD_L = 4  # 좌측 정렬 본문의 박스 내부 여백
TEXT_DY = 1.6  # 셀 세로 중앙 대비 잉크 중심 보정 (원본 실측)
TEXT_DY_R2 = 1.0  # 2단 본문만 보정량이 다르다 (여러 줄 블록이라 원본 여백이 달랐다)


class Renderer:
    def __init__(self, scale: float):
        self.s = scale
        self.img = Image.new("RGB", (round(CANVAS[0] * scale), round(CANVAS[1] * scale)), BG)
        self.d = ImageDraw.Draw(self.img)
        self.font = ImageFont.truetype(FONT_PATH, round(FONT_PX * scale))

    # ── 선 ──
    def _w(self) -> int:
        return max(1, round(LINE_W * self.s))

    def hline(self, x0: float, x1: float, y: float) -> None:
        s, w = self.s, self._w()
        yy = round(y * s) - w // 2
        self.d.rectangle([round(x0 * s), yy, round(x1 * s) - 1, yy + w - 1], fill=FG)

    def vline(self, x: float, y0: float, y1: float) -> None:
        s, w = self.s, self._w()
        xx = round(x * s) - w // 2
        self.d.rectangle([xx, round(y0 * s), xx + w - 1, round(y1 * s) - 1], fill=FG)

    def box(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self.hline(x0, x1, y0)
        self.hline(x0, x1, y1)
        self.vline(x0, y0, y1)
        self.vline(x1, y0, y1)

    # ── 글자 ──
    def _block_ink(self, lines: list[str], pitch: float) -> tuple[float, float, float, float]:
        """원점(0,0)에 그렸을 때 블록 전체 잉크 bbox (1x 좌표계).

        `font.getbbox()` 는 안티에일리어싱 여유를 포함해 실제 잉크보다 1~2px 크다.
        원본을 임계값 잉크로 실측했으므로 여기서도 같은 방식(프로브 렌더 후 임계)으로 잰다.
        """
        s = self.s
        pad = round(40 * s)
        w = round(max(self.font.getbbox(ln)[2] for ln in lines)) + 2 * pad
        h = round(self.font.getbbox(lines[0])[3] + (len(lines) - 1) * pitch * s) + 2 * pad
        probe = Image.new("L", (w, h), 255)
        pd = ImageDraw.Draw(probe)
        for i, ln in enumerate(lines):
            pd.text((pad, pad + round(i * pitch * s)), ln, font=self.font, fill=0)
        bb = probe.point(lambda v: 255 if v < 160 else 0).getbbox()
        return (
            (bb[0] - pad) / s,
            (bb[1] - pad) / s,
            (bb[2] - 1 - pad) / s,
            (bb[3] - 1 - pad) / s,
        )

    def text_block(
        self,
        lines: list[str],
        *,
        cy: float,
        pitch: float = PITCH_R2,
        cx: float | None = None,
        left: float | None = None,
    ) -> None:
        """잉크 bbox 기준으로 배치한다 — 원본이 잉크 위치로 실측됐기 때문."""
        s = self.s
        ix0, iy0, ix1, iy1 = self._block_ink(lines, pitch)
        ox = (cx - (ix0 + ix1) / 2) if cx is not None else (left - ix0)
        oy = cy - (iy0 + iy1) / 2
        for i, ln in enumerate(lines):
            self.d.text(
                (round(ox * s), round((oy + i * pitch) * s)), ln, font=self.font, fill=FG
            )


def build(scale: float) -> Image.Image:
    r = Renderer(scale)

    # 최상단 3박스 + 사이 연결선
    for (x0, x1), (head, body) in zip(TOP_BOXES, TOP_CELLS):
        r.box(x0, x1, TOP_Y0, TOP_Y1)
        r.hline(x0, x1, TOP_DIV)
        r.text_block([head], cx=(x0 + x1) / 2, cy=(TOP_Y0 + TOP_DIV) / 2 - TEXT_DY)
    body_cy = (TOP_DIV + TOP_Y1) / 2 - TEXT_DY
    r.text_block(TOP_CELLS[0][1], cx=(TOP_BOXES[0][0] + TOP_BOXES[0][1]) / 2, cy=body_cy)
    r.text_block(TOP_CELLS[1][1], cx=(TOP_BOXES[1][0] + TOP_BOXES[1][1]) / 2, cy=body_cy)
    r.text_block(TOP_CELLS[2][1], left=TOP_BOXES[2][0] + 5, cy=body_cy, pitch=PITCH_TOP)

    r.hline(TOP_BOXES[0][1], TOP_BOXES[1][0], body_cy + 1)
    r.hline(TOP_BOXES[1][1], TOP_BOXES[2][0], body_cy + 1)

    # 소속 박스 → 가로 버스 → 5갈래
    centers = [(x0 + x1) / 2 for x0, x1 in COLS]
    r.vline(centers[0], TOP_Y1, BUS_Y)
    r.hline(centers[0], centers[-1], BUS_Y)
    for cx in centers:
        r.vline(cx, BUS_Y, R1_Y0)

    # 1단: 파트명 / 책임자
    for (x0, x1), (head, who) in zip(COLS, R1_CELLS):
        r.box(x0, x1, R1_Y0, R1_Y1)
        r.hline(x0, x1, R1_DIV)
        cx = (x0 + x1) / 2
        r.text_block([head], cx=cx, cy=(R1_Y0 + R1_DIV) / 2 - TEXT_DY)
        r.text_block([who], cx=cx, cy=(R1_DIV + R1_Y1) / 2 - TEXT_DY)

    # 1단 → 2단 연결선
    for cx in centers:
        r.vline(cx, R1_Y1, R2_Y0)

    # 2단: 역할 및 담당 업무
    for (x0, x1), body in zip(COLS, R2_BODIES):
        r.box(x0, x1, R2_Y0, R2_Y1)
        r.hline(x0, x1, R2_DIV)
        r.text_block([R2_HEAD], cx=(x0 + x1) / 2, cy=(R2_Y0 + R2_DIV) / 2 - TEXT_DY)
        r.text_block(
            body, left=x0 + BODY_PAD_L, cy=(R2_DIV + R2_Y1) / 2 - TEXT_DY_R2, pitch=PITCH_R2
        )

    return r.img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="docs/figures/org_chart.png")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--also-1x", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    img = build(args.scale)
    img.save(out)
    print(f"{out}  {img.size[0]}x{img.size[1]}  (scale={args.scale})")

    if args.also_1x and args.scale != 1.0:
        one = out.with_name(out.stem + "_1x" + out.suffix)
        img1 = build(1.0)
        img1.save(one)
        print(f"{one}  {img1.size[0]}x{img1.size[1]}  (원본 대조용)")


if __name__ == "__main__":
    main()
