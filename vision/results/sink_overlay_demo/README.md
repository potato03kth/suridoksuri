# 유도 발행 상태 오버레이 — 화면 버퍼 증거 (2026-07-28)

헤드리스 환경이라 창(`--display window`)을 띄울 수 없어, **저장된 프레임**으로
`utils/visualize.py::draw_sink_status`가 실제로 화면에 그려짐을 남긴다.
근거·설계는 `vision/CLAUDE.md`의 "bind 하드 페일 + 유도 발행 상태 오버레이" 절.

| 파일 | 상태 | 재현 명령 |
|---|---|---|
| `overlay_noconsumer.png` | 🔴 sink는 떴는데 **소비자 0명** — 빨강 + 큰 글씨 `CONSUMERS 0 - GUIDANCE GOES NOWHERE`. 사용자가 지목한 사각지대(화면은 뜨는데 유도 좌표는 허공) | `python -m vision.main <img> --display file --output overlay_noconsumer.png --target-sink --target-sink-port 18095` |
| `overlay_nosink.png` | 🟠 `--target-sink` 미지정 — 주황 `SINK OFF - NO GUIDANCE OUT`. 같은 "유도가 안 나간다"지만 운영자의 명시적 선택이라 색으로 구분 | `python -m vision.main <img> --display file --output overlay_nosink.png` |
| `replay_frame60.png` | 🟢 **소비자가 실제로 붙은** 재생의 60번째 프레임 — 초록 `CONSUMERS 1` + `sink 127.0.0.1:18097 seq 122 dropped 0`. **ArUco 검출 박스와 신뢰도 라벨이 그대로 살아 있다**(오버레이가 검출 그리기를 가리지 않는다는 증거) | `python -m vision.replay <rec> --preset vision/presets/vertiport_fine.yaml --target-sink --target-sink-port 18097 --display stream --stream-port 0 --output replay_overlay.mp4` + 별도 프로세스의 stdlib 소비자 |

입력 프레임(합성 ArUco 80장 + `telemetry.jsonl`)은 커밋 저장소 밖에서 임시 생성했다 —
`state_machine_demo/`·`distress_fine_demo/` 전례와 동일하게 결과물만 남긴다.
