# 개발 환경 정리

## 목표 플랫폼

- **하드웨어**: 라즈베리파이4
- **OS**: Raspberry Pi OS (Linux, ARM)
- **언어**: Python 3
- **주요 라이브러리**: OpenCV (`opencv-python-headless`)

## 개발 환경

- **개발 머신**: Windows
- **운영 머신**: 라즈베리파이4 (Linux, 헤드리스 SSH 운용 가능)

## OpenCV 설치

```bash
# 헤드리스 환경 (SSH, GUI 없음) 권장
pip install opencv-python-headless

# 또는 apt
sudo apt install python3-opencv
```

## 주의사항

| 항목 | 내용 |
|---|---|
| `imshow()` | 헤드리스 환경에서 사용 불가 — 파일 저장 또는 스트림으로 대체 |
| 비디오 코덱 | `libx264` 등 별도 설치 필요. 없으면 조용히 실패함 |
| 카메라 인덱스 | Windows와 번호 다를 수 있음 (`/dev/video0` 등) |
| 성능 | pip 설치본은 ARM NEON 최적화 미포함 — 최대 성능 필요 시 소스 빌드 권장 |
| 한글 렌더링 | OpenCV 기본 폰트는 ASCII만 지원 — PIL/Pillow로 우회 |

## 성능 기대치

- 단순 파이프라인 (캡처 → resize → 추론): **15~30fps** 수준

## C++ 포팅

Python OpenCV API와 C++ API가 1:1 대응되므로 포팅 난이도 낮음.
알고리즘 완성 후 성능 부족 시 C++ 포팅 고려.
