# Offboard 스테이지 3에 필요한 사안

> 가속도 세트포인트 이상의 제어 레이어로 전환 시 반영해야 할 설계 요구사항  
> 배경: 현재 스테이지(속도 세트포인트)는 PX4 내부 루프가 에너지 결합을 대신 처리함

---

## 제어 레이어별 책임 변화

| 스테이지 | 제어 레이어 | 에너지 결합 처리 주체 |
|---|---|---|
| 1–2 (현재) | 속도 세트포인트 (`vx, vy, vz`) | PX4 내부 루프 |
| 3 | 가속도 세트포인트 (`ax, ay, az`) | PX4 일부 처리 / 일부 직접 구현 필요 |
| 4 | 추력 + 자세 직접 명령 | 전부 직접 구현 |

---

## 핵심 문제: L1–TECS 에너지 결합

뱅크각을 취하면 양력의 수직 성분이 감소한다.

```
수직 양력 = L · cos(φ)   →   φ 증가 시 고도 침하 발생
```

TECS는 이 감소분을 부하 계수(load factor)로 보정한다.

### 결합 신호 흐름

```
L1 guidance
  → 횡방향 가속도 a_lat
  → 뱅크각  φ = arctan(a_lat / g)
  → 부하 계수 n = 1 / cos(φ)        ← TECS로 전달

TECS
  → 에너지 기준 × n 으로 상향
  → 피치 / 스로틀 명령 생성

합산 → 3축 가속도 또는 추력 + 자세 명령
```

### 수직축 피드포워드 보정량

```
Δa_z = g · (1 / cos(φ) - 1)
```

이 값을 수직 가속도 명령에 더하지 않으면 뱅크 구간에서 고도 침하가 발생한다.

---

## 스테이지 3 전환 시 구현 항목

### 1. L1Guidance 인터페이스 변경

- 현재: 3D 속도 명령 반환 (`ned_velocity_cmd`)
- 필요: 횡방향 가속도 `a_lat` 반환으로 변경 또는 추가
- 위치: `fc_bridge/guidance/l1_guidance.py`

### 2. 부하 계수 보정 모듈 추가

```python
def load_factor_correction(a_lat: float, g: float = 9.81) -> float:
    phi = np.arctan2(a_lat, g)
    return 1.0 / max(np.cos(phi), 0.1)   # 0으로 나누기 방지
```

### 3. TECS 스타일 에너지 관리 루프

- 총 에너지 기준: `E_ref = ½mv_ref² + mgh_ref`
- 스로틀 채널: 총 에너지 오차 제어
- 피치 채널: 에너지 분배 (속도 ↔ 고도) 제어
- 부하 계수 `n`을 에너지 기준에 피드포워드

### 4. gamma_profile 대체

- 현재: 경로점별 상승각을 정적으로 미리 정의
- 스테이지 3: TECS 출력이 수직 채널을 실시간으로 계산
- 현재 `gamma_profile` 파라미터 자리를 TECS 출력으로 교체하면 됨
- 위치: `fc_bridge/execution/offboard_follower.py:56–57`, `:240–242`

---

## 현재 코드 설계와의 연속성

현재 `OffboardFollower`에서 수직 채널(`gamma_profile`)을 수평(`path_pts`)과 분리한 구조는  
나중에 TECS 출력으로 교체할 수 있도록 인터페이스가 이미 분리되어 있어 재설계 부담이 낮다.

---

## 참고 문헌

| 항목 | 출처 |
|---|---|
| L1 Guidance Law | Park, S., Deyst, J., How, J.P. (2004). AIAA 2004-4900 |
| TECS 원론 | Lambregts, A.A. (1983). AIAA Paper 83-2561 |
| PX4 구현 | `src/lib/l1/`, `src/lib/tecs/` |
| 교재 | Stevens, Lewis, Johnson — *Aircraft Control and Simulation*, Wiley 3rd ed. |
