"""
좌표 프레임 체인: 카메라 광학 프레임(cam) → 기체 body FLU → 기체 body FRD.

`docs/vision_fc_interface.md` §4.3의 명세를 그대로 코드로 옮긴 것이다(작업 V3). 이 파일은
**순수 회전 계산만** 한다 — `vision/CLAUDE.md` import 규칙("core/ ← numpy, opencv만 허용")과
`core/target.py`/`core/state_machine.py`가 세운 패턴을 그대로 따라 파일 I/O·로깅·wall-clock·
난수를 쓰지 않는다(같은 입력 → 항상 같은 출력, §7.5 결정론).

## 프레임 정의 (§4.3, 재논의 대상 아님)

```
① cam (카메라 광학 프레임, OpenCV 표준)
     X = 이미지 오른쪽, Y = 이미지 아래, Z = 렌즈 바깥(전방)
     `TargetEstimate.position`이 바로 이 프레임(`core/target.py` docstring "확정 전제")
② body FLU (baselink, ROS 표준)  X = 전방(기수), Y = 좌, Z = 상
③ body FRD (aircraft, PX4)       X = 전방,      Y = 우, Z = 하
```

단위는 전부 미터(SI), 각도는 라디안. 회전은 능동회전(active)이고 `p_B = R_{B←A} · p_A` 표기를
쓴다.

## ⚠️ 이 파일이 하지 않는 것 — ENU/NED 변환 (의도적 미구현)

`docs/vision_fc_interface.md` §4.4의 정식 권고는 **"vision은 body FLU/FRD 상대 벡터까지만 낸다.
NED 변환은 하지 않는다"** 이다. 이유는 body 프레임 경로가 기체 자세 구독과 시간 동기를 통째로
없애주기 때문이다(기울기 보정은 PX4가 자기 IMU 자세로 지연 없이 한다). ENU/NED로 변환해 넘기려면
vision이 기체 자세를 읽어 곱해야 하고 그 순간 §4.2(자세 오차)와 §4.6(시간 동기) 문제가 vision
쪽으로 이관된다. 그래서 §4.3 왕복검증 표의 RT-7(ENU 경로)은 **구현하지 않았다** — 안 쓰는 경로를
만들어두면 누군가 쓰게 된다.

🔴 **이 저장소 안의 `pos_ned`/`vel_ned`를 참조하지 말 것.** `fc_bridge` 쪽 내부 표현은
`pos_ned = [N, E, h_up]`(3번째가 **위** 양수)인데 `vel_ned = [vN, vE, vD]`(3번째가 **아래** 양수)
라 **같은 `_ned` 접미사가 위치와 속도에서 반대 부호 규약**을 쓴다(§4.2). vision이 이 배열을
그대로 받아 쓰면 고도 부호를 반드시 틀린다. vision 계약은 이 내부 표현을 참조하지 않는다.

## ⚠️ 물리 가정과 그 리스크

1. **나디르 하드마운트 가정(§4.2/§4.3).** `R_{frd←cam}`을 **상수 회전**으로 두는 것 자체가
   "카메라가 기체에 강체(rigid)로 붙어 있고, 마운트 회전이 시간에 따라 변하지 않는다"는 가정이다.
2. 🔴 **고무 마운트의 역설(§4.5).** 진동댐핑 마운트는 정확히 그 강체 가정을 깬다 — 카메라가
   IMU와 물리적으로 분리돼 **IMU 자세 ≠ 실제 카메라 자세** 잔차가 남는다. 이 파일의 상수 회전은
   그 잔차를 **없애지 못한다**. 정량 영향은 `지상오차 = 고도 × tan(θ_잔차)`:
   `40m·5° → 3.50m`, `40m·2° → 1.40m`, `40m·1° → 0.70m`, `3m·1° → 0.052m`.
   **고도가 낮아질수록 선형으로 줄어든다** — 즉 이 리스크는 "40m에서 정확히 찍는 것"을 막지
   "3m에서 폐루프로 수렴하는 것"은 막지 않는다. 대응은 회피가 아니라 **게이팅**이다(고고도
   추정치로 최종 커밋하지 않는다 — `core/wire.py`의 `closed_loop_floor_agl_m()` 계약과 상태머신
   커밋 게이트가 이미 그 역할). 실제 잔차 각도는 **아무도 측정한 적 없다**(§7 U4).
3. 🔴 **마운트 요각 ψ_m 은 미측정이다(§7 U3).** 아래 `MOUNT_YAW_PSI_M_RAD_DEFAULT` 참조.
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 마운트 요각 ψ_m — 🔴 미측정 (§7 U3)
# ──────────────────────────────────────────────────────────────────────────────
# "기수 기준으로 카메라를 몇 도 돌려 달았는가". `docs/vision_fc_interface.md` §7 U3가
# **저장소 어디에도 이 값이 기록돼 있지 않다**고 확정했고, 물리 측정(기수 방향에 표식을 두고
# 1프레임 촬영)이 필요한 값이라 이 세션에서는 확인할 수 없었다.
#
# 그래서 **값을 지어내지 않고 설정 파라미터로 빼두고 공칭값 0.0만 기본값으로 준다.**
# 아래 `MOUNT_YAW_PSI_M_MEASURED = False`가 "이건 실측값이 아니다"를 코드로 들고 다니며,
# 이 플래그는 와이어 레코드(`core/wire.py`)에도 그대로 실려 소비자와 로그까지 전파된다.
#
# ψ_m = 0 의 뜻: 이미지의 위쪽(-Y_cam 방향)이 기수 전방을 가리키도록 장착된 경우.
#
# ⚠️ ψ_m 이 틀리면 **착륙 오프셋의 방향 자체가 틀린다** — "정확히 90° 틀린 방향으로 정확하게"
# 이동한다. 90°/180° 오차는 실비행에서 아주 알아보기 쉬운 증상(수정할수록 멀어짐)이니 첫
# 폐루프 시험의 1번 체크항목이다(§4.3).
MOUNT_YAW_PSI_M_RAD_DEFAULT = 0.0
MOUNT_YAW_PSI_M_MEASURED = False

# R_{frd←cam}(ψ_m = 0) — §4.3 명세 그대로. Z축 +90° 회전과 같다. det = +1.
_R_FRD_CAM_0 = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# FLU ↔ FRD : (x, -y, -z). mavros `ftf::transform_frame_baselink_aircraft`와 같은 것.
# diag(1,-1,-1) = Rx(180°) 이라 det = +1 인 진짜 회전이고, 자기 자신이 역행렬이다(involution).
_R_FLU_FRD = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _rz(psi: float) -> np.ndarray:
    """Z축 둘레 능동회전 Rz(ψ)."""
    c, s = np.cos(float(psi)), np.sin(float(psi))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def R_frd_cam(psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """`R_{frd←cam}(ψ_m) = Rz(ψ_m) · R_{frd←cam}(0)` (§4.3)."""
    return _rz(psi_m) @ _R_FRD_CAM_0


def R_flu_cam(psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """`R_{flu←cam}(ψ_m)`.

    FRD 쪽을 1차 정의로 두고 `R_{flu←frd}`를 곱해 **유도**한다 — 두 행렬을 각각 손으로 적어두면
    한쪽만 고쳤을 때 조용히 어긋난다. 이렇게 두면 FLU/FRD 일관성이 구조적으로 보장된다.
    (ψ_m=0에서 §4.3이 적어둔 `[[0,-1,0],[-1,0,0],[0,0,-1]]`과 같은 값이 나오는지는
    `tests/test_frames.py`가 문서 리터럴과 직접 대조해 회귀로 박아 뒀다.)
    """
    return _R_FLU_FRD @ R_frd_cam(psi_m)


def _as_vec3(p) -> np.ndarray:
    v = np.asarray(p, dtype=np.float64).reshape(-1)
    if v.size != 3:
        raise ValueError(f"3차원 벡터가 필요하다 — 받은 shape={np.asarray(p).shape}")
    return v


def cam_to_frd(p_cam, psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """카메라 광학 프레임 상대벡터 → 기체 body FRD 상대벡터 (미터)."""
    return R_frd_cam(psi_m) @ _as_vec3(p_cam)


def frd_to_cam(p_frd, psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """`cam_to_frd`의 역변환. 회전행렬이므로 역행렬 = 전치."""
    return R_frd_cam(psi_m).T @ _as_vec3(p_frd)


def cam_to_flu(p_cam, psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """카메라 광학 프레임 상대벡터 → 기체 body FLU 상대벡터 (미터)."""
    return R_flu_cam(psi_m) @ _as_vec3(p_cam)


def flu_to_cam(p_flu, psi_m: float = MOUNT_YAW_PSI_M_RAD_DEFAULT) -> np.ndarray:
    """`cam_to_flu`의 역변환."""
    return R_flu_cam(psi_m).T @ _as_vec3(p_flu)


def flu_to_frd(p_flu) -> np.ndarray:
    """body FLU → body FRD: `(x, -y, -z)`. ψ_m 무관(마운트가 아니라 프레임 규약 차이)."""
    return _R_FLU_FRD @ _as_vec3(p_flu)


def frd_to_flu(p_frd) -> np.ndarray:
    """body FRD → body FLU. `_R_FLU_FRD`가 involution이라 같은 행렬을 곱한다."""
    return _R_FLU_FRD @ _as_vec3(p_frd)


# ──────────────────────────────────────────────────────────────────────────────
# 쿼터니언 성분 순서 어댑터
# ──────────────────────────────────────────────────────────────────────────────
# 🔴 **두 규약이 이 저장소 안에서 파일 하나 사이로 다르다**(§4.2):
#   - `TargetEstimate.orientation` = **(x, y, z, w)**  (`core/target.py`) — ROS
#     `geometry_msgs/Quaternion`도 같다.
#   - `fc_bridge/utils/rotation.py::quat_to_euler_xyz(w, x, y, z)` = **w가 먼저**.
# 순서를 암묵적으로 넘기면 조용히 틀린 자세가 나온다(노름이 1이라 sanity check에도 안 걸린다).
# 그래서 어댑터를 **명시적 함수로** 두고 왕복검증(RT-6)을 건다.
#
# ⚠️ vision은 `fc_bridge`를 import하지 않는다(도메인 격리 — 루트 `CLAUDE.md` "도메인 간 교차
# import가 없다"). 여기 있는 건 순수 재배열일 뿐이고, 실제 `quat_to_euler_xyz` 호출은 이 값을
# 받는 쪽(Phase 2 shim / FC 도메인)의 책임이다.


def quat_xyzw_to_wxyz(q) -> tuple[float, float, float, float]:
    """(x, y, z, w) → (w, x, y, z). `fc_bridge.utils.rotation` 계열에 넘기기 직전에 쓴다."""
    x, y, z, w = (float(v) for v in _as_quat4(q))
    return (w, x, y, z)


def quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    """(w, x, y, z) → (x, y, z, w). `TargetEstimate.orientation`/ROS 규약으로 되돌린다."""
    w, x, y, z = (float(v) for v in _as_quat4(q))
    return (x, y, z, w)


def _as_quat4(q) -> np.ndarray:
    v = np.asarray(q, dtype=np.float64).reshape(-1)
    if v.size != 4:
        raise ValueError(f"4성분 쿼터니언이 필요하다 — 받은 크기={v.size}")
    return v
