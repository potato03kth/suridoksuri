---
doc_type: procedure
project: suridoksuri-1
scope: "실기체(RPi5 doksuri) 코드 배포 — git push → pull → colcon build → 검증"
last_updated: 2026-07-25
---

# 실기체 배포 절차

> **규칙:** 코드를 고쳤으면 **바로** 여기까지 한다. 유일한 예외는 사용자가
> **"현재 비행중"**이라고 말한 경우. (프로젝트 루트 `CLAUDE.md` §공통 규칙)
>
> 이유: 2026-07-25에 offboard 좌표계 수정을 커밋만 하고 배포를 안 해서
> "고쳤다"와 "기체가 고쳐졌다" 사이에 간극이 생겼다. 과거에도 **stale colcon
> build**가 실비행 8건의 근본원인이었다(커밋 `4dc30f9`).

---

## 0. 환경 사실 (외우지 말고 여기 보고 하기)

| 항목 | 값 |
|---|---|
| 원격 | `suri@100.67.27.83` (tailscale, 호스트명 `doksuri`) |
| 호스트 저장소 | `~/drone_ws/src/suridoksuri` |
| 컨테이너 | `fc` (docker), 마운트 `/home/suri/drone_ws → /drone_ws` |
| 컨테이너 내 저장소 | `/drone_ws/src/suridoksuri` (= 호스트와 같은 실체) |
| 브랜치 | `dev--vision-computing-module` |
| sudo | 무암호 가능 |

**마운트가 같은 실체이므로 호스트에서 `git pull`하면 컨테이너 소스도 같이 바뀐다.**

---

## 1. 무엇이 언제 반영되는가 (중요)

| 패키지 | 반영 시점 | 이유 |
|---|---|---|
| `fc_bridge` | **`git pull`만으로 즉시** | colcon install 사본이 import에 안 쓰인다(§4 참조) — 소스에서 직접 로드된다. **build 대상에 넣을 필요 없음** |
| `fc_ros` | **`colcon build` 필요** | `install/fc_ros/.../site-packages/fc_ros/`의 **복사본**이 로드된다 |
| `*.launch.py`, `params/*.yaml` | **`colcon build` 필요** | `install/fc_ros/share/fc_ros/`로 복사된다 |

즉 `fc_ros`나 yaml을 건드렸으면 build를 빼먹으면 안 된다.

---

## 2. 절차

```bash
# ── (1) 개발컴: push ──────────────────────────────────────────
git push origin HEAD:dev--vision-computing-module

# ── (2) RPi: pull ────────────────────────────────────────────
# 미커밋 로그 폴더가 pull을 막으면 지우지 말고 백업 (§3)
ssh suri@100.67.27.83 'cd ~/drone_ws/src/suridoksuri && git fetch origin && git pull --ff-only'

# ── (3) RPi: colcon build (컨테이너 안) ──────────────────────
ssh suri@100.67.27.83 'docker exec fc bash -lc "
  cd /drone_ws && source /opt/ros/humble/setup.bash &&
  colcon build --packages-select fc_ros"'

# ── (4) 검증 — 반드시 한다 (§5) ──────────────────────────────
```

---

## 3. `git pull`이 막힐 때

RPi에는 `record_flight.sh`가 만든 로그 폴더가 미커밋 상태로 남아 있다. 개발컴에서
같은 경로를 커밋하면 pull이 거부된다(`untracked working tree files would be overwritten`).

**지우지 말고 백업한다** — 일부는 root 소유라 `sudo`가 필요하다.

```bash
ssh suri@100.67.27.83 '
cd ~/drone_ws/src/suridoksuri
BK=~/drone_ws/_pull_backup_$(date +%Y%m%d_%H%M%S); mkdir -p "$BK/logs"
for d in $(git status --porcelain | awk "/^\?\? logs\//{print \$2}"); do
  sudo mv "$d" "$BK/logs/" && echo "백업: $d"
done
echo "백업 위치: $BK"
git pull --ff-only'
```

pull 성공 후 백업 폴더의 내용이 커밋본과 같은지 확인했으면 지워도 된다.

---

## 4. ⚠️ 절대 하지 말 것

### `--symlink-install` 금지

기존 배포는 **복사본 설치**다. `--symlink-install`로 빌드하면
`install/fc_ros/lib/python3.10/site-packages/fc_ros/` 실제 복사본이 사라지고
`.egg-link`로 대체돼 레이아웃이 통째로 바뀐다 (2026-07-25에 실제로 겪음).

되돌리려면 해당 패키지의 build/install을 지우고 다시 빌드해야 한다:

```bash
docker exec fc bash -lc '
  cd /drone_ws && rm -rf build/fc_ros install/fc_ros &&
  source /opt/ros/humble/setup.bash && colcon build --packages-select fc_ros'
```

### `fc_bridge`는 colcon install 사본으로 import되지 않는다

`fc_bridge/setup.py`가 `fc_bridge/` 디렉터리 안에서 `find_packages()`를 돌리기 때문에
하위 패키지가 **최상위로 평평하게** 깔린다 — `site-packages/execution/state_logic.py`이지
`site-packages/fc_bridge/execution/state_logic.py`가 아니다. 그래서
`import fc_bridge`는 colcon install 경로로는 **절대 안 된다**.

실제 메커니즘은 **운용자가 컨테이너 셸에서 손으로 넣는 `PYTHONPATH`** 다 —
`docs/mc_flight_procedure.md:50, 83`에 절차로 적혀 있다:

```bash
export PYTHONPATH=/drone_ws/src/suridoksuri:$PYTHONPATH
```

`docs/session_status.md:109`도 같은 사실을 기록한다: "fc_ros는 colcon 빌드,
fc_bridge+vtol_sim은 `PYTHONPATH=/drone_ws/src/suridoksuri`".

**cwd로는 해결되지 않는다** (2026-07-25 실측). console_scripts 엔트리 스크립트는
`sys.path[0]`이 **스크립트 디렉터리**이지 cwd가 아니므로, 저장소 루트에서 실행해도
`import fc_bridge`는 실패한다. 컨테이너에 `.pth` 파일도 없다:

```
$ cd /drone_ws/src/suridoksuri && python3 -c "import sys;
  sys.path=[p for p in sys.path if p not in ('','.', '/drone_ws/src/suridoksuri')];
  import fc_bridge"
ModuleNotFoundError: No module named 'fc_bridge'
```

> ⚠️ **이 export를 빠뜨리면 `offboard_node`가 import 단계에서 죽는다.** 강제하는
> 장치가 없고 운용자 기억에만 의존한다 — launch 래퍼나 `.pth`로 고정하는 게 맞다(미조치).

> 같은 함정을 노트북 SITL 환경에서도 겪었고 거기선 `.pth` 파일로 해결했다 —
> `docs/wsl_dev_env_setup.md` 섹션 F, 메모리 `project_fc_sitl_laptop_env`.

---

## 5. 검증 (배포했다고 말하기 전에)

```bash
ssh suri@100.67.27.83 'docker exec fc bash -lc "
  source /opt/ros/humble/setup.bash; source /drone_ws/install/setup.bash
  cd /drone_ws/src/suridoksuri
  echo \"--- 소스 vs install 사본 md5 (같아야 함) ---\"
  md5sum fc_ros/fc_ros/nodes/offboard_node.py \
         /drone_ws/install/fc_ros/lib/python3.10/site-packages/fc_ros/nodes/offboard_node.py
  echo \"--- import 검증 (노드 인스턴스화 안 함 = 기체 무동작) ---\"
  python3 -c \"import fc_ros.nodes.offboard_node as m, fc_bridge, os;
print(m.__file__); print(os.path.dirname(fc_bridge.__file__))\"
"'
```

**두 md5가 같아야 하고**, import가 통과해야 한다. 여기까지 확인해야 "배포 완료"다.

> ⚠️ 검증에 `ros2 run fc_ros offboard_node`나 launch를 쓰지 말 것 — 노드가 뜨는 순간
> **ARM 명령을 보낸다.** 모듈 import까지만 하면 노드가 인스턴스화되지 않아 안전하다.

---

## 6. 새 파라미터를 추가했을 때

`fc_ros_params.yaml`에 넣고, launch 인자로도 주고 싶으면
`fc_ros/launch/phase2.launch.py`에 `DeclareLaunchArgument` + `overrides` 두 곳을 같이 고친다.
빈 문자열이면 YAML 값을 쓰도록 하는 게 이 저장소 관례다(기본값 이중 관리 금지).

배포 후 반영 확인:

```bash
ssh suri@100.67.27.83 'docker exec fc bash -lc \
  "grep -n 새파라미터명 /drone_ws/install/fc_ros/share/fc_ros/params/fc_ros_params.yaml"'
```
