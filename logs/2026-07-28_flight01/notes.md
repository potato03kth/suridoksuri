# 2026-07-28_flight01

- **비행 조건:** (기체/모드/launch 인자: vehicle_type:=vtol   transition_alt:=20.0   waypoints:=[0.0,0.0,20.0, -60.0,-20.0,20.0])
- **관찰:** **비행 없음.** launch가 뜨자마자 죽었다 — `launch.log` 전체가 한 줄이다:

  ```
  malformed launch argument ' ', expected format '<name>:=<value>'
  ```

  rosbag은 118KB(녹화 10여 초, arm 전)만 남았고 ulog도 없다.
- **결론:** 인자 구분 공백이 **U+00A0(non-breaking space)** 여서 쉘이 단어 분리를 못 했고,
  ROS launch가 `:=`가 없는 토큰을 거부한 것. 같은 원인이 flight02에서는 **에러 없이 조용히**
  `transition_alt`를 삼켜 천이 고도가 YAML 기본값 50m로 실행됐다 —
  근거·물증은 `logs/2026-07-28_flight02/notes.md` ①.
