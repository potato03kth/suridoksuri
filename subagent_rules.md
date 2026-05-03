# Subagent 트리거 규칙

기본값은 직접 작업. Subagent는 컨텍스트 보호 또는 병렬 처리가 필요할 때만 사용.

## 규칙표

| Subagent / Skill | 트리거 조건 |
|---|---|
| **Explore** | "어디에 있지?" 질문 + Grep 1~2번으로 못 찾을 것 같을 때. 명확한 패턴이면 직접 Grep. |
| **Plan** | 새 모듈 설계, 3개 이상 파일 수정 예상, "어떻게 접근할까?" 질문, FC/CC 아키텍처 결정. |
| **general-purpose** | 웹 검색(논문/라이브러리), 멀티홉 조사, 오픈엔드 리서치. 코드베이스 내부 탐색엔 사용 금지 (Explore가 빠름). |
| **simplify skill** | 모듈 하나 완성 직후, 다음으로 넘어가기 전. |
| **security-review skill** | FC/CC 코드 → main 머지 직전 항상. |
| **review skill** | dev 브랜치 PR 올리기 전. |

## 투명성 규칙

Subagent 호출 직전에 반드시 텍스트로 먼저 알릴 것.
예: "Explore 에이전트로 탐색하겠습니다."

## 판단 기준

- "이 탐색이 컨텍스트를 오염시킬 만큼 넓은가?" → Yes면 Explore / general-purpose
- "설계 결정이 필요한가?" → Yes면 Plan

---

*2026-04-30 정립*
