# 하니스 CPU 오버헤드 — LPU 회사 관점에서 왜 중요한가

"에이전틱 AI와 CPU 재부상" 조사에서 확인한 3-레이어 모델(하니스①-추론②-도구실행③) 중,
LPU 같은 추론 가속기가 직접 손댈 수 없는 **레이어①(하니스)의 CPU 오버헤드**를 실제로 줄여보는 실습.

## 왜 이게 LPU 회사에 중요한가 — Amdahl's Law

LPU가 레이어②(추론)를 아무리 빠르게 만들어도, 하니스(①③)의 CPU 오버헤드가 그대로면
사용자가 체감하는 전체 속도 개선은 훨씬 작아진다.

```
예시: 추론 200ms + 하니스 50ms = 총 250ms (추론이 전체의 80%)
LPU가 추론을 10배 가속(20ms)  → 20ms + 50ms = 70ms
체감 개선: 250/70 = 3.6배  (10배가 아니라 3.6배)
하니스 비중: 80% → 71%로 오히려 커짐
```

**LPU가 빨라질수록 하니스 CPU 오버헤드의 상대적 비중이 커진다** — 칩만 잘 만들어도 이 병목을
누군가 안 풀면 고객이 실제로 느끼는 이득이 깎인다. Compute Library Engineer가 커널 바깥
(하니스 쪽 CPU 작업)까지 신경 써야 하는 이유.

## 두 가지 실측

### [artifact1_incremental_history](artifact1_incremental_history/) — 요청 조립 비용

매 턴 `{system + tools + 전체 히스토리}`를 JSON으로 조립하는 비용. vLLM n-gram proposer가
매 스텝 "지금까지 전체 컨텍스트"를 처음부터 재스캔했던 것과 같은 구조적 문제가 하니스의
요청 조립에도 있다 — 과거 메시지까지 매번 Python dict → JSON 텍스트로 재인코딩.

```
=== 누적 비용 (400턴, 매 턴 요청 조립) ===
A. naive json.dumps(전체):        455.82 ms
C. naive orjson.dumps(전체):       47.60 ms  ( 9.58x vs A, 라이브러리만 교체)
B. incremental(캐시+join):         14.69 ms  (31.04x vs A, 아키텍처 변경)
D. incremental+orjson(결합):       11.10 ms  (41.05x vs A, 아키텍처+라이브러리 둘 다)
```

**B가 C보다 3.24배 빠름** — 아키텍처 변경(과거 메시지 재인코딩 회피)이 순수 라이브러리 교체보다
효과가 큼. 단 둘을 결합(D)하면 B보다 1.32배 더 빠름 — 상호 배타적이지 않고 누적됨.

### [artifact2_structured_validation](artifact2_structured_validation/) — tool_use 검증 비용

모델 응답의 tool_use 블록이 도구 JSON Schema에 맞는지 검증하는 비용. 에이전틱 세션은 도구
호출이 수백~수천 번 반복되므로(챗봇과 다른 패턴) 이 검증 비용이 누적됨.

```
=== 누적 비용 (5000회 tool_use 검증) ===
A. jsonschema (순수 Python):     257.68 ms  (51.54 us/call)
B. pydantic v2 (Rust core):       16.12 ms  ( 3.22 us/call)

비율: 15.99x -- pydantic-core가 더 빠름
```

이건 새로운 발견이 아니라 업계에서 이미 증명된 사실(pydantic이 v1→v2에서 코어를 통째로
Rust로 재작성한 이유)을 하니스의 실제 사용 패턴(반복되는 tool_use 검증)에 맞춰 직접
재현/측정한 것.

## 정직한 한계

- **artifact1은 빅오 클래스를 바꾸는 최적화가 아님.** LLM API는 스테이트리스라 매 호출마다
  전체 히스토리 텍스트를 실제로 만들어서 보내야 한다 — incremental 방식도 결국 매 턴
  O(전체 길이)만큼의 바이트를 join해야 함. 다만 "Python dict를 순회하며 타입 검사+이스케이프
  처리해서 인코딩"하는 비용(A)과 "이미 인코딩된 바이트 문자열을 이어붙이기만 하는" 비용(B)의
  상수 계수 차이를 측정한 것 — 복잡도 클래스는 그대로, 상수 계수만 크게 줄임.
- 실제 Claude Code/LangGraph 하니스 코드를 patch로 수정해서 프로덕션에 검증한 게 아니라,
  구조를 흉내낸 독립 벤치마크.
- artifact2는 pydantic v2가 이미 Rust 코어를 쓴다는 점에서 "직접 만든" 최적화가 아니라
  "이미 있는 올바른 도구를 골라 쓴" 것에 가까움 — 그럼에도 실제로 하니스가 jsonschema처럼
  느린 검증기를 쓰고 있다면 교체만으로 16배 이득이라는 걸 실측으로 확인한 데 의미가 있음.
- 메시지/스키마 크기, 턴 수는 임의로 정한 것 — 실제 프로덕션 워크로드의 정확한 분포를
  반영한 건 아님.

## 재현 방법
```bash
cd artifact1_incremental_history && python3 benchmark.py
cd ../artifact2_structured_validation && python3 benchmark.py
```
