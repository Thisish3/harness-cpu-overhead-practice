# pipeline — 실제로 돌아가는 하니스 + 서빙 엔진

artifact1/artifact2의 최적화(증분 직렬화, pydantic 검증)를 실제로 HuggingFace 모델을 로딩해서
동작하는 에이전트 하니스 안에 넣고 end-to-end로 돌려본다.

## 왜 진짜 vLLM이 아니라 스텁 서버인가 — 정직하게

이 머신(M2 Mac, 16GB, CUDA 없음)에서 vLLM을 직접 돌리는 걸 시도했지만 막혔다:

1. **`pip install vllm`**: macOS용 wheel 자체가 없음 (Linux 전용 배포)
2. **`Dockerfile.cpu` 빌드**: Apple Silicon에서 [알려진 빌드 실패 이슈](https://github.com/vllm-project/vllm/issues/21714)
3. **Docker Model Runner + vllm-metal**(2026년에 나온 Apple Silicon 전용 경로): 설치된 Docker Desktop(4.38.0)을
   최신 버전으로 업데이트하려 했으나 `docker desktop update`가 매번 `install error: problem while
   unmounting the disk: exit status 16`로 실패 — 이전 실패 시도가 남긴 디스크 이미지 마운트가
   `diskutil unmountDisk force`로도 안 풀려서(디스크 이미지 헬퍼 프로세스가 붙잡고 있음) 재부팅
   없이는 해결 불가로 판단, 보류함

그래서 **API 표면(엔드포인트 경로, 요청/응답 스키마, `tool_calls` 형식)을 vLLM의 OpenAI 호환
서버와 동일하게 맞춘 스텁**(`server.py`)을 직접 만들었다. `harness.py`는 `SERVER_URL`만 실제
vLLM 서버 주소로 바꾸면 코드 변경 없이 그대로 동작한다 — 로컬에서는 transformers+MPS/CPU로
Qwen2.5-0.5B-Instruct를 돌리고, 프로덕션에서는 같은 코드가 진짜 vLLM을 호출하는 구조.

## 구성
| 파일 | 역할 |
|---|---|
| [`tools.py`](tools.py) | 샌드박스 파일 도구(read/write/list) + pydantic 입력 검증 모델 |
| [`server.py`](server.py) | vLLM/OpenAI 호환 `/v1/chat/completions` 스텁 (내부는 HF Qwen2.5-0.5B-Instruct) |
| [`harness.py`](harness.py) | 실제 에이전트 루프 — 증분 직렬화(artifact1) + pydantic 검증(artifact2) 적용, 하니스 CPU 시간과 모델 추론 시간을 분리 계측 |
| [`run_demo.py`](run_demo.py) | 데모 실행: "fib.py를 써라" 태스크를 실제로 시켜본다 |

## 실행 방법
```bash
pip install -r requirements.txt
python3 server.py &          # 모델 로딩 후 127.0.0.1:8000에서 대기
python3 run_demo.py          # 실제 에이전틱 태스크 1턴 실행
```

## 실측 결과 (CPU 추론, 실제 1턴 실행)
```
태스크: write_file로 fib.py를 만들고 read_file로 다시 읽어 확인하라

=== 실측 시간 분해 ===
하니스 CPU(요청조립+검증):        0.30 ms
서버 왕복(네트워크+추론):     19323.11 ms
  └ 그중 모델 추론(순수):     19289.95 ms
하니스 CPU 비중: 0.0%
```

**이게 오히려 Amdahl's Law 논지를 반대 방향에서 확인해준다**: 가속 없는 CPU 추론이 압도적으로
느리니(19.3초) 하니스 비용(0.3ms, 약 64,000분의 1)은 지금 당장은 완전히 무시할 수 있는 수준.
**바로 이게 LPU/GPU 가속이 중요한 이유 — 그리고 그 가속이 실제로 이뤄지고 나면(추론이 20ms대로
떨어지면) artifact1/2에서 측정한 하니스 비용(수십 ms 단위)이 무시 못할 비중으로 드러난다.**
지금 이 데모는 "가속 전"의 스냅샷이고, artifact1/2의 400턴/5000콜 누적 벤치마크가 "가속 후"
시나리오를 미리 보여주는 셈.

## 정직한 한계
- Qwen2.5-**0.5B**는 아주 작은 모델이라 tool-call 체이닝이 불안정함 — 이번 데모에서도
  `write_file`은 정상 호출했지만 이어서 `read_file`을 호출하라는 지시는 따르지 않고 텍스트로
  응답을 끝냄. 하니스/파이프라인 자체는 정상 동작(실제 파일이 올바르게 써졌음을 확인함) —
  모델 능력 한계이지 하니스 버그가 아님.
- 실제 vLLM 서버가 아니라 API 호환 스텁이라, vLLM 자체의 스케줄링/배치/PagedAttention 관련
  지연 특성은 전혀 반영 안 됨 — 이 데모가 보여주는 건 "하니스 쪽 구조"뿐, 서빙 엔진 성능은 아님.
- 단일 턴 1회 실행 결과라 통계적으로 유의미한 반복 측정은 아님(artifact1/2는 각각 400턴/5000회
  누적 측정이라 더 신뢰할 수 있음).
- MPS(Apple GPU) 백엔드로 처음 시도했으나 서버가 조용히 크래시함(트레이스백 없이 프로세스
  종료) — 원인 특정 안 하고 CPU로 우회함. MPS 관련 PyTorch 연산 지원 이슈로 추정되나 확인은
  못 했음.
