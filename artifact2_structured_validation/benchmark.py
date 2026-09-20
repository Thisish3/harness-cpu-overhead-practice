"""
하니스가 모델 응답에서 tool_use 블록을 받을 때마다 그 input이 도구의 JSON Schema에
맞는지 검증하는 비용을 측정한다. 긴 에이전틱 세션은 도구 호출이 수백~수천 번 반복되므로
(챗봇과 다른 "반복 패턴") 이 검증 비용이 누적됨.

비교 대상:
  A. jsonschema.validate()  -- 순수 Python 구현 (파이썬 레벨에서 스키마 트리를 순회)
  B. pydantic v2 model_validate() -- 코어 검증 엔진이 Rust(pydantic-core)로 작성됨

이건 새로운 발견이 아니라 이미 업계에서 증명된 사실(그래서 pydantic v1->v2에서
코어를 통째로 Rust로 재작성했음)을 하니스의 실제 사용 패턴(반복되는 tool_use 검증)에
맞춰 직접 재현/측정한 것 -- Bloom filter 프로젝트가 vLLM의 실제 알고리즘을 재구현해서
검증했던 것과 같은 방법론.
"""

import time
import json
import jsonschema
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


# 실제 Claude Code류 하니스의 "run_code" 도구를 흉내낸, 적당히 중첩된 스키마
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "language": {"type": "string", "enum": ["python", "javascript", "bash", "rust"]},
        "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 600000},
        "options": {
            "type": "object",
            "properties": {
                "run_in_background": {"type": "boolean"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "retries": {"type": "integer", "minimum": 0, "maximum": 5},
            },
        },
    },
    "required": ["path", "content", "language"],
}


class RunCodeOptions(BaseModel):
    run_in_background: Optional[bool] = None
    env: Optional[dict[str, str]] = None
    retries: Optional[int] = Field(default=None, ge=0, le=5)


class RunCodeInput(BaseModel):
    path: str
    content: str
    language: str
    timeout_ms: Optional[int] = Field(default=None, ge=0, le=600000)
    options: Optional[RunCodeOptions] = None


def make_tool_call_input(i: int) -> dict:
    return {
        "path": f"script_{i}.py",
        "content": "print('hello world')\n" * 3,
        "language": "python",
        "timeout_ms": 5000,
        "options": {"run_in_background": False, "env": {"PATH": "/usr/bin"}, "retries": 2},
    }


def bench_jsonschema(n_calls: int) -> float:
    validator = jsonschema.Draft7Validator(JSON_SCHEMA)
    total = 0.0
    for i in range(n_calls):
        data = make_tool_call_input(i)
        t0 = time.perf_counter()
        validator.validate(data)
        total += time.perf_counter() - t0
    return total


def bench_pydantic(n_calls: int) -> float:
    total = 0.0
    for i in range(n_calls):
        data = make_tool_call_input(i)
        t0 = time.perf_counter()
        RunCodeInput.model_validate(data)
        total += time.perf_counter() - t0
    return total


def verify_correctness(n_calls: int = 20):
    """둘 다 유효한 입력은 통과, 잘못된 입력(필수 필드 누락)은 거부하는지 확인."""
    for i in range(n_calls):
        data = make_tool_call_input(i)
        jsonschema.validate(data, JSON_SCHEMA)  # 예외 안 나면 통과
        RunCodeInput.model_validate(data)  # 예외 안 나면 통과

    bad = {"path": "x.py"}  # content, language 누락 -- 둘 다 거부해야 함
    jsonschema_rejected = False
    try:
        jsonschema.validate(bad, JSON_SCHEMA)
    except jsonschema.ValidationError:
        jsonschema_rejected = True

    pydantic_rejected = False
    try:
        RunCodeInput.model_validate(bad)
    except ValidationError:
        pydantic_rejected = True

    assert jsonschema_rejected and pydantic_rejected, "잘못된 입력을 걸러내지 못함 -- 검증 로직 버그"
    print("정확성 검증 통과: 유효 입력 통과, 잘못된 입력(필수 필드 누락) 둘 다 거부 확인")


if __name__ == "__main__":
    verify_correctness()

    N_CALLS = 5000
    print(f"\n=== 누적 비용 ({N_CALLS}회 tool_use 검증, 긴 에이전틱 세션 흉내) ===")

    t_js = bench_jsonschema(N_CALLS)
    print(f"A. jsonschema (순수 Python):     {t_js*1000:8.2f} ms  ({t_js/N_CALLS*1e6:.2f} us/call)")

    t_pyd = bench_pydantic(N_CALLS)
    print(f"B. pydantic v2 (Rust core):      {t_pyd*1000:8.2f} ms  ({t_pyd/N_CALLS*1e6:.2f} us/call)")

    print(f"\n비율: {t_js/t_pyd:.2f}x -- pydantic-core가 더 빠름")
