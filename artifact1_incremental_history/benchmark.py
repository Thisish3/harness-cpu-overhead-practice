"""
하니스가 매 턴 LLM 호출 전에 {system + tools + 전체 히스토리}를 JSON으로 조립하는 비용을 측정한다.

가설: vLLM n-gram proposer가 매 스텝 "지금까지 전체 컨텍스트"를 처음부터 재스캔했던 것과
같은 구조적 문제가, 하니스의 요청 조립에도 있다 -- 매 턴마다 이미 확정된 과거 메시지까지
Python dict -> JSON 텍스트로 처음부터 다시 인코딩함.

세 가지 방식 비교:
  A. naive        : 매 턴 json.dumps(전체 payload) -- 과거 메시지도 매번 재인코딩
  B. incremental   : 메시지별로 한 번만 json.dumps 하고, 문자열 캐시를 이어붙이기(join)만 함
  C. orjson-naive  : A와 동일한 구조지만 Rust로 구현된 orjson으로 교체(아키텍처 변경 없이 라이브러리만 교체)

정직한 주의점: 이건 "빅오 클래스를 바꾸는" 최적화가 아니다. LLM API는 스테이트리스라서
매 호출마다 전체 히스토리 텍스트를 실제로 만들어서 보내야 한다 -- B도 결국 매 턴 O(전체 길이)
만큼의 바이트를 join해야 한다. 다만 "Python dict를 순회하며 타입 검사 + 이스케이프 처리해서
인코딩"하는 비용(A)과 "이미 인코딩된 바이트 문자열을 이어붙이기만 하는"비용(B)은 상수 계수가
크게 다르다 -- 이 상수 계수 차이를 측정하는 것이 이 벤치마크의 목적.
"""

import json
import time
import orjson

SYSTEM_PROMPT = (
    "You are Claude Code, an interactive CLI tool. " * 40
)  # 실제 시스템 프롬프트 규모 흉내 (약 2KB)

TOOLS = [
    {
        "name": f"tool_{i}",
        "description": f"Tool number {i} does something useful with files or shell commands. " * 3,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "options": {
                    "type": "object",
                    "properties": {
                        "recursive": {"type": "boolean"},
                        "timeout": {"type": "integer"},
                    },
                },
            },
            "required": ["path"],
        },
    }
    for i in range(12)  # Claude Code류 하니스의 실제 도구 개수 규모 흉내
]


def make_user_message(turn: int) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"turn {turn}: please do something with file_{turn}.py " * 3}],
    }


def make_assistant_message(turn: int) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": f"I'll handle turn {turn} now."},
            {
                "type": "tool_use",
                "id": f"toolu_{turn}",
                "name": f"tool_{turn % 12}",
                "input": {"path": f"file_{turn}.py", "content": "print('hello')\n" * 5},
            },
            {
                "type": "tool_result",
                "tool_use_id": f"toolu_{turn}",
                "content": [{"type": "text", "text": "Done. " * 10}],
            },
        ],
    }


def bench_naive_json(n_turns: int) -> float:
    messages = []
    total = 0.0
    for turn in range(n_turns):
        messages.append(make_user_message(turn))
        messages.append(make_assistant_message(turn))
        t0 = time.perf_counter()
        body = json.dumps({"system": SYSTEM_PROMPT, "tools": TOOLS, "messages": messages})
        total += time.perf_counter() - t0
    return total, len(body)


def bench_naive_orjson(n_turns: int) -> float:
    messages = []
    total = 0.0
    for turn in range(n_turns):
        messages.append(make_user_message(turn))
        messages.append(make_assistant_message(turn))
        t0 = time.perf_counter()
        body = orjson.dumps({"system": SYSTEM_PROMPT, "tools": TOOLS, "messages": messages})
        total += time.perf_counter() - t0
    return total, len(body)


def bench_incremental(n_turns: int) -> float:
    serialized_system = json.dumps(SYSTEM_PROMPT)
    serialized_tools = json.dumps(TOOLS)
    cached_msgs = []
    total = 0.0
    for turn in range(n_turns):
        new_msgs = [make_user_message(turn), make_assistant_message(turn)]
        t0 = time.perf_counter()
        for m in new_msgs:
            cached_msgs.append(json.dumps(m))
        body = (
            '{"system":' + serialized_system
            + ',"tools":' + serialized_tools
            + ',"messages":[' + ",".join(cached_msgs) + "]}"
        )
        total += time.perf_counter() - t0
    return total, len(body)


def bench_incremental_orjson(n_turns: int) -> float:
    """B + C 결합: 증분 캐싱 아키텍처 + 개별 메시지 인코딩도 orjson(Rust)으로."""
    serialized_system = orjson.dumps(SYSTEM_PROMPT)
    serialized_tools = orjson.dumps(TOOLS)
    cached_msgs = []
    total = 0.0
    for turn in range(n_turns):
        new_msgs = [make_user_message(turn), make_assistant_message(turn)]
        t0 = time.perf_counter()
        for m in new_msgs:
            cached_msgs.append(orjson.dumps(m))
        body = (
            b'{"system":' + serialized_system
            + b',"tools":' + serialized_tools
            + b',"messages":[' + b",".join(cached_msgs) + b"]}"
        )
        total += time.perf_counter() - t0
    return total, len(body)


def verify_correctness(n_turns: int = 5):
    """세 방식이 같은 논리적 payload를 만들어내는지 확인 (Bloom filter 프로젝트의
    false-negative 검증과 같은 정신 -- 속도만 재고 정확성을 안 재면 의미 없음)."""
    messages = []
    for turn in range(n_turns):
        messages.append(make_user_message(turn))
        messages.append(make_assistant_message(turn))
    naive = json.loads(json.dumps({"system": SYSTEM_PROMPT, "tools": TOOLS, "messages": messages}))

    serialized_system = json.dumps(SYSTEM_PROMPT)
    serialized_tools = json.dumps(TOOLS)
    cached_msgs = [json.dumps(m) for m in messages]
    body = (
        '{"system":' + serialized_system
        + ',"tools":' + serialized_tools
        + ',"messages":[' + ",".join(cached_msgs) + "]}"
    )
    incremental = json.loads(body)
    assert naive == incremental, "incremental 방식이 naive와 다른 payload를 만들어냄 -- 버그"
    print("정확성 검증 통과: naive == incremental (논리적으로 동일한 payload)")


if __name__ == "__main__":
    verify_correctness()

    N_TURNS = 400
    print(f"\n=== 누적 비용 ({N_TURNS}턴, 매 턴 요청 조립) ===")

    t_naive, size = bench_naive_json(N_TURNS)
    print(f"A. naive json.dumps(전체):      {t_naive*1000:8.2f} ms  (마지막 payload 크기: {size/1024:.1f} KB)")

    t_orjson, size = bench_naive_orjson(N_TURNS)
    print(f"C. naive orjson.dumps(전체):    {t_orjson*1000:8.2f} ms  ({t_naive/t_orjson:.2f}x vs A, 순수 라이브러리 교체)")

    t_incr, size = bench_incremental(N_TURNS)
    print(f"B. incremental(캐시+join):      {t_incr*1000:8.2f} ms  ({t_naive/t_incr:.2f}x vs A, 아키텍처 변경)")

    t_incr_orjson, size = bench_incremental_orjson(N_TURNS)
    print(f"D. incremental+orjson(결합):    {t_incr_orjson*1000:8.2f} ms  ({t_naive/t_incr_orjson:.2f}x vs A, 아키텍처+라이브러리 둘 다)")

    print(f"\nB가 C보다 {'빠름' if t_incr < t_orjson else '느림'} ({t_orjson/t_incr:.2f}x)")
    print(f"D가 B보다 {'빠름' if t_incr_orjson < t_incr else '느림'} ({t_incr/t_incr_orjson:.2f}x)")
