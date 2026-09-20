"""
실제로 돌아가는 에이전트 하니스 -- vLLM/OpenAI 호환 서버(server.py 또는 진짜 vLLM)에
HTTP로 요청을 보내고, tool_calls를 검증/실행하고, 결과를 다시 히스토리에 넣어 루프를 돈다.

CPU 재부상 조사에서 확인한 두 가지 최적화를 실제 루프 안에 적용한다:
  1. artifact1_incremental_history -- 매 턴 전체 히스토리를 json.dumps로 재인코딩하지 않고,
     메시지별로 한 번만 직렬화한 캐시를 이어붙인다.
  2. artifact2_structured_validation -- tool_call의 arguments를 jsonschema가 아니라
     pydantic(Rust 코어)으로 검증한다.

그리고 이 벤치마크들이 "합성 상황에서 몇 배 빠르다"고 주장했던 걸, 진짜 모델을 실제로
호출하는 루프 안에서 하니스 CPU 시간 vs 모델 추론 시간으로 다시 측정해서 Amdahl's Law
논지가 실측으로도 성립하는지 검증한다.
"""

import json
import time

import httpx

from tools import ARG_MODELS, TOOL_IMPLS, TOOL_SCHEMAS

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
SYSTEM_PROMPT = "You are a helpful coding assistant with access to a sandboxed filesystem."


class Harness:
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.client = httpx.Client(timeout=120.0)
        self._serialized_msgs: list[str] = []  # 증분 직렬화 캐시 (artifact1)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._serialized_msgs.append(json.dumps(self.messages[0]))

        # 계측: 하니스 CPU 시간(요청 조립+검증+도구 실행) vs 모델 추론 시간(서버 왕복)
        self.timing = {"harness_cpu_ms": 0.0, "server_roundtrip_ms": 0.0, "model_inference_ms": 0.0}

    def _append_and_serialize(self, msg: dict) -> None:
        """artifact1의 증분 캐싱 -- 새 메시지 하나만 json.dumps, 과거 메시지는 재인코딩 안 함."""
        t0 = time.perf_counter()
        self.messages.append(msg)
        self._serialized_msgs.append(json.dumps(msg))
        self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000

    def _build_request_body(self) -> bytes:
        t0 = time.perf_counter()
        body = (
            b'{"messages":[' + ",".join(self._serialized_msgs).encode() + b"],"
            + b'"tools":' + json.dumps(TOOL_SCHEMAS).encode() + b"}"
        )
        self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000
        return body

    def _call_model(self) -> dict:
        body = self._build_request_body()
        t0 = time.perf_counter()
        resp = self.client.post(self.server_url, content=body, headers={"Content-Type": "application/json"})
        roundtrip_ms = (time.perf_counter() - t0) * 1000
        self.timing["server_roundtrip_ms"] += roundtrip_ms
        data = resp.json()
        self.timing["model_inference_ms"] += data.get("_server_timing_ms", {}).get("inference", 0.0)
        return data

    def _execute_tool_call(self, tool_call: dict) -> str:
        """pydantic 검증(artifact2) 후 실행."""
        t0 = time.perf_counter()
        name = tool_call["function"]["name"]
        raw_args = json.loads(tool_call["function"]["arguments"])
        model_cls = ARG_MODELS.get(name)
        if model_cls is None:
            self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000
            return f"오류: 알 수 없는 도구 {name}"
        try:
            validated = model_cls.model_validate(raw_args)
        except Exception as e:
            self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000
            return f"검증 실패: {e}"
        self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000

        # 실제 도구 실행(파일 I/O)은 "하니스 CPU 작업"과 별개로 계측하지 않음 -- I/O bound
        return TOOL_IMPLS[name](**validated.model_dump())

    def run_turn(self, user_text: str, max_tool_hops: int = 5) -> str:
        self._append_and_serialize({"role": "user", "content": user_text})

        for _ in range(max_tool_hops):
            data = self._call_model()
            choice = data["choices"][0]
            message = choice["message"]
            self._append_and_serialize(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message.get("content") or ""

            for tc in tool_calls:
                result = self._execute_tool_call(tc)
                self._append_and_serialize(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )

        return "(max_tool_hops 도달)"

    def report_timing(self) -> None:
        t = self.timing
        total = t["harness_cpu_ms"] + t["server_roundtrip_ms"]
        print("\n=== 실측 시간 분해 ===")
        print(f"하니스 CPU(요청조립+검증):  {t['harness_cpu_ms']:8.2f} ms")
        print(f"서버 왕복(네트워크+추론):    {t['server_roundtrip_ms']:8.2f} ms")
        print(f"  └ 그중 모델 추론(순수):    {t['model_inference_ms']:8.2f} ms")
        if total > 0:
            print(f"하니스 CPU 비중: {t['harness_cpu_ms']/total*100:.1f}%")
