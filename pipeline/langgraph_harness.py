"""
진짜 LangGraph(StateGraph, Pregel 실행 엔진)로 짠 하니스 -- "🔄 요청 처리 파이프라인 비교" 페이지의
4-1(LangGraph는 사실 구글 Pregel이다) 딥다이브에서 확인한 구조(super-step: Plan→Execute→Update)를
직접 코드로 재현. harness.py(직접 짠 커스텀 루프)와 달리 이번엔 실제 langgraph 라이브러리를 쓴다.

서빙 엔진은 server.py(vLLM API 호환 스텁, 내부는 HF Qwen2.5)를 그대로 쓴다 -- langchain_openai의
ChatOpenAI가 base_url만 바꾸면 붙는 걸 이용해서, 이 스텁이든 진짜 vLLM/SGLang이든(둘 다
OpenAI 호환 엔드포인트를 노출하므로) 코드 변경 없이 서빙 엔진을 스왑할 수 있게 짰다.

도구 실행 노드에서는 artifact2의 pydantic 검증을 그대로 적용한다.
"""

import time
from typing import Annotated, TypedDict

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import ARG_MODELS, TOOL_IMPLS, TOOL_SCHEMAS

SERVER_BASE_URL = "http://127.0.0.1:8000/v1"  # vLLM/SGLang 실서버로 바꾸려면 이 줄만 교체
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphHarness:
    def __init__(self, base_url: str = SERVER_BASE_URL):
        self.llm = ChatOpenAI(base_url=base_url, api_key="none", model=MODEL_NAME, max_tokens=200).bind_tools(
            TOOL_SCHEMAS
        )
        self.timing = {"harness_cpu_ms": 0.0, "model_roundtrip_ms": 0.0, "super_steps": 0}
        self.graph = self._build_graph()

    def _agent_node(self, state: AgentState) -> dict:
        t0 = time.perf_counter()
        response = self.llm.invoke(state["messages"])
        self.timing["model_roundtrip_ms"] += (time.perf_counter() - t0) * 1000
        self.timing["super_steps"] += 1
        return {"messages": [response]}

    def _tools_node(self, state: AgentState) -> dict:
        """pydantic 검증(artifact2) 후 실행 -- 이 노드 안의 시간만 '하니스 CPU'로 계측."""
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            t0 = time.perf_counter()
            name = tc["name"]
            model_cls = ARG_MODELS.get(name)
            if model_cls is None:
                content = f"오류: 알 수 없는 도구 {name}"
            else:
                try:
                    validated = model_cls.model_validate(tc["args"])
                    self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000
                    content = TOOL_IMPLS[name](**validated.model_dump())  # 파일 I/O는 별도 -- CPU 계측 밖
                except Exception as e:
                    self.timing["harness_cpu_ms"] += (time.perf_counter() - t0) * 1000
                    content = f"검증 실패: {e}"
            results.append(ToolMessage(content=str(content), tool_call_id=tc["id"]))
        self.timing["super_steps"] += 1
        return {"messages": results}

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    def run(self, user_text: str) -> str:
        result = self.graph.invoke({"messages": [{"role": "user", "content": user_text}]})
        return result["messages"][-1].content

    def report_timing(self) -> None:
        t = self.timing
        print("\n=== LangGraph 실행 실측 ===")
        print(f"super-step 수(Plan→Execute→Update 반복):  {t['super_steps']}")
        print(f"하니스 CPU(도구 검증):          {t['harness_cpu_ms']:8.2f} ms")
        print(f"모델 왕복(agent 노드, 네트워크+추론): {t['model_roundtrip_ms']:8.2f} ms")


if __name__ == "__main__":
    h = LangGraphHarness()
    task = "Use the write_file tool to create greeting.txt in the sandbox with the text 'hello from langgraph'."
    print(f"태스크: {task}\n")
    print("최종 응답:", h.run(task))
    h.report_timing()
