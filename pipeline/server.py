"""
vLLM/OpenAI 호환 /v1/chat/completions 서버 -- 안에서는 HuggingFace Qwen2.5-0.5B-Instruct를
transformers + MPS(Apple Silicon GPU)로 직접 돌린다.

이 머신(M2 Mac)은 vLLM 자체를 못 돌린다: (1) macOS용 pip wheel이 없고, (2) Dockerfile.cpu를
Apple Silicon에서 빌드하는 건 알려진 실패 이슈가 있고, (3) Docker Model Runner(vllm-metal)는
설치된 Docker Desktop 버전에 없었음. 그래서 API 표면(요청/응답 스키마, 엔드포인트 경로,
tool_calls 형식)을 vLLM의 OpenAI 호환 서버와 동일하게 맞춘 스텁을 직접 만들었다 --
하니스(harness.py) 코드는 이 스텁이든 진짜 vLLM 서버든 base_url만 바꾸면 그대로 동작한다.
"""

import json
import re
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
import os
DEVICE = os.environ.get("HARNESS_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI()

print(f"모델 로딩: {MODEL_NAME} (device={DEVICE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()
print("모델 로딩 완료")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[dict]
    tools: list[dict] | None = None
    max_tokens: int = 256
    temperature: float = 0.2


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Qwen의 <tool_call>{"name":..,"arguments":{..}}</tool_call> 출력을
    OpenAI tool_calls 형식으로 변환. content와 tool_calls를 분리해서 반환."""
    matches = TOOL_CALL_RE.findall(text)
    tool_calls = []
    for m in matches:
        try:
            obj = json.loads(m)
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))},
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    remaining_text = TOOL_CALL_RE.sub("", text).strip()
    return remaining_text, tool_calls


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt_text = tokenizer.apply_chat_template(
        req.messages, tools=req.tools, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_ms = (time.perf_counter() - t0) * 1000

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    content, tool_calls = parse_tool_calls(raw_text)

    message = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(new_tokens.shape[0])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "_server_timing_ms": {"inference": inference_ms},  # 표준 스펙 밖 -- 하니스 CPU vs 추론 CPU 분리 측정용
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
