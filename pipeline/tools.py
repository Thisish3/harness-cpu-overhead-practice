"""
하니스가 실제로 실행하는 도구들 -- sandbox/ 디렉토리 밖으로 못 나가게 경로를 강제한다.
각 도구는 pydantic 모델로 입력을 검증한 뒤에만 실행된다 (artifact2_structured_validation의
pydantic-core 검증을 실제 파이프라인에 적용한 것).
"""

from pathlib import Path
from pydantic import BaseModel

SANDBOX_DIR = Path(__file__).parent / "sandbox"
SANDBOX_DIR.mkdir(exist_ok=True)


def _resolve(path: str) -> Path:
    resolved = (SANDBOX_DIR / path).resolve()
    if not str(resolved).startswith(str(SANDBOX_DIR.resolve())):
        raise ValueError(f"경로가 샌드박스({SANDBOX_DIR}) 밖을 가리킴: {path}")
    return resolved


class ReadFileArgs(BaseModel):
    path: str


class WriteFileArgs(BaseModel):
    path: str
    content: str


class ListDirArgs(BaseModel):
    path: str = "."


def read_file(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"오류: {path} 없음"
    return p.read_text()


def write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"{path}에 {len(content)}바이트 씀"


def list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.exists():
        return f"오류: {path} 없음"
    return "\n".join(sorted(e.name for e in p.iterdir()))


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "샌드박스 안의 파일 내용을 읽는다.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "샌드박스 안에 파일을 쓴다(덮어씀).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "샌드박스 안의 디렉토리 내용을 나열한다.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
]

ARG_MODELS = {"read_file": ReadFileArgs, "write_file": WriteFileArgs, "list_dir": ListDirArgs}
TOOL_IMPLS = {"read_file": read_file, "write_file": write_file, "list_dir": list_dir}
