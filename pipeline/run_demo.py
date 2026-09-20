"""
실제 데모: 하니스가 로컬 모델(server.py, 또는 base_url을 바꾸면 진짜 vLLM)에게 파일을
쓰고 읽게 시키는 에이전틱 태스크를 실제로 한 번 실행하고, 하니스 CPU 시간 vs 모델 추론
시간을 실측한다.

먼저 다른 터미널에서 서버를 띄워야 함:
    python3 server.py
"""

from harness import Harness

TASK = (
    "Use the write_file tool to create fib.py in the sandbox that prints the first "
    "10 Fibonacci numbers. Then use read_file to read it back and confirm the content."
)


def main():
    h = Harness()
    print(f"태스크: {TASK}\n")
    result = h.run_turn(TASK)
    print(f"\n최종 응답: {result}")
    h.report_timing()


if __name__ == "__main__":
    main()
