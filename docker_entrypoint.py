import os
import signal
import subprocess
import sys
import time


SERVICES = (
    ("api", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]),
    ("chat-ui", [sys.executable, "-m", "streamlit", "run", "UI/app.py", "--server.address=0.0.0.0", "--server.port=8502"]),
    ("eval-ui", [sys.executable, "-m", "streamlit", "run", "evals/app.py", "--server.address=0.0.0.0", "--server.port=8501"]),
)

processes = {}
stopping = False


def stop_services(signum=None, frame=None):
    global stopping
    stopping = True
    for process in processes.values():
        if process.poll() is None:
            process.terminate()


def main():
    os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:8000")
    os.environ.setdefault("COGNIVAULT_API_URL", "http://127.0.0.1:8000/query")
    signal.signal(signal.SIGTERM, stop_services)
    signal.signal(signal.SIGINT, stop_services)

    try:
        for name, command in SERVICES:
            processes[name] = subprocess.Popen(command)

        while not stopping:
            for name, process in processes.items():
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"{name} exited with status {exit_code}; stopping remaining services.", flush=True)
                    stop_services()
                    return exit_code or 1
            time.sleep(1)
    finally:
        stop_services()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
