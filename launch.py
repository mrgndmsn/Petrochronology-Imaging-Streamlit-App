import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"


def environment_python():
    return ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def requirements_digest():
    content = b"".join(
        (ROOT / name).read_bytes() for name in ("requirements.txt", "requirements-local.txt")
    )
    return hashlib.sha256(content).hexdigest()


def setup():
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Install Python 3.13 from python.org, then open the launcher again.")
    python = environment_python()
    if not python.exists():
        print("Creating the local Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    stamp = ENV / "requirements.sha256"
    digest = requirements_digest()
    if not stamp.exists() or stamp.read_text() != digest:
        print(
            "Installing dependencies. The first start needs internet access...",
            flush=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements-local.txt"),
            ],
            check=True,
        )
        stamp.write_text(digest)
    return python


def server_command(python, port):
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=false",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the app locally.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    python = setup()
    if args.setup_only:
        return
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
    except OSError as exc:
        raise RuntimeError(
            f"Port {args.port} is busy. Close the earlier app, or use --port 8502."
        ) from exc
    print(
        f"Opening http://localhost:{args.port}\nKeep this window open. Save your project before stopping the app.",
        flush=True,
    )
    subprocess.run(server_command(python, args.port), cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Could not start the app: {error}", file=sys.stderr)
        sys.exit(1)
