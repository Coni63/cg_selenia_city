#!/usr/bin/env python3
"""
Benchmark the Python solution against the Selenia City game engine.

Builds the game's benchmark jar (FallChallenge2024-SeleniaCity/) if needed,
then runs the solution in python_solution/main.py against every test case
in FallChallenge2024-SeleniaCity/config/ (or a single one with --test),
reporting per-test and total scores.

Usage:
    python bench_full.py                # run all tests
    python bench_full.py --test 8       # run only test8.json
    python bench_full.py --rebuild      # force a fresh `mvn package`
    python bench_full.py --solution rust_solution/target/release/agent.exe
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAME_DIR = ROOT / "FallChallenge2024-SeleniaCity"
CONFIG_DIR = GAME_DIR / "config"
JAR_PATH = GAME_DIR / "target" / "fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar"
OUTPUT_FILE = ROOT / "ref_scores.txt"

DEFAULT_VENV_PYTHON = ROOT / "python_solution" / ".venv" / "Scripts" / "python.exe"
DEFAULT_MAIN = "python_solution/main.py"


def find_python() -> Path:
    if DEFAULT_VENV_PYTHON.exists():
        return DEFAULT_VENV_PYTHON
    posix_venv = ROOT / "python_solution" / ".venv" / "bin" / "python"
    if posix_venv.exists():
        return posix_venv
    return Path(sys.executable)


def build_jar():
    print(f"[bench] building jar in {GAME_DIR} ...", flush=True)
    result = subprocess.run(
        ["mvn", "-q", "-DskipTests", "package"],
        cwd=GAME_DIR,
        check=False,
    )
    if result.returncode != 0:
        print("[bench] maven build failed", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=int, default=None, help="run only testN.json instead of the full suite")
    parser.add_argument("--rebuild", action="store_true", help="force rebuilding the benchmark jar with mvn package")
    parser.add_argument(
        "--solution",
        default=None,
        help="agent command to run (default: '<venv python> python_solution/main.py')",
    )
    parser.add_argument("--config-dir", default=None, help="override the config/ directory used for test cases")
    parser.add_argument("--output", default=None, help="override the score report output file")
    args = parser.parse_args()

    if not shutil.which("java"):
        print("[bench] java not found on PATH", file=sys.stderr)
        sys.exit(1)

    if args.rebuild or not JAR_PATH.exists():
        if not shutil.which("mvn"):
            print("[bench] mvn not found on PATH, cannot build the jar", file=sys.stderr)
            sys.exit(1)
        build_jar()

    agent_command = args.solution
    if agent_command is None:
        python_exe = find_python()
        agent_command = f"{python_exe} {DEFAULT_MAIN}"

    config_dir = Path(args.config_dir) if args.config_dir else CONFIG_DIR
    output_file = Path(args.output) if args.output else OUTPUT_FILE

    cmd = ["java", "-jar", str(JAR_PATH), agent_command, str(config_dir), str(output_file)]
    if args.test is not None:
        cmd.append(str(args.test))

    print(f"[bench] agent: {agent_command}")
    print(f"[bench] config: {config_dir}")
    print("[bench] running...", flush=True)

    result = subprocess.run(cmd, cwd=ROOT, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
