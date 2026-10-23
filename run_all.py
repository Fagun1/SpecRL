#!/usr/bin/env python3
"""Full reproduction: unit tests, training (profile + Q-learning), held-out evaluation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/main")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    models = ["--target", args.target.resolve(), "--draft", args.draft.resolve(),
              "--run-dir", args.run_dir.resolve(), "--seed", args.seed]
    run("-m", "unittest", "discover", "-s", "tests")
    run("train.py", *models)
    run("evaluate.py", *models, "--prompts", ROOT / "prompts/test_diverse.tsv", "--name", "test_diverse")
    run("evaluate.py", *models, "--prompts", ROOT / "prompts/dev.tsv", "--name", "dev")


if __name__ == "__main__":
    main()
