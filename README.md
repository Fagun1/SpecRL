# SpecRL

A reinforcement learning (Q-learning) agent that speeds up text generation with speculative decoding.
A small model (Qwen3-0.6B) guesses tokens and a large model (Qwen3-4B) checks them. After each guess,
the agent decides whether to guess one more token or stop and check. The trained agent is
`runs/main/policy.json`.

## Dependencies

- Windows 10/11 with an NVIDIA GPU (6 GB or more)
- Python 3.11 or newer, with the packages in `requirements.txt` (matplotlib)
- [Ollama 0.30.10](https://github.com/ollama/ollama/releases/tag/v0.30.10). Only its llama.cpp/CUDA
  runtime files are used; the Ollama app does not need to run.
- Two model files:
  - `Qwen3-4B-Q4_K_M.gguf` from [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
  - `Qwen3-0.6B-Q8_0.gguf` from [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)

## Installation

1. Install Ollama 0.30.10.
2. Download the two model files (for example into `Downloads`).
3. Install the Python packages:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`native/bridge.exe` is already built. To rebuild it, use an x64 MinGW-w64 compiler:
`.venv\Scripts\python.exe build_native.py --compiler C:\path\to\g++.exe`

## Running

Set the model paths once in PowerShell:

```powershell
$t = "C:\path\to\Qwen3-4B-Q4_K_M.gguf"; $d = "C:\path\to\Qwen3-0.6B-Q8_0.gguf"
```

| Task | Command |
| --- | --- |
| Check the GPU setup | `.venv\Scripts\python.exe smoke_bridge.py --target $t --draft $d` |
| Run the unit tests | `.venv\Scripts\python.exe -m unittest discover -s tests` |
| Train a new agent | `.venv\Scripts\python.exe train.py --target $t --draft $d --run-dir runs/new` |
| Test an agent | `.venv\Scripts\python.exe evaluate.py --target $t --draft $d --run-dir runs/new` |
| Live demo (browser) | double-click `start_demo.bat`, or `.venv\Scripts\python.exe demo_app.py --target $t --draft $d --open` |
| Demo with one prompt | `.venv\Scripts\python.exe demo_app.py --target $t --draft $d --prompt "Write a Python function that reverses a list."` |

The demo opens at http://127.0.0.1:7860. It uses the trained agent in `runs/main` and compares it with
the large model alone and with fixed guess lengths. `start_demo.bat` looks for the model files in
`Downloads`; edit the two paths inside it if yours are elsewhere.

## Reproducing the results

```powershell
.venv\Scripts\python.exe run_all.py --target $t --draft $d --run-dir runs/repro
```

This runs the tests, trains a new agent (about 40 minutes on an RTX 3050) and tests it on
`prompts/test_diverse.tsv` and `prompts/dev.tsv` (about 10 minutes each). The results go to
`runs/repro/eval_test_diverse/results.md`. The results of the final agent are in
`runs/main/eval_test_diverse/results.md`.

Speeds depend on the GPU, its temperature and other running programs. For stable numbers, plug in the
charger, use Windows "Best performance" mode and close other heavy apps.
