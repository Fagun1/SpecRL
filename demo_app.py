#!/usr/bin/env python3
"""Live demo: load the trained RL agent and test it on any prompt.

Browser dashboard (default):
    python demo_app.py --target <Qwen3-4B.gguf> --draft <Qwen3-0.6B.gguf>
    then open http://127.0.0.1:7860

Terminal only, one prompt:
    python demo_app.py --target ... --draft ... --prompt "Write a Python function that reverses a list."

Both models stay loaded on the GPU. The Q-table in runs/main/policy.json is used
frozen: no exploration and no learning.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import statistics
import threading
import webbrowser

from agent import (CONF_THRESHOLDS, MAX_DRAFT, CostModel, FixedController, QController, QLearner,
                   load_prompts, run_episode)
from native_decoding import NativeDecoder

ROOT = Path(__file__).resolve().parent
CODE_WORDS = ("python", "function", "code", "program", "script", "class ", "def ", "javascript", "java",
              "c++", "sql", "implement", "algorithm", "compile", "variable", "loop", "array", "list ")
METHOD_NAMES = {"rl": "RL agent", "target_only": "Large model only"}


def guess_task(prompt):
    text = prompt.lower()
    return "code" if any(w in text for w in CODE_WORDS) else "fact"


def method_name(method):
    return METHOD_NAMES.get(method) or f"Fixed length {method.split('_')[1]}"


class Demo:
    def __init__(self, args):
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        self.policy_info = {"file": str(Path(args.policy).relative_to(ROOT)) if Path(args.policy).is_relative_to(ROOT)
                            else str(args.policy),
                            "states": len(policy["q"]), "episodes": policy["episodes"],
                            "best_fixed": policy["best_fixed_budget"], "trained_tokens": policy["tokens"]}
        self.learner = QLearner.from_dict(policy)
        self.cost = CostModel(**policy["cost_model"])
        self.best = policy["best_fixed_budget"]
        print("Loading both models on the GPU (about 20 s)...", flush=True)
        self.decoder = NativeDecoder(args.target, args.draft, context=args.context,
                                     log_path=ROOT / "runs/demo/native-runtime.log")
        self.decoder.configure(MAX_DRAFT, CONF_THRESHOLDS)
        trained = policy["provenance"]["models"]
        self.model_warning = None
        if (self.decoder.metadata["target_sha256"], self.decoder.metadata["draft_sha256"]) != \
                (trained["target_sha256"], trained["draft_sha256"]):
            self.model_warning = "These model files differ from the ones the agent was trained with."
            print("WARNING:", self.model_warning, flush=True)
        self.models = {"target": self.decoder.metadata["target_file"], "draft": self.decoder.metadata["draft_file"],
                       "gpu_layers": self.decoder.metadata["offloaded_layers"]}
        self.lock = threading.Lock()
        print("Warming up the GPU...", flush=True)
        for row in load_prompts(ROOT / "prompts/train.tsv")[:3]:
            for budget in (0, self.best):
                run_episode(self.decoder, row, 32, FixedController(budget), self.cost)
        print("Ready.", flush=True)

    def controller(self, method):
        if method == "rl":
            return QController(self.learner)
        return FixedController(0 if method == "target_only" else int(method.split("_")[1]))

    def run(self, prompt, task, tokens, methods, repeats):
        task = task if task in ("code", "fact") else guess_task(prompt)
        row = {"task": task, "prompt": prompt}
        with self.lock:
            runs = {m: [] for m in methods}
            for _ in range(repeats):
                order = list(methods)
                random.shuffle(order)  # spread GPU warm-up and drift across methods
                for method in order:
                    runs[method].append(run_episode(self.decoder, row, tokens, self.controller(method), self.cost))
            results = [self.summarize(method, runs[method], method == "rl") for method in methods]
        base = next((r for r in results if r["method"] == "target_only"), None)
        for r in results:
            r["speedup"] = r["tokens_per_second"] / base["tokens_per_second"] if base else None
            r["same_text_as_large_model"] = (r["text"] == base["text"]) if base else None
        return {"task": task, "tokens": tokens, "repeats": repeats, "results": results}

    def summarize(self, method, episodes, with_trace):
        produced = sum(e["tokens_generated"] for e in episodes)
        seconds = sum(e["generation_seconds"] for e in episodes)
        last = episodes[-1]
        lengths = [l for e in episodes for l in e["draft_lengths"]]
        drafted, accepted = sum(e["drafted"] for e in episodes), sum(e["accepted"] for e in episodes)
        out = {"method": method, "name": method_name(method), "text": last["response"],
               "tokens_generated": last["tokens_generated"],
               "tokens_per_second": produced / seconds,
               "runs": [round(e["tokens_per_second"], 2) for e in episodes],
               "rounds": last["cycles"], "mean_guesses": statistics.mean(lengths) if lengths else 0,
               "acceptance": accepted / drafted if drafted else None}
        if with_trace:
            out["trace"] = self.trace(last)
        return out

    def trace(self, episode):
        """Round-by-round view of the last RL run: guesses, confidences, decisions, Q-values."""
        transitions = iter(episode["transitions"])
        rounds = []
        for cycle in episode["cycle_records"]:
            guesses = self.decoder.pieces(cycle["proposals"])
            emitted = self.decoder.pieces(cycle["emitted_ids"])
            prefix, items = 1.0, []
            for i, (text, conf) in enumerate(zip(guesses, cycle["confidences"])):
                prefix *= conf
                items.append({"text": text, "confidence": conf, "combined": prefix,
                              "accepted": i < cycle["accepted"]})
            decisions = []
            for _ in cycle["decisions"]:
                t = next(transitions)
                q = self.learner.q.get(t["state"], [0.0, 0.0])
                decisions.append({"state": t["state"], "action": "DRAFT" if t["action"] else "STOP",
                                  "forced": t["forced"], "q_stop": q[0], "q_draft": q[1]})
            rounds.append({"guesses": items, "large_model_token": emitted[-1] if emitted else "",
                           "large_model_token_is_correction": len(emitted) <= len(guesses),
                           "decisions": decisions, "lag": cycle["lag_before"],
                           "ms": 1000 * cycle["wall_seconds"]})
        return rounds

    def info(self):
        return {"models": self.models, "policy": self.policy_info, "warning": self.model_warning,
                "methods": ["rl", "target_only", *[f"fixed_{b}" for b in range(1, 5)]],
                "default_methods": ["rl", "target_only", f"fixed_{self.best}"]}

    def close(self):
        self.decoder.close()


def make_handler(demo):
    page = (ROOT / "demo_page.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(200, demo.info())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/run":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    return self._send(400, {"error": "Type a prompt first."})
                tokens = max(8, min(int(body.get("tokens", 96)), 200))
                repeats = max(1, min(int(body.get("repeats", 1)), 5))
                allowed = set(demo.info()["methods"])
                methods = [m for m in body.get("methods", []) if m in allowed] or ["rl"]
                if "rl" not in methods:
                    methods.insert(0, "rl")
                self._send(200, demo.run(prompt, body.get("task", "auto"), tokens, methods, repeats))
            except Exception as error:  # shown in the page instead of crashing the server
                message = str(error)
                if "context" in message:
                    message = "Prompt plus answer is too long. Shorten the prompt or lower max tokens."
                self._send(500, {"error": message})

        def log_message(self, fmt, *args):
            pass

    return Handler


def print_result(summary):
    print(f"\nTask type: {summary['task']}   max tokens: {summary['tokens']}")
    for r in summary["results"]:
        extra = f"  x{r['speedup']:.2f} vs large model only" if r.get("speedup") else ""
        print(f"  {r['name']:18s} {r['tokens_per_second']:6.1f} tokens/s{extra}")
    rl = summary["results"][0]
    print(f"\nRL answer:\n{rl['text']}\n\nRL rounds (guesses → large model token):")
    for i, rnd in enumerate(rl["trace"], 1):
        guesses = " ".join(f"{g['text']!r}({g['confidence']:.2f}){'✓' if g['accepted'] else '✗'}"
                           for g in rnd["guesses"]) or "(no guess)"
        path = " ".join(d["action"][0] for d in rnd["decisions"])
        print(f"  {i:2d}. [{path}] {guesses} → {rnd['large_model_token']!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=Path, required=True, help="Qwen3-4B GGUF file")
    p.add_argument("--draft", type=Path, required=True, help="Qwen3-0.6B GGUF file")
    p.add_argument("--policy", type=Path, default=ROOT / "runs/main/policy.json", help="Trained Q-table")
    p.add_argument("--context", type=int, default=512, help="Room for prompt + answer, in tokens")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--prompt", help="Run one prompt in the terminal instead of starting the dashboard")
    p.add_argument("--task", default="auto", choices=("auto", "code", "fact"))
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--open", action="store_true", help="Open the dashboard in the default browser when ready")
    args = p.parse_args()
    demo = Demo(args)
    try:
        if args.prompt:
            print_result(demo.run(args.prompt, args.task, args.tokens, ["rl", "target_only", f"fixed_{demo.best}"], 1))
            return
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
        print(f"Dashboard: http://127.0.0.1:{args.port}   (Ctrl+C to stop)", flush=True)
        if args.open:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        demo.close()


if __name__ == "__main__":
    main()
