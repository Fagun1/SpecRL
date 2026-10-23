#!/usr/bin/env python3
"""Compare the frozen RL policy with non-RL baselines on held-out prompts.

All resident methods share one process with both models loaded and use the same
bridge code path; prompt and method order are randomized per repeat. Afterwards
the target is measured alone in a fresh process that never loads the draft.
Speed is real wall-clock generation throughput (prefill excluded).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics

from agent import (CostModel, FixedController, QController, QLearner, RandomController,
                   load_prompts, run_episode, strip, write_json)
from train import SCHEMA, common_arguments, make_decoder, throughput

ROOT = Path(__file__).resolve().parent


def bootstrap_ratio(runs, method, baseline, rng, samples=5000):
    """Prompt-cluster bootstrap of aggregate-throughput ratio method/baseline."""
    by_prompt = {}
    for r in runs:
        by_prompt.setdefault(r["prompt_id"], {}).setdefault(r["method"], []).append(r)
    ids = sorted(by_prompt)

    def ratio(chosen):
        a = [r for i in chosen for r in by_prompt[i][method]]
        b = [r for i in chosen for r in by_prompt[i][baseline]]
        return throughput(a) / throughput(b)
    draws = sorted(ratio([rng.choice(ids) for _ in ids]) for _ in range(samples))
    return {"ratio": ratio(ids), "ci95": [draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]]}


def summarize(runs, method, reference):
    sel = [r for r in runs if r["method"] == method]
    ref = {(r["repeat"], r["prompt_id"]): r for r in runs if r["method"] == reference}
    lengths = [l for r in sel for l in r["draft_lengths"]]
    out = {"method": method, "runs": len(sel), "tokens_per_second": throughput(sel),
           "by_task": {t: throughput([r for r in sel if r["task"] == t]) for t in sorted({r["task"] for r in sel})},
           "mean_draft_length": statistics.mean(lengths) if lengths else 0,
           "drafted": sum(r["drafted"] for r in sel), "accepted": sum(r["accepted"] for r in sel),
           "draft_length_counts": dict(sorted(Counter(lengths).items())),
           "fallback_decisions": sum(r["label_counts"].get("fallback", 0) for r in sel)}
    if ref:
        out["exact_token_matches_vs_target_only"] = sum(
            r["token_ids"] == ref[r["repeat"], r["prompt_id"]]["token_ids"] for r in sel)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_arguments(p)
    p.add_argument("--prompts", type=Path, default=ROOT / "prompts/test_diverse.tsv")
    p.add_argument("--name", default="test_diverse", help="Output folder name inside the run directory")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--fixed", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--skip-standalone", action="store_true")
    args = p.parse_args()
    policy_path = args.run_dir / "policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != SCHEMA:
        p.error("policy.json was not produced by this train.py")
    if args.tokens != policy["tokens"]:
        p.error(f"Use --tokens {policy['tokens']} (as in training)")
    out = args.run_dir / f"eval_{args.name}"
    if (out / "comparison.json").exists():
        p.error(f"{out} already has results; pick a new --name")
    out.mkdir(parents=True, exist_ok=True)
    train_prompts = {r["prompt"] for r in load_prompts(ROOT / "prompts" / policy["provenance"]["train_prompts"])}
    rows = load_prompts(args.prompts)
    if any(r["prompt"] in train_prompts for r in rows):
        p.error("Evaluation prompts overlap training prompts")
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    cost = CostModel(**policy["cost_model"])
    learner = QLearner.from_dict(policy)
    best = policy["best_fixed_budget"]
    fixed = sorted(set(args.fixed) | {best})
    methods = ["target_only", *[f"fixed_{b}" for b in fixed], "random", "rl"]
    rng = random.Random(args.seed)
    runs = []

    decoder = make_decoder(args, "native-compare.log")
    if decoder.metadata["target_sha256"] != policy["provenance"]["models"]["target_sha256"] or \
            decoder.metadata["draft_sha256"] != policy["provenance"]["models"]["draft_sha256"]:
        decoder.close()
        p.error("Model files differ from training")
    try:
        warm_rows = load_prompts(ROOT / "prompts" / policy["provenance"]["train_prompts"])[:4]
        for row in warm_rows:
            for b in (0, best):
                run_episode(decoder, row, args.tokens, FixedController(b), cost)
        rl = QController(learner)
        with (out / "comparison_runs.jsonl").open("w", encoding="utf-8") as log:
            for repeat in range(1, args.repeats + 1):
                order = list(enumerate(rows, 1))
                rng.shuffle(order)
                for prompt_id, row in order:
                    names = list(methods)
                    rng.shuffle(names)
                    for name in names:
                        controller = (rl if name == "rl" else
                                      RandomController(random.Random(args.seed * 1000 + repeat * 100 + prompt_id))
                                      if name == "random" else
                                      FixedController(0 if name == "target_only" else int(name.split("_")[1])))
                        result = strip(run_episode(decoder, row, args.tokens, controller, cost))
                        record = {"repeat": repeat, "prompt_id": prompt_id, **result}
                        runs.append(record)
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"repeat {repeat} prompt {prompt_id:2d} {name:11s} "
                              f"{result['tokens_per_second']:6.2f} tok/s", flush=True)
    finally:
        decoder.close()

    if not args.skip_standalone:
        decoder = make_decoder(args, "native-standalone.log", standalone=True)
        try:
            for row in warm_rows:
                run_episode(decoder, row, args.tokens, FixedController(0), cost)
            alone = FixedController(0)
            alone.name = "standalone_target"
            for repeat in range(1, args.repeats + 1):
                order = list(enumerate(rows, 1))
                rng.shuffle(order)
                for prompt_id, row in order:
                    result = strip(run_episode(decoder, row, args.tokens, alone, cost))
                    runs.append({"repeat": repeat, "prompt_id": prompt_id, **result})
                    print(f"repeat {repeat} prompt {prompt_id:2d} standalone  "
                          f"{result['tokens_per_second']:6.2f} tok/s", flush=True)
        finally:
            decoder.close()
        methods = ["standalone_target", *methods]
        with (out / "comparison_runs.jsonl").open("a", encoding="utf-8") as log:
            for r in runs:
                if r["method"] == "standalone_target":
                    log.write(json.dumps(r, ensure_ascii=False) + "\n")

    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != policy_sha:
        raise RuntimeError("policy.json changed during evaluation")
    boot = random.Random(args.seed + 7)
    baselines = [m for m in methods if m != "rl"]
    summary = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "policy_sha256": policy_sha, "prompts": args.prompts.name,
              "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
              "prompt_count": len(rows), "repeats": args.repeats, "tokens": args.tokens,
              "best_fixed_budget_from_training_profile": best,
              "methods": [summarize(runs, m, "target_only") for m in methods],
              "rl_vs": {m: bootstrap_ratio(runs, "rl", m, boot) for m in baselines},
              "protocol": "Frozen greedy policy, no exploration or updates. Resident methods share one "
                          "process and code path; randomized prompt/method order. Standalone target "
                          "measured in a separate process without the draft model. Throughput = total "
                          "output tokens / total generation wall seconds (prefill excluded)."}
    write_json(out / "comparison.json", summary)
    from plots import render_eval
    render_eval(args.run_dir, out)
    print((out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
