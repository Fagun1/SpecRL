#!/usr/bin/env python3
"""Profile the GPU cost model on training prompts, then train the Q-learning drafter.

Phases (one model load):
  1. unscored warm-up
  2. profiling: every training prompt with fixed budgets 0..MAX_DRAFT plus random
     drafting -> cost model (time per decision), best fixed budget
  3. epsilon-greedy average-reward Q-learning with experience replay, with
     periodic frozen-greedy checks on the training prompts for the learning curve
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics

import agent
from agent import (CONF_THRESHOLDS, MAX_DRAFT, CostModel, FixedController, QController,
                   QLearner, RandomController, load_prompts, run_episode, strip, write_json)

ROOT = Path(__file__).resolve().parent
SCHEMA = "ltd-adaptive-draft-q-v3"


def common_arguments(p):
    p.add_argument("--target", type=Path, required=True, help="Qwen3-4B GGUF file")
    p.add_argument("--draft", type=Path, required=True, help="Qwen3-0.6B GGUF file")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/main")
    p.add_argument("--tokens", type=int, default=48)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)


def throughput(results):
    return sum(r["tokens_generated"] for r in results) / sum(r["generation_seconds"] for r in results)


def make_decoder(args, log_name, standalone=False):
    from native_decoding import NativeDecoder
    decoder = NativeDecoder(args.target, args.draft, args.context, args.threads,
                            log_path=args.run_dir / log_name, standalone=standalone)
    decoder.configure(MAX_DRAFT, CONF_THRESHOLDS)
    return decoder


def modeled_throughput(results):
    return sum(r["tokens_generated"] for r in results) / sum(r["modeled_seconds"] for r in results)


def subset(rows, limit=24):
    """Every n-th prompt, so profiling and checks stay short with many prompts."""
    step = -(-len(rows) // limit)
    return rows[::step]


def greedy_check(decoder, rows, tokens, learner, cost):
    results = [run_episode(decoder, row, tokens, QController(learner), cost) for row in subset(rows, 16)]
    return {"tokens_per_second": throughput(results),
            "modeled_tokens_per_second": modeled_throughput(results), "rho": learner.rho,
            "mean_draft_length": statistics.mean(l for r in results for l in r["draft_lengths"]),
            "fallback_decisions": sum(r["label_counts"].get("fallback", 0) for r in results)}


def profile(decoder, rows, tokens, rng):
    plan = [(row, FixedController(b)) for row in rows for b in range(MAX_DRAFT + 1)]
    plan += [(row, RandomController(random.Random(rng.random()))) for row in rows]
    rng.shuffle(plan)
    results = []
    for i, (row, controller) in enumerate(plan, 1):
        result = run_episode(decoder, row, tokens, controller, None)
        results.append(result)
        print(f"profile {i}/{len(plan)} {row['task']} {controller.name}: "
              f"{result['tokens_per_second']:.2f} tok/s", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_arguments(p)
    p.add_argument("--train-prompts", type=Path, default=ROOT / "prompts/train_diverse.tsv")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--epsilon-start", type=float, default=0.5)
    p.add_argument("--epsilon-end", type=float, default=0.02)
    p.add_argument("--alpha-min", type=float, default=0.05)
    p.add_argument("--rho-step", type=float, default=0.05, help="Step size of the average-reward estimate")
    p.add_argument("--replay-ratio", type=int, default=4, help="Replay updates per real transition")
    p.add_argument("--check-every", type=int, default=50,
                   help="Episodes between greedy checks; the best checked Q-table is saved (0 = keep the last)")
    args = p.parse_args()
    if (args.run_dir / "policy.json").exists():
        p.error(f"{args.run_dir} already has a policy; pick a new --run-dir")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompts(args.train_prompts)
    rng = random.Random(args.seed)
    decoder = make_decoder(args, "native-train.log")
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "models": decoder.metadata,
            "train_prompts": args.train_prompts.name,
            "train_prompts_sha256": hashlib.sha256(args.train_prompts.read_bytes()).hexdigest()}
    try:
        warm = [strip(run_episode(decoder, row, args.tokens, FixedController(b), None))
                for row in rows[:3] for b in (0, 2, MAX_DRAFT)]
        write_json(args.run_dir / "warmup.json", warm)

        # ---- profiling -> cost model -------------------------------------------
        prof = profile(decoder, subset(rows), args.tokens, rng)
        rates = {str(b): throughput([r for r in prof if r["method"] == FixedController(b).name])
                 for b in range(MAX_DRAFT + 1)}
        best_fixed = max(range(1, MAX_DRAFT + 1), key=lambda b: rates[str(b)])
        cost = CostModel.fit([c for r in prof for c in r["cycle_records"]])
        for r in prof:
            r["modeled_seconds"] = sum(cost.cycle_seconds(c) for c in r["cycle_records"])
        modeled = {str(b): modeled_throughput([r for r in prof if r["method"] == FixedController(b).name])
                   for b in range(MAX_DRAFT + 1)}
        errors = [(r["modeled_seconds"] - r["generation_seconds"]) / r["generation_seconds"] for r in prof]
        write_json(args.run_dir / "profile.json", {
            "rates_tokens_per_second": rates, "modeled_rates_tokens_per_second": modeled,
            "best_fixed_budget": best_fixed, "cost_model": asdict(cost),
            "cost_model_check": {"mean_relative_error": statistics.mean(errors),
                                 "mean_abs_relative_error": statistics.mean(abs(e) for e in errors)},
            "episodes": [strip(r) for r in prof],
            "cycles": [{k: c[k] for k in ("drafted", "prev_drafted", "accepted", "catchup", "draft_times",
                                          "draft_seconds", "target_seconds", "wall_seconds")}
                       for r in prof for c in r["cycle_records"]]})
        print(json.dumps({"rates": rates, "modeled_rates": modeled, "best_fixed": best_fixed,
                          "cost_model": asdict(cost), "cost_model_mean_abs_error":
                          statistics.mean(abs(e) for e in errors)}, indent=2), flush=True)

        # ---- Q-learning ---------------------------------------------------------
        # rho starts at the best fixed budget's modeled throughput (training profile).
        learner = QLearner(args.alpha_min, modeled[str(best_fixed)], args.rho_step)
        for s in agent.all_states(sorted({r["task"] for r in rows})):
            learner.row(s)
        replay, replay_rng = [], random.Random(args.seed + 1)
        summaries, checks = [], [{"episode": 0, **greedy_check(decoder, rows, args.tokens, learner, cost)}]
        best = None  # best trained checkpoint; the untrained table at episode 0 is never chosen
        schedule = []
        with (args.run_dir / "transitions.jsonl").open("w", encoding="utf-8") as log:
            for episode in range(1, args.episodes + 1):
                if not schedule:
                    schedule = list(rows)
                    rng.shuffle(schedule)
                row = schedule.pop()
                frac = (episode - 1) / max(1, args.episodes - 1)
                epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)
                result = run_episode(decoder, row, args.tokens,
                                     QController(learner, epsilon, rng), cost, learner)
                for t in result["transitions"]:
                    log.write(json.dumps({"episode": episode, **t}) + "\n")
                    replay.append(t)
                # Replay reuses genuine transitions (stored as tokens and modeled
                # seconds, so rewards use the current rho). No extra model runs.
                for _ in range(args.replay_ratio * len(result["transitions"])):
                    learner.update(replay_rng.choice(replay))
                learner.recenter(row["task"])
                summaries.append({"episode": episode, "epsilon": epsilon, **strip(result)})
                print(f"episode {episode}/{args.episodes} {row['task']}: {result['tokens_per_second']:.2f} tok/s "
                      f"(modeled {result['modeled_tokens_per_second']:.2f}), rho {learner.rho:.2f}, mean draft "
                      f"{statistics.mean(result['draft_lengths']):.2f}, eps {epsilon:.2f}", flush=True)
                if args.check_every and (episode % args.check_every == 0 or episode == args.episodes):
                    checks.append({"episode": episode, **greedy_check(decoder, rows, args.tokens, learner, cost)})
                    print(f"greedy check after {episode}: {checks[-1]['tokens_per_second']:.2f} tok/s, modeled "
                          f"{checks[-1]['modeled_tokens_per_second']:.2f} (fixed_{best_fixed} modeled: "
                          f"{modeled[str(best_fixed)]:.2f})", flush=True)
                    # Keep the best Q-table seen so far. Scored with the cost model on
                    # training prompts, so GPU heat does not affect the choice.
                    if best is None or checks[-1]["modeled_tokens_per_second"] > best["modeled_tokens_per_second"]:
                        best = {"episode": episode,
                                "modeled_tokens_per_second": checks[-1]["modeled_tokens_per_second"],
                                "learner": copy.deepcopy(learner.to_dict())}
                write_json(args.run_dir / "training.json", {"episodes": summaries, "greedy_checks": checks})
    finally:
        decoder.close()
    selected = best or {"episode": args.episodes, "learner": learner.to_dict()}
    print(f"Selected the Q-table from episode {selected['episode']}", flush=True)
    policy = {"schema": SCHEMA,
              "algorithm": "average-reward tabular Q-learning (SMART rho, RVI anchoring; "
                           "epsilon-greedy, experience replay)",
              "max_draft": MAX_DRAFT, "confidence_thresholds": list(CONF_THRESHOLDS),
              "tokens": args.tokens, "seed": args.seed, "episodes": args.episodes,
              "epsilon_start": args.epsilon_start, "epsilon_end": args.epsilon_end,
              "replay_ratio": args.replay_ratio, "replay_updates": args.replay_ratio * len(replay),
              "real_transitions": len(replay),
              "state_space": {"k=0": "(task, draft lag 0/1/2+, previous cycle: none or all/rej x length 1/2/3+)",
                              "k>=1": "(task, k, cumulative draft-confidence bin, previous cycle)",
                              "states": len(learner.q)},
              "action_space": "STOP (verify now) or DRAFT one more token",
              "reward": "tokens emitted - rho * modeled seconds of the decision; rho = running "
                        "throughput of greedy cycles; token cap treated as truncation",
              "cost_model": asdict(cost), "best_fixed_budget": best_fixed, "profile_rates": rates,
              "profile_modeled_rates": modeled,
              "selected_episode": selected["episode"],
              "selection": "best greedy check on training prompts (cost-model tokens/s)",
              "provenance": info, **selected["learner"]}
    write_json(args.run_dir / "policy.json", policy)
    from plots import render_training
    render_training(args.run_dir)
    print(f"Saved trained policy to {args.run_dir / 'policy.json'}")


if __name__ == "__main__":
    main()
