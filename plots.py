#!/usr/bin/env python3
"""Graphs and Markdown summaries generated only from recorded runs.

    python plots.py --run-dir runs/main [--eval test]
"""
import argparse
import json
from pathlib import Path

from agent import BUCKETS, CONF_THRESHOLDS, MAX_DRAFT, DRAFT, STOP, state_k

RL_COLOR, BASE_COLOR, ACCENT = "#167b81", "#8b9bb1", "#d17b0f"


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    return plt


def moving_mean(values, width=18):
    return [sum(values[max(0, i - width + 1):i + 1]) / len(values[max(0, i - width + 1):i + 1])
            for i in range(len(values))]


def bucket_labels():
    edges = [0.0, *CONF_THRESHOLDS, 1.0]
    return [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(BUCKETS)]


def top_prev(policy, task, n):
    """The n previous-cycle values most visited in k>=1 states of this task."""
    totals = {}
    for state, visits in policy["visits"].items():
        parts = state.split("|")
        if parts[0] == task and parts[1] != "k=0":
            prev = parts[-1].split("=", 1)[1]
            totals[prev] = totals.get(prev, 0) + sum(visits)
    return sorted(totals, key=totals.get, reverse=True)[:n]


def render_training(run_dir):
    run_dir = Path(run_dir)
    training = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
    policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    episodes, checks = training["episodes"], training["greedy_checks"]
    best = policy["best_fixed_budget"]
    plt = plotting()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    x = [e["episode"] for e in episodes]
    modeled = [e["modeled_tokens_per_second"] for e in episodes]
    axes[0].plot(x, modeled, alpha=.25, color=BASE_COLOR)
    axes[0].plot(x, moving_mean(modeled), color=RL_COLOR, label="behaviour policy, 18-episode mean")
    axes[0].plot(x, [e["rho"] for e in episodes], color=ACCENT, label="rho (greedy-cycle estimate)")
    axes[0].axhline(policy["profile_modeled_rates"][str(best)], color=ACCENT, ls="--", lw=.8,
                    label=f"fixed_{best} (profile)")
    axes[0].set(xlabel="Training episode", ylabel="Modeled tokens / s",
                title="Training (with exploration)")
    axes[0].legend(fontsize=7)
    # Modeled throughput is immune to GPU clock drift between the profile and later
    # checks; measured speed is shown faintly for reference.
    modeled_rates = policy["profile_modeled_rates"]
    axes[1].plot([c["episode"] for c in checks], [c["modeled_tokens_per_second"] for c in checks],
                 marker="o", color=RL_COLOR, label="greedy RL policy (modeled)")
    axes[1].plot([c["episode"] for c in checks], [c["tokens_per_second"] for c in checks],
                 marker=".", color=RL_COLOR, alpha=.35, label="greedy RL policy (measured)")
    axes[1].axhline(modeled_rates[str(best)], color=ACCENT, ls="--", label=f"fixed_{best} (modeled)")
    axes[1].axhline(modeled_rates["0"], color=BASE_COLOR, ls=":", label="target only (modeled)")
    axes[1].set(xlabel="Training episode", ylabel="Tokens / s (training prompts)",
                title="Frozen-greedy checks")
    axes[1].legend(fontsize=7)
    axes[2].plot(x, [e["epsilon"] for e in episodes], color=ACCENT)
    axes[2].set(xlabel="Training episode", ylabel="epsilon", title="Exploration schedule", ylim=(0, 1))
    fig.savefig(run_dir / "learning_curve.png")
    plt.close(fig)

    tasks = sorted({s.split("|")[0] for s in policy["q"]})
    fig, axes = plt.subplots(len(tasks), 2, figsize=(9, 2.6 * len(tasks)), constrained_layout=True, squeeze=False)
    for i, task in enumerate(tasks):
        for j, prev in enumerate(top_prev(policy, task, 2)):
            grid, notes = [], []
            for k in range(1, MAX_DRAFT):
                row, note = [], []
                for b in range(BUCKETS):
                    s = state_k(task, k, b, prev)
                    q, n = policy["q"].get(s, [0, 0]), policy["visits"].get(s, [0, 0])
                    advantage = q[DRAFT] - q[STOP] if n[DRAFT] and n[STOP] else float("nan")
                    row.append(advantage)
                    note.append("" if advantage != advantage else ("D" if advantage > 0 else "S"))
                grid.append(row)
                notes.append(note)
            ax = axes[i][j]
            image = ax.imshow(grid, cmap="BrBG", vmin=-1, vmax=1, aspect="auto")
            for r, note in enumerate(notes):
                for c, text in enumerate(note):
                    ax.text(c, r, text, ha="center", va="center", fontsize=8)
            ax.set(xticks=range(BUCKETS), xticklabels=bucket_labels(), yticks=range(MAX_DRAFT - 1),
                   yticklabels=range(1, MAX_DRAFT), xlabel="Product of draft confidences, tokens 1..k",
                   ylabel="Tokens drafted so far (k)", title=f"{task}, previous cycle: {prev}")
            ax.tick_params(axis="x", labelsize=7)
    fig.colorbar(image, ax=axes, label="Q(draft) - Q(stop)  [tokens]")
    fig.savefig(run_dir / "policy_map.png")
    plt.close(fig)


def render_eval(run_dir, out):
    run_dir, out = Path(run_dir), Path(out)
    summary = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    methods = summary["methods"]
    names = [m["method"] for m in methods]
    plt = plotting()

    fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    tasks = sorted(methods[0]["by_task"])
    groups = ["all", *tasks]
    width = 0.8 / len(groups)
    for g, group in enumerate(groups):
        values = [m["tokens_per_second"] if group == "all" else m["by_task"][group] for m in methods]
        ax.bar([i + (g - (len(groups) - 1) / 2) * width for i in range(len(names))], values, width,
               label=group, color=[RL_COLOR, ACCENT, BASE_COLOR][g % 3], alpha=1 if group == "all" else .55)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set(ylabel="Output tokens / generation second",
           title=f"Held-out comparison ({summary['prompt_count']} prompts x {summary['repeats']} repeats)")
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(out / "comparison.png")
    plt.close(fig)

    best = f"fixed_{summary['best_fixed_budget_from_training_profile']}"
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    for name, color in (("rl", RL_COLOR), (best, ACCENT)):
        counts = next(m for m in methods if m["method"] == name)["draft_length_counts"]
        total = sum(counts.values())
        offset = -0.2 if name == "rl" else 0.2
        ax.bar([int(k) + offset for k in counts], [v / total for v in counts.values()], 0.4,
               color=color, label=name)
    ax.set(xlabel="Tokens drafted in a cycle", ylabel="Share of cycles", xticks=range(MAX_DRAFT + 1),
           title="Draft lengths chosen")
    ax.legend(fontsize=7)
    fig.savefig(out / "draft_lengths.png")
    plt.close(fig)

    reference = "standalone_target" if "standalone_target" in names else "target_only"
    base = next(m for m in methods if m["method"] == reference)["tokens_per_second"]
    lines = [f"# Results: {summary['prompts']} ({summary['prompt_count']} prompts x {summary['repeats']} repeats)", "",
             f"Best fixed budget chosen on the training profile: **{best}**. "
             f"Speed-up column is relative to **{reference}**.", "",
             "| Method | Tokens/s | Speed-up | " + " | ".join(f"{t} tok/s" for t in tasks) +
             " | Mean draft | Acceptance | Same tokens as target |",
             "| --- | ---: | ---: | " + " | ".join("---:" for _ in tasks) + " | ---: | ---: | ---: |"]
    for m in methods:
        acc = f"{m['accepted'] / m['drafted']:.2f}" if m["drafted"] else "-"
        same = f"{m['exact_token_matches_vs_target_only']}/{m['runs']}" if "exact_token_matches_vs_target_only" in m else "-"
        label = f"**{m['method']}**" if m["method"] == "rl" else m["method"]
        lines.append(f"| {label} | {m['tokens_per_second']:.2f} | {m['tokens_per_second'] / base:.3f}x | " +
                     " | ".join(f"{m['by_task'][t]:.2f}" for t in tasks) +
                     f" | {m['mean_draft_length']:.2f} | {acc} | {same} |")
    lines += ["", "RL / baseline throughput ratio (prompt-cluster bootstrap 95% interval):", "",
              "| Baseline | Ratio | 95% interval |", "| --- | ---: | --- |"]
    for name, r in summary["rl_vs"].items():
        lines.append(f"| {name} | {r['ratio']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    rl = next(m for m in methods if m["method"] == "rl")
    lines += ["", f"RL decisions that hit an unvisited state (fallback rule): {rl['fallback_decisions']}.",
              "", summary["protocol"], "",
              "![comparison](comparison.png)", "", "![draft lengths](draft_lengths.png)", ""]
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval", help="Evaluation name to re-render, e.g. test")
    args = p.parse_args()
    render_training(args.run_dir)
    if args.eval:
        render_eval(args.run_dir, args.run_dir / f"eval_{args.eval}")


if __name__ == "__main__":
    main()
