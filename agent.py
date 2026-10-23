"""SMDP definition, cost-model reward and average-reward Q-learning for adaptive drafting.

Decision points occur inside every speculative-decoding cycle:

* k = 0 (before drafting): DRAFT a first token, or STOP (= one target-only step).
* k >= 1 (after drafting token k): DRAFT one more token, or STOP and verify.
  At k = MAX_DRAFT only STOP is available.

State at k = 0:   (task, draft-lag bin, previous cycle)
State at k >= 1:  (task, k, bin of the product of the k draft confidences, previous cycle)
"Previous cycle" = how many tokens were drafted last cycle (none, 1, 2, 3+) and whether
all were accepted. The length matters because changing the verification batch size
costs extra GPU time (llama.cpp re-records its CUDA graph); the cost model charges it.

Objective: output tokens per second. Each decision earns tokens - rho * seconds,
where seconds come from a fitted cost model and rho is the running throughput
of the greedy policy (SMART average-reward Q-learning, Das et al. 1999).
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import statistics
import time

STOP, DRAFT = 0, 1
ACTION_NAMES = {STOP: "stop", DRAFT: "draft"}
MAX_DRAFT = 6
CONF_THRESHOLDS = (0.3, 0.6, 0.8, 0.95)
BUCKETS = len(CONF_THRESHOLDS) + 1
LENGTHS = ("1", "2", "3+")
PREV = ("none", *[f"{o}{d}" for o in ("rej", "all") for d in LENGTHS])


def conf_bucket(confidence):
    return sum(confidence >= t for t in CONF_THRESHOLDS)


def lag_bin(lag):
    return "2+" if lag >= 2 else str(lag)


def prev_outcome(drafted, accepted):
    if drafted == 0:
        return "none"
    length = "3+" if drafted >= 3 else str(drafted)
    return ("all" if accepted == drafted else "rej") + length


def state0(task, lag, prev):
    return f"{task}|k=0|lag={lag_bin(lag)}|prev={prev}"


def state_k(task, k, bucket, prev):
    return f"{task}|k={k}|conf={bucket}|prev={prev}"


def position(state):
    return int(state.split("|")[1][2:])


def available(state):
    return (STOP,) if position(state) >= MAX_DRAFT else (STOP, DRAFT)


def all_states(tasks):
    states = []
    for task in tasks:
        for prev in PREV:
            states += [state0(task, lag, prev) for lag in (0, 1, 2)]
            states += [state_k(task, k, b, prev) for k in range(1, MAX_DRAFT + 1) for b in range(BUCKETS)]
    return states


def load_prompts(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            task, prompt = line.split("\t", 1)
            rows.append({"task": task, "prompt": prompt})
    if not rows:
        raise ValueError(f"No prompts in {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


# --------------------------------------------------------------------------
# Time: a cost model fitted once on training-prompt profiling runs
# --------------------------------------------------------------------------

@dataclass
class CostModel:
    """Mean measured component times (seconds). Rewards use these expected times
    instead of each cycle's own noisy wall-clock time; only accepted-token counts
    vary. Means, not medians: GPU call times are right-skewed and the objective
    is expected throughput."""
    draft_first: list      # first draft eval, indexed by catch-up tokens 0, 1, 2, 3+
    draft_step: float      # each further draft token
    verify: list           # target verification, same batch size as the previous cycle
    overhead: float        # per-cycle IPC and bookkeeping
    verify_switch: list = None  # target verification after the batch size changed

    def first(self, catchup):
        return self.draft_first[min(catchup, len(self.draft_first) - 1)]

    def verify_time(self, drafted, prev_drafted):
        if self.verify_switch is None or prev_drafted == drafted:
            return self.verify[drafted]
        return self.verify_switch[drafted]

    def cycle_seconds(self, cycle):
        drafted = cycle["drafted"]
        seconds = self.verify_time(drafted, cycle.get("prev_drafted")) + self.overhead
        if drafted:
            seconds += self.first(cycle["catchup"]) + (drafted - 1) * self.draft_step
        return seconds

    @classmethod
    def fit(cls, cycles):
        def mean(values, name):
            if not values:
                raise ValueError(f"Profiling produced no samples for {name}")
            return statistics.mean(values)
        firsts = {c: [] for c in range(4)}
        steps, overhead = [], []
        same, switch = ({d: [] for d in range(MAX_DRAFT + 1)} for _ in range(2))
        for c in cycles:
            if c["draft_times"]:
                firsts[min(c["catchup"], 3)].append(c["draft_times"][0])
                steps += c["draft_times"][1:]
            (same if c.get("prev_drafted") == c["drafted"] else switch)[c["drafted"]].append(c["target_seconds"])
            overhead.append(c["wall_seconds"] - c["draft_seconds"] - c["target_seconds"])
        step = mean(steps, "draft steps")
        first = []
        for c in range(4):
            # Unseen catch-up sizes: previous size plus one draft step's work.
            first.append(statistics.mean(firsts[c]) if firsts[c] else first[-1] + step if first else step)
        # Switching penalty where both kinds were observed; used to fill gaps.
        gaps = [statistics.mean(switch[d]) - statistics.mean(same[d])
                for d in same if same[d] and switch[d]]
        penalty = statistics.mean(gaps) if gaps else 0.0
        same_t, switch_t = [], []
        for d in range(MAX_DRAFT + 1):
            s_mean = statistics.mean(same[d]) if same[d] else None
            w_mean = statistics.mean(switch[d]) if switch[d] else None
            if s_mean is None and w_mean is None:
                s_mean, w_mean = same_t[-1], switch_t[-1]
            same_t.append(s_mean if s_mean is not None else w_mean - penalty)
            switch_t.append(w_mean if w_mean is not None else s_mean + penalty)
        return cls(first, step, same_t, mean(overhead, "overhead"), switch_t)


# --------------------------------------------------------------------------
# Average-reward tabular Q-learning (SMART)
# --------------------------------------------------------------------------

@dataclass
class QLearner:
    alpha_min: float = 0.05
    rho: float = 50.0       # tokens / modeled second of the greedy policy
    rho_step: float = 0.05  # step size for the rho running estimate (per episode)
    q: dict = field(default_factory=dict)
    visits: dict = field(default_factory=dict)

    def row(self, state):
        if state not in self.q:
            self.q[state] = [0.0, 0.0]
            self.visits[state] = [0, 0]
        return self.q[state]

    def value(self, state):
        if state is None:
            return 0.0
        row = self.row(state)
        return max(row[a] for a in available(state))

    def reward(self, t):
        return t["tokens"] - self.rho * t["seconds"]

    def update(self, t):
        """Relative-value Q-learning step; returns the TD error."""
        state, action = t["state"], t["action"]
        row = self.row(state)
        self.visits[state][action] += 1
        # Sample-average step size early on, then a constant floor.
        alpha = max(self.alpha_min, 1.0 / self.visits[state][action])
        error = self.reward(t) + self.value(t["next_state"]) - row[action]
        row[action] += alpha * error
        return error

    def observe_greedy(self, tokens, seconds):
        """SMART: rho tracks reward per unit time of non-exploratory cycles."""
        if seconds > 0:
            self.rho += self.rho_step * (tokens / seconds - self.rho)

    def recenter(self, task):
        """Relative-value (RVI) anchoring. Average-reward Q-values are defined only
        up to a constant per task; without an anchor that constant drifts with rho,
        and rarely visited states fall behind. Shifting the whole task table so
        its reference state has value 0 keeps every state on the same level and
        never changes the greedy policy."""
        shift = self.value(state0(task, 1, "all1"))
        for state, row in self.q.items():
            if state.startswith(task + "|"):
                row[0] -= shift
                row[1] -= shift

    def greedy(self, state):
        """Best visited action; (action, used_fallback). Ties go to STOP."""
        self.row(state)
        tried = [a for a in available(state) if self.visits[state][a] > 0]
        if not tried:
            return (DRAFT if position(state) == 0 else STOP), True
        return max(tried, key=lambda a: (self.q[state][a], -a)), False

    def to_dict(self):
        return {"alpha_min": self.alpha_min, "rho": self.rho, "rho_step": self.rho_step,
                "q": self.q, "visits": self.visits}

    @classmethod
    def from_dict(cls, data):
        return cls(data["alpha_min"], data["rho"], data["rho_step"], data["q"], data["visits"])


# --------------------------------------------------------------------------
# Controllers: produce the k=0 choice plus the per-(k, bucket) continue table
# --------------------------------------------------------------------------

class Controller:
    name = "controller"

    def choose(self, state):
        """Return (action, label)."""
        raise NotImplementedError

    def plan(self, task, lag, prev):
        labels = {}
        s0 = state0(task, lag, prev)
        first, labels[s0] = self.choose(s0)
        bits = []
        for k in range(1, MAX_DRAFT):
            for b in range(BUCKETS):
                s = state_k(task, k, b, prev)
                action, labels[s] = self.choose(s)
                bits.append("1" if action == DRAFT else "0")
        return first, "".join(bits), labels


class FixedController(Controller):
    """Non-RL baseline: always draft exactly `budget` tokens (0 = target only)."""

    def __init__(self, budget):
        self.budget = budget
        self.name = "target_only" if budget == 0 else f"fixed_{budget}"
        self.cache = {}

    def choose(self, state):
        return (DRAFT if position(state) < self.budget else STOP), "fixed"

    def plan(self, task, lag, prev):
        key = (task, lag_bin(lag), prev)
        if key not in self.cache:
            self.cache[key] = super().plan(task, lag, prev)
        return self.cache[key]


class RandomController(Controller):
    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def choose(self, state):
        return self.rng.choice(available(state)), "random"


class QController(Controller):
    """Epsilon-greedy over the Q table. With epsilon=0 the policy is frozen and
    its tables are cached, so per-cycle Python cost matches the fixed baselines."""
    name = "rl"

    def __init__(self, learner, epsilon=0.0, rng=None):
        self.learner, self.epsilon, self.rng = learner, epsilon, rng or random.Random(0)
        self.cache = {}

    def choose(self, state):
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(available(state)), "explore"
        action, fallback = self.learner.greedy(state)
        return action, "fallback" if fallback else "greedy"

    def plan(self, task, lag, prev):
        if self.epsilon:
            return super().plan(task, lag, prev)
        key = (task, lag_bin(lag), prev)
        if key not in self.cache:
            self.cache[key] = super().plan(task, lag, prev)
        return self.cache[key]


# --------------------------------------------------------------------------
# Episode: one generation; converts bridge cycles into MDP transitions
# --------------------------------------------------------------------------

def cycle_transitions(task, prev, cycle, cost):
    """Decision-level transitions of one cycle with their earned tokens and
    modeled seconds. The last one (STOP) leads to the next cycle's k=0 state.
    Hitting the output-token cap is a truncation, so that state is still
    bootstrapped; only end-of-sequence is terminal (next_state None)."""
    lag = cycle["lag_before"]
    states = [state0(task, lag, prev) if k == 0 else state_k(task, k, b, prev)
              for k, b, _, _ in cycle["decisions"]]
    new_prev = prev_outcome(cycle["drafted"], cycle["accepted"])
    next0 = None if cycle["stop_reason"] == "eos" else state0(task, cycle["lag_after"], new_prev)
    out = []
    for i, (k, _, action, forced) in enumerate(cycle["decisions"]):
        if action == DRAFT:
            tokens, seconds = 0, cost.first(cycle["catchup"]) if k == 0 else cost.draft_step
            next_state = states[i + 1]
        else:
            tokens = len(cycle["emitted_ids"])
            seconds = cost.verify_time(cycle["drafted"], cycle.get("prev_drafted")) + cost.overhead
            next_state = next0
        out.append({"state": states[i], "action": action, "tokens": tokens, "seconds": seconds,
                    "next_state": next_state, "forced": bool(forced),
                    "truncated": bool(cycle["done"] and next_state is not None and action == STOP)})
    return out, new_prev


def run_episode(decoder, row, tokens, controller, cost, learner=None):
    prefill = decoder.reset(row["prompt"], tokens)
    task, prev = row["task"], "none"
    transitions, cycles = [], []
    greedy_tokens = greedy_seconds = 0.0
    start = time.perf_counter()
    while not decoder.done:
        first, bits, labels = controller.plan(task, decoder.lag, prev)
        cycle = decoder.cycle(first, bits)
        cycle["prev_drafted"] = cycles[-1]["drafted"] if cycles else None
        if cost is None:  # profiling, before the cost model exists
            steps, prev = [], prev_outcome(cycle["drafted"], cycle["accepted"])
        else:
            steps, prev = cycle_transitions(task, prev, cycle, cost)
        for t in steps:
            t["label"] = "forced" if t["forced"] else labels.get(t["state"], "?")
            if learner:
                t["td_error"] = learner.update(t)
        if steps and all(t["label"] != "explore" for t in steps):
            greedy_tokens += len(cycle["emitted_ids"])
            greedy_seconds += cost.cycle_seconds(cycle)
        transitions += steps
        cycles.append(cycle)
    generation = time.perf_counter() - start
    if learner:
        learner.observe_greedy(greedy_tokens, greedy_seconds)
        learner.recenter(task)
    produced = len(decoder.output)
    modeled = sum(cost.cycle_seconds(c) for c in cycles) if cost else None
    return {**row, "method": controller.name, "token_ids": list(decoder.output), "response": decoder.text(),
            "tokens_generated": produced, "prefill_seconds": prefill,
            "generation_seconds": generation, "tokens_per_second": produced / generation,
            "modeled_seconds": modeled, "modeled_tokens_per_second": produced / modeled if cost else None,
            "rho": learner.rho if learner else None,
            "cycles": len(cycles), "drafted": sum(c["drafted"] for c in cycles),
            "accepted": sum(c["accepted"] for c in cycles),
            "draft_lengths": [c["drafted"] for c in cycles],
            "decisions": len(transitions),
            "label_counts": {l: sum(t["label"] == l for t in transitions)
                             for l in sorted({t["label"] for t in transitions})},
            "stop_reason": cycles[-1]["stop_reason"],
            "cycle_records": cycles, "transitions": transitions}


def strip(result):
    """Episode summary without bulky per-cycle records."""
    return {k: v for k, v in result.items() if k not in ("cycle_records", "transitions")}
