"""Logic checks with a fake decoder that mirrors bridge.cpp's CYCLE semantics.
These verify bookkeeping only; they make no performance claims."""
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from agent import (BUCKETS, DRAFT, MAX_DRAFT, STOP, CostModel, FixedController, QController,
                   QLearner, RandomController, run_episode, state0, state_k)


class FakeDecoder:
    """Mirrors bridge.cpp: buckets use the product of draft confidences so far.
    Draft confidence follows a two-state Markov chain over output positions
    (confident runs persist); a confident token is accepted with probability
    0.95, an unsure one with 0.2. Seeded, so episodes are reproducible."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def position(self, pos):
        while len(self.chain) <= pos:
            high = self.rng.random() < (0.85 if self.chain[-1][0] >= 0.5 else 0.3)
            conf = 0.99 if high else 0.4
            self.chain.append((conf, self.rng.random() < (0.95 if high else 0.2)))
        return self.chain[pos]

    def reset(self, prompt, limit):
        self.limit, self.output, self.done, self.lag = limit, [], False, 0
        self.chain = [(0.99, True)]
        return 0.0

    def cycle(self, draft_first, bits):
        remaining = self.limit - len(self.output)
        cap = min(MAX_DRAFT, remaining - 1)
        lag_before = self.lag
        decisions, confs, times = [], [], []
        if draft_first and cap > 0:
            decisions.append([0, -1, 1, 0])
            catchup = self.lag
            k, prefix = 1, 1.0
            while True:
                confs.append(self.position(len(self.output) + k - 1)[0])
                times.append(0.006 + (0.001 * catchup if k == 1 else 0))
                prefix *= confs[-1]
                b = agent.conf_bucket(prefix)
                if k >= cap:
                    decisions.append([k, b, 0, 1])
                    break
                action = int(bits[(k - 1) * BUCKETS + b])
                decisions.append([k, b, action, 0])
                if not action:
                    break
                k += 1
        else:
            catchup = 0
            decisions.append([0, -1, 0, int(bool(draft_first))])
        accepted = 0
        for k in range(1, len(confs) + 1):
            if not self.position(len(self.output) + k - 1)[1]:
                break
            accepted += 1
        emitted = accepted + 1
        self.output += [7] * emitted
        drafted = len(confs)
        self.lag = 1 if drafted and accepted == drafted else (0 if drafted else self.lag + 1)
        self.done = len(self.output) >= self.limit
        return {"drafted": drafted, "accepted": accepted, "emitted_ids": [7] * emitted,
                "confidences": confs, "decisions": decisions, "draft_times": times,
                "draft_seconds": sum(times), "target_seconds": 0.02 + 0.002 * drafted,
                "wall_seconds": sum(times) + 0.021 + 0.002 * drafted, "seconds": 0,
                "lag_before": lag_before, "catchup": catchup, "lag_after": self.lag,
                "done": self.done, "stop_reason": "token_limit" if self.done else None}

    def text(self):
        return ""


COST = CostModel([0.006, 0.007, 0.008, 0.009], 0.006, [0.02 + 0.002 * d for d in range(MAX_DRAFT + 1)],
                 0.001)
ROW = {"task": "code", "prompt": "x"}


class AgentTests(unittest.TestCase):
    def test_states_and_actions(self):
        self.assertEqual(agent.available(state_k("code", MAX_DRAFT, 2, "all")), (STOP,))
        self.assertEqual(agent.available(state0("code", 5, "none")), (STOP, DRAFT))
        self.assertEqual(state0("fact", 7, "none"), "fact|k=0|lag=2+|prev=none")
        self.assertEqual([agent.conf_bucket(c) for c in (0.1, 0.3, 0.7, 0.9, 0.99)], [0, 1, 2, 3, 4])
        self.assertEqual(len(agent.all_states(["code", "fact"])), 2 * 7 * (3 + MAX_DRAFT * BUCKETS))
        self.assertEqual([agent.prev_outcome(d, a) for d, a in ((0, 0), (1, 1), (2, 1), (5, 5))],
                         ["none", "all1", "rej2", "all3+"])

    def test_fixed_controller_drafts_exact_budget(self):
        for budget in range(MAX_DRAFT + 1):
            result = run_episode(FakeDecoder(), ROW, 30, FixedController(budget), COST)
            self.assertTrue(all(l == budget or l < budget for l in result["draft_lengths"]))
            self.assertEqual(result["tokens_generated"], 30)

    def test_transition_chain_and_rewards(self):
        result = run_episode(FakeDecoder(), ROW, 20, FixedController(3), COST)
        ts = result["transitions"]
        for a, b in zip(ts, ts[1:]):
            self.assertEqual(a["next_state"], b["state"])
        # The token cap is a truncation: the last transition is still bootstrapped.
        self.assertTrue(ts[-1]["truncated"])
        self.assertIsNotNone(ts[-1]["next_state"])
        self.assertEqual(sum(t["tokens"] for t in ts), 20)
        self.assertAlmostEqual(sum(t["seconds"] for t in ts), result["modeled_seconds"], places=9)

    def test_q_learning_learns_confidence_rule(self):
        learner = QLearner(0.05)
        rng = random.Random(0)
        for episode in range(600):
            epsilon = max(0.05, 0.6 - episode / 600)
            run_episode(FakeDecoder(episode), ROW, 40, QController(learner, epsilon, rng), COST, learner)
        self.assertGreater(learner.rho, 50)
        self.assertAlmostEqual(learner.value(state0("code", 1, "all1")), 0.0)
        # Unsure tokens are usually rejected: the policy should keep drafting far
        # more often after confident tokens than after unsure ones.
        rate = {}
        for bucket in (agent.conf_bucket(0.4), agent.conf_bucket(0.99)):
            choices = [t["action"] for i in range(50)
                       for t in run_episode(FakeDecoder(20_000 + i), ROW, 40, QController(learner), COST)["transitions"]
                       if not t["forced"] and agent.position(t["state"]) > 0
                       and t["state"].split("|")[2] == f"conf={bucket}"]
            rate[bucket] = sum(choices) / len(choices)
        self.assertLess(rate[agent.conf_bucket(0.4)] + 0.3, rate[agent.conf_bucket(0.99)])

        def modeled_rate(controller):
            results = [run_episode(FakeDecoder(10_000 + i), ROW, 40, controller, COST) for i in range(200)]
            return sum(r["tokens_generated"] for r in results) / sum(r["modeled_seconds"] for r in results)
        best_fixed = max(modeled_rate(FixedController(b)) for b in range(MAX_DRAFT + 1))
        self.assertGreater(modeled_rate(QController(learner)), 1.05 * best_fixed)

    def test_cost_model_fit(self):
        cycles = run_episode(FakeDecoder(), ROW, 60, RandomController(random.Random(1)), None)["cycle_records"]
        cycles += run_episode(FakeDecoder(), ROW, 40, FixedController(MAX_DRAFT), None)["cycle_records"]
        cost = CostModel.fit(cycles)
        self.assertAlmostEqual(cost.draft_step, 0.006)
        self.assertAlmostEqual(cost.verify[2], 0.024)
        self.assertEqual(len(cost.draft_first), 4)

    def test_switch_cost(self):
        cost = CostModel([0.006] * 4, 0.006, [0.02] * (MAX_DRAFT + 1), 0.0, [0.022] * (MAX_DRAFT + 1))
        self.assertEqual(cost.verify_time(2, 2), 0.02)
        self.assertEqual(cost.verify_time(2, 1), 0.022)
        self.assertEqual(cost.verify_time(2, None), 0.022)
        legacy = CostModel([0.006] * 4, 0.006, [0.02] * (MAX_DRAFT + 1), 0.0)
        self.assertEqual(legacy.verify_time(2, 1), 0.02)

    def test_plan_bits_layout(self):
        first, bits, _ = FixedController(2).plan("code", 0, "none")
        self.assertEqual(first, DRAFT)
        self.assertEqual(len(bits), (MAX_DRAFT - 1) * BUCKETS)
        self.assertEqual(bits[:BUCKETS], "1" * BUCKETS)
        self.assertEqual(set(bits[BUCKETS:]), {"0"})


if __name__ == "__main__":
    unittest.main()
