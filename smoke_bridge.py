"""Check that the native bridge loads both models on the GPU and runs a few cycles."""
import argparse
import json
from pathlib import Path

from native_decoding import NativeDecoder

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    args = p.parse_args()
    d = NativeDecoder(args.target, args.draft, log_path=ROOT / "runs/smoke/native-runtime.log")
    try:
        print(json.dumps(d.metadata, indent=2))
        d.configure(6, [0.3, 0.6, 0.8, 0.95])
        d.reset("Write a Python function that adds two integers.", 48)
        always, two = "1" * 25, "1" * 5 + "0" * 20
        i = 0
        while not d.done:
            c = d.cycle(i % 3 != 2, always if i % 2 else two)
            print({k: c[k] for k in ("drafted", "accepted", "lag_before", "catchup", "lag_after", "decisions")},
                  "conf", [round(x, 3) for x in c["confidences"]],
                  f"draft {1000 * c['draft_seconds']:.1f} ms, target {1000 * c['target_seconds']:.1f} ms, "
                  f"wall {1000 * c['wall_seconds']:.1f} ms")
            i += 1
        print(repr(d.text()))
    finally:
        d.close()


if __name__ == "__main__":
    main()
