"""Bootstrap a 95% confidence interval for execution accuracy from a saved run.

Resamples the per-question correct/wrong outcomes with replacement (default
2000 times) and reports the 2.5th / 97.5th percentiles. No model calls.

Usage: uv run python eval/bootstrap_ci.py eval/results/<run>.jsonl
"""

import json
import random
import sys
from pathlib import Path


def bootstrap_ci(correct: list[int], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    n = len(correct)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    point = sum(correct) / n
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return point, lo, hi


def main(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = [1 if r.get("correct") else 0 for r in rows]
    point, lo, hi = bootstrap_ci(correct)
    print(f"file: {Path(path).name}")
    print(f"n = {len(correct)}")
    print(f"accuracy = {point:.1%}")
    print(f"95% CI (bootstrap) = [{lo:.1%}, {hi:.1%}]  (±{(hi - lo) / 2:.1%})")


if __name__ == "__main__":
    main(sys.argv[1])
