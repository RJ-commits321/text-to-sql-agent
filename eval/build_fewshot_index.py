"""Build the dynamic few-shot retrieval index (run once).

Embeds every Spider-train question with nomic-embed-text and saves the vectors
(+ their questions and SQL) to config paths.fewshot_index. The Spider *train*
split is used, which is disjoint from the dev/test questions we evaluate on.

Usage: uv run python eval/build_fewshot_index.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text2sql.config import load_config  # noqa: E402
from text2sql.fewshot import build_index  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    train = ROOT / cfg["paths"]["spider_train"]
    out = ROOT / cfg["paths"]["fewshot_index"]
    print(f"embedding Spider-train questions from {train.name} ...")
    t0 = time.monotonic()
    n = build_index(str(train), str(out))
    print(f"indexed {n} examples in {time.monotonic() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
