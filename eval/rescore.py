"""Re-score saved eval runs with the permutation-tolerant comparator.

Re-executes the stored predicted and gold SQL against the same databases —
no LLM involved, so this takes seconds and can fairly re-score every past
run after a comparator fix.

Usage:
  uv run python eval/rescore.py                 # re-score all subset/full runs
"""

import glob
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from compare import results_match  # noqa: E402
from text2sql.config import load_config  # noqa: E402
from text2sql.guardrails import execute  # noqa: E402


def rescore_file(path: str, spider_dir: Path) -> None:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    old_correct = new_correct = 0

    for r in rows:
        old_correct += bool(r["correct"])
        is_correct = False
        if r["status"] == "ok" and r["predicted"]:
            db = str(spider_dir / "database" / r["db_id"] / f"{r['db_id']}.sqlite")
            try:
                _, pred_rows = execute(db, r["predicted"], timeout_s=30, max_rows=100_000)
                _, gold_rows = execute(db, r["gold"], timeout_s=30, max_rows=100_000)
                is_correct = results_match(pred_rows, gold_rows)
            except sqlite3.Error:
                is_correct = False  # a query that no longer runs counts as wrong
        r["correct"] = is_correct
        new_correct += is_correct

    out = path.replace(".jsonl", "_rescored.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    print(
        f"{Path(path).name}: {old_correct}/{n} ({old_correct / n:.1%}) "
        f"-> {new_correct}/{n} ({new_correct / n:.1%})"
    )


if __name__ == "__main__":
    cfg = load_config(ROOT / "config.yaml")
    spider_dir = ROOT / cfg["paths"]["spider_dir"]
    files = [
        f
        for f in sorted(glob.glob(str(ROOT / "eval" / "results" / "*.jsonl")))
        if "_rescored" not in f
    ]
    if not files:
        print("no result files found")
    for f in files:
        rescore_file(f, spider_dir)
