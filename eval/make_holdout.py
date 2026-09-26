"""Build a clean held-out TEST set for honest measurement.

The 200-question dev_subset.json was used during prompt iteration (we looked at
its failures), so it is no longer truly unseen. This script samples a separate
~200 questions from the *remaining* dev questions (the 834 we never tuned on),
seeded and stratified by database. Same 20 dev databases, comparable to the
reported subset-200 number, but never used for tuning -> a fair held-out.

Usage: uv run python eval/make_holdout.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text2sql.config import load_config  # noqa: E402

HOLDOUT_SIZE = 200
HOLDOUT_SEED = 123  # distinct from the tuned subset's seed (42)


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    dev = json.loads((ROOT / cfg["paths"]["spider_dir"] / "dev.json").read_text(encoding="utf-8"))
    tuned = json.loads((ROOT / cfg["eval"]["subset_file"]).read_text(encoding="utf-8"))

    # exclude the exact questions used during tuning
    used = {(t["db_id"], t["question"], t["query"]) for t in tuned}
    pool = [d for d in dev if (d["db_id"], d["question"], d["query"]) not in used]
    print(f"dev={len(dev)}  tuned-out={len(tuned)}  eligible pool={len(pool)}")

    by_db: dict[str, list] = defaultdict(list)
    for item in pool:
        by_db[item["db_id"]].append(item)

    rng = random.Random(HOLDOUT_SEED)
    holdout: list[dict] = []
    for db_id in sorted(by_db):
        group = by_db[db_id]
        take = max(1, round(len(group) * HOLDOUT_SIZE / len(pool)))
        holdout.extend(rng.sample(group, min(take, len(group))))
    rng.shuffle(holdout)
    holdout = holdout[:HOLDOUT_SIZE]

    slim = [{"question": i["question"], "query": i["query"], "db_id": i["db_id"]} for i in holdout]
    out = ROOT / "eval" / "test_holdout.json"
    out.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

    # sanity: zero overlap with the tuned subset
    overlap = len({(i["db_id"], i["question"]) for i in slim} & {(t["db_id"], t["question"]) for t in tuned})
    print(f"wrote {len(slim)} held-out questions across {len({i['db_id'] for i in slim})} dbs -> {out}")
    print(f"overlap with tuned subset: {overlap} (must be 0)")


if __name__ == "__main__":
    main()
