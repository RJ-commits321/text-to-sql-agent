"""Build the fixed development subset of Spider dev (default 200 questions).

Sampling is seeded and stratified proportionally by database, then the file is
committed so every eval run scores the exact same questions. The full 1,034-
question dev set is reserved for the single final run.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from text2sql.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    dev_path = ROOT / cfg["paths"]["spider_dir"] / "dev.json"
    subset_path = ROOT / cfg["eval"]["subset_file"]
    size = cfg["eval"]["subset_size"]

    items = json.loads(dev_path.read_text(encoding="utf-8"))
    print(f"Spider dev: {len(items)} questions, {len({i['db_id'] for i in items})} databases")

    by_db: dict[str, list] = defaultdict(list)
    for item in items:
        by_db[item["db_id"]].append(item)

    rng = random.Random(cfg["eval"]["seed"])
    subset: list[dict] = []
    for db_id in sorted(by_db):
        group = by_db[db_id]
        take = max(1, round(len(group) * size / len(items)))
        subset.extend(rng.sample(group, min(take, len(group))))

    rng.shuffle(subset)
    subset = subset[:size]

    slim = [
        {"question": i["question"], "query": i["query"], "db_id": i["db_id"]} for i in subset
    ]
    subset_path.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(slim)} questions across {len({i['db_id'] for i in slim})} dbs -> {subset_path}")


if __name__ == "__main__":
    main()
