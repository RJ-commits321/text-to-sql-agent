"""Evaluate the agent on the Spider dev subset and log everything to MLflow.

Execution accuracy: the predicted SQL and the gold SQL are both executed on the
same database; the row multisets must match (order-insensitive). This is the
standard Spider "execution accuracy" metric.

Usage:
  uv run python eval/run_eval.py --model qwen2.5:3b
  uv run python eval/run_eval.py --model llama3.2:3b --limit 20   # smoke test
  uv run python eval/run_eval.py --full                           # final 1,034-question run
  uv run python eval/run_eval.py --offtopic                       # refusal test on Chinook
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compare import results_match  # noqa: E402
from text2sql.agent import answer_question  # noqa: E402
from text2sql.config import load_config  # noqa: E402
from text2sql.guardrails import execute  # noqa: E402
from text2sql.prompts import PROMPT_VERSION  # noqa: E402
from text2sql.schema import get_schema  # noqa: E402


def run_spider(
    cfg: dict, model: str, limit: int | None, full: bool, source_file: str | None = None
) -> None:
    spider_dir = ROOT / cfg["paths"]["spider_dir"]
    if source_file:
        source = ROOT / source_file
    else:
        source = spider_dir / "dev.json" if full else ROOT / cfg["eval"]["subset_file"]
    items = json.loads(source.read_text(encoding="utf-8"))
    if limit:
        items = items[:limit]

    # eval uses huge limits so neither LIMIT injection nor the fetch cap can
    # distort the comparison against gold
    cfg = {
        **cfg,
        "guardrails": {
            **cfg["guardrails"],
            "row_limit": cfg["eval"]["row_limit"],
            "max_result_rows": cfg["eval"]["row_limit"],
        },
    }

    # Dynamic few-shot: retrieve all examples up front (embed model only), then free
    # the embed model, so the SQL model runs alone — avoids two models thrashing in RAM.
    example_map: dict[str, list] = {}
    if cfg["agent"].get("dynamic_fewshot"):
        import subprocess

        from text2sql.fewshot import retrieve_batch

        questions = [it["question"] for it in items]
        print(f"pre-retrieving few-shot examples for {len(questions)} questions...")
        batched = retrieve_batch(
            questions, str(ROOT / cfg["paths"]["fewshot_index"]), cfg["agent"].get("fewshot_k", 3)
        )
        example_map = dict(zip(questions, batched))
        subprocess.run(["ollama", "stop", "nomic-embed-text"], capture_output=True)  # noqa: S603,S607
        print("embed model unloaded; running generation.")

    schemas: dict[str, str] = {}
    results = []
    correct = ok = cannot = attempts_total = 0
    t0 = time.monotonic()

    for i, item in enumerate(items, 1):
        db_path = str(spider_dir / "database" / item["db_id"] / f"{item['db_id']}.sqlite")
        if item["db_id"] not in schemas:
            schemas[item["db_id"]] = get_schema(db_path)

        kwargs = {"examples": example_map[item["question"]]} if example_map else {}
        res = answer_question(
            item["question"], db_path, cfg, schema=schemas[item["db_id"]], model=model, **kwargs
        )
        attempts_total += res.attempts

        is_correct = False
        if res.status == "ok":
            ok += 1
            try:
                _, gold_rows = execute(db_path, item["query"], timeout_s=30, max_rows=100_000)
                is_correct = results_match(res.rows, gold_rows)
            except Exception as e:
                print(f"  [warn] gold query failed on {item['db_id']}: {e}")
        elif res.status == "cannot_answer":
            cannot += 1
        correct += is_correct

        results.append(
            {
                "db_id": item["db_id"],
                "question": item["question"],
                "gold": item["query"],
                "predicted": res.sql,
                "status": res.status,
                "correct": is_correct,
                "attempts": res.attempts,
                "latency_s": round(res.latency_s, 2),
                "error": res.error,
            }
        )
        print(f"[{i}/{len(items)}] acc={correct / i:.1%} | {item['question'][:60]}")

    n = len(items)
    metrics = {
        "execution_accuracy": correct / n,
        "valid_sql_rate": ok / n,
        "cannot_answer_rate": cannot / n,
        "avg_attempts": attempts_total / n,
        "avg_latency_s": (time.monotonic() - t0) / n,
    }

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(source_file).stem if source_file else ("full" if full else f"subset{n}")
    out_file = out_dir / f"{model.replace(':', '_')}_{tag}_{int(time.time())}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mlflow.set_experiment("text2sql-spider")
    with mlflow.start_run(run_name=f"{model}-{tag}"):
        mlflow.log_params(
            {
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "num_ctx": cfg["llm"]["num_ctx"],
                "temperature": cfg["llm"]["temperature"],
                "max_attempts": cfg["agent"]["max_attempts"],
                "self_consistency": cfg["agent"].get("self_consistency", 1),
                "dynamic_fewshot": cfg["agent"].get("dynamic_fewshot", False),
                "n_questions": n,
                "dataset": tag,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_file))

    print("\n=== results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print(f"per-question details: {out_file}")


def run_offtopic(cfg: dict, model: str) -> None:
    """Off-topic questions asked against Chinook — correct behaviour is refusal."""
    items = [
        json.loads(line)
        for line in (ROOT / "eval" / "offtopic.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    db_path = str(ROOT / cfg["paths"]["demo_db"])
    schema = get_schema(db_path)

    refused = 0
    for i, item in enumerate(items, 1):
        res = answer_question(item["question"], db_path, cfg, schema=schema, model=model)
        is_refusal = res.status in ("cannot_answer", "failed")
        refused += is_refusal
        print(f"[{i}/{len(items)}] {'REFUSED' if is_refusal else 'ANSWERED'} | {item['question']}")

    rate = refused / len(items)
    mlflow.set_experiment("text2sql-offtopic")
    with mlflow.start_run(run_name=f"{model}-offtopic"):
        mlflow.log_params(
            {"model": model, "prompt_version": PROMPT_VERSION, "n_questions": len(items)}
        )
        mlflow.log_metric("refusal_rate", rate)
    print(f"\nrefusal_rate: {rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="ollama model (default: config.yaml)")
    parser.add_argument("--limit", type=int, default=None, help="only first N questions")
    parser.add_argument("--full", action="store_true", help="full 1,034-question dev set")
    parser.add_argument("--offtopic", action="store_true", help="run the refusal test instead")
    parser.add_argument("--sc", type=int, default=None, help="self-consistency samples (e.g. 3)")
    parser.add_argument("--set", dest="set_file", default=None,
                        help="run on a specific question file, e.g. eval/test_holdout.json")
    parser.add_argument("--dynamic", action="store_true", help="enable dynamic few-shot retrieval")
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    if args.sc:
        cfg["agent"]["self_consistency"] = args.sc
    if args.dynamic:
        cfg["agent"]["dynamic_fewshot"] = True
    model = args.model or cfg["llm"]["model"]

    if args.offtopic:
        run_offtopic(cfg, model)
    else:
        run_spider(cfg, model, args.limit, args.full, args.set_file)
