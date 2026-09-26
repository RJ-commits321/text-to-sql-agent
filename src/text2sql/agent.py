"""The core agent loop: generate SQL, validate, execute, and feed errors back
to the model for self-correction. The database is the feedback signal.

Optional self-consistency mode (config agent.self_consistency > 1): sample N
independent queries at temperature > 0, execute each, and answer with the
majority result — two wrong queries rarely agree, so agreement is evidence of
correctness. Falls back to the standard greedy loop if no sample succeeds."""

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import llm
from .guardrails import GuardrailError, enforce_limit, execute, extract_sql, validate
from .prompts import CANNOT_ANSWER, build_messages, error_feedback
from .schema import column_catalog, get_schema


@dataclass
class AgentResult:
    question: str
    status: str  # "ok" | "cannot_answer" | "failed"
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    attempts: int = 0
    latency_s: float = 0.0
    error: str | None = None


_AUTO = object()  # sentinel: retrieve examples internally unless the caller supplies them


def answer_question(
    question: str,
    db_path: str,
    cfg: dict,
    schema: str | None = None,
    model: str | None = None,
    examples=_AUTO,
) -> AgentResult:
    schema = schema or get_schema(db_path)
    model = model or cfg["llm"]["model"]
    if examples is _AUTO:  # app path: retrieve live; eval pre-retrieves and passes them in
        examples = _retrieve_examples(question, cfg)

    k = cfg["agent"].get("self_consistency", 1)
    if k > 1:
        return _answer_self_consistency(question, db_path, cfg, schema, model, k, examples)
    return _answer_greedy(question, db_path, cfg, schema, model, examples)


def _retrieve_examples(question: str, cfg: dict) -> list[tuple[str, str]] | None:
    """Top-k similar Spider-train examples when dynamic few-shot is enabled, else None."""
    if not cfg["agent"].get("dynamic_fewshot"):
        return None
    from .fewshot import retrieve

    index_path = str(Path(cfg["paths"]["fewshot_index"]))
    return retrieve(question, index_path, k=cfg["agent"].get("fewshot_k", 3))


def _validated_execute(
    sql: str, db_path: str, cfg: dict
) -> tuple[str, list[str], list[tuple]]:
    """The one guardrailed path from raw SQL to results, shared by both modes."""
    tree = validate(sql)
    final_sql = enforce_limit(tree, cfg["guardrails"]["row_limit"])
    columns, rows = execute(
        db_path,
        final_sql,
        timeout_s=cfg["guardrails"]["timeout_seconds"],
        max_rows=cfg["guardrails"].get("max_result_rows", 500),
    )
    return final_sql, columns, rows


def _answer_greedy(
    question: str, db_path: str, cfg: dict, schema: str, model: str, examples=None
) -> AgentResult:
    start = time.monotonic()
    messages = build_messages(question, schema, examples)

    last_error = None
    last_sql = None
    catalog = None
    for attempt in range(1, cfg["agent"]["max_attempts"] + 1):
        raw = llm.chat(
            messages,
            model=model,
            num_ctx=cfg["llm"]["num_ctx"],
            temperature=cfg["llm"]["temperature"],
        )
        if CANNOT_ANSWER in raw:
            return AgentResult(
                question=question,
                status="cannot_answer",
                attempts=attempt,
                latency_s=time.monotonic() - start,
            )

        last_sql = extract_sql(raw)
        try:
            final_sql, columns, rows = _validated_execute(last_sql, db_path, cfg)
            return AgentResult(
                question=question,
                status="ok",
                sql=final_sql,
                columns=columns,
                rows=rows,
                attempts=attempt,
                latency_s=time.monotonic() - start,
            )
        except (GuardrailError, sqlite3.Error) as e:
            last_error = str(e)
            if catalog is None:
                catalog = column_catalog(db_path)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": error_feedback(last_error, catalog)})

    return AgentResult(
        question=question,
        status="failed",
        sql=last_sql,
        attempts=cfg["agent"]["max_attempts"],
        latency_s=time.monotonic() - start,
        error=last_error,
    )


def result_signature(rows: list[tuple]) -> tuple:
    """Hashable, order-insensitive fingerprint of a result set, for vote grouping."""
    return tuple(sorted(tuple(str(v) for v in row) for row in rows))


def majority_pick(candidates: list[tuple]) -> tuple:
    """candidates: (signature, sql, columns, rows). Return the first candidate
    whose signature occurs most often."""
    counts = Counter(sig for sig, *_ in candidates)
    best_sig, _ = counts.most_common(1)[0]
    return next(c for c in candidates if c[0] == best_sig)


def _answer_self_consistency(
    question: str, db_path: str, cfg: dict, schema: str, model: str, k: int, examples=None
) -> AgentResult:
    start = time.monotonic()
    messages = build_messages(question, schema, examples)
    temperature = cfg["agent"].get("sc_temperature", 0.7)

    candidates = []
    refusals = 0
    for _ in range(k):
        raw = llm.chat(
            messages, model=model, num_ctx=cfg["llm"]["num_ctx"], temperature=temperature
        )
        if CANNOT_ANSWER in raw:
            refusals += 1
            continue
        try:
            final_sql, columns, rows = _validated_execute(extract_sql(raw), db_path, cfg)
        except (GuardrailError, sqlite3.Error):
            continue
        candidates.append((result_signature(rows), final_sql, columns, rows))

    if refusals > k // 2:
        return AgentResult(
            question=question,
            status="cannot_answer",
            attempts=k,
            latency_s=time.monotonic() - start,
        )

    if candidates:
        _, sql, columns, rows = majority_pick(candidates)
        return AgentResult(
            question=question,
            status="ok",
            sql=sql,
            columns=columns,
            rows=rows,
            attempts=k,
            latency_s=time.monotonic() - start,
        )

    # no sample produced working SQL — fall back to the greedy self-correction loop
    result = _answer_greedy(question, db_path, cfg, schema, model, examples)
    result.attempts += k
    result.latency_s = time.monotonic() - start
    return result
