# text-to-sql-agent

Ask a database questions in plain English. A local LLM (via [Ollama](https://ollama.com))
translates the question to SQL, guardrails validate it, it runs against the database,
and you get a one-sentence answer, the results table, and the SQL itself. Fully local:
no API keys, no cloud, no data leaving the machine.

> **You:** Which 3 countries have the most customers?
> **App:** The three countries with the most customers are the USA, Canada and France.

## How it works

```
question → prompt (schema + sample rows + few-shot examples + rules)
         → LLM writes SQL → guardrails validate → execute (read-only)
         → on error: feed the DB error back, retry (max 3)
         → results + one-sentence answer
```

- **Self-correction** — the database's own error messages are fed back to the model.
- **Guardrails** — single read-only SELECT only (validated by parsing, not keyword
  matching), row-limit injection, 10s timeout. Off-topic questions are refused.
- **Self-consistency mode** (optional) — sample 3 queries, run all, take the majority
  result; higher accuracy for higher latency.
- **Evaluation** — scored on the [Spider](https://yale-lily.github.io/spider) benchmark
  by execution accuracy (the generated SQL must return the same rows as the reference
  SQL), with every run tracked in MLflow.

## Results

Execution accuracy on the Spider dev benchmark (fixed 200-question subset),
single query per question (deterministic — the same every run):

| Model | Accuracy | Off-topic refusal |
|---|---|---|
| **qwen2.5-coder:3b** | **76.0%** | 100% |
| llama3.2:3b | 66.0% | 80% |
| qwen2.5:3b | 65.5% | 100% |
| qwen2.5:1.5b | 52.5% | 85% |

All models run locally on a laptop. `qwen2.5-coder:3b` is the default and is shown
in its shipped configuration (which adds schema foreign-key hints — these don't
change its accuracy but raise its off-topic refusal to 100%); the other models are
in the base configuration, so the refusal column is not a like-for-like comparison.

Enabling self-consistency (sample 3, majority vote) typically adds a small boost,
but it samples with randomness so the exact figure varies run to run (~76–78%);
the 76.0% above is the reproducible single-query number.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Ollama (`ollama pull qwen2.5-coder:3b`).

```bash
uv sync
uv run python eval/download_data.py     # Chinook demo DB + Spider benchmark
uv run streamlit run app.py             # chat UI at http://localhost:8501
```

Reproduce the benchmark:

```bash
uv run python eval/make_subset.py                                # build the fixed subset
uv run python eval/run_eval.py --model qwen2.5-coder:3b --sc 3   # run + log to MLflow
uv run mlflow ui                                                 # view results
```

## Structure

```
src/text2sql/    llm client, schema extraction, prompts, guardrails, agent loop
app.py           Streamlit UI (chat + runtime stats)
eval/            benchmark harness, result comparison, off-topic probes
tests/           unit tests (guardrail cases run without an LLM)
config.yaml      all settings in one place
```

## Design

- **Evaluation-driven** — measured on a standard benchmark with a reproducible harness
  and every run tracked in MLflow.
- **Safe by construction** — model output is parsed and constrained to a single
  read-only SELECT; write and injection attempts are rejected and unit-tested.
- **Verifiable** — the generated SQL is always shown, so the answer can be checked;
  built as an analyst-assist tool rather than a black box.
- **Deliberately focused** — single agent, local-first, no framework. A Dockerfile
  covers packaging; CI runs lint and tests on every push.
