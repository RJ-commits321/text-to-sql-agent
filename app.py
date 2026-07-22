"""Streamlit UI: a chat over the demo database, plus a stats page reading the
runtime log. Run from the project root: streamlit run app.py"""

from pathlib import Path

import pandas as pd
import streamlit as st

from text2sql.agent import answer_question
from text2sql.answer import summarize
from text2sql.config import load_config
from text2sql.prompts import PROMPT_VERSION
from text2sql.runlog import log_query, read_log
from text2sql.schema import get_schema, list_tables, table_overview

st.set_page_config(page_title="Text-to-SQL Agent", page_icon="🗃️", layout="wide")

cfg = load_config()

EXAMPLE_QUESTIONS = [
    "Which 3 countries have the most customers?",
    "List the 5 longest tracks with their album names",
    "Which employee supports the most customers?",
    "What is the weather today?",
]


@st.cache_data
def cached_schema(db_path: str) -> str:
    return get_schema(db_path)


@st.cache_data
def cached_overview(db_path: str) -> list[dict]:
    return table_overview(db_path)


with st.sidebar:
    st.title("🗃️ Text-to-SQL Agent")
    st.caption("Ask a database questions in plain English — a local LLM writes and runs the SQL.")
    page = st.radio("Page", ["Chat", "Stats"], label_visibility="collapsed")
    model = st.selectbox("Model", cfg["llm"]["available_models"])
    st.caption(
        f"prompt {PROMPT_VERSION} · retries ≤{cfg['agent']['max_attempts']} · "
        "answers take ~5–15 s (local model)"
    )

db_path = cfg["paths"]["demo_db"]

if not Path(db_path).exists():
    st.error(
        f"Demo database not found at `{db_path}`. "
        "Run `uv run python eval/download_data.py` first."
    )
    st.stop()


def render_result(result, answer, df):
    st.write(answer)
    if df is not None and not df.empty:
        st.dataframe(df, width="stretch")
    if result.sql:
        with st.expander("Show the SQL it wrote"):
            st.code(result.sql, language="sql")
    st.caption(
        f"{model} · {result.attempts} attempt(s) · {result.latency_s:.1f}s · {result.status}"
    )


def handle_question(question: str):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("Writing SQL and running it on the database…"):
        try:
            result = answer_question(
                question, db_path, cfg, schema=cached_schema(db_path), model=model
            )
        except Exception as e:
            st.error(
                f"Couldn't reach the model. Is Ollama running and `{model}` pulled? "
                f"(`ollama pull {model}`)\n\nDetails: {e}"
            )
            return

        df = None
        if result.status == "ok":
            answer = summarize(question, result.columns, result.rows, cfg, model=model)
            df = pd.DataFrame(result.rows, columns=result.columns)
        elif result.status == "cannot_answer":
            tables = ", ".join(list_tables(db_path))
            answer = (
                "That's outside what this database knows. "
                f"It only contains music-store data — tables: {tables}."
            )
        else:
            answer = (
                "I couldn't produce a working query for that question — "
                "try rephrasing it or being more specific."
            )
        render_result(result, answer, df)

    log_query(
        {
            "question": question,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "self_consistency": cfg["agent"].get("self_consistency", 1),
            "status": result.status,
            "sql": result.sql,
            "attempts": result.attempts,
            "latency_s": round(result.latency_s, 2),
            "error": result.error,
        },
        cfg["paths"]["runtime_log"],
    )
    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "df": df,
            "sql": result.sql,
            "meta": f"{model} · {result.attempts} attempt(s) · {result.latency_s:.1f}s",
        }
    )


if page == "Chat":
    st.markdown(
        "**You're chatting with a music-store database** — 11 tables of artists, "
        "albums, tracks, customers, invoices and employees."
    )
    with st.expander("💡 What can I ask? (see the database contents)"):
        st.dataframe(pd.DataFrame(cached_overview(db_path)), width="stretch")
        st.caption("Below is the exact schema text the AI receives with every question:")
        st.code(cached_schema(db_path), language="sql")

    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry.get("df") is not None and not entry["df"].empty:
                st.dataframe(entry["df"], width="stretch")
            if entry.get("sql"):
                with st.expander("Show the SQL it wrote"):
                    st.code(entry["sql"], language="sql")
            st.caption(entry["meta"])

    if not st.session_state.history:
        st.write("Try one of these:")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(example, width="stretch"):
                st.session_state.pending_question = example
                st.rerun()

    typed = st.chat_input("Ask a question about the music store…")
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        handle_question(question)

else:  # Stats
    st.header("Runtime stats")
    st.caption("Every question asked in the app is logged locally — this page reads that log.")
    records = read_log(cfg["paths"]["runtime_log"])
    if not records:
        st.info("No queries logged yet — ask something on the Chat page first.")
        st.stop()

    df = pd.DataFrame(records)
    ok = (df["status"] == "ok").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", len(df))
    c2.metric("Success rate", f"{ok:.0%}")
    c3.metric("Avg attempts", f"{df['attempts'].mean():.2f}")
    c4.metric("Avg latency", f"{df['latency_s'].mean():.1f}s")

    st.subheader("By status")
    st.bar_chart(df["status"].value_counts())

    st.subheader("Recent queries")
    st.dataframe(
        df[["timestamp", "model", "question", "status", "attempts", "latency_s"]].tail(50),
        width="stretch",
    )
