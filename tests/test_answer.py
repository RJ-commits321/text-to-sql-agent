"""Tests for the result-summarizing branches that don't need an LLM.
(The small-result path calls the model, so it isn't unit-tested here.)"""

from text2sql.answer import _MAX_PREVIEW_ROWS, summarize

CFG = {"llm": {"model": "x", "num_ctx": 8192, "temperature": 0.0}}


def test_empty_result_needs_no_model():
    assert summarize("q", ["c"], [], CFG) == "The query ran successfully but returned no results."


def test_large_result_points_to_table_without_calling_model():
    rows = [(i,) for i in range(_MAX_PREVIEW_ROWS + 1)]
    msg = summarize("list everything", ["id"], rows, CFG)
    assert str(len(rows)) in msg
    assert "table" in msg.lower()
