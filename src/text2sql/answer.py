"""Turn query results into a one-sentence natural-language answer."""

from . import llm

_MAX_PREVIEW_ROWS = 10


def summarize(
    question: str,
    columns: list[str],
    rows: list[tuple],
    cfg: dict,
    model: str | None = None,
) -> str:
    if not rows:
        return "The query ran successfully but returned no results."

    # A one-sentence summary only makes sense for a small result. For a large
    # one, any sentence built from a preview would misrepresent the whole, so
    # point the user at the full table instead.
    if len(rows) > _MAX_PREVIEW_ROWS:
        return f"The query returned {len(rows)} rows — see the full table below."

    table = "\n".join(
        [" | ".join(columns)] + [" | ".join(str(v) for v in row) for row in rows]
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"A database query was run and these results ARE the correct answer "
                f"to the question:\n{table}\n\n"
                "State the answer in one short, direct sentence. "
                "Do not doubt or qualify the results."
            ),
        }
    ]
    return llm.chat(
        messages,
        model=model or cfg["llm"]["model"],
        num_ctx=cfg["llm"]["num_ctx"],
        temperature=cfg["llm"]["temperature"],
    ).strip()
