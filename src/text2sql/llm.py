"""Thin wrapper around the Ollama chat API.

num_ctx is passed explicitly on every call: Ollama runs models with a small
default context window and silently truncates longer prompts, which would cut
off the schema without any error.
"""

import ollama


def chat(
    messages: list[dict],
    model: str,
    num_ctx: int = 8192,
    temperature: float = 0.0,
    keep_alive: str = "30m",
) -> str:
    # keep_alive stops Ollama unloading the model after idle — without it the
    # first question after a pause pays a multi-second model reload
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"num_ctx": num_ctx, "temperature": temperature},
        keep_alive=keep_alive,
    )
    return response["message"]["content"]
