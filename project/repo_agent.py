"""
PydanticAI agent over repo documentation chunks (keyword + vector search).

Typical use from a notebook after building ``transformers_chunks``::

    from repo_agent import create_repo_agent, ask_sync

    agent = create_repo_agent(transformers_chunks)
    answer = ask_sync(agent, "How do I use LoRA with PEFT?")

Requires ``OPENAI_API_KEY`` (or the provider your ``llm_model`` uses) in the environment.

In Jupyter, you can use ``await ask(agent, ...)`` instead of ``ask_sync`` if you prefer.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import numbers
from typing import Any, cast

from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from search import (
    build_text_index,
    build_vector_index,
    search_text_index,
    search_vector_index,
)

DEFAULT_TEXT_FIELDS = ("filename", "section")

DEFAULT_REPO_INSTRUCTIONS = """You are a helpful assistant answering questions using retrieved sections from documentation (file path + markdown chunk per result).

Always call the search tools to find relevant sections before answering. If the first search is not enough, try different wording or the other search tool.

Mention source filenames when it helps the user locate content. If the retrieved sections do not answer the question, say so clearly."""

SYSTEM_PROMPT_STRICT_CITATIONS = """You are a helpful assistant answering questions using retrieved sections from Hugging Face Transformers documentation.

If you can find specific information through search, use it to provide accurate answers.

Always include references by citing the filename of the source material you used.  
When citing the reference, replace "transformers-main" by the full path to the GitHub repository: "https://github.com/huggingface/transformers/tree/main/"
Format: [LINK TITLE](FULL_GITHUB_LINK)

If the search doesn't return relevant results, let the user know and provide general guidance
"""


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert tool outputs to values that ``json.dumps(..., allow_nan=False)`` accepts.

    OpenAI's API rejects bodies with NaN/Infinity or non-standard numeric types; minsearch / numpy
    can otherwise leak into tool return payloads.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return _json_safe(float(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _tool_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = _json_safe(rows)
    json.dumps(out, ensure_ascii=False, allow_nan=False)
    return cast(list[dict[str, Any]], out)


def create_repo_agent(
    chunks: list[dict[str, Any]],
    *,
    text_fields: list[str] | tuple[str, ...] | None = None,
    embedding_model: SentenceTransformer | None = None,
    embedding_model_name: str = "multi-qa-distilbert-cos-v1",
    llm_model: str = "openai:gpt-4o-mini",
    instructions: str | None = None,
    keyword_search: bool = False,
    vector_search: bool = True,
    name: str = "repo_docs_agent",
) -> Agent:
    """
    Build search indices once and return an :class:`~pydantic_ai.Agent` with tools.

    Parameters
    ----------
    chunks
        List of dicts with at least ``filename`` and ``section`` (same as in ``search.py``).
    text_fields
        Fields passed to minsearch keyword index; default ``(\"filename\", \"section\")``.
    embedding_model
        Optional pre-loaded model. If ``None`` and ``vector_search`` is True, loads
        ``embedding_model_name``.
    llm_model
        Model string for pydantic-ai, e.g. ``\"openai:gpt-4o-mini\"``.
    keyword_search / vector_search
        Which tools to register (at least one must be True).
    """
    if not keyword_search and not vector_search:
        raise ValueError("At least one of keyword_search or vector_search must be True")

    fields = list(text_fields) if text_fields is not None else list(DEFAULT_TEXT_FIELDS)
    tools: list[Any] = []

    text_index = None
    if keyword_search:
        text_index = build_text_index(chunks, fields)

        def keyword_search_docs(query: str) -> list[dict[str, Any]]:
            """Keyword search over doc paths and section text (good for names, APIs, exact terms)."""
            assert text_index is not None
            return _tool_results(search_text_index(text_index, query))

        tools.append(keyword_search_docs)

    emb = embedding_model
    vector_vindex = None
    if vector_search:
        if emb is None:
            emb = SentenceTransformer(embedding_model_name)
        vector_vindex = build_vector_index(chunks, emb)

        def semantic_search_docs(query: str) -> list[dict[str, Any]]:
            """Semantic search over doc paths and section text (semantic similarity)."""
            assert vector_vindex is not None and emb is not None
            return _tool_results(search_vector_index(vector_vindex, emb, query))

        tools.append(semantic_search_docs)

    return Agent(
        name=name,
        instructions=instructions or SYSTEM_PROMPT_STRICT_CITATIONS,
        tools=tools,
        model=llm_model,
    )


def ask_sync(agent: Agent, user_prompt: str) -> str:
    """
    Run the agent and return the final text output.

    Uses :meth:`~pydantic_ai.Agent.run_sync` when no asyncio loop is running (scripts, REPL).
    When a loop is already running (e.g. Jupyter), runs :meth:`~pydantic_ai.Agent.run` in a
    worker thread with its own loop so we avoid ``RuntimeError: This event loop is already running``.
    """
    prompt = str(user_prompt)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return agent.run_sync(prompt).output

    async def _run() -> str:
        return (await agent.run(prompt)).output

    def _in_thread() -> str:
        return asyncio.run(_run())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


async def ask(agent: Agent, user_prompt: str) -> str:
    """Run the agent asynchronously; returns the final text output."""
    return (await agent.run(str(user_prompt))).output


if __name__ == "__main__":
    _agent = create_repo_agent(
        [{"filename": "docs/example.md", "section": "## Hello\n\nSmoke test chunk."}],
        keyword_search=True,
        vector_search=True,
    )
    print("Agent ready:", _agent.name)
