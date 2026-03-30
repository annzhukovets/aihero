"""
CLI: index docs, run the agent in the terminal (no Streamlit).

Usage (from ``project/app``)::

    uv run python main.py

For the web UI::

    uv run streamlit run app.py
"""
from __future__ import annotations

import asyncio
import os

import ingest
import logs
import search_agent


def main() -> None:
    print("Indexing huggingface/transformers (English docs)…")

    vindex, embedding_model = ingest.index_data(
        "huggingface",
        "transformers",
        folder_filter="docs/source/en",
        artifacts_repo_id=os.getenv("HF_ARTIFACTS_REPO_ID"),
        artifacts_revision=os.getenv("HF_ARTIFACTS_REVISION"),
        artifacts_subdir=os.getenv("HF_ARTIFACTS_SUBDIR", ""),
        artifacts_repo_type=os.getenv("HF_ARTIFACTS_REPO_TYPE", "dataset"),
    )
    print("Vector index ready.")

    agent = search_agent.init_agent(
        vindex,
        embedding_model,
        "huggingface",
        "transformers",
    )
    print(f"Agent ready: {agent.name}. Type 'stop' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "stop":
            print("Goodbye.")
            break

        response = asyncio.run(agent.run(user_prompt=question))
        logs.log_interaction_to_file(agent, response.new_messages())
        print("\n", response.output, "\n", "=" * 50, "\n", sep="")


if __name__ == "__main__":
    main()
