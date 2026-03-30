"""
Streamlit UI: Transformers documentation assistant (streaming replies).
"""
from __future__ import annotations

import concurrent.futures
import os
import traceback

import streamlit as st

import ingest
import logs
import search_agent

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def init_agent():
    # Cap the number of chunks we embed so Streamlit startup doesn't time out.
    # Override with `MAX_CHUNKS=...` if you want a larger index.
    default_max_chunks = 100

    max_chunks_env = os.getenv("MAX_CHUNKS")
    max_chunks = int(max_chunks_env) if max_chunks_env else default_max_chunks
    chunk_level = int(os.getenv("CHUNK_LEVEL", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))

    vindex, embedding_model = ingest.index_data(
        "huggingface",
        "transformers",
        folder_filter="docs/source/en",
        level=chunk_level,
        max_chunks=max_chunks,
        batch_size=batch_size,
    )
    return search_agent.init_agent(
        vindex,
        embedding_model,
        "huggingface",
        "transformers",
    )


st.set_page_config(
    page_title="Transformers Docs Assistant",
    page_icon="🤗",
    layout="centered",
)
st.title("🤗 Transformers Docs Assistant")
st.caption("Questions are answered using retrieved sections from huggingface/transformers (English docs).")

if "agent" not in st.session_state:
    if "init_future" not in st.session_state:
        st.session_state.init_future = _EXECUTOR.submit(init_agent)

    if st.session_state.init_future.done():
        try:
            st.session_state.agent = st.session_state.init_future.result()
        except Exception as e:
            st.error("Failed to initialize the agent.")
            st.text(f"{type(e).__name__}: {e}")
            st.text(traceback.format_exc())
            st.stop()
    else:
        with st.spinner("Indexing docs (this can take a while)…"):
            st.write(
                "The UI stays responsive while the embedding index is built in the background. "
                "If this takes too long, set `MAX_CHUNKS`."
            )
        st.stop()

agent = st.session_state.agent

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def stream_response(prompt: str):
    full_text = ""
    result = agent.run_stream_sync(user_prompt=prompt)

    for delta in result.stream_text(delta=True):
        full_text += delta
        yield delta

    logs.log_interaction_to_file(agent, result.new_messages())
    st.session_state._last_response = full_text


if prompt := st.chat_input("Ask about the Transformers library…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(prompt))

    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})
