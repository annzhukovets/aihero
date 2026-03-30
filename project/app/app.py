"""
Streamlit UI: Transformers documentation assistant (streaming replies).
"""
from __future__ import annotations

import os

import streamlit as st

import ingest
import logs
import search_agent


def init_agent():
    vindex, embedding_model = ingest.index_data(
        "huggingface",
        "transformers",
        folder_filter="docs/source/en",
        artifacts_repo_id=os.getenv("HF_ARTIFACTS_REPO_ID"),
        artifacts_revision=os.getenv("HF_ARTIFACTS_REVISION"),
        artifacts_subdir=os.getenv("HF_ARTIFACTS_SUBDIR", ""),
        artifacts_repo_type=os.getenv("HF_ARTIFACTS_REPO_TYPE", "dataset"),
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
    with st.spinner("Loading embedding model and indexing docs…"):
        st.session_state.agent = init_agent()

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
