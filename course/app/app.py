import streamlit as st

import ingest
import search_agent
import logs


def init_agent():
    repo_owner = "DataTalksClub"
    repo_name = "faq"

    def filter(doc):
        return "data-engineering" in doc["filename"]

    index = ingest.index_data(repo_owner, repo_name, filter=filter)
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    return agent


st.set_page_config(page_title="AI FAQ Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI FAQ Assistant")
st.caption("Ask me anything about the DataTalksClub/faq repository")

if "agent" not in st.session_state:
    with st.spinner("🔄 Indexing repo..."):
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


if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(prompt))

    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})
    