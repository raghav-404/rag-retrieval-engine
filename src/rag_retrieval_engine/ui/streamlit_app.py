import requests
import streamlit as st

from ..config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    st.set_page_config(page_title="RAG Chat", page_icon=":books:")
    st.title("Document Q&A")
    st.caption(f"Demo UI for the local RAG API via Ollama ({config.model_name})")
    history = st.session_state.setdefault("history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            data = requests.post(config.api_url, json={"question": question}, timeout=180).json()
            if "detail" in data:
                st.error(data["detail"])
                return
            st.markdown(data.get("answer", "No answer returned."))
            st.caption(
                f"Sources: {', '.join(data.get('sources', [])) or 'N/A'} | "
                f"Rewritten: {data.get('rewritten_query', '')} | Score: {data.get('eval_score', 0)}"
            )
            history.append({"role": "assistant", "content": data.get("answer", "No answer returned.")})
        except Exception as exc:
            st.error(f"Backend error: {exc}")
