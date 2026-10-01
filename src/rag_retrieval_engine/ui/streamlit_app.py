import requests
import streamlit as st

from ..config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    st.set_page_config(page_title="RAG Chat", page_icon=":books:")
    st.title("Document Q&A")
    st.caption("Ask questions about the indexed documents")
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
            response = requests.post(config.api_url, json={"question": question}, timeout=180)
            data = response.json()
            if "detail" in data:
                detail = data["detail"]
                st.error(detail.get("message", detail) if isinstance(detail, dict) else detail)
                return
            response.raise_for_status()
            answer = data["answer"]
            st.markdown(answer)
            sources = ", ".join(item["source"] for item in data["sources"])
            st.caption(
                f"Sources: {sources or 'N/A'} | "
                f"Latency: {data['metrics']['total_ms']:.1f} ms"
            )
            history.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"Backend error: {exc}")
