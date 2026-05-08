"""
Streamlit frontend — chat UI that calls the FastAPI /ask endpoint.
Run with: streamlit run app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="centered")
st.title("📚 Document Q&A")
st.caption("Powered by local RAG — FAISS + Ollama (qwen2.5:7b)")

if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Details", expanded=False):
                m = msg["meta"]
                st.write(f"**Rewritten query:** {m['rewritten_query']}")
                st.write(f"**Sources:** {', '.join(m['sources']) or 'N/A'}")
                st.write(f"**Eval score:** {m['eval_score']}")

# Input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                resp = requests.post(API_URL, json={"question": question}, timeout=180)
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)

                meta = {
                    "rewritten_query": data.get("rewritten_query", ""),
                    "sources": data.get("sources", []),
                    "eval_score": data.get("eval_score", 0),
                }
                with st.expander("🔍 Details", expanded=False):
                    st.write(f"**Rewritten query:** {meta['rewritten_query']}")
                    st.write(f"**Sources:** {', '.join(meta['sources']) or 'N/A'}")
                    st.write(f"**Eval score:** {meta['eval_score']}")

                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is `uvicorn backend:app` running?")
            except Exception as e:
                st.error(f"Error: {e}")
