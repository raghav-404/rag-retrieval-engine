"""Thin Streamlit entrypoint kept for `streamlit run app.py`."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_retrieval_engine.ui import main


main()
