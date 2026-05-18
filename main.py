from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_retrieval_engine.config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    print(f"rag-retrieval-engine ready. Docs: {config.docs_dir}")


if __name__ == "__main__":
    main()
