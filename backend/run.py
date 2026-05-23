import sys
from pathlib import Path

# Add the project root (AmCham/) to sys.path so the `chatbot` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_config import setup_logging
setup_logging()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
