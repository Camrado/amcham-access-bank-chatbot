"""
logging_config.py
-----------------
Call setup_logging() once at startup (run.py) and once at the top of main.py
(covers uvicorn hot-reload workers that never pass through run.py).

What is scrubbed before any log line is written:
  - JSON fields named password / token / access_token / api_key / secret
  - Authorization: Bearer <token>  headers
  - OpenAI/Anthropic sk-... key patterns
"""

import logging
import logging.handlers
import re
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_SCRUB_RULES = [
    # JSON "sensitive_key": "value"
    (re.compile(
        r'("(?:password|hashed_password|secret|api_key|access_token|token|authorization)"\s*:\s*)"[^"]*"',
        re.IGNORECASE,
    ), r'\1"[REDACTED]"'),
    # Bearer <token>
    (re.compile(r'(Bearer\s+)[A-Za-z0-9\-._~+/]+=*', re.IGNORECASE), r'\1[REDACTED]'),
    # sk-... API key pattern (OpenAI / Anthropic)
    (re.compile(r'\b(sk-[A-Za-z0-9]{4})[A-Za-z0-9\-]+\b'), r'\1[REDACTED]'),
]

_configured = False


class _SensitiveFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _scrub(str(record.msg))
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(_scrub(a) if isinstance(a, str) else a for a in args)
        return True


def _scrub(text: str) -> str:
    for pattern, repl in _SCRUB_RULES:
        text = pattern.sub(repl, text)
    return text


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)-22s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    sensitive = _SensitiveFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(sensitive)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(sensitive)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers — their INFO is just noise
    for name in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore", "multipart"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
