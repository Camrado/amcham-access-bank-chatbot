"""
chatbot package initializer.

Exports commonly referenced symbols at package level so tests that do
`__import__('chatbot.anomaly').SPIKE_THRESHOLD` (legacy) continue to work.
"""
from . import anomaly as anomaly

# Re-export SPIKE_THRESHOLD at package level for compatibility with tests
try:
    SPIKE_THRESHOLD = anomaly.SPIKE_THRESHOLD
except Exception:
    SPIKE_THRESHOLD = None

__all__ = ["anomaly", "SPIKE_THRESHOLD"]
