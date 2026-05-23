"""
anomaly.py
----------
Detects anomalous spikes in case volume per department within a rolling time window.

If N or more cases arrive for the same department within WINDOW_MINUTES,
an anomaly is recorded and returned so the admin panel can show an alert banner.

No ML needed — pure counter logic on the SQLite cases table.

Public API:
    from anomaly import check_anomaly, get_active_anomalies

    # Call after every new case is created
    anomaly = check_anomaly(department="Card Operations", db_path="cases.db")
    if anomaly:
        print(anomaly["message"])  # "⚠️ 5 Card Operations cases in the last 30 minutes"

    # Call from admin panel to show active alert banners
    alerts = get_active_anomalies(db_path="cases.db")
"""

import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger("anomaly")

# ─── Config ───────────────────────────────────────────────────────────────────
WINDOW_MINUTES = 30          # rolling window to count cases in
SPIKE_THRESHOLD = 4          # cases in window → anomaly triggered
ANOMALY_COOLDOWN_MINUTES = 60  # don't re-alert for same dept within this period

# ─── DB setup ─────────────────────────────────────────────────────────────────

def init_anomaly_table(db_path: str) -> None:
    """Create anomalies table if not exists. Called by init_db in agent.py."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id              TEXT PRIMARY KEY,
                department      TEXT NOT NULL,
                case_count      INTEGER NOT NULL,
                window_minutes  INTEGER NOT NULL,
                message         TEXT NOT NULL,
                resolved        INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL
            )
        """)
        conn.commit()


# ─── Core logic ───────────────────────────────────────────────────────────────

def check_anomaly(department: str, db_path: str) -> dict | None:
    """
    Count cases for `department` in the last WINDOW_MINUTES.
    If count >= SPIKE_THRESHOLD and no recent anomaly for this dept, record and return it.
    Returns the anomaly dict if triggered, else None.
    """
    init_anomaly_table(db_path)
    now = datetime.utcnow()
    window_start = (now - timedelta(minutes=WINDOW_MINUTES)).isoformat()
    cooldown_start = (now - timedelta(minutes=ANOMALY_COOLDOWN_MINUTES)).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Count recent cases for this department
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM cases
            WHERE department = ? AND created_at >= ?
            """,
            (department, window_start),
        ).fetchone()
        count = row["cnt"]

        if count < SPIKE_THRESHOLD:
            return None

        # Check if we already alerted for this dept recently (cooldown)
        existing = conn.execute(
            """
            SELECT id FROM anomalies
            WHERE department = ? AND created_at >= ? AND resolved = 0
            ORDER BY created_at DESC LIMIT 1
            """,
            (department, cooldown_start),
        ).fetchone()

        if existing:
            logger.debug("Anomaly cooldown active for dept=%s", department)
            return None

        # Record the anomaly
        import uuid
        anomaly_id = f"ANO-{uuid.uuid4().hex[:8].upper()}"
        message = (
            f"⚠️ Spike detected: {count} '{department}' cases "
            f"in the last {WINDOW_MINUTES} minutes. "
            f"Possible systemic issue — review immediately."
        )
        conn.execute(
            """
            INSERT INTO anomalies (id, department, case_count, window_minutes, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (anomaly_id, department, count, WINDOW_MINUTES, message, now.isoformat()),
        )
        conn.commit()

    logger.warning("ANOMALY triggered: %s", message)
    return {
        "id": anomaly_id,
        "department": department,
        "case_count": count,
        "window_minutes": WINDOW_MINUTES,
        "message": message,
        "created_at": now.isoformat(),
    }


def get_active_anomalies(db_path: str) -> list[dict]:
    """
    Return all unresolved anomalies — used by admin panel to show alert banners.
    """
    init_anomaly_table(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM anomalies WHERE resolved = 0 ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def resolve_anomaly(anomaly_id: str, db_path: str) -> None:
    """Mark an anomaly as resolved (admin dismissed the alert)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE anomalies SET resolved = 1 WHERE id = ?",
            (anomaly_id,),
        )
        conn.commit()
    logger.info("Anomaly %s resolved", anomaly_id)


def get_department_volume(db_path: str, window_minutes: int = 60) -> list[dict]:
    """
    Returns case counts per department for the last `window_minutes`.
    Used by admin panel dashboard charts.
    """
    init_anomaly_table(db_path)
    window_start = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT department, COUNT(*) as count
            FROM cases
            WHERE created_at >= ?
            GROUP BY department
            ORDER BY count DESC
            """,
            (window_start,),
        ).fetchall()
    return [dict(r) for r in rows]
