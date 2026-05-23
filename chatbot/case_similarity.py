"""
case_similarity.py
------------------
Finds past resolved cases similar to a new case summary using embeddings.
Reuses the same embed_text() from rag_loader — no extra model needed.

The admin panel shows: "Similar past case CASE-XXXX was resolved in 2 days."
This gives bank staff instant institutional memory.

Public API:
    from case_similarity import find_similar_cases, index_case

    # Call after a new case is created to find similar past cases
    similar = find_similar_cases(
        summary="Customer card declined at POS, 47 AZN deducted",
        db_path="cases.db",
        top_k=3,
        min_score=0.55,
    )
    for s in similar:
        print(s["case_id"], s["score"], s["summary"])

    # Call after a case is created to add it to the similarity index
    index_case(case_id="CASE-ABC123", summary="...", department="Card Operations", db_path="cases.db")
"""

import json
import logging
import math
import sqlite3
from datetime import datetime

logger = logging.getLogger("case_similarity")

# Lazy import — rag_loader builds the embedding index on import,
# so we import only when first needed to avoid startup cost if unused.
_embed_text = None

def _get_embedder():
    global _embed_text
    if _embed_text is None:
        from chatbot.rag_loader import embed_text
        _embed_text = embed_text
    return _embed_text


# ─── DB setup ─────────────────────────────────────────────────────────────────

def _init_similarity_table(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS case_embeddings (
                case_id     TEXT PRIMARY KEY,
                department  TEXT NOT NULL,
                summary     TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()


# ─── Cosine similarity ────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─── Public API ───────────────────────────────────────────────────────────────

def index_case(case_id: str, summary: str, department: str, db_path: str) -> None:
    """
    Embed the case summary and store it in case_embeddings.
    Call this right after create_case() in agent.py.
    """
    _init_similarity_table(db_path)
    embed = _get_embedder()
    vector = embed(summary)
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO case_embeddings (case_id, department, summary, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (case_id, department, summary, json.dumps(vector), now),
        )
        conn.commit()
    logger.info("Case %s indexed for similarity search", case_id)


def find_similar_cases(
    summary: str,
    db_path: str,
    top_k: int = 3,
    min_score: float = 0.55,
    exclude_case_id: str | None = None,
) -> list[dict]:
    """
    Find past cases similar to the given summary.

    Returns list of dicts:
        [{ "case_id", "department", "summary", "score", "created_at" }, ...]
    sorted by score descending, filtered by min_score.
    Returns [] if no similar cases found or index is empty.
    """
    _init_similarity_table(db_path)
    embed = _get_embedder()

    # Load all indexed cases
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT case_id, department, summary, embedding, created_at FROM case_embeddings"
        ).fetchall()

    if not rows:
        return []

    query_vec = embed(summary)

    scored = []
    for row in rows:
        if exclude_case_id and row["case_id"] == exclude_case_id:
            continue
        stored_vec = json.loads(row["embedding"])
        score = _cosine(query_vec, stored_vec)
        if score >= min_score:
            scored.append({
                "case_id": row["case_id"],
                "department": row["department"],
                "summary": row["summary"],
                "score": round(score, 4),
                "created_at": row["created_at"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:top_k]

    if result:
        logger.info(
            "Found %d similar cases for summary (top score=%.2f)",
            len(result), result[0]["score"],
        )
    return result


def format_similarity_hint(similar_cases: list[dict]) -> str | None:
    """
    Format similar cases into a readable hint for the admin panel.
    Returns None if no similar cases.
    """
    if not similar_cases:
        return None

    lines = ["**Similar past cases:**"]
    for c in similar_cases:
        score_pct = int(c["score"] * 100)
        date = c["created_at"][:10]
        lines.append(
            f"• {c['case_id']} ({c['department']}, {date}) — {score_pct}% match\n"
            f"  _{c['summary'][:120]}{'...' if len(c['summary']) > 120 else ''}_"
        )
    return "\n".join(lines)
