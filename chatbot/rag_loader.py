"""
rag_loader.py
-------------
Loads knowledge_base.json, generates embeddings via OpenAI,
and provides a retrieval function for the AccessBank support agent.

Usage:
    from rag_loader import retrieve

    results = retrieve("What are your loan interest rates?", top_k=3)
    for r in results:
        print(r["title"], r["score"])
        print(r["content"])
"""

import json
import os
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

KNOWLEDGE_BASE_PATH = "chatbot/knowledge_base.json"
SIMILARITY_THRESHOLD = 0.40  # below this → flag for human review (calibrated for text-embedding-3-small)

# Model candidates in preference order — first one that works is used
_EMBED_MODEL_CANDIDATES = [
    "text-embedding-3-small",
    "text-embedding-ada-002",
]

def _detect_embed_model() -> str:
    """Try each candidate model with a short probe and return the first that works."""
    for model in _EMBED_MODEL_CANDIDATES:
        try:
            client.embeddings.create(input="test", model=model)
            print(f"Embedding model: {model}")
            return model
        except Exception:
            continue
    raise RuntimeError(
        "No supported embedding model found. "
        "Tried: " + ", ".join(_EMBED_MODEL_CANDIDATES)
    )

EMBED_MODEL = _detect_embed_model()

# ─── Load & embed on startup ──────────────────────────────────────────────────

def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(chunks: list[dict]) -> list[dict]:
    """Embed every chunk once and attach the vector."""
    print(f"Building RAG index for {len(chunks)} chunks...")
    for chunk in chunks:
        text = f"{chunk['title']}. {chunk['content']}"
        chunk["embedding"] = embed_text(text)
    print("Index ready.")
    return chunks


# Build index once at module load
_chunks = load_knowledge_base()
_index = build_index(_chunks)


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Returns:
        {
            "results": [{ "id", "title", "content", "category", "score" }, ...],
            "top_score": float,
            "flag_for_human": bool  # True if best match is below threshold
        }
    """
    query_embedding = embed_text(query)

    scored = []
    for chunk in _index:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "content": chunk["content"],
            "category": chunk["category"],
            "tags": chunk["tags"],
            "score": round(score, 4)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored[:top_k]
    top_score = top_results[0]["score"] if top_results else 0.0

    return {
        "results": top_results,
        "top_score": top_score,
        "flag_for_human": top_score < SIMILARITY_THRESHOLD
    }


# ─── Add new chunk (feedback loop) ────────────────────────────────────────────

def add_chunk(title: str, content: str, category: str = "learned", tags: list[str] = []) -> dict:
    """
    Adds a new Q&A pair to the live index (from admin corrections).
    Also persists it to knowledge_base.json so it survives restarts.
    """
    new_id = f"kb_{len(_index) + 1:03d}"
    new_chunk = {
        "id": new_id,
        "category": category,
        "title": title,
        "content": content,
        "tags": tags
    }

    # Embed and add to live index
    text = f"{title}. {content}"
    new_chunk_with_embedding = {**new_chunk, "embedding": embed_text(text)}
    _index.append(new_chunk_with_embedding)

    # Persist to file
    raw = load_knowledge_base()
    raw.append(new_chunk)
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    print(f"New chunk added: {new_id} — {title}")
    return new_chunk


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What are your working hours?",
        "My card was blocked and money was taken",
        "How do I apply for a mortgage?",
        "I can't log in to the app",
        "My transfer is missing"
    ]
    for q in test_queries:
        result = retrieve(q, top_k=2)
        print(f"\nQuery: {q}")
        print(f"Top score: {result['top_score']} | Flag for human: {result['flag_for_human']}")
        for r in result["results"]:
            print(f"  [{r['score']}] {r['title']}")