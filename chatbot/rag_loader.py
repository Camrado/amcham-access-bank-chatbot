"""
rag_loader.py
-------------
Loads knowledge_base.json, generates embeddings via OpenAI, and provides
hybrid retrieval (dense cosine + BM25 sparse) merged with Reciprocal Rank
Fusion, with query expansion and overlapping sub-chunk splits.

New in this version:
  - Chunk overlap: long chunks are split into overlapping sub-chunks (20% overlap)
    so answers that span chunk boundaries are not missed.
  - Query expansion: the original query is rephrased into 2 variants via
    gpt-4o-mini; all variants are searched and the best dense score per chunk
    is kept before RRF merging.
  - Hybrid retrieval: BM25 (sparse keyword) results are fused with dense
    (cosine) results using Reciprocal Rank Fusion before the top-k are picked.
    Falls back gracefully to dense-only if rank_bm25 is not installed.

Usage:
    from rag_loader import retrieve

    results = retrieve("What are your loan interest rates?", top_k=3)
    for r in results["results"]:
        print(r["title"], r["score"])
        print(r["content"])
"""

import json
import os
import math
import re
import logging
from openai import OpenAI

logger = logging.getLogger("rag_loader")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning(
        "rank_bm25 not installed — hybrid retrieval will use dense-only. "
        "Install with: pip install rank-bm25"
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

KNOWLEDGE_BASE_PATH = "chatbot/knowledge_base.json"

# Below this dense cosine score → flag for human review
SIMILARITY_THRESHOLD = 0.40

# Set False to skip query expansion (saves ~1 fast LLM call per request)
QUERY_EXPANSION_ENABLED = True

# ─── Chunk overlap settings ───────────────────────────────────────────────────
# Chunks whose content exceeds CHUNK_MAX_WORDS are split into overlapping
# sub-chunks of CHUNK_MAX_WORDS words each, with CHUNK_OVERLAP_WORDS reused
# from the previous window.  Shorter chunks are kept as-is.
CHUNK_MAX_WORDS = 100
CHUNK_OVERLAP_WORDS = 20   # 20% of max — matches the "20% overlap" target

_EMBED_MODEL_CANDIDATES = [
    "text-embedding-3-small",
    "text-embedding-ada-002",
]


# ─── Model detection ──────────────────────────────────────────────────────────

def _detect_embed_model() -> str:
    """Try each candidate embedding model and return the first that succeeds."""
    for model in _EMBED_MODEL_CANDIDATES:
        try:
            client.embeddings.create(input="test", model=model)
            logger.info("Embedding model selected: %s", model)
            return model
        except Exception:
            continue
    raise RuntimeError(
        "No supported embedding model found. Tried: " + ", ".join(_EMBED_MODEL_CANDIDATES)
    )


EMBED_MODEL = _detect_embed_model()


# ─── Tokenizer (for BM25) ─────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ─── Chunk overlap splitter ───────────────────────────────────────────────────

def _split_with_overlap(chunk: dict) -> list[dict]:
    """
    Split a KB chunk into overlapping sub-chunks when its content is long enough.

    Sub-chunks inherit all fields of the parent chunk (title, category, tags)
    and get a modified id: "<parent_id>_s0", "<parent_id>_s1", etc.
    Chunks at or below CHUNK_MAX_WORDS are returned unchanged in a 1-element list.
    """
    words = chunk["content"].split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [chunk]

    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    sub_chunks: list[dict] = []
    sub_idx = 0

    for start in range(0, len(words), step):
        end = min(start + CHUNK_MAX_WORDS, len(words))
        sub_content = " ".join(words[start:end])
        sub_chunks.append({
            **chunk,
            "id": f"{chunk['id']}_s{sub_idx}",
            "content": sub_content,
            # title is inherited — keeps context for the reader / prompt
        })
        sub_idx += 1
        if end >= len(words):
            break

    logger.debug(
        "Chunk %s (%d words) → %d sub-chunks", chunk["id"], len(words), len(sub_chunks)
    )
    return sub_chunks


# ─── Load & embed at startup ──────────────────────────────────────────────────

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


def build_index(raw_chunks: list[dict]) -> list[dict]:
    """
    1. Expand each raw chunk into overlapping sub-chunks.
    2. Embed every sub-chunk once.
    Returns the enriched list ready for retrieval.
    """
    expanded: list[dict] = []
    for chunk in raw_chunks:
        expanded.extend(_split_with_overlap(chunk))

    logger.info(
        "Building index: %d raw chunks → %d sub-chunks after overlap split",
        len(raw_chunks), len(expanded),
    )

    for chunk in expanded:
        text = f"{chunk['title']}. {chunk['content']}"
        chunk["embedding"] = embed_text(text)

    logger.info("Dense index ready (%d vectors).", len(expanded))
    return expanded


# Build once at module load
_raw_chunks: list[dict] = load_knowledge_base()
_index: list[dict] = build_index(_raw_chunks)

# BM25 sparse index — built from the same expanded sub-chunks
if BM25_AVAILABLE:
    _bm25_corpus: list[list[str]] = [
        _tokenize(f"{c['title']}. {c['content']}") for c in _index
    ]
    _bm25 = BM25Okapi(_bm25_corpus)
    logger.info("BM25 sparse index ready (%d docs).", len(_bm25_corpus))
else:
    _bm25 = None
    _bm25_corpus = []


# ─── Query expansion ──────────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """
    Generate up to 2 alternative phrasings via gpt-4o-mini.
    Returns [original_query, variant_1, variant_2].
    Fails silently — original query is always included.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are helping a retrieval system for a bank customer support knowledge base.\n"
                    "Rephrase the following customer query into exactly 2 alternative versions that "
                    "use different words but ask the same thing.\n"
                    "Return ONLY a JSON object with this shape: "
                    "{\"expansions\": [\"phrasing1\", \"phrasing2\"]}\n\n"
                    f"Query: {query}"
                ),
            }],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.4,
        )
        data = json.loads(resp.choices[0].message.content)
        expansions = data.get("expansions", [])
        if isinstance(expansions, list) and expansions:
            variants = [query] + [e for e in expansions if isinstance(e, str)]
            logger.debug("Query expanded to %d variants: %s", len(variants), variants)
            return variants[:3]  # at most 3 total
    except Exception as exc:
        logger.warning("Query expansion failed (%s) — using original query only", exc)
    return [query]


# ─── Dense and sparse search primitives ──────────────────────────────────────

def _dense_search(query_embedding: list[float]) -> list[tuple[str, float]]:
    """Return (chunk_id, cosine_score) pairs sorted descending."""
    scored = [
        (chunk["id"], cosine_similarity(query_embedding, chunk["embedding"]))
        for chunk in _index
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def _sparse_search(query: str) -> list[tuple[str, float]]:
    """Return (chunk_id, bm25_score) pairs sorted descending."""
    if not BM25_AVAILABLE or _bm25 is None:
        return []
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked = [
        (_index[i]["id"], float(scores[i]))
        for i in range(len(_index))
    ]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _rrf_merge(
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.
    Each list contributes 1/(k + rank) to the combined score.
    k=60 is the standard RRF constant that balances high-rank and low-rank items.
    """
    rrf: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_ranked):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(sparse_ranked):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ─── Lookup helper ────────────────────────────────────────────────────────────

def _chunk_by_id(chunk_id: str) -> dict:
    for chunk in _index:
        if chunk["id"] == chunk_id:
            return chunk
    return {}


# ─── Public retrieval API ─────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Full retrieval pipeline:
      1. Expand query into up to 3 variants (gpt-4o-mini, ~50ms)
      2. Dense cosine search on every variant; keep best score per chunk
      3. BM25 sparse search on the original query
      4. Merge dense + sparse rankings via RRF
      5. Deduplicate sub-chunks from the same parent; return top_k

    Returns:
        {
            "results":       [{ "id", "title", "content", "category", "tags", "score" }, ...],
            "top_score":     float,   # best cosine score (used for threshold gating)
            "flag_for_human": bool    # True when top_score < SIMILARITY_THRESHOLD
        }
    """
    # ── Step 1: query expansion ───────────────────────────────────────────────
    queries = _expand_query(query) if QUERY_EXPANSION_ENABLED else [query]

    # ── Step 2: dense retrieval — best score per chunk across all variants ────
    best_dense_score: dict[str, float] = {}
    for q in queries:
        q_emb = embed_text(q)
        for chunk_id, score in _dense_search(q_emb):
            if score > best_dense_score.get(chunk_id, 0.0):
                best_dense_score[chunk_id] = score
    dense_ranked = sorted(best_dense_score.items(), key=lambda x: x[1], reverse=True)

    # ── Step 3: sparse retrieval on original query ────────────────────────────
    sparse_ranked = _sparse_search(query)

    # ── Step 4: RRF merge ─────────────────────────────────────────────────────
    if BM25_AVAILABLE and sparse_ranked:
        merged = _rrf_merge(dense_ranked, sparse_ranked)
    else:
        merged = dense_ranked  # graceful dense-only fallback

    # ── Step 5: assemble results, deduplicate sub-chunks ─────────────────────
    results: list[dict] = []
    seen_base_ids: set[str] = set()  # one result per original KB chunk

    for chunk_id, _ in merged:
        chunk = _chunk_by_id(chunk_id)
        if not chunk:
            continue

        # Derive the parent chunk ID: "kb_007_s1" → "kb_007"
        base_id = chunk_id.split("_s")[0]
        if base_id in seen_base_ids:
            continue
        seen_base_ids.add(base_id)

        results.append({
            "id": chunk_id,
            "title": chunk["title"],
            "content": chunk["content"],
            "category": chunk.get("category", ""),
            "tags": chunk.get("tags", []),
            # Expose the dense cosine score (the threshold is calibrated for it)
            "score": round(best_dense_score.get(chunk_id, 0.0), 4),
        })

        if len(results) >= top_k:
            break

    top_score = results[0]["score"] if results else 0.0

    logger.debug(
        "retrieve(%r) → top_score=%.4f, %d results, flag=%s",
        query, top_score, len(results), top_score < SIMILARITY_THRESHOLD,
    )

    return {
        "results": results,
        "top_score": top_score,
        "flag_for_human": top_score < SIMILARITY_THRESHOLD,
    }


# ─── Feedback loop: add admin-contributed chunk ───────────────────────────────

def add_chunk(
    title: str,
    content: str,
    category: str = "learned",
    tags: list[str] = [],
) -> dict:
    """
    Adds a new Q&A pair to the live index (from admin corrections).
    Applies the same overlap split as the startup index.
    Also persists the raw chunk to knowledge_base.json.
    """
    global _bm25

    new_id = f"kb_{len(_raw_chunks) + 1:03d}"
    new_chunk = {
        "id": new_id,
        "category": category,
        "title": title,
        "content": content,
        "tags": tags,
    }

    # Expand, embed, and append to live index
    for sub in _split_with_overlap(new_chunk):
        sub["embedding"] = embed_text(f"{sub['title']}. {sub['content']}")
        _index.append(sub)
        if BM25_AVAILABLE:
            _bm25_corpus.append(_tokenize(f"{sub['title']}. {sub['content']}"))

    # Rebuild BM25 index to include new documents
    if BM25_AVAILABLE:
        _bm25 = BM25Okapi(_bm25_corpus)

    # Persist raw chunk (without embedding) to file
    raw = load_knowledge_base()
    raw.append(new_chunk)
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    logger.info("New chunk added and indexed: %s — %s", new_id, title)
    return new_chunk


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What are your working hours?",
        "My card was blocked and money was taken",
        "How do I apply for a mortgage?",
        "I can't log in to the app",
        "My transfer is missing",
    ]
    for q in test_queries:
        result = retrieve(q, top_k=2)
        print(f"\nQuery: {q}")
        print(f"Top score: {result['top_score']} | Flag: {result['flag_for_human']}")
        for r in result["results"]:
            print(f"  [{r['score']}] {r['title']}")