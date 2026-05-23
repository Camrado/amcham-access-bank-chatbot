"""
rag_loader.py
-------------
Loads knowledge_base.json, generates embeddings via OpenAI, and provides
hybrid retrieval (dense cosine + BM25 sparse) merged with Reciprocal Rank
Fusion, with query expansion and overlapping sub-chunk splits.

Logging:
  - INFO  : startup events, retrieve() pipeline summary, add_chunk()
  - DEBUG : per-step scores (dense top-3, BM25 top-3, RRF top-5, final results)
  - WARNING : degraded modes (BM25 unavailable, query expansion failure)
  - ERROR : unexpected exceptions caught during retrieval
"""

import json
import os
import math
import re
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("rag_loader")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning(
        "rank_bm25 not installed — falling back to dense-only retrieval. "
        "Install with: pip install rank-bm25"
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

KNOWLEDGE_BASE_PATH = "chatbot/knowledge_base.json"
SIMILARITY_THRESHOLD = 0.40
QUERY_EXPANSION_ENABLED = True
CHUNK_MAX_WORDS = 100
CHUNK_OVERLAP_WORDS = 20

_EMBED_MODEL_CANDIDATES = [
    "text-embedding-3-small",
    "text-embedding-ada-002",
]


# ─── Model detection ──────────────────────────────────────────────────────────

def _detect_embed_model() -> str:
    for model in _EMBED_MODEL_CANDIDATES:
        try:
            client.embeddings.create(input="test", model=model)
            logger.info("Embedding model selected: %s", model)
            return model
        except Exception as exc:
            logger.debug("Embedding model %s unavailable: %s", model, exc)
    raise RuntimeError(
        "No supported embedding model found. Tried: " + ", ".join(_EMBED_MODEL_CANDIDATES)
    )


EMBED_MODEL = _detect_embed_model()


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ─── Chunk overlap splitter ───────────────────────────────────────────────────

def _split_with_overlap(chunk: dict) -> list[dict]:
    """Split a long chunk into overlapping sub-chunks; short chunks pass through."""
    words = chunk["content"].split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [chunk]

    step = CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    sub_chunks: list[dict] = []
    sub_idx = 0

    for start in range(0, len(words), step):
        end = min(start + CHUNK_MAX_WORDS, len(words))
        sub_content = " ".join(words[start:end])
        sub_chunks.append({**chunk, "id": f"{chunk['id']}_s{sub_idx}", "content": sub_content})
        sub_idx += 1
        if end >= len(words):
            break

    logger.debug(
        "Chunk split | id=%s | words=%d | sub_chunks=%d",
        chunk["id"], len(words), len(sub_chunks),
    )
    return sub_chunks


# ─── Load & embed on startup ──────────────────────────────────────────────────

def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Knowledge base loaded | path=%s | chunks=%d", path, len(data))
    return data


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
    """Expand chunks with overlap splits, then embed each sub-chunk."""
    t0 = time.time()
    expanded: list[dict] = []
    for chunk in raw_chunks:
        expanded.extend(_split_with_overlap(chunk))

    logger.info(
        "Index build start | raw_chunks=%d | expanded_sub_chunks=%d",
        len(raw_chunks), len(expanded),
    )

    for i, chunk in enumerate(expanded):
        text = f"{chunk['title']}. {chunk['content']}"
        chunk["embedding"] = embed_text(text)
        if (i + 1) % 10 == 0 or (i + 1) == len(expanded):
            logger.debug("Embedding progress | %d / %d chunks", i + 1, len(expanded))

    elapsed = time.time() - t0
    logger.info(
        "Index build complete | sub_chunks=%d | elapsed=%.2fs",
        len(expanded), elapsed,
    )
    return expanded


# Build once at module load
_raw_chunks: list[dict] = load_knowledge_base()
_index: list[dict] = build_index(_raw_chunks)

if BM25_AVAILABLE:
    _bm25_corpus: list[list[str]] = [
        _tokenize(f"{c['title']}. {c['content']}") for c in _index
    ]
    _bm25 = BM25Okapi(_bm25_corpus)
    logger.info("BM25 sparse index ready | docs=%d", len(_bm25_corpus))
else:
    _bm25 = None
    _bm25_corpus = []


# ─── Query expansion ──────────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """Generate 2 alternative phrasings via gpt-4o-mini. Fails silently."""
    t0 = time.time()
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
            variants = variants[:3]
            logger.debug(
                "Query expansion | original=%.50r | variants=%d | elapsed=%.2fs | queries=%s",
                query, len(variants), time.time() - t0, variants,
            )
            return variants
    except Exception as exc:
        logger.warning(
            "Query expansion failed | query=%.50r | error=%s | fallback=original_only",
            query, exc,
        )
    return [query]


# ─── Dense and sparse search primitives ──────────────────────────────────────

def _dense_search(query_embedding: list[float]) -> list[tuple[str, float]]:
    """Return (chunk_id, cosine_score) sorted descending."""
    scored = [
        (chunk["id"], cosine_similarity(query_embedding, chunk["embedding"]))
        for chunk in _index
    ]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    logger.debug(
        "Dense search | top3=%s",
        [(cid, round(s, 4)) for cid, s in ranked[:3]],
    )
    return ranked


def _sparse_search(query: str) -> list[tuple[str, float]]:
    """Return (chunk_id, bm25_score) sorted descending."""
    if not BM25_AVAILABLE or _bm25 is None:
        logger.debug("Sparse search skipped | reason=bm25_unavailable")
        return []
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        [(_index[i]["id"], float(scores[i])) for i in range(len(_index))],
        key=lambda x: x[1],
        reverse=True,
    )
    logger.debug(
        "BM25 sparse search | tokens=%d | top3=%s",
        len(tokens),
        [(cid, round(s, 4)) for cid, s in ranked[:3]],
    )
    return ranked


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _rrf_merge(
    dense_ranked: list[tuple[str, float]],
    sparse_ranked: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge dense and sparse rankings with RRF (k=60)."""
    rrf: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_ranked):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(sparse_ranked):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    logger.debug(
        "RRF merge | dense_inputs=%d | sparse_inputs=%d | merged_total=%d | top5=%s",
        len(dense_ranked), len(sparse_ranked), len(merged),
        [(cid, round(s, 5)) for cid, s in merged[:5]],
    )
    return merged


# ─── Lookup helper ────────────────────────────────────────────────────────────

def _chunk_by_id(chunk_id: str) -> dict:
    for chunk in _index:
        if chunk["id"] == chunk_id:
            return chunk
    return {}


# ─── Public retrieval API ─────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> dict:
    """
    Full hybrid retrieval pipeline: expand → dense → BM25 → RRF → deduplicate.

    Returns:
        { "results": [...], "top_score": float, "flag_for_human": bool }
    """
    t0 = time.time()
    logger.info(
        "RAG retrieve | query=%.60r | top_k=%d | expansion=%s | bm25=%s",
        query, top_k, QUERY_EXPANSION_ENABLED, BM25_AVAILABLE,
    )

    # ── Step 1: query expansion ───────────────────────────────────────────────
    queries = _expand_query(query) if QUERY_EXPANSION_ENABLED else [query]
    if len(queries) == 1:
        logger.debug("Query expansion disabled or returned single variant")

    # ── Step 2: dense retrieval — best score per chunk across all variants ────
    best_dense_score: dict[str, float] = {}
    for q in queries:
        q_emb = embed_text(q)
        for chunk_id, score in _dense_search(q_emb):
            if score > best_dense_score.get(chunk_id, 0.0):
                best_dense_score[chunk_id] = score
    dense_ranked = sorted(best_dense_score.items(), key=lambda x: x[1], reverse=True)
    logger.debug(
        "Dense multi-variant | variants=%d | unique_chunks=%d | best_score=%.4f",
        len(queries), len(dense_ranked),
        dense_ranked[0][1] if dense_ranked else 0.0,
    )

    # ── Step 3: sparse retrieval on original query ────────────────────────────
    sparse_ranked = _sparse_search(query)

    # ── Step 4: RRF merge ─────────────────────────────────────────────────────
    if BM25_AVAILABLE and sparse_ranked:
        merged = _rrf_merge(dense_ranked, sparse_ranked)
        retrieval_mode = "hybrid_rrf"
    else:
        merged = dense_ranked
        retrieval_mode = "dense_only"
        logger.debug("Retrieval mode | mode=dense_only | reason=bm25_unavailable_or_empty")

    # ── Step 5: assemble results, deduplicate sub-chunks ─────────────────────
    results: list[dict] = []
    seen_base_ids: set[str] = set()
    skipped_duplicates = 0

    for chunk_id, _ in merged:
        chunk = _chunk_by_id(chunk_id)
        if not chunk:
            continue
        base_id = chunk_id.split("_s")[0]
        if base_id in seen_base_ids:
            skipped_duplicates += 1
            continue
        seen_base_ids.add(base_id)
        results.append({
            "id": chunk_id,
            "title": chunk["title"],
            "content": chunk["content"],
            "category": chunk.get("category", ""),
            "tags": chunk.get("tags", []),
            "score": round(best_dense_score.get(chunk_id, 0.0), 4),
        })
        if len(results) >= top_k:
            break

    top_score = results[0]["score"] if results else 0.0
    flag = top_score < SIMILARITY_THRESHOLD
    elapsed = time.time() - t0

    # ── Summary log ───────────────────────────────────────────────────────────
    logger.info(
        "RAG result | query=%.50r | mode=%s | top_score=%.4f | threshold=%.2f "
        "| flag_for_human=%s | results=%d | skipped_dupes=%d | elapsed=%.3fs",
        query, retrieval_mode, top_score, SIMILARITY_THRESHOLD,
        flag, len(results), skipped_duplicates, elapsed,
    )
    for i, r in enumerate(results):
        logger.debug(
            "  result[%d] | id=%s | score=%.4f | title=%s",
            i, r["id"], r["score"], r["title"],
        )

    if flag:
        logger.warning(
            "RAG score below threshold | query=%.60r | top_score=%.4f | threshold=%.2f "
            "| will_flag_for_human=True",
            query, top_score, SIMILARITY_THRESHOLD,
        )

    return {"results": results, "top_score": top_score, "flag_for_human": flag}


# ─── Feedback loop: add admin-contributed chunk ───────────────────────────────

def add_chunk(
    title: str,
    content: str,
    category: str = "learned",
    tags: list[str] = [],
) -> dict:
    """Add an admin-contributed chunk to the live index and persist to disk."""
    global _bm25

    logger.info(
        "add_chunk | title=%.60r | category=%s | tags=%s",
        title, category, tags,
    )

    new_id = f"kb_{len(_raw_chunks) + 1:03d}"
    new_chunk = {"id": new_id, "category": category, "title": title, "content": content, "tags": tags}

    sub_count = 0
    for sub in _split_with_overlap(new_chunk):
        sub["embedding"] = embed_text(f"{sub['title']}. {sub['content']}")
        _index.append(sub)
        if BM25_AVAILABLE:
            _bm25_corpus.append(_tokenize(f"{sub['title']}. {sub['content']}"))
        sub_count += 1

    if BM25_AVAILABLE:
        _bm25 = BM25Okapi(_bm25_corpus)
        logger.debug("BM25 index rebuilt | total_docs=%d", len(_bm25_corpus))

    raw = load_knowledge_base()
    raw.append(new_chunk)
    with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    logger.info(
        "add_chunk complete | id=%s | sub_chunks_added=%d | index_size=%d | kb_size=%d",
        new_id, sub_count, len(_index), len(raw),
    )
    return new_chunk


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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