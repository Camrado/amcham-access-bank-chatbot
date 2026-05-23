# AccessBank AI Support Agent — Chatbot Module

Backend integration reference for the `chatbot/` package.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Public API](#public-api)
   - [Agent.handle()](#agenthandle)
   - [AgentResponse](#agentresponse)
   - [Database helpers](#database-helpers)
   - [RAG retrieve()](#rag-retrieve)
6. [Message Pipeline](#message-pipeline)
7. [Multi-turn State Management](#multi-turn-state-management)
8. [Database Schema](#database-schema)
9. [Knowledge Base](#knowledge-base)
10. [Flagging & Admin Queue](#flagging--admin-queue)
11. [Running Tests](#running-tests)

---

## Overview

This module is the core AI layer of the AccessBank customer support chatbot. It:

- **Classifies** each incoming message as a `question`, `issue`, or `unclear`
- **Answers questions** using RAG (Retrieval-Augmented Generation) against a bank knowledge base
- **Collects missing details** across multiple turns when a customer reports an issue
- **Creates support cases** in SQLite once enough information is gathered
- **Routes flagged conversations** to a human admin queue
- **Runs a safety guardrail** on every outgoing message to block any credential requests

The `Agent` class is stateless. The **caller** (Telegram bot, REST API, admin panel) owns conversation history and must pass it in on every call.

---

## File Structure

```
chatbot/
├── agent.py              # Main Agent class and all pipeline logic
├── rag_loader.py         # Knowledge base loading, embedding, and retrieval
├── prompts.py            # All LLM system prompts (imported by agent.py)
├── knowledge_base.json   # Bank FAQ entries used by the RAG system
├── sample_conversations.json  # Labelled example conversations for development/testing
└── test_agent.py         # Full test suite (RAG, intent, safety, end-to-end, edge cases)
```

---

## Setup

### Requirements

```
openai>=1.0.0
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `DB_PATH` | No | `cases.db` | Path to the SQLite database file |

```bash
export OPENAI_API_KEY=sk-...
export DB_PATH=/var/data/cases.db   # optional
```

The database file is created automatically on first run if it does not exist.

### Python path

`agent.py` imports from `chatbot.rag_loader`, so the project root (the directory that **contains** `chatbot/`) must be on `sys.path`:

```python
import sys
sys.path.insert(0, "/path/to/project/root")
from chatbot.agent import Agent
```

---

## Quick Start

```python
from chatbot.agent import Agent

agent = Agent()  # initialises DB, loads RAG index

# Turn 1 — user reports an issue
response = agent.handle(
    user_id="telegram_12345",
    message="My card was declined but money was taken",
    history=[],
)
print(response.text)          # ask for more details
print(response.department)    # "Card Operations"

# Turn 2 — user provides details; pass pending state from turn 1
history = [
    {"role": "user",      "content": "My card was declined but money was taken"},
    {"role": "assistant", "content": response.text},
]
response2 = agent.handle(
    user_id="telegram_12345",
    message="It was today at 3pm, amount 85 AZN, card ending 4421",
    history=history,
    pending_department=response.department,
    pending_missing_info=[],       # signal: all info collected
)
print(response2.case_id)       # "CASE-A1B2C3D4"
```

---

## Public API

### `Agent.handle()`

```python
agent.handle(
    user_id: str,
    message: str,
    history: list[dict],
    pending_department: str | None = None,
    pending_missing_info: list[str] | None = None,
) -> AgentResponse
```

| Parameter | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique identifier for the user (Telegram ID, session token, etc.) |
| `message` | `str` | The user's latest message |
| `history` | `list[dict]` | Prior turns **not including** the current message. Each dict: `{"role": "user"/"assistant", "content": "..."}` |
| `pending_department` | `str \| None` | Pass the `department` from the previous turn while collecting issue details |
| `pending_missing_info` | `list[str] \| None` | Remaining fields still needed. Pass `[]` once all info is collected to trigger case creation |

**Returns** an `AgentResponse`.

**Raises** `KeyError` if `OPENAI_API_KEY` is not set.

---

### `AgentResponse`

```python
@dataclass
class AgentResponse:
    text: str                       # Message to send back to the user
    intent: str                     # "question" | "issue" | "unclear"
    case_id: str | None             # Set when a new support case is created
    department: str | None          # Set when the issue is routed to a department
    flagged: bool                   # True → conversation is in the admin queue
    flag_reason: str | None         # Human-readable reason for flagging
    rag_top_score: float | None     # Cosine similarity of the best RAG match
```

**Decision table for the caller:**

| `flagged` | `case_id` | `intent` | Action |
|---|---|---|---|
| `True` | `None` | any | Notify admin queue; show `text` to user |
| `False` | `None` | `"question"` | Send `text` to user; no follow-up needed |
| `False` | `None` | `"issue"` | Send `text` to user; persist `department` and re-send on next turn as `pending_department` |
| `False` | `"CASE-..."` | `"issue"` | Case created; send `text`; conversation is complete |

**Valid `department` values:**

```
Digital Banking
Card Operations
Transfers & Payments
Loans & Applications
Customer Service
```

---

### Database helpers

These are exported from `agent.py` for use by the admin panel.

```python
from chatbot.agent import get_case, update_case_status, init_db
```

#### `get_case(case_id, db_path?) -> dict | None`

Returns the full case record or `None` if not found.

```python
{
    "id": "CASE-A1B2C3D4",
    "user_id": "telegram_12345",
    "department": "Card Operations",
    "summary": "Customer reports ...",
    "status": "open",               # open | pending | resolved | closed
    "history": [...],               # deserialized list of message dicts
    "created_at": "2024-05-22T14:30:00",
    "updated_at": "2024-05-22T14:30:00",
    "email_ref": null
}
```

#### `update_case_status(case_id, status, db_path?)`

Valid statuses: `open`, `pending`, `resolved`, `closed`.

#### `init_db(db_path?)`

Creates the `cases` and `flagged_conversations` tables if they do not exist. Called automatically by `Agent.__init__()`.

---

### RAG `retrieve()`

```python
from chatbot.rag_loader import retrieve

result = retrieve("What are your loan interest rates?", top_k=3)
```

**Returns:**

```python
{
    "results": [
        {
            "id": "kb_012",
            "title": "Consumer Loan",
            "content": "...",
            "category": "loans",
            "tags": ["loan", "interest", "application"],
            "score": 0.8412
        },
        ...
    ],
    "top_score": 0.8412,
    "flag_for_human": False   # True if top_score < 0.40
}
```

The RAG index is built once at **module import time**. All embeddings are generated via `text-embedding-3-small` (falls back to `text-embedding-ada-002`). Startup takes a few seconds while the knowledge base is embedded.

#### `add_chunk()` — live knowledge base updates

```python
from chatbot.rag_loader import add_chunk

add_chunk(
    title="New FAQ Entry",
    content="Answer text here...",
    category="general",
    tags=["tag1", "tag2"],
)
```

Adds the chunk to both the live in-memory index and persists it to `knowledge_base.json`. Use this for admin-driven corrections without restarting the service.

---

## Message Pipeline

Each call to `agent.handle()` runs this sequence:

```
User message
     │
     ▼
[1] classify_intent()          model: gpt-4o-mini
     │                         output: intent, department, missing_info, flag_for_human
     │
     ├─ flag_for_human=True ──► save_flagged() → AgentResponse(flagged=True)
     │
     ├─ intent="question" ────► answer_question() via RAG
     │                              ├─ RAG score OK ──► run_safety_check() → AgentResponse
     │                              └─ RAG score low ─► save_flagged() → AgentResponse(flagged=True)
     │
     └─ intent="issue" ───────► collect_missing_info() or create_case()
                                     │
                                     ├─ still missing info ──► run_safety_check() → AgentResponse
                                     └─ all info collected ──► summarise_case() (gpt-4o)
                                                                 → create_case() → AgentResponse(case_id=...)
```

**Models used:**

| Step | Model |
|---|---|
| Intent classification | `gpt-4o-mini` |
| RAG answer generation | `gpt-4o-mini` |
| Issue info collection | `gpt-4o-mini` |
| Safety guardrail | `gpt-4o-mini` |
| Case summarization | `gpt-4o` |

---

## Multi-turn State Management

The `Agent` is stateless — it holds no per-user memory. The **backend/caller** must:

1. Store `history` per `user_id` between turns (in-memory dict, Redis, or DB).
2. When `response.intent == "issue"` and `response.case_id is None`, persist `response.department` as `pending_department` for the user.
3. On the next turn from that user, pass `pending_department` and the updated `pending_missing_info`.
4. When you judge all info is collected (or when the user has provided enough turns), pass `pending_missing_info=[]` to trigger case creation.

**Minimal backend session structure:**

```python
sessions = {}   # user_id → {"history": [], "pending_dept": None, "pending_missing": None}

def on_message(user_id, text):
    session = sessions.setdefault(user_id, {"history": [], "pending_dept": None, "pending_missing": None})
    
    response = agent.handle(
        user_id=user_id,
        message=text,
        history=session["history"],
        pending_department=session["pending_dept"],
        pending_missing_info=session["pending_missing"],
    )
    
    session["history"].append({"role": "user", "content": text})
    session["history"].append({"role": "assistant", "content": response.text})
    
    if response.case_id:
        session["pending_dept"] = None
        session["pending_missing"] = None
    elif response.intent == "issue" and not response.flagged:
        session["pending_dept"] = response.department
        # decrement or clear pending_missing based on your logic
    
    return response.text
```

History is trimmed to the last 10 turns internally before being sent to the LLM.

---

## Database Schema

Two tables are created in the SQLite file at `DB_PATH`.

### `cases`

| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | `CASE-` + 8 uppercase hex chars |
| `user_id` | TEXT | Caller-supplied user identifier |
| `department` | TEXT | One of the five valid departments |
| `summary` | TEXT | AI-generated 2–3 sentence case brief |
| `status` | TEXT | `open` / `pending` / `resolved` / `closed` |
| `history` | TEXT | JSON-serialized conversation array |
| `created_at` | TEXT | UTC ISO-8601 |
| `updated_at` | TEXT | UTC ISO-8601 |
| `email_ref` | TEXT | Optional — for linking to email tickets |

### `flagged_conversations`

| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | `FLAG-` + 8 uppercase hex chars |
| `user_id` | TEXT | Caller-supplied user identifier |
| `history` | TEXT | JSON-serialized conversation array |
| `flag_reason` | TEXT | Why the conversation was flagged |
| `admin_reply` | TEXT | Human agent's reply (set by admin panel) |
| `resolved` | INTEGER | `0` = open, `1` = resolved |
| `created_at` | TEXT | UTC ISO-8601 |
| `updated_at` | TEXT | UTC ISO-8601 |

---

## Knowledge Base

`knowledge_base.json` contains an array of chunks:

```json
{
    "id": "kb_001",
    "category": "general",
    "title": "Working Hours & Branches",
    "content": "Full answer text...",
    "tags": ["hours", "branch", "location"]
}
```

**Categories:** `general`, `cards`, `digital_banking`, `transfers`, `loans`, `accounts`, `learned`

The file is read once at startup. To add entries at runtime without restart, use `add_chunk()` (see [RAG retrieve()](#rag-retrieve)).

---

## Flagging & Admin Queue

A conversation is automatically flagged and saved to `flagged_conversations` in these cases:

- Intent confidence below 75% (`flag_for_human=true` from the LLM)
- Intent is `"unclear"`
- RAG cosine similarity below `0.40` for a question
- Department cannot be determined for an issue

When `AgentResponse.flagged is True`, the backend should:
1. Show `response.text` to the user (a polite hold message is already in it)
2. Alert the admin queue (the record is already in `flagged_conversations`)
3. Pause automated responses for that user until an admin resolves it

The `ADMIN_REPLY_PROMPT` in `prompts.py` is available for admin panels that want to offer AI-suggested replies to human agents.

---

## Running Tests

The test suite makes real OpenAI API calls and covers RAG retrieval, intent classification, safety guardrail, end-to-end flows, and edge cases.

```bash
cd /path/to/project/root
export OPENAI_API_KEY=sk-...
python chatbot/test_agent.py
```

Test databases (`test_agent_flow.db`, `test_edge_cases.db`) are created and cleaned up automatically.

Expected output:

```
AccessBank Agent — Test Suite

────────────────────────────────────────────────────────────
SUITE 1 — RAG Retrieval Performance
  ✓ RAG: Basic hours query should hit kb_001 (312ms)
  ...

SUMMARY
  Total:   28
  Passed:  28
  Failed:  0

  Total time: 45.2s
```
