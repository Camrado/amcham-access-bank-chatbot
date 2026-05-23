# AccessBank AI Support Agent

An AI-powered customer support system for AccessBank built during an AI hackathon. The system handles multilingual customer inquiries in Azerbaijani, Russian, and English — answering questions from a knowledge base, guiding customers through issue reporting, creating support cases, and routing unanswerable queries to the right department when human expertise is needed.

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key
- Google Cloud credentials for Gmail-based email routing

### 1. Clone and install

```bash
git clone <repo-url>
cd amcham-access-bank-chatbot
pip install -r backend/requirements.txt
```

### 2. Configure environment variables

**Chatbot** — create `chatbot/.env` from the example:

```bash
cp chatbot/.env.example chatbot/.env
# Edit chatbot/.env and set:
#   OPENAI_API_KEY=sk-...
```

**Backend** — create `backend/.env` from the example:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set:
#   JWT_SECRET=<random strong secret>
#   GMAIL_SENDER=your-sender@gmail.com
#   EMAIL_DIGITAL_BANKING=...
#   EMAIL_CARD_OPERATIONS=...
#   EMAIL_TRANSFERS=...
#   EMAIL_LOANS=...
#   EMAIL_CUSTOMER_SERVICE=...
```

Email routing is optional — the chatbot works fully without it. If `credentials.json` is absent or the variables are blank, email dispatch is silently skipped.

### 3. Set up Google Gmail credentials

If you want department email routing, place `credentials.json` (downloaded from Google Cloud Console, OAuth 2.0 desktop app) in the `backend/` directory. On first run the browser will open for authentication and create `token.json` automatically.

### 4. Seed the admin account

```bash
cd backend
python seed_admin.py
```

This creates the default admin user for the admin panel.

### 5. Start the server

```bash
cd backend
python run.py
```

The server starts at `http://localhost:8000`. The frontend is served automatically from `http://localhost:8000/index.html`.

### 6. Open the app

| Page | URL | Description |
|------|-----|-------------|
| Landing | `http://localhost:8000/` | Product overview |
| Register | `http://localhost:8000/register.html` | Create a customer account |
| Login | `http://localhost:8000/login.html` | Customer login |
| Chat | `http://localhost:8000/chat.html` | Customer chat interface |
| Admin | `http://localhost:8000/admin.html` | Admin dashboard |

---

## Directory Structure

```
amcham-access-bank-chatbot/
├── backend/                    # FastAPI server + auth + database
│   ├── main.py                 # App entry point, all HTTP routes
│   ├── run.py                  # Uvicorn launcher (adds project root to sys.path)
│   ├── models.py               # SQLAlchemy ORM models (User, Conversation, Message, Case)
│   ├── database.py             # SQLAlchemy engine + session factory
│   ├── init_db.py              # One-shot DB initialisation script
│   ├── seed_admin.py           # Creates default admin user
│   ├── email_service.py        # Gmail API sender + department email map
│   ├── logging_config.py       # Structured logging setup (rotating file + console)
│   ├── auth/                   # JWT authentication (router + service)
│   ├── conversations/          # Conversation management routes
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variable template
│   ├── accessbank.db           # SQLite: users, conversations, messages
│   └── cases.db                # SQLite: cases, flagged conversations, anomalies
│
├── chatbot/                    # AI pipeline (stateless, called by backend)
│   ├── agent.py                # Core Agent class — full pipeline orchestration
│   ├── rag_loader.py           # Knowledge base loader + hybrid retrieval
│   ├── prompts.py              # All LLM system prompts + JSON schemas
│   ├── anomaly.py              # Case volume spike detector
│   ├── case_similarity.py      # Embedding-based similar-case search
│   ├── knowledge_base.json     # Authoritative product knowledge (cards, loans, etc.)
│   ├── sample_conversations.json  # Example conversations for testing
│   ├── test_agent.py           # Agent integration tests
│   └── .env.example            # OPENAI_API_KEY template
│
└── frontend/                   # Static HTML/JS/CSS (served by FastAPI)
    ├── index.html              # Landing page
    ├── login.html              # Customer login
    ├── register.html           # Customer registration
    ├── chat.html               # Customer chat UI
    ├── admin.html              # Admin dashboard (cases, flags, anomalies, stats)
    ├── logger.js               # Browser-side error reporter → POST /api/logs
    └── accessbank-logo.svg     # Brand asset
```

---

## What Was Implemented and Where

### Backend (`backend/`)

**`main.py`** — FastAPI application with all routes:
- `POST /chat` — main customer endpoint; loads conversation history from the database, calls the agent, persists messages, returns the structured response
- `GET /conversations/{id}/messages` — fetch message history for a conversation
- `GET /cases`, `GET /cases/{id}`, `PATCH /cases/{id}/status` — admin case management
- `GET /flagged`, `GET /flagged/{id}`, `POST /flagged/{id}/reply` — admin flagged-conversation queue
- `GET /anomalies`, `POST /anomalies/{id}/resolve` — anomaly alert management
- `GET /dashboard/stats` — aggregate counts for the admin dashboard
- `POST /api/logs` — receives browser-side error reports from `logger.js`
- `StaticFiles` mount — serves the entire `frontend/` directory at `/`

**`email_service.py`** — Gmail API integration using OAuth 2.0. Sends two kinds of emails: escalation emails for newly created support cases, and routing alert emails for unanswerable questions or unroutable issues. Department email addresses are configured via environment variables.

**`models.py`** — SQLAlchemy models for `User`, `Conversation`, `Message`, and `Case` stored in `accessbank.db`.

**`logging_config.py`** — Rotating file handler writing to `backend/logs/` plus console output. Separate loggers for the main app, agent pipeline, RAG, and frontend.

**`auth/`** — JWT-based authentication. Tokens are issued on login, validated as a FastAPI dependency on protected routes.

**`conversations/`** — Routes for creating and listing conversations per user.

---

### Chatbot (`chatbot/`)

**`agent.py`** — The entire AI pipeline lives here. The `Agent` class is stateless; all conversation state is passed in by the caller. `handle()` runs five sequential steps for every message:

1. Intent classification (GPT-4o-mini, structured output)
2. Sentiment and urgency detection (GPT-4o-mini, structured output)
3. Routing: greeting → greeting handler / flagged → admin queue + email / question → RAG / issue → info collection or case creation
4. Safety guardrail on every outgoing response
5. Anomaly check after every case creation

**`rag_loader.py`** — Knowledge base retrieval. On startup: loads `knowledge_base.json`, splits long chunks with 20-word overlap, embeds every sub-chunk via `text-embedding-3-small`, and builds a BM25 sparse index. On each query: expands to 3 variants, runs dense cosine search across all variants, runs BM25 on the original query, merges with Reciprocal Rank Fusion (k=60), deduplicates sub-chunks, and thresholds at cosine score 0.40.

**`prompts.py`** — All system prompts and JSON schemas:
- `INTENT_PROMPT` / `INTENT_SCHEMA` — 4-class intent classifier with language, confidence, department, missing fields, and exploratory flag
- `SENTIMENT_PROMPT` / `SENTIMENT_SCHEMA` — 5-point sentiment, 4-point urgency, financial loss flag, priority boost
- `ANSWER_PROMPT` — RAG answer generator with exploratory/specific dual mode
- `COLLECTOR_PROMPT` — one-field-at-a-time issue detail collector
- `SUMMARY_PROMPT` — 2–3 sentence case brief generator (GPT-4o)
- `GREETING_PROMPT` — warm multilingual greeting generator
- `SAFETY_PROMPT` / `SAFETY_SCHEMA` — credential request detector and cleaner

**`anomaly.py`** — Counts cases per department in a 30-minute rolling window. Triggers an alert when 4+ cases arrive from the same department. Respects a 60-minute cooldown so the same spike doesn't alert repeatedly.

**`case_similarity.py`** — Embeds each case summary on creation and stores the vector. When a case is opened in the admin panel, the top-3 most similar past cases (cosine ≥ 0.55) are returned alongside it.

---

### Frontend (`frontend/`)

**`chat.html`** — Single-page chat UI. Manages multiple conversations in a sidebar, sends messages to `POST /chat`, renders markdown in responses, displays case IDs, department routing notices, and sentiment badges.

**`admin.html`** — Admin dashboard with four tabs: Cases (filterable list with status management), Flagged Conversations (human review queue with reply capability), Anomaly Alerts (active spike alerts with resolve action), and Stats (counts by status, today's case volume, department breakdown).

**`login.html` / `register.html`** — Standard JWT-based auth forms.

**`logger.js`** — Attaches to `window.onerror` and `window.onunhandledrejection`, batches browser errors, and ships them to `POST /api/logs` so browser-side failures appear in server logs.

---

## AI Concepts

### Large Language Models

Two OpenAI models are used, selected by task complexity:

**GPT-4o-mini** handles all time-sensitive steps in the hot path: intent classification, sentiment detection, answer generation from retrieved context, greeting responses, multi-turn issue collection, query expansion, and the safety guardrail. It is fast and cheap enough to run multiple times per user message without noticeable latency.

**GPT-4o** is reserved for case summarisation only — the step where a full conversation transcript is compressed into a 2–3 sentence case brief that becomes a permanent record reviewed by human agents. Quality matters more than speed here, so the stronger model is justified.

---

### Intent Classification

Every message is classified into one of four intents before any other processing happens:

- **Greeting** — conversational openers with no specific request yet
- **Question** — the customer wants information or is exploring a product
- **Issue** — a real problem that needs escalation and case creation
- **Unclear** — genuinely ambiguous even with full conversation history

The classifier also extracts the most relevant internal department, the customer's language (Azerbaijani, Russian, English, or other), a confidence score, a list of missing details still needed to open a case, and an `is_exploratory` flag for messages that just name a product topic without asking a specific question.

Confidence below 0.75, or an unclear intent, triggers the human escalation path regardless of what the message says.

---

### Structured Outputs

Intent classification, sentiment detection, and the safety check all use OpenAI's `json_schema` response format with strict mode enabled. This is meaningfully different from asking the model to "return JSON" in the prompt.

In strict mode, the API enforces the schema at the model sampling level — every field in `required` is guaranteed to be present, `additionalProperties` is blocked, and nullable fields are constrained to specific types. The model cannot produce malformed output. This eliminates an entire category of runtime errors that `json_object` mode is susceptible to and removes the need for defensive JSON parsing and fallback logic.

---

### Sentiment & Urgency Detection

Alongside intent classification, every message is analysed for emotional signal. The output is a five-point sentiment scale (positive → neutral → frustrated → angry → distressed), a four-point urgency level (low → medium → high → critical), a flag for whether financial loss was mentioned, and a boolean priority boost.

Priority-boosted conversations surface at the top of the admin queue and trigger high-priority subject lines in department escalation emails. This ensures that a customer who has lost money or is in distress is seen by a human agent faster than a customer asking a routine question.

---

### Retrieval-Augmented Generation (RAG)

Questions are answered using RAG rather than relying on the model's parametric knowledge. The knowledge base — covering cards, loans, transfers, accounts, fees, digital banking, and general product information — is the authoritative source. The model is only ever asked to synthesise an answer from retrieved context, never to generate facts independently.

#### Dense Retrieval

Each knowledge base chunk is embedded once at startup using `text-embedding-3-small` (with automatic fallback to `text-embedding-ada-002`). On each question, the query is embedded with the same model and cosine similarity is computed against every chunk vector. This catches semantic matches even when the customer uses completely different words than the knowledge base.

#### Sparse Retrieval (BM25)

A BM25 index is built over the same chunks at startup. BM25 is a keyword-frequency ranking function — it captures exact terminology matches that dense retrieval can miss. Banking queries often contain precise terms (SWIFT, IBAN, MIDA, OTP) where keyword matching outperforms semantic similarity.

#### Reciprocal Rank Fusion

The dense and sparse rankings are merged using Reciprocal Rank Fusion (RRF). Each list contributes a score of `1 / (k + rank)` per chunk, where k=60 is the standard constant. A chunk that ranks highly in both lists receives a substantially boosted combined score. A chunk strong in only one signal still contributes but is outranked by chunks that appear strong in both. This consistently outperforms either signal alone on queries that contain both conceptual meaning and specific terminology.

#### Similarity Threshold & Routing

If the best cosine score after retrieval falls below 0.40, the system considers the question unanswerable from the knowledge base. Rather than returning a vague "I don't know" message, the conversation is flagged and routed to the relevant department via email so the customer's question is answered by a human with full context.

---

### Query Expansion

Banking customer queries are often vague or underspecified. "My card isn't working" could refer to six different problems with different solutions. Before retrieval, the original query is rephrased into two alternative variants by the LLM. All three versions are searched independently and the best cosine score per chunk is kept before fusion. This improves recall on queries where the customer's phrasing doesn't closely match the knowledge base wording, without any changes to the knowledge base itself.

---

### Chunk Overlap

The knowledge base uses fixed-size chunks. Hard chunk boundaries create a specific failure mode: if the answer to a question spans the boundary between two chunks, neither chunk alone scores well enough to be retrieved. The solution is overlapping sub-chunks — longer content is split into sliding windows with a 20% word overlap between consecutive windows. A question that matches the end of one conceptual section and the beginning of the next will now score well against the overlapping sub-chunk that contains both. At retrieval time, at most one sub-chunk per original article is returned, so results remain distinct.

---

### Promotional & Exploratory Response Mode

When a customer sends a very short message that just names a product — "Debit card", "Loans", "Mobile app" — they are not asking a specific question. They are expressing interest and expecting to be informed. The `is_exploratory` flag in the intent classification output signals this. The answer prompt has two explicit modes: for exploratory messages it responds enthusiastically with a feature highlight and a call to action; for specific questions it answers directly and factually. The model detects which mode applies from the message content — no branching logic is needed in code.

---

### Multi-Turn Issue Collection

When a customer has a problem that needs to be escalated, the agent needs specific details to create a useful support case — transaction date, amount, card last four digits, reference number, and so on. Rather than asking for everything at once (which feels like a form), the agent asks for one missing detail per conversation turn. This is handled by a dedicated collector prompt that knows which department the issue belongs to and which fields are still missing. The conversation continues naturally until enough information is collected, at which point a case is created and the customer receives a case ID.

---

### Case Summarisation

When a case is ready to be created, the full conversation transcript is summarised into a 2–3 sentence case brief. This brief becomes the permanent case record in the database, the input for similarity search, and what the human agent sees first when reviewing a case. It is generated by the stronger model to ensure quality — human agents rely on it to understand the issue without reading the full conversation.

---

### Semantic Case Similarity

Every case summary is embedded and stored. When a new case is created, its summary is compared against all previous case embeddings using cosine similarity. The top matching past cases are surfaced to the admin agent. This helps agents spot recurring patterns, apply previously successful resolutions, and identify when multiple customers are experiencing the same issue before it becomes large enough to trigger the anomaly detector.

---

### Anomaly Detection

After every case is created, the system checks whether the volume of new cases in the same department over the last 30 minutes has exceeded a threshold of 4 cases. A sudden spike in Card Operations cases, for example, may indicate a systemic card processing issue rather than individual customer problems. When a spike is detected, an alert is created and surfaced on the admin dashboard so the team can investigate proactively rather than reactively. A 60-minute cooldown prevents the same spike from alerting repeatedly.

---

### Safety Guardrail

Every outgoing message — regardless of which path generated it — passes through a safety check before being sent to the customer. The check specifically looks for requests for sensitive credentials: PIN, CVV, OTP, password, or full card number. If a violation is found, the problematic content is replaced with a safe alternative that still helps the customer. This runs as a hard gate on every response, not as a soft instruction in the main prompt, so it cannot be bypassed by prompt injection or unusual conversation flows.

---

### Department Email Routing

When the AI cannot help a customer, the conversation is not silently dropped into an admin queue — it is actively routed to the relevant department via email with full context. The email contains the conversation history, the customer's latest message, the reason the AI could not answer, the urgency level, and the sentiment signal. High-priority cases (financial loss, distressed sentiment, critical urgency) are marked clearly in the subject line. This ensures human agents have everything they need to respond without asking the customer to repeat themselves.

---

### Language Detection & Multilingual Responses

Language is detected automatically as part of intent classification and carried through the entire pipeline. Every answer prompt, greeting prompt, and sorry-message generator is instructed to respond in the same language the customer used. No separate translation step is needed — the models handle Azerbaijani, Russian, and English natively, which covers the full customer base of AccessBank.

---

## Next Steps

**Conversation memory compression.** After many turns, passing the full history to every model call increases cost and can dilute intent classification as the model attends to irrelevant early turns. Compressing older turns into a rolling summary while keeping recent turns verbatim would maintain quality while significantly reducing token usage.

**Per-department confidence thresholds.** A single flat threshold for flagging treats a loan application query the same as a working hours question. Higher-stakes departments should require higher confidence before the AI responds autonomously.

**Entity extraction on cases.** Conversation history contains amounts, dates, card last-four digits, and reference numbers that currently exist only as unstructured text. Extracting these into structured fields at case creation time would enable better similarity matching, richer admin views, and downstream analytics.

**Knowledge gap detection.** Queries that score just below the retrieval threshold are the strongest signal that the knowledge base has a gap. Logging these and surfacing them as suggested additions would give the support team a data-driven way to improve coverage over time.

**Topic clustering.** Running unsupervised clustering over case embeddings weekly would surface the top recurring issue categories without any manual tagging — giving operations teams early visibility into systemic problems.

**Streaming responses.** Token-by-token streaming would significantly improve perceived responsiveness for longer answers without changing any of the underlying AI logic.

**Voice input.** The Whisper API could transcribe customer voice messages into text before feeding them into the existing pipeline unchanged, extending the same AI logic to voice channels.
