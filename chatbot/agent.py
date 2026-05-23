"""
agent.py
--------
Core agent logic for the AccessBank AI Support Agent.

Responsibilities:
  - Classify intent (question / issue / unclear)
  - Route to RAG for questions
  - Collect missing info for issues (multi-turn)
  - Summarize and create cases in SQLite when ready
  - Run safety guardrail on every outgoing message
  - Return structured AgentResponse to callers (Telegram bot, admin panel, etc.)

Public API:
    from agent import Agent

    agent = Agent()
    response = agent.handle(
        user_id="user_123",
        message="My card was declined but money was taken",
        history=[{"role": "user", "content": "..."}, ...]
    )
    print(response.text)
    print(response.case_id)       # set when a case is created
    print(response.flagged)       # True → goes to admin queue
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from openai import OpenAI

from chatbot.prompts import (
    ANSWER_PROMPT,
    COLLECTOR_PROMPT,
    INTENT_PROMPT,
    SAFETY_PROMPT,
    SUMMARY_PROMPT,
)
from chatbot.rag_loader import retrieve
from chatbot.anomaly import check_anomaly, init_anomaly_table
from chatbot.case_similarity import index_case, find_similar_cases, format_similarity_hint
from chatbot.prompts import SENTIMENT_PROMPT

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_PATH = os.environ.get("DB_PATH", "cases.db")

FAST_MODEL = "gpt-4o-mini"   # intent, safety, collector
SMART_MODEL = "gpt-4o"       # summarization (official case record)

# How many conversation turns to include in prompts (keep cost under control)
MAX_HISTORY_TURNS = 10

DEPARTMENTS = {
    "Digital Banking",
    "Card Operations",
    "Transfers & Payments",
    "Loans & Applications",
    "Customer Service",
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    intent: str                        # "question" | "issue" | "unclear"
    confidence: float
    department: Optional[str]
    missing_info: list[str]
    flag_for_human: bool
    reasoning: str
    language: str = "en"              # detected: az | ru | en | other


@dataclass
class AgentResponse:
    text: str                          # message to send back to the user
    intent: str                        # "question" | "issue" | "unclear"
    case_id: Optional[str] = None      # set when a new case is created
    department: Optional[str] = None   # set when escalation is determined
    flagged: bool = False              # True → route to admin queue
    flag_reason: Optional[str] = None  # why it was flagged
    rag_top_score: Optional[float] = None
    language: Optional[str] = None     # detected language: az | ru | en | other
    sentiment: Optional[str] = None    # positive | neutral | frustrated | angry | distressed
    urgency: Optional[str] = None      # low | medium | high | critical
    priority_boost: bool = False       # True → surface at top of admin queue
    similar_cases: Optional[list] = None  # similar past cases from similarity index
    anomaly: Optional[dict] = None     # set if a department spike was detected


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """Create the cases table if it does not exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                department  TEXT NOT NULL,
                summary     TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'open',
                history     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                email_ref   TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flagged_conversations (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                history         TEXT NOT NULL,
                flag_reason     TEXT,
                admin_reply     TEXT,
                resolved        INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.commit()
    init_anomaly_table(db_path)
    logger.info("Database initialised at %s", db_path)


def create_case(
    user_id: str,
    department: str,
    summary: str,
    history: list[dict],
    db_path: str = DB_PATH,
) -> str:
    """Insert a new case and return its ID."""
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (id, user_id, department, summary, status, history, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
            """,
            (case_id, user_id, department, summary, json.dumps(history), now, now),
        )
        conn.commit()
    logger.info("Case created: %s | dept=%s | user=%s", case_id, department, user_id)
    return case_id


def save_flagged(
    user_id: str,
    history: list[dict],
    flag_reason: str,
    db_path: str = DB_PATH,
) -> str:
    """Save a flagged conversation to the admin queue."""
    flag_id = f"FLAG-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO flagged_conversations (id, user_id, history, flag_reason, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (flag_id, user_id, json.dumps(history), flag_reason, now, now),
        )
        conn.commit()
    logger.info("Flagged conversation saved: %s | user=%s | reason=%s", flag_id, user_id, flag_reason)
    return flag_id


def get_case(case_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Retrieve a single case by ID."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["history"] = json.loads(result["history"])
    return result


def update_case_status(case_id: str, status: str, db_path: str = DB_PATH) -> None:
    """Update case status: open | pending | resolved | closed."""
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, case_id),
        )
        conn.commit()
    logger.info("Case %s status → %s", case_id, status)


# ─── OpenAI helpers ───────────────────────────────────────────────────────────

def _chat(
    system: str,
    messages: list[dict],
    model: str = FAST_MODEL,
    json_mode: bool = False,
) -> str:
    """Single OpenAI chat call. Returns the text content."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 1000,
        "temperature": 0.2,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY_TURNS turns to control token usage."""
    return history[-(MAX_HISTORY_TURNS * 2):]


def _history_to_text(history: list[dict]) -> str:
    """Convert history list to readable transcript for prompts."""
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ─── AI pipeline steps ────────────────────────────────────────────────────────

def classify_intent(message: str, history: list[dict]) -> IntentResult:
    """Step 1: Classify intent and route to department."""
    logger.info("Classifying intent for message: %.80s", message)

    trimmed = _trim_history(history)
    raw = _chat(
        system=INTENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        json_mode=True,
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Intent JSON parse failed: %s", raw)
        return IntentResult(
            intent="unclear",
            confidence=0.0,
            department=None,
            missing_info=[],
            flag_for_human=True,
            reasoning="Failed to parse intent classification response.",
        )

    # Sanitise department value
    department = data.get("department")
    if department not in DEPARTMENTS:
        department = None

    result = IntentResult(
        intent=data.get("intent", "unclear"),
        confidence=float(data.get("confidence", 0.0)),
        department=department,
        missing_info=data.get("missing_info", []),
        flag_for_human=bool(data.get("flag_for_human", True)),
        reasoning=data.get("reasoning", ""),
        language=data.get("language") or "en",
    )
    logger.info(
        "Intent=%s | confidence=%.2f | dept=%s | flag=%s",
        result.intent, result.confidence, result.department, result.flag_for_human,
    )
    return result


def answer_question(message: str, history: list[dict]) -> tuple[str, float]:
    """Step 2a: RAG-based answer for questions. Returns (answer_text, top_rag_score)."""
    rag = retrieve(message, top_k=3)
    top_score = rag["top_score"]

    if rag["flag_for_human"]:
        logger.info("RAG score too low (%.2f), flagging for human", top_score)
        return None, top_score  # caller will handle flagging

    context = "\n\n".join(
        f"[{r['title']}]\n{r['content']}" for r in rag["results"]
    )
    system = ANSWER_PROMPT.replace("{context}", context)

    trimmed = _trim_history(history)
    answer = _chat(
        system=system,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
    )
    logger.info("RAG answer generated (top_score=%.2f)", top_score)
    return answer, top_score


def collect_missing_info(
    message: str,
    history: list[dict],
    department: str,
    missing_info: list[str],
) -> str:
    """Step 2b: Ask for one missing detail needed to create a case."""
    system = COLLECTOR_PROMPT.replace(
        "{department}", department
    ).replace(
        "{missing_info}", ", ".join(missing_info) if missing_info else "none"
    )
    trimmed = _trim_history(history)
    reply = _chat(
        system=system,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
    )
    logger.info("Collector reply generated for dept=%s", department)
    return reply


def summarise_case(history: list[dict]) -> str:
    """Step 3: Summarise conversation into a case brief using the smarter model."""
    transcript = _history_to_text(history)
    system = SUMMARY_PROMPT.replace("{conversation}", transcript)
    summary = _chat(
        system=system,
        messages=[{"role": "user", "content": "Summarise the above conversation."}],
        model=SMART_MODEL,
    )
    logger.info("Case summary generated (%d chars)", len(summary))
    return summary


def _generate_sorry_message(language: str) -> str:
    """Generate a polite specialist-redirect message in the detected language."""
    lang_name = {"ru": "Russian", "az": "Azerbaijani", "en": "English"}.get(language, "English")
    return _chat(
        system=(
            f"You are a customer support assistant for AccessBank. "
            f"Write a short, polite message (2–3 sentences) telling the customer you want to connect "
            f"them with a specialist for accurate information, and that they can call *8880 for immediate "
            f"assistance. Write ONLY in {lang_name}."
        ),
        messages=[{"role": "user", "content": "Generate the message."}],
        model=FAST_MODEL,
    )


def run_safety_check(draft: str) -> str:
    """Step 4: Safety guardrail — blocks any sensitive credential requests."""
    raw = _chat(
        system=SAFETY_PROMPT,
        messages=[{"role": "user", "content": f"Draft response to check:\n\n{draft}"}],
        model=FAST_MODEL,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        if not data.get("safe", True):
            logger.warning("Safety violation detected: %s", data.get("violation"))
        return data.get("cleaned_response", draft)
    except json.JSONDecodeError:
        logger.error("Safety check JSON parse failed, returning original draft")
        return draft


# ─── Sentiment & urgency detection ───────────────────────────────────────────

def detect_sentiment(message: str, history: list[dict]) -> dict:
    """Detect sentiment, urgency, and whether a financial loss is mentioned."""
    trimmed = _trim_history(history)
    raw = _chat(
        system=SENTIMENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        logger.info(
            "Sentiment=%s | urgency=%s | priority_boost=%s",
            data.get("sentiment"), data.get("urgency"), data.get("priority_boost"),
        )
        return data
    except json.JSONDecodeError:
        logger.error("Sentiment JSON parse failed")
        return {"sentiment": "neutral", "urgency": "low", "priority_boost": False,
                "financial_loss_mentioned": False, "reason": ""}


# ─── Case-readiness check ─────────────────────────────────────────────────────

def _is_case_ready(history: list[dict], missing_info: list[str]) -> bool:
    """
    Heuristic: a case is ready when there are no declared missing fields
    AND the conversation has at least 3 user turns (enough context).
    """
    user_turns = sum(1 for m in history if m["role"] == "user")
    return len(missing_info) == 0 and user_turns >= 2


# ─── Main Agent class ─────────────────────────────────────────────────────────

class Agent:
    """
    Stateless agent — all state (history, pending intent) is passed in by the caller.
    The caller (Telegram bot, admin panel) is responsible for persisting history
    per user_id between turns.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db(db_path)
        logger.info("Agent ready. DB=%s", db_path)

    def handle(
        self,
        user_id: str,
        message: str,
        history: list[dict],
        pending_department: Optional[str] = None,
        pending_missing_info: Optional[list[str]] = None,
    ) -> AgentResponse:
        """
        Process one user message and return an AgentResponse.

        Args:
            user_id:               unique identifier for the user (Telegram ID, session ID, etc.)
            message:               latest user message text
            history:               full conversation history so far (NOT including current message)
            pending_department:    if we're mid-issue-collection, pass the dept here
            pending_missing_info:  remaining fields still needed

        Returns:
            AgentResponse with .text to send back, and optional .case_id / .flagged
        """
        logger.info("Handling message for user=%s: %.80s", user_id, message)

        # Append current message to history for context building
        full_history = history + [{"role": "user", "content": message}]

        # ── Step 1: Classify intent + detect language ────────────────────────────
        intent_result = classify_intent(message, history)
        detected_language = getattr(intent_result, "language", "en")

        # ── Step 1b: Sentiment & urgency (runs in parallel conceptually) ─────────
        sentiment_data = detect_sentiment(message, history)

        # ── Step 2: Route based on intent ─────────────────────────────────────

        # ── 2A: Flagged → admin queue ──────────────────────────────────────────
        if intent_result.flag_for_human:
            flag_reason = (
                f"Low confidence ({intent_result.confidence:.0%}): {intent_result.reasoning}"
            )
            save_flagged(user_id, full_history, flag_reason, self.db_path)
            text = (
                "Thank you for reaching out. Your message has been forwarded to one of our "
                "support specialists who will get back to you shortly. "
                "For urgent matters, please call us at *8880."
            )
            safe_text = run_safety_check(text)
            return AgentResponse(
                text=safe_text,
                intent=intent_result.intent,
                flagged=True,
                flag_reason=flag_reason,
                department=intent_result.department,
                language=detected_language,
                sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
            )

        # ── 2B: Question → RAG answer ──────────────────────────────────────────
        if intent_result.intent == "question":
            answer, top_score = answer_question(message, history)

            if answer is None:
                # RAG score too low even for a question → flag
                flag_reason = f"RAG score too low ({top_score:.2f}) for question"
                save_flagged(user_id, full_history, flag_reason, self.db_path)
                text = _generate_sorry_message(detected_language)
                safe_text = run_safety_check(text)
                return AgentResponse(
                    text=safe_text,
                    intent="question",
                    flagged=True,
                    flag_reason=flag_reason,
                    rag_top_score=top_score,
                    language=detected_language,
                    sentiment=sentiment_data.get("sentiment"),
                    urgency=sentiment_data.get("urgency"),
                    priority_boost=bool(sentiment_data.get("priority_boost", False)),
                )

            safe_text = run_safety_check(answer)
            return AgentResponse(
                text=safe_text,
                intent="question",
                rag_top_score=top_score,
                language=detected_language,
                sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
            )

        # ── 2C: Issue → collect info then create case ──────────────────────────
        if intent_result.intent == "issue":
            # Use pending state if caller is continuing a multi-turn collection
            department = pending_department or intent_result.department
            missing_info = (
                pending_missing_info
                if pending_missing_info is not None
                else intent_result.missing_info
            )

            if not department:
                # Department could not be determined — flag for human
                flag_reason = "Could not determine department for issue"
                save_flagged(user_id, full_history, flag_reason, self.db_path)
                text = (
                    "I want to make sure your issue reaches the right team. "
                    "A support specialist will review your case shortly. "
                    "You can also call *8880 for immediate help."
                )
                safe_text = run_safety_check(text)
                return AgentResponse(
                    text=safe_text,
                    intent="issue",
                    flagged=True,
                    flag_reason=flag_reason,
                )

            # Check if we have enough info to create the case
            if _is_case_ready(full_history, missing_info):
                summary = summarise_case(full_history)

                # Find similar past cases BEFORE creating the new one
                similar = find_similar_cases(
                    summary=summary,
                    db_path=self.db_path,
                    top_k=3,
                    min_score=0.55,
                )

                case_id = create_case(
                    user_id=user_id,
                    department=department,
                    summary=summary,
                    history=full_history,
                    db_path=self.db_path,
                )

                # Index the new case for future similarity searches
                index_case(case_id=case_id, summary=summary, department=department, db_path=self.db_path)

                # Check for anomaly (department volume spike)
                anomaly = check_anomaly(department=department, db_path=self.db_path)
                if anomaly:
                    logger.warning("Anomaly detected after case %s: %s", case_id, anomaly["message"])

                text = (
                    f"I've created support case **{case_id}** and escalated it to our "
                    f"**{department}** team. They will review your case and contact you "
                    f"within 1–2 business days. Please save your case ID for reference. "
                    f"Is there anything else I can help you with?"
                )
                safe_text = run_safety_check(text)
                return AgentResponse(
                    text=safe_text,
                    intent="issue",
                    case_id=case_id,
                    department=department,
                    language=detected_language,
                    sentiment=sentiment_data.get("sentiment"),
                    urgency=sentiment_data.get("urgency"),
                    priority_boost=bool(sentiment_data.get("priority_boost", False)),
                    similar_cases=similar,
                    anomaly=anomaly,
                )

            # Still need more info — ask for the next missing field
            reply = collect_missing_info(message, history, department, missing_info)
            safe_reply = run_safety_check(reply)
            return AgentResponse(
                text=safe_reply,
                intent="issue",
                department=department,
                language=detected_language,
                sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
                # case_id is None — not yet created
            )

        # ── Fallback (should not normally be reached) ──────────────────────────
        logger.warning("Unhandled intent state for user=%s", user_id)
        text = (
            "I'm sorry, I didn't quite understand that. Could you please rephrase, "
            "or call us at *8880 for immediate support?"
        )
        return AgentResponse(text=text, intent="unclear")


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = Agent(db_path="test_cases.db")

    print("\n── Test 1: Simple question ──")
    r = agent.handle("user_001", "What are your working hours?", [])
    print(f"Intent: {r.intent} | Flagged: {r.flagged}")
    print(f"Response: {r.text}\n")

    print("── Test 2: Issue — card declined ──")
    history = []
    msg = "My card was declined at a supermarket but money was deducted from my account"
    r = agent.handle("user_002", msg, history)
    print(f"Intent: {r.intent} | Dept: {r.department} | Case: {r.case_id}")
    print(f"Response: {r.text}\n")

    # Simulate second turn providing details
    history = [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": r.text},
    ]
    r2 = agent.handle(
        "user_002",
        "It was today at 3pm, the amount was 85 AZN, card ending 4421",
        history,
        pending_department=r.department,
        pending_missing_info=[],
    )
    print(f"Turn 2 — Case: {r2.case_id} | Flagged: {r2.flagged}")
    print(f"Response: {r2.text}\n")