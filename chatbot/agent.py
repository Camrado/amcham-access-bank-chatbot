"""
agent.py
--------
Core agent logic for the AccessBank AI Support Agent.

Changes in this version:
  - Structured outputs: INTENT_PROMPT, SENTIMENT_PROMPT, and SAFETY_PROMPT
    now use OpenAI's response_format=json_schema (strict mode) instead of
    json_object mode, eliminating parse errors at the source.
  - Department email routing: when the AI cannot answer a question (RAG score
    too low) or cannot determine a department for an issue, it sends an
    escalation email to the relevant department via gmail_sender and sets
    email_routed=True on the response instead of silently flagging.
  - AgentResponse gains an email_routed field.
  - _chat() gains a `schema` parameter for structured outputs.

Public API:
    from agent import Agent

    agent = Agent()
    response = agent.handle(
        user_id="user_123",
        message="My card was declined but money was taken",
        history=[{"role": "user", "content": "..."}, ...]
    )
    print(response.text)
    print(response.case_id)
    print(response.flagged)
    print(response.email_routed)
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

from chatbot.prompts import (
    ANSWER_PROMPT,
    COLLECTOR_PROMPT,
    INTENT_PROMPT,
    INTENT_SCHEMA,
    SAFETY_PROMPT,
    SAFETY_SCHEMA,
    SENTIMENT_PROMPT,
    SENTIMENT_SCHEMA,
    SUMMARY_PROMPT,
)
from chatbot.rag_loader import retrieve
from chatbot.anomaly import check_anomaly, init_anomaly_table
from chatbot.case_similarity import index_case, find_similar_cases, format_similarity_hint

# ─── Optional email routing (soft dependency) ─────────────────────────────────
# email_service handles Gmail auth and department mapping.
# Imported lazily so the agent starts up even if Gmail credentials are not
# yet configured — any send failure is caught and logged non-fatally.
try:
    from email_service import send_email, DEPARTMENT_EMAILS
    _EMAIL_AVAILABLE = True
except Exception:
    _EMAIL_AVAILABLE = False
    logging.getLogger("agent").warning(
        "email_service not available — email routing disabled. "
        "Ensure credentials.json is present and google-auth libraries are installed."
    )

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

# ─── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_PATH = os.environ.get("DB_PATH", "cases.db")

FAST_MODEL = "gpt-4o-mini"   # intent, safety, collector, sentiment
SMART_MODEL = "gpt-4o"       # summarisation (official case record)

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
    language: str = "en"              # az | ru | en | other


@dataclass
class AgentResponse:
    text: str                          # message to send back to the user
    intent: str                        # "question" | "issue" | "unclear"
    case_id: Optional[str] = None
    department: Optional[str] = None
    flagged: bool = False
    flag_reason: Optional[str] = None
    rag_top_score: Optional[float] = None
    language: Optional[str] = None
    sentiment: Optional[str] = None
    urgency: Optional[str] = None
    priority_boost: bool = False
    similar_cases: Optional[list] = None
    anomaly: Optional[dict] = None
    # NEW: set True when an escalation email was dispatched to a department
    email_routed: bool = False


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
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
    logger.info("Flagged: %s | user=%s | reason=%s", flag_id, user_id, flag_reason)
    return flag_id


def get_case(case_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["history"] = json.loads(result["history"])
    return result


def update_case_status(case_id: str, status: str, db_path: str = DB_PATH) -> None:
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
    schema: Optional[dict] = None,
) -> str:
    """
    Single OpenAI chat completion.

    Priority order for response_format:
      1. `schema` provided → json_schema (structured outputs, strict mode)
      2. `json_mode=True`  → json_object (legacy, kept for SUMMARY_PROMPT etc.)
      3. Neither           → plain text

    Returns the text content of the first completion choice.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": schema,
        }
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _trim_history(history: list[dict]) -> list[dict]:
    return history[-(MAX_HISTORY_TURNS * 2):]


def _history_to_text(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ─── AI pipeline steps ────────────────────────────────────────────────────────

def classify_intent(message: str, history: list[dict]) -> IntentResult:
    """
    Step 1: Classify intent and route to department.
    Uses structured output (INTENT_SCHEMA) — no JSON parse errors possible.
    """
    logger.info("Classifying intent: %.80s", message)

    trimmed = _trim_history(history)
    raw = _chat(
        system=INTENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        schema=INTENT_SCHEMA,   # ← structured output
    )

    # With json_schema the response is always valid JSON — no try/except needed.
    # We still guard defensively in case of unexpected API behaviour.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Intent JSON parse failed (unexpected): %s", raw)
        return IntentResult(
            intent="unclear",
            confidence=0.0,
            department=None,
            missing_info=[],
            flag_for_human=True,
            reasoning="Failed to parse intent classification response.",
        )

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


def answer_question(message: str, history: list[dict]) -> tuple[Optional[str], float]:
    """Step 2a: Hybrid RAG answer. Returns (answer_text | None, top_rag_score)."""
    rag = retrieve(message, top_k=3)
    top_score = rag["top_score"]

    if rag["flag_for_human"]:
        logger.info("RAG score too low (%.2f) — routing to department.", top_score)
        return None, top_score

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
    lang_name = {"ru": "Russian", "az": "Azerbaijani", "en": "English"}.get(language, "English")
    return _chat(
        system=(
            f"You are a customer support assistant for AccessBank. "
            f"Write a short, polite message (2–3 sentences) telling the customer that "
            f"their query has been forwarded to the relevant specialist team and that "
            f"they can call *8880 for immediate assistance. "
            f"Write ONLY in {lang_name}."
        ),
        messages=[{"role": "user", "content": "Generate the message."}],
        model=FAST_MODEL,
    )


def run_safety_check(draft: str) -> str:
    """
    Safety guardrail — strips credential requests from every outgoing message.
    Uses structured output (SAFETY_SCHEMA) for guaranteed valid JSON.
    """
    raw = _chat(
        system=SAFETY_PROMPT,
        messages=[{"role": "user", "content": f"Draft response to check:\n\n{draft}"}],
        model=FAST_MODEL,
        schema=SAFETY_SCHEMA,   # ← structured output
    )
    try:
        data = json.loads(raw)
        if not data.get("safe", True):
            logger.warning("Safety violation: %s", data.get("violation"))
        return data.get("cleaned_response", draft)
    except json.JSONDecodeError:
        logger.error("Safety check JSON parse failed — returning original draft")
        return draft


def detect_sentiment(message: str, history: list[dict]) -> dict:
    """
    Detect sentiment, urgency, and financial loss mention.
    Uses structured output (SENTIMENT_SCHEMA) for guaranteed valid JSON.
    """
    trimmed = _trim_history(history)
    raw = _chat(
        system=SENTIMENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        schema=SENTIMENT_SCHEMA,   # ← structured output
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
        return {
            "sentiment": "neutral",
            "urgency": "low",
            "priority_boost": False,
            "financial_loss_mentioned": False,
            "reason": "",
        }


# ─── Case-readiness check ─────────────────────────────────────────────────────

def _is_case_ready(history: list[dict], missing_info: list[str]) -> bool:
    user_turns = sum(1 for m in history if m["role"] == "user")
    return len(missing_info) == 0 and user_turns >= 2


# ─── Email routing helper ─────────────────────────────────────────────────────

def _build_routing_html(
    flag_id: str,
    user_id: str,
    department: str,
    message: str,
    full_history: list[dict],
    flag_reason: str,
    urgency: str,
    priority_boost: bool,
    sentiment: str,
    routing_type: str,
) -> str:
    """Build a branded HTML email body for department routing alerts."""
    priority_badge = (
        '<span style="background:#ef4444;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:12px;font-weight:600;">HIGH PRIORITY</span>'
        if priority_boost else
        '<span style="background:#f59e0b;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:12px;font-weight:600;">STANDARD</span>'
    )

    routing_reason = {
        "unanswerable_question": (
            "The AI agent could not find a relevant answer in the knowledge base. "
            "Human expertise is required to respond."
        ),
        "unroutable_issue": (
            "The AI agent could not determine the correct department for this issue. "
            "Please review and assign manually."
        ),
    }.get(routing_type, flag_reason)

    # Last 10 turns of conversation formatted as HTML rows
    recent = full_history[-10:]
    history_rows = "".join(
        f"""<tr>
              <td style="padding:6px 8px;color:#64748b;white-space:nowrap;vertical-align:top;">
                {'Customer' if m['role'] == 'user' else 'Agent'}
              </td>
              <td style="padding:6px 8px;color:#0f172a;">{m['content'][:400]}</td>
            </tr>"""
        for m in recent
    )

    return f"""
    <div style="font-family:sans-serif;max-width:640px;margin:0 auto;">
      <div style="background:#0f1f35;padding:20px 24px;border-radius:8px 8px 0 0;">
        <img src="https://upload.wikimedia.org/wikipedia/en/4/46/AccessBank_Azerbaijan_logo.svg"
             alt="AccessBank" style="height:28px;filter:brightness(0) invert(1);">
      </div>
      <div style="border:1px solid #e2e8f0;border-top:none;padding:28px 24px;border-radius:0 0 8px 8px;">
        <h2 style="margin:0 0 4px;color:#0f172a;font-size:18px;">
          AI Routing Alert &nbsp;{priority_badge}
        </h2>
        <p style="margin:0 0 24px;color:#64748b;font-size:13px;">
          {flag_id} &middot; {department}
        </p>

        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;width:140px;">User ID</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;font-weight:500;">{user_id}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;">Urgency</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;font-weight:500;">{urgency.upper()}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;">Sentiment</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;">{sentiment}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;vertical-align:top;">Reason</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;">{routing_reason}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;color:#64748b;vertical-align:top;">Latest Message</td>
            <td style="padding:10px 0;color:#0f172a;">{message[:500]}</td>
          </tr>
        </table>

        <h3 style="margin:24px 0 8px;font-size:14px;color:#0f172a;">
          Conversation History (last 10 turns)
        </h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;
                      background:#f8fafc;border-radius:6px;overflow:hidden;">
          {history_rows}
        </table>

        <p style="margin:28px 0 0;font-size:12px;color:#94a3b8;">
          This is an automated message from AccessBank AI Support.
          Log in to the admin panel to respond.
        </p>
      </div>
    </div>
    """


def _route_via_email(
    user_id: str,
    message: str,
    full_history: list[dict],
    flag_id: str,
    flag_reason: str,
    department: Optional[str],
    sentiment_data: dict,
    routing_type: str,
) -> bool:
    """
    Send a routing alert email to the relevant department via email_service.send_email().
    Uses send_email() directly (not send_escalation_email()) because routing alerts
    are flagged conversations — they don't have a SQLAlchemy Case DB record.
    Returns True on success, False on any error.  Never raises.
    """
    if not _EMAIL_AVAILABLE:
        return False

    target_dept = department or "Customer Service"
    to = DEPARTMENT_EMAILS.get(target_dept, DEPARTMENT_EMAILS.get("Customer Service", ""))
    if not to:
        logger.warning("No email address configured for department '%s'. Skipping.", target_dept)
        return False

    urgency = sentiment_data.get("urgency", "medium")
    priority_boost = bool(sentiment_data.get("priority_boost", False))
    sentiment = sentiment_data.get("sentiment", "neutral")

    priority_marker = "[HIGH PRIORITY] " if priority_boost else ""
    subject = (
        f"{priority_marker}[AccessBank AI] Routing Alert — {target_dept} | "
        f"{flag_id} | Urgency: {urgency.upper()}"
    )

    html_body = _build_routing_html(
        flag_id=flag_id,
        user_id=user_id,
        department=target_dept,
        message=message,
        full_history=full_history,
        flag_reason=flag_reason,
        urgency=urgency,
        priority_boost=priority_boost,
        sentiment=sentiment,
        routing_type=routing_type,
    )

    try:
        send_email(to, subject, html_body)
        logger.info(
            "Routing email sent → %s | flag=%s | dept=%s | urgency=%s",
            to, flag_id, target_dept, urgency,
        )
        return True
    except Exception as exc:
        logger.error("Email routing error (non-fatal): %s", exc)
        return False


# ─── Main Agent class ─────────────────────────────────────────────────────────

class Agent:
    """
    Stateless agent — all state (history, pending intent) is passed in by the
    caller. The caller is responsible for persisting history per user_id.
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
            user_id:               unique identifier for the user
            message:               latest user message text
            history:               full conversation history (NOT including current message)
            pending_department:    department saved from a previous issue-collection turn
            pending_missing_info:  remaining fields still needed (from previous turn)
        """
        logger.info("Handling message for user=%s: %.80s", user_id, message)

        full_history = history + [{"role": "user", "content": message}]

        # ── Step 1: Classify intent + detect language ────────────────────────
        intent_result = classify_intent(message, history)
        detected_language = getattr(intent_result, "language", "en")

        # ── Step 1b: Sentiment & urgency ─────────────────────────────────────
        sentiment_data = detect_sentiment(message, history)

        # ── Step 2: Route based on intent ────────────────────────────────────

        # ── 2A: Flagged → admin queue ─────────────────────────────────────────
        if intent_result.flag_for_human:
            flag_reason = (
                f"Low confidence ({intent_result.confidence:.0%}): {intent_result.reasoning}"
            )
            flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

            # Email the best-guess department so a human sees it immediately
            email_routed = _route_via_email(
                user_id=user_id,
                message=message,
                full_history=full_history,
                flag_id=flag_id,
                flag_reason=flag_reason,
                department=intent_result.department,
                sentiment_data=sentiment_data,
                routing_type="unroutable_issue",
            )

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
                email_routed=email_routed,
            )

        # ── 2B: Question → RAG answer ─────────────────────────────────────────
        if intent_result.intent == "question":
            answer, top_score = answer_question(message, history)

            if answer is None:
                # RAG score too low — flag AND route via email to relevant department.
                # intent_result.department is now always set even for questions (prompt updated).
                flag_reason = (
                    f"RAG score too low ({top_score:.2f}) — no knowledge base match. "
                    f"Routed to {intent_result.department or 'Customer Service'}."
                )
                flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

                email_routed = _route_via_email(
                    user_id=user_id,
                    message=message,
                    full_history=full_history,
                    flag_id=flag_id,
                    flag_reason=flag_reason,
                    department=intent_result.department,
                    sentiment_data=sentiment_data,
                    routing_type="unanswerable_question",
                )

                text = _generate_sorry_message(detected_language)
                safe_text = run_safety_check(text)
                return AgentResponse(
                    text=safe_text,
                    intent="question",
                    flagged=True,
                    flag_reason=flag_reason,
                    department=intent_result.department,
                    rag_top_score=top_score,
                    language=detected_language,
                    sentiment=sentiment_data.get("sentiment"),
                    urgency=sentiment_data.get("urgency"),
                    priority_boost=bool(sentiment_data.get("priority_boost", False)),
                    email_routed=email_routed,
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

        # ── 2C: Issue → collect info then create case ─────────────────────────
        if intent_result.intent == "issue":
            department = pending_department or intent_result.department
            missing_info = (
                pending_missing_info
                if pending_missing_info is not None
                else intent_result.missing_info
            )

            if not department:
                # Cannot determine department → flag and email Customer Service
                flag_reason = "Could not determine department for issue"
                flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

                email_routed = _route_via_email(
                    user_id=user_id,
                    message=message,
                    full_history=full_history,
                    flag_id=flag_id,
                    flag_reason=flag_reason,
                    department="Customer Service",
                    sentiment_data=sentiment_data,
                    routing_type="unroutable_issue",
                )

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
                    email_routed=email_routed,
                )

            if _is_case_ready(full_history, missing_info):
                summary = summarise_case(full_history)

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

                index_case(
                    case_id=case_id,
                    summary=summary,
                    department=department,
                    db_path=self.db_path,
                )

                anomaly = check_anomaly(department=department, db_path=self.db_path)
                if anomaly:
                    logger.warning("Anomaly after case %s: %s", case_id, anomaly["message"])

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

            # Still collecting — ask for next missing field
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
            )

        # ── Fallback ──────────────────────────────────────────────────────────
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
    print(f"Intent: {r.intent} | Flagged: {r.flagged} | Email: {r.email_routed}")
    print(f"Response: {r.text}\n")

    print("── Test 2: Issue — card declined ──")
    history: list[dict] = []
    msg = "My card was declined at a supermarket but money was deducted from my account"
    r = agent.handle("user_002", msg, history)
    print(f"Intent: {r.intent} | Dept: {r.department} | Case: {r.case_id}")
    print(f"Response: {r.text}\n")

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
    print(f"Turn 2 — Case: {r2.case_id} | Flagged: {r2.flagged} | Email: {r2.email_routed}")
    print(f"Response: {r2.text}\n")