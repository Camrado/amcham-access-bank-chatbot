"""
agent.py
--------
Core agent logic for the AccessBank AI Support Agent.

Logging strategy:
  - INFO  : every pipeline entry/exit, routing decisions, case/flag creation,
            email routing outcomes — enough to reconstruct any conversation from logs alone
  - DEBUG : intermediate data (intent fields, sentiment fields, RAG scores, history length)
  - WARNING : safety violations, degraded modes, missing configuration
  - ERROR : caught exceptions that were handled without crashing

Department routing logs use the prefix [DEPT ROUTING] for easy grep/filter.
Email routing logs use [EMAIL ROUTING].
Safety check logs use [SAFETY].
"""

import json
import logging
import os
import sqlite3
import time
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
    GREETING_PROMPT,
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

# ─── Optional email routing ────────────────────────────────────────────────────
try:
    from email_service import send_email, DEPARTMENT_EMAILS
    _EMAIL_AVAILABLE = True
except Exception as _email_import_err:
    _EMAIL_AVAILABLE = False
    logging.getLogger("agent").warning(
        "[EMAIL ROUTING] email_service import failed — email routing disabled | error=%s",
        _email_import_err,
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

FAST_MODEL = "gpt-4o-mini"
SMART_MODEL = "gpt-4o"
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
    intent: str                        # "greeting" | "question" | "issue" | "unclear"
    confidence: float
    department: Optional[str]
    missing_info: list[str]
    flag_for_human: bool
    reasoning: str
    language: str = "en"
    is_exploratory: bool = False


@dataclass
class AgentResponse:
    text: str
    intent: str
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
    email_routed: bool = False


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, department TEXT NOT NULL,
                summary TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
                history TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                email_ref TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flagged_conversations (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, history TEXT NOT NULL,
                flag_reason TEXT, admin_reply TEXT, resolved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
    init_anomaly_table(db_path)
    logger.info("Database initialised | path=%s", db_path)


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
            "INSERT INTO cases (id, user_id, department, summary, status, history, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'open', ?, ?, ?)",
            (case_id, user_id, department, summary, json.dumps(history), now, now),
        )
        conn.commit()
    logger.info(
        "Case created | case_id=%s | dept=%s | user=%s | summary_len=%d",
        case_id, department, user_id, len(summary),
    )
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
            "INSERT INTO flagged_conversations (id, user_id, history, flag_reason, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (flag_id, user_id, json.dumps(history), flag_reason, now, now),
        )
        conn.commit()
    logger.info(
        "Flagged conversation saved | flag_id=%s | user=%s | reason=%.80s",
        flag_id, user_id, flag_reason,
    )
    return flag_id


def get_case(case_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        logger.debug("get_case | case_id=%s | found=False", case_id)
        return None
    result = dict(row)
    result["history"] = json.loads(result["history"])
    logger.debug("get_case | case_id=%s | dept=%s | status=%s", case_id, result["department"], result["status"])
    return result


def update_case_status(case_id: str, status: str, db_path: str = DB_PATH) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, case_id),
        )
        conn.commit()
    logger.info("Case status updated | case_id=%s | new_status=%s", case_id, status)


# ─── OpenAI helpers ───────────────────────────────────────────────────────────

def _chat(
    system: str,
    messages: list[dict],
    model: str = FAST_MODEL,
    json_mode: bool = False,
    schema: Optional[dict] = None,
) -> str:
    """Single OpenAI chat completion. schema takes priority over json_mode."""
    t0 = time.time()
    client_instance = OpenAI(api_key=OPENAI_API_KEY)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 1000,
        "temperature": 0.2,
    }
    if schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        mode = "json_schema"
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        mode = "json_object"
    else:
        mode = "text"

    response = client_instance.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    elapsed = time.time() - t0

    logger.debug(
        "_chat | model=%s | mode=%s | input_msgs=%d | output_len=%d | elapsed=%.3fs",
        model, mode, len(messages) + 1, len(content), elapsed,
    )
    return content


def _trim_history(history: list[dict]) -> list[dict]:
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    if len(trimmed) < len(history):
        logger.debug(
            "History trimmed | original=%d | trimmed=%d | max_turns=%d",
            len(history), len(trimmed), MAX_HISTORY_TURNS,
        )
    return trimmed


def _history_to_text(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Agent"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ─── AI pipeline steps ────────────────────────────────────────────────────────

def classify_intent(message: str, history: list[dict]) -> IntentResult:
    """Step 1: Classify intent, language, exploratory flag, and routing department."""
    logger.info(
        "classify_intent | msg=%.80r | history_turns=%d",
        message, len(history),
    )
    t0 = time.time()
    trimmed = _trim_history(history)
    raw = _chat(
        system=INTENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        schema=INTENT_SCHEMA,
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "classify_intent | JSON parse failed (unexpected) | error=%s | raw=%.200s",
            exc, raw,
        )
        return IntentResult(
            intent="unclear", confidence=0.0, department=None, missing_info=[],
            flag_for_human=True, reasoning="Failed to parse intent classification response.",
        )

    department = data.get("department")
    if department not in DEPARTMENTS:
        if department is not None:
            logger.warning(
                "classify_intent | invalid department value | received=%r | nulled",
                department,
            )
        department = None

    result = IntentResult(
        intent=data.get("intent", "unclear"),
        confidence=float(data.get("confidence", 0.0)),
        department=department,
        missing_info=data.get("missing_info", []),
        flag_for_human=bool(data.get("flag_for_human", True)),
        reasoning=data.get("reasoning", ""),
        language=data.get("language") or "en",
        is_exploratory=bool(data.get("is_exploratory", False)),
    )

    logger.info(
        "classify_intent result | intent=%s | confidence=%.2f | dept=%s | lang=%s "
        "| flag=%s | exploratory=%s | missing_info=%s | elapsed=%.3fs",
        result.intent, result.confidence, result.department, result.language,
        result.flag_for_human, result.is_exploratory,
        result.missing_info, time.time() - t0,
    )
    return result


def answer_question(message: str, history: list[dict]) -> tuple[Optional[str], float]:
    """Step 2a: Hybrid RAG answer. Returns (answer_text | None, top_rag_score)."""
    logger.info("answer_question | msg=%.80r", message)
    t0 = time.time()

    rag = retrieve(message, top_k=3)
    top_score = rag["top_score"]

    if rag["flag_for_human"]:
        logger.warning(
            "answer_question | RAG score below threshold | score=%.4f | threshold=%.2f "
            "| returning None — will trigger department routing",
            top_score, 0.40,
        )
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

    logger.info(
        "answer_question | rag_score=%.4f | context_chunks=%d | answer_len=%d | elapsed=%.3fs",
        top_score, len(rag["results"]), len(answer), time.time() - t0,
    )
    return answer, top_score


def generate_greeting(message: str, history: list[dict], language: str) -> str:
    """Generate a warm welcome for greetings and conversation openers."""
    logger.info("generate_greeting | lang=%s | msg=%.60r", language, message)
    trimmed = _trim_history(history)
    reply = _chat(
        system=GREETING_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
    )
    logger.debug("generate_greeting | reply_len=%d", len(reply))
    return reply


def collect_missing_info(
    message: str,
    history: list[dict],
    department: str,
    missing_info: list[str],
) -> str:
    """Ask for the next missing detail needed to create a case."""
    logger.info(
        "collect_missing_info | dept=%s | missing_fields=%s",
        department, missing_info,
    )
    system = COLLECTOR_PROMPT.replace("{department}", department).replace(
        "{missing_info}", ", ".join(missing_info) if missing_info else "none"
    )
    trimmed = _trim_history(history)
    reply = _chat(
        system=system,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
    )
    logger.debug("collect_missing_info | reply_len=%d", len(reply))
    return reply


def summarise_case(history: list[dict]) -> str:
    """Summarise conversation into a case brief (uses SMART_MODEL)."""
    logger.info("summarise_case | history_turns=%d", len(history))
    t0 = time.time()
    transcript = _history_to_text(history)
    system = SUMMARY_PROMPT.replace("{conversation}", transcript)
    summary = _chat(
        system=system,
        messages=[{"role": "user", "content": "Summarise the above conversation."}],
        model=SMART_MODEL,
    )
    logger.info(
        "summarise_case | summary_len=%d | elapsed=%.3fs",
        len(summary), time.time() - t0,
    )
    return summary


def _generate_sorry_message(language: str) -> str:
    """Polite specialist-redirect message in the detected language."""
    logger.debug("_generate_sorry_message | lang=%s", language)
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
    """Safety guardrail — removes credential requests from every outgoing message."""
    raw = _chat(
        system=SAFETY_PROMPT,
        messages=[{"role": "user", "content": f"Draft response to check:\n\n{draft}"}],
        model=FAST_MODEL,
        schema=SAFETY_SCHEMA,
    )
    try:
        data = json.loads(raw)
        is_safe = data.get("safe", True)
        if not is_safe:
            logger.warning(
                "[SAFETY] Violation detected | violation=%s | original_len=%d | cleaned_len=%d",
                data.get("violation"), len(draft), len(data.get("cleaned_response", "")),
            )
        else:
            logger.debug("[SAFETY] Check passed | response_len=%d", len(draft))
        return data.get("cleaned_response", draft)
    except json.JSONDecodeError as exc:
        logger.error(
            "[SAFETY] JSON parse failed — returning original draft | error=%s", exc
        )
        return draft


def detect_sentiment(message: str, history: list[dict]) -> dict:
    """Detect sentiment, urgency, and financial loss mention."""
    logger.debug("detect_sentiment | msg=%.80r", message)
    trimmed = _trim_history(history)
    raw = _chat(
        system=SENTIMENT_PROMPT,
        messages=trimmed + [{"role": "user", "content": message}],
        model=FAST_MODEL,
        schema=SENTIMENT_SCHEMA,
    )
    try:
        data = json.loads(raw)
        logger.info(
            "detect_sentiment | sentiment=%s | urgency=%s | priority_boost=%s "
            "| financial_loss=%s | reason=%.60s",
            data.get("sentiment"), data.get("urgency"),
            data.get("priority_boost"), data.get("financial_loss_mentioned"),
            data.get("reason", ""),
        )
        return data
    except json.JSONDecodeError as exc:
        logger.error("detect_sentiment | JSON parse failed | error=%s", exc)
        return {
            "sentiment": "neutral", "urgency": "low",
            "priority_boost": False, "financial_loss_mentioned": False, "reason": "",
        }


# ─── Case-readiness check ─────────────────────────────────────────────────────

def _is_case_ready(history: list[dict], missing_info: list[str]) -> bool:
    user_turns = sum(1 for m in history if m["role"] == "user")
    ready = len(missing_info) == 0 and user_turns >= 2
    logger.debug(
        "_is_case_ready | user_turns=%d | missing_info=%s | ready=%s",
        user_turns, missing_info, ready,
    )
    return ready


# ─── Email routing ────────────────────────────────────────────────────────────

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
    """Build branded HTML email body for a department routing alert."""
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
          Automated message from AccessBank AI Support. Log in to the admin panel to respond.
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
    Send a routing alert email to the relevant department.
    Uses send_email() directly since routing alerts don't have a SQLAlchemy Case record.
    Returns True on success, False on any error. Never raises.
    """
    target_dept = department or "Customer Service"
    urgency = sentiment_data.get("urgency", "medium")
    priority_boost = bool(sentiment_data.get("priority_boost", False))
    sentiment = sentiment_data.get("sentiment", "neutral")

    logger.info(
        "[DEPT ROUTING] Routing decision | flag_id=%s | target_dept=%s | type=%s "
        "| user=%s | urgency=%s | priority_boost=%s | sentiment=%s",
        flag_id, target_dept, routing_type,
        user_id, urgency, priority_boost, sentiment,
    )

    if not _EMAIL_AVAILABLE:
        logger.warning(
            "[EMAIL ROUTING] Skipped | flag_id=%s | reason=email_service_unavailable",
            flag_id,
        )
        return False

    to = DEPARTMENT_EMAILS.get(target_dept, DEPARTMENT_EMAILS.get("Customer Service", ""))
    if not to:
        logger.warning(
            "[EMAIL ROUTING] Skipped | flag_id=%s | dept=%s | reason=no_email_address_configured",
            flag_id, target_dept,
        )
        return False

    priority_marker = "[HIGH PRIORITY] " if priority_boost else ""
    subject = (
        f"{priority_marker}[AccessBank AI] Routing Alert — {target_dept} | "
        f"{flag_id} | Urgency: {urgency.upper()}"
    )

    html_body = _build_routing_html(
        flag_id=flag_id, user_id=user_id, department=target_dept,
        message=message, full_history=full_history, flag_reason=flag_reason,
        urgency=urgency, priority_boost=priority_boost,
        sentiment=sentiment, routing_type=routing_type,
    )

    t0 = time.time()
    try:
        send_email(to, subject, html_body)
        logger.info(
            "[EMAIL ROUTING] Sent | flag_id=%s | dept=%s | recipient=%s "
            "| subject=%.80s | elapsed=%.3fs",
            flag_id, target_dept, to, subject, time.time() - t0,
        )
        return True
    except Exception as exc:
        logger.error(
            "[EMAIL ROUTING] Failed | flag_id=%s | dept=%s | recipient=%s | error=%s",
            flag_id, target_dept, to, exc,
        )
        return False


# ─── Main Agent class ─────────────────────────────────────────────────────────

class Agent:
    """Stateless agent. All state (history, pending intent) is passed in by the caller."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db(db_path)
        logger.info("Agent initialised | db=%s | fast_model=%s | smart_model=%s",
                    db_path, FAST_MODEL, SMART_MODEL)

    def handle(
        self,
        user_id: str,
        message: str,
        history: list[dict],
        pending_department: Optional[str] = None,
        pending_missing_info: Optional[list[str]] = None,
    ) -> AgentResponse:
        """Process one user message and return an AgentResponse."""
        t_total = time.time()
        logger.info(
            "handle | user=%s | msg=%.80r | history_turns=%d | pending_dept=%s | pending_missing=%s",
            user_id, message, len(history), pending_department, pending_missing_info,
        )

        full_history = history + [{"role": "user", "content": message}]

        # ── Step 1: Classify intent + detect language ────────────────────────
        intent_result = classify_intent(message, history)
        detected_language = getattr(intent_result, "language", "en")

        # ── Step 1b: Sentiment & urgency ─────────────────────────────────────
        sentiment_data = detect_sentiment(message, history)

        # ── Step 2: Route based on intent ────────────────────────────────────

        # ── 2A: Greeting → warm welcome ───────────────────────────────────────
        if intent_result.intent == "greeting":
            logger.info(
                "Pipeline | intent=greeting | user=%s | lang=%s | routing=greeting_handler",
                user_id, detected_language,
            )
            text = generate_greeting(message, history, detected_language)
            safe_text = run_safety_check(text)
            resp = AgentResponse(
                text=safe_text, intent="greeting", language=detected_language,
                sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=False,
            )
            logger.info(
                "handle complete | user=%s | intent=greeting | elapsed=%.3fs",
                user_id, time.time() - t_total,
            )
            return resp

        # ── 2B: Flagged → admin queue + email ─────────────────────────────────
        if intent_result.flag_for_human:
            flag_reason = (
                f"Low confidence ({intent_result.confidence:.0%}): {intent_result.reasoning}"
            )
            logger.warning(
                "Pipeline | intent=%s | confidence=%.2f | flag_for_human=True "
                "| user=%s | reason=%.80s",
                intent_result.intent, intent_result.confidence, user_id, flag_reason,
            )
            flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

            email_routed = _route_via_email(
                user_id=user_id, message=message, full_history=full_history,
                flag_id=flag_id, flag_reason=flag_reason,
                department=intent_result.department, sentiment_data=sentiment_data,
                routing_type="unroutable_issue",
            )

            text = (
                "Thank you for reaching out. Your message has been forwarded to one of our "
                "support specialists who will get back to you shortly. "
                "For urgent matters, please call us at *8880."
            )
            safe_text = run_safety_check(text)
            resp = AgentResponse(
                text=safe_text, intent=intent_result.intent, flagged=True,
                flag_reason=flag_reason, department=intent_result.department,
                language=detected_language, sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
                email_routed=email_routed,
            )
            logger.info(
                "handle complete | user=%s | intent=%s | flagged=True | flag_id=%s "
                "| email_routed=%s | elapsed=%.3fs",
                user_id, intent_result.intent, flag_id, email_routed, time.time() - t_total,
            )
            return resp

        # ── 2C: Question → RAG answer ─────────────────────────────────────────
        if intent_result.intent == "question":
            logger.info(
                "Pipeline | intent=question | exploratory=%s | dept=%s | user=%s",
                intent_result.is_exploratory, intent_result.department, user_id,
            )
            answer, top_score = answer_question(message, history)

            if answer is None:
                # RAG score too low → flag AND email the relevant department
                flag_reason = (
                    f"RAG score too low ({top_score:.2f}) — no knowledge base match. "
                    f"Routed to {intent_result.department or 'Customer Service'}."
                )
                logger.warning(
                    "[DEPT ROUTING] Unanswerable question | user=%s | rag_score=%.4f "
                    "| dept=%s | flag_reason=%.80s",
                    user_id, top_score, intent_result.department, flag_reason,
                )
                flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

                email_routed = _route_via_email(
                    user_id=user_id, message=message, full_history=full_history,
                    flag_id=flag_id, flag_reason=flag_reason,
                    department=intent_result.department, sentiment_data=sentiment_data,
                    routing_type="unanswerable_question",
                )

                text = _generate_sorry_message(detected_language)
                safe_text = run_safety_check(text)
                resp = AgentResponse(
                    text=safe_text, intent="question", flagged=True, flag_reason=flag_reason,
                    department=intent_result.department, rag_top_score=top_score,
                    language=detected_language, sentiment=sentiment_data.get("sentiment"),
                    urgency=sentiment_data.get("urgency"),
                    priority_boost=bool(sentiment_data.get("priority_boost", False)),
                    email_routed=email_routed,
                )
                logger.info(
                    "handle complete | user=%s | intent=question | flagged=True | flag_id=%s "
                    "| rag_score=%.4f | email_routed=%s | elapsed=%.3fs",
                    user_id, flag_id, top_score, email_routed, time.time() - t_total,
                )
                return resp

            safe_text = run_safety_check(answer)
            resp = AgentResponse(
                text=safe_text, intent="question", rag_top_score=top_score,
                language=detected_language, sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
            )
            logger.info(
                "handle complete | user=%s | intent=question | rag_score=%.4f "
                "| exploratory=%s | answer_len=%d | elapsed=%.3fs",
                user_id, top_score, intent_result.is_exploratory,
                len(safe_text), time.time() - t_total,
            )
            return resp

        # ── 2D: Issue → collect info then create case ─────────────────────────
        if intent_result.intent == "issue":
            department = pending_department or intent_result.department
            missing_info = (
                pending_missing_info
                if pending_missing_info is not None
                else intent_result.missing_info
            )
            logger.info(
                "Pipeline | intent=issue | dept=%s | missing_info=%s | user=%s",
                department, missing_info, user_id,
            )

            if not department:
                flag_reason = "Could not determine department for issue"
                logger.warning(
                    "[DEPT ROUTING] Department unknown | user=%s | falling_back=Customer Service",
                    user_id,
                )
                flag_id = save_flagged(user_id, full_history, flag_reason, self.db_path)

                email_routed = _route_via_email(
                    user_id=user_id, message=message, full_history=full_history,
                    flag_id=flag_id, flag_reason=flag_reason,
                    department="Customer Service", sentiment_data=sentiment_data,
                    routing_type="unroutable_issue",
                )

                text = (
                    "I want to make sure your issue reaches the right team. "
                    "A support specialist will review your case shortly. "
                    "You can also call *8880 for immediate help."
                )
                safe_text = run_safety_check(text)
                resp = AgentResponse(
                    text=safe_text, intent="issue", flagged=True,
                    flag_reason=flag_reason, email_routed=email_routed,
                )
                logger.info(
                    "handle complete | user=%s | intent=issue | flagged=True | flag_id=%s "
                    "| email_routed=%s | elapsed=%.3fs",
                    user_id, flag_id, email_routed, time.time() - t_total,
                )
                return resp

            if _is_case_ready(full_history, missing_info):
                logger.info(
                    "Issue ready to create case | user=%s | dept=%s",
                    user_id, department,
                )
                summary = summarise_case(full_history)

                similar = find_similar_cases(
                    summary=summary, db_path=self.db_path, top_k=3, min_score=0.55,
                )
                logger.info(
                    "Similar cases found | count=%d | dept=%s",
                    len(similar), department,
                )

                case_id = create_case(
                    user_id=user_id, department=department,
                    summary=summary, history=full_history, db_path=self.db_path,
                )

                index_case(
                    case_id=case_id, summary=summary,
                    department=department, db_path=self.db_path,
                )
                logger.debug("Case indexed for similarity | case_id=%s", case_id)

                anomaly = check_anomaly(department=department, db_path=self.db_path)
                if anomaly:
                    logger.warning(
                        "Anomaly detected | case_id=%s | dept=%s | message=%s",
                        case_id, department, anomaly.get("message"),
                    )
                else:
                    logger.debug("No anomaly detected | dept=%s", department)

                text = (
                    f"I've created support case **{case_id}** and escalated it to our "
                    f"**{department}** team. They will review your case and contact you "
                    f"within 1–2 business days. Please save your case ID for reference. "
                    f"Is there anything else I can help you with?"
                )
                safe_text = run_safety_check(text)
                resp = AgentResponse(
                    text=safe_text, intent="issue", case_id=case_id, department=department,
                    language=detected_language, sentiment=sentiment_data.get("sentiment"),
                    urgency=sentiment_data.get("urgency"),
                    priority_boost=bool(sentiment_data.get("priority_boost", False)),
                    similar_cases=similar, anomaly=anomaly,
                )
                logger.info(
                    "handle complete | user=%s | intent=issue | case_id=%s | dept=%s "
                    "| similar=%d | anomaly=%s | elapsed=%.3fs",
                    user_id, case_id, department, len(similar),
                    bool(anomaly), time.time() - t_total,
                )
                return resp

            # Still collecting — ask for next missing field
            logger.info(
                "Issue collection in progress | user=%s | dept=%s | still_missing=%s",
                user_id, department, missing_info,
            )
            reply = collect_missing_info(message, history, department, missing_info)
            safe_reply = run_safety_check(reply)
            resp = AgentResponse(
                text=safe_reply, intent="issue", department=department,
                language=detected_language, sentiment=sentiment_data.get("sentiment"),
                urgency=sentiment_data.get("urgency"),
                priority_boost=bool(sentiment_data.get("priority_boost", False)),
            )
            logger.info(
                "handle complete | user=%s | intent=issue | collecting | dept=%s | elapsed=%.3fs",
                user_id, department, time.time() - t_total,
            )
            return resp

        # ── Fallback ──────────────────────────────────────────────────────────
        logger.warning(
            "handle | unhandled intent state | user=%s | intent=%s | elapsed=%.3fs",
            user_id, intent_result.intent, time.time() - t_total,
        )
        return AgentResponse(
            text=(
                "I'm sorry, I didn't quite understand that. Could you please rephrase, "
                "or call us at *8880 for immediate support?"
            ),
            intent="unclear",
        )


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = Agent(db_path="test_cases.db")

    print("\n── Test 1: Greeting ──")
    r = agent.handle("user_001", "Hello!", [])
    print(f"Intent: {r.intent} | Lang: {r.language}")
    print(f"Response: {r.text}\n")

    print("── Test 2: Exploratory ──")
    r = agent.handle("user_001", "Debit card", [])
    print(f"Intent: {r.intent} | Score: {r.rag_top_score}")
    print(f"Response: {r.text[:200]}\n")

    print("── Test 3: Issue — card declined ──")
    history: list[dict] = []
    msg = "My card was declined at a supermarket but money was deducted from my account"
    r = agent.handle("user_002", msg, history)
    print(f"Intent: {r.intent} | Dept: {r.department} | Case: {r.case_id}")
    print(f"Response: {r.text}\n")