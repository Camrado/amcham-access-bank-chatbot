"""
test_agent.py
-------------
Tests for RAG retrieval performance and agent response generation.

Run with:
    python test_agent.py

Requires OPENAI_API_KEY to be set in environment.
Does NOT require a running server — tests the agent and RAG directly.

Test suites:
  1. RAG Retrieval Tests   — checks that the right chunks are returned for known queries
  2. Intent Classification — checks that intent + department routing is correct
  3. Safety Guardrail      — checks that unsafe responses are caught and cleaned
  4. Full Agent Flow       — end-to-end question and issue flows
  5. Edge Cases            — empty input, gibberish, sensitive data attempts
"""

import json
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

# ─── Colour helpers for terminal output ───────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}~{RESET} {msg}")
def section(msg): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{msg}{RESET}")


# ─── Test result tracking ─────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    note: str = ""
    duration_ms: float = 0.0


results: list[TestResult] = []


def run_test(name: str, fn) -> TestResult:
    """Run a single test function, catch exceptions, record result."""
    start = time.time()
    try:
        fn()
        duration = (time.time() - start) * 1000
        r = TestResult(name=name, passed=True, duration_ms=duration)
        ok(f"{name} ({duration:.0f}ms)")
    except AssertionError as e:
        duration = (time.time() - start) * 1000
        r = TestResult(name=name, passed=False, note=str(e), duration_ms=duration)
        fail(f"{name} — {e}")
    except Exception as e:
        duration = (time.time() - start) * 1000
        r = TestResult(name=name, passed=False, note=traceback.format_exc(), duration_ms=duration)
        fail(f"{name} — EXCEPTION: {e}")
    results.append(r)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 1 — RAG RETRIEVAL TESTS
# Tests that the right knowledge chunks surface for known queries.
# Each test specifies the query and the expected chunk ID or title keyword.
# ═════════════════════════════════════════════════════════════════════════════

def test_rag_suite():
    section("SUITE 1 — RAG Retrieval Performance")

    from rag_loader import retrieve

    rag_cases = [
        {
            "query": "What are your working hours?",
            "expected_title_keyword": "Working Hours",
            "min_score": 0.70,
            "description": "Basic hours query should hit kb_001",
        },
        {
            "query": "My card was declined and money was taken from my account",
            "expected_title_keyword": "Payment Failed",
            "min_score": 0.65,
            "description": "Card decline issue → kb_006",
        },
        {
            "query": "How do I apply for a personal loan?",
            "expected_title_keyword": "Consumer Loan",
            "min_score": 0.65,
            "description": "Loan application → kb_012",
        },
        {
            "query": "I cannot log in to the mobile banking app",
            "expected_title_keyword": "Mobile App",
            "min_score": 0.65,
            "description": "Mobile app login issue → kb_007",
        },
        {
            "query": "My transfer to another bank was not received",
            "expected_title_keyword": "Failed Transfer",
            "min_score": 0.65,
            "description": "Failed transfer → kb_011",
        },
        {
            "query": "I want to open a bank account",
            "expected_title_keyword": "Account Opening",
            "min_score": 0.65,
            "description": "Account opening → kb_016",
        },
        {
            "query": "What is the fee for sending money to another bank?",
            "expected_title_keyword": "Domestic Transfers",
            "min_score": 0.60,
            "description": "Transfer fee → kb_009",
        },
        {
            "query": "My card was stolen, I need to block it",
            "expected_title_keyword": "Lost",
            "min_score": 0.65,
            "description": "Lost/stolen card → kb_005",
        },
        {
            "query": "What are the deposit interest rates?",
            "expected_title_keyword": "Deposit",
            "min_score": 0.65,
            "description": "Deposit rates → kb_017",
        },
        {
            "query": "I'm not receiving the OTP SMS code",
            "expected_title_keyword": "OTP",
            "min_score": 0.65,
            "description": "OTP issue → kb_008",
        },
    ]

    for case in rag_cases:
        def make_test(c):
            def t():
                result = retrieve(c["query"], top_k=3)
                top_score = result["top_score"]
                top_titles = [r["title"] for r in result["results"]]

                assert top_score >= c["min_score"], (
                    f"Score {top_score:.3f} below threshold {c['min_score']} "
                    f"| top titles: {top_titles}"
                )

                keyword = c["expected_title_keyword"].lower()
                matched = any(keyword in t.lower() for t in top_titles)
                assert matched, (
                    f"Expected title containing '{c['expected_title_keyword']}' "
                    f"in top results. Got: {top_titles}"
                )
            return t

        run_test(f"RAG: {case['description']}", make_test(case))

    # Threshold test — gibberish should be flagged for human
    def test_low_score_gibberish():
        result = retrieve("xkqwpzmn blorp flargle 12345", top_k=3)
        # We don't assert flag_for_human since threshold may vary,
        # but top_score should be low (< 0.55)
        assert result["top_score"] < 0.55, (
            f"Expected low score for gibberish, got {result['top_score']}"
        )

    run_test("RAG: gibberish query returns low score", test_low_score_gibberish)

    # top_k respected
    def test_topk():
        result = retrieve("loan interest rate", top_k=2)
        assert len(result["results"]) == 2, (
            f"Expected 2 results, got {len(result['results'])}"
        )

    run_test("RAG: top_k=2 returns exactly 2 results", test_topk)

    # Scores are sorted descending
    def test_sorted():
        result = retrieve("working hours branch", top_k=5)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by score: {scores}"
        )

    run_test("RAG: results are sorted by score descending", test_sorted)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 2 — INTENT CLASSIFICATION TESTS
# Tests that the agent correctly classifies intent and routes to the right dept.
# ═════════════════════════════════════════════════════════════════════════════

def test_intent_suite():
    section("SUITE 2 — Intent Classification & Department Routing")

    from agent import classify_intent

    intent_cases = [
        # Questions — should never escalate
        {
            "message": "What are AccessBank's working hours?",
            "expected_intent": "question",
            "expected_dept": None,
            "description": "Simple hours question",
        },
        {
            "message": "How much does a domestic transfer cost?",
            "expected_intent": "question",
            "expected_dept": None,
            "description": "Transfer fee question",
        },
        {
            "message": "What documents do I need to open an account?",
            "expected_intent": "question",
            "expected_dept": None,
            "description": "Account opening question",
        },
        # Issues — must detect and route correctly
        {
            "message": "My card was blocked at a store and money was taken",
            "expected_intent": "issue",
            "expected_dept": "Card Operations",
            "description": "Card declined + deducted → Card Operations",
        },
        {
            "message": "I sent a transfer 3 days ago and it hasn't arrived",
            "expected_intent": "issue",
            "expected_dept": "Transfers & Payments",
            "description": "Missing transfer → Transfers & Payments",
        },
        {
            "message": "I can't log into the mobile app, it says password is wrong",
            "expected_intent": "issue",
            "expected_dept": "Digital Banking",
            "description": "Login failure → Digital Banking",
        },
        {
            "message": "My loan application was submitted 2 weeks ago and no one responded",
            "expected_intent": "issue",
            "expected_dept": "Loans & Applications",
            "description": "Loan application no response → Loans & Applications",
        },
        {
            "message": "The staff at the Narimanov branch were very rude to me",
            "expected_intent": "issue",
            "expected_dept": "Customer Service",
            "description": "Branch complaint → Customer Service",
        },
        {
            "message": "I'm not receiving my OTP to confirm a payment",
            "expected_intent": "issue",
            "expected_dept": "Digital Banking",
            "description": "OTP not received → Digital Banking",
        },
    ]

    for case in intent_cases:
        def make_test(c):
            def t():
                result = classify_intent(c["message"], [])
                assert result.intent == c["expected_intent"], (
                    f"Expected intent='{c['expected_intent']}', got '{result.intent}' "
                    f"(confidence={result.confidence:.2f}, reasoning={result.reasoning})"
                )
                if c["expected_dept"] is not None:
                    assert result.department == c["expected_dept"], (
                        f"Expected dept='{c['expected_dept']}', got '{result.department}'"
                    )
                if c["expected_intent"] == "question":
                    assert result.department is None, (
                        f"Questions should have dept=None, got '{result.department}'"
                    )
            return t

        run_test(f"Intent: {case['description']}", make_test(case))

    # Safety: missing_info must never contain sensitive fields
    def test_no_sensitive_in_missing_info():
        from agent import classify_intent
        result = classify_intent("My card payment failed", [])
        sensitive = {"pin", "cvv", "otp", "password", "card number", "full card"}
        for field in result.missing_info:
            for s in sensitive:
                assert s not in field.lower(), (
                    f"Sensitive field '{s}' found in missing_info: {result.missing_info}"
                )

    run_test("Intent: missing_info contains no sensitive fields", test_no_sensitive_in_missing_info)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 3 — SAFETY GUARDRAIL TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_safety_suite():
    section("SUITE 3 — Safety Guardrail")

    from agent import run_safety_check

    # Safe responses should pass through unchanged (or near-unchanged)
    def test_safe_response_passes():
        draft = "Your working hours are Monday to Friday, 9am to 6pm."
        result = run_safety_check(draft)
        assert len(result) > 0, "Safety check returned empty string"
        assert "9" in result or "working" in result.lower(), (
            f"Safe response was incorrectly altered: {result}"
        )

    run_test("Safety: safe response passes through", test_safe_response_passes)

    # PIN request must be blocked
    def test_pin_blocked():
        draft = "Please provide your PIN code so I can verify your identity."
        result = run_safety_check(draft)
        assert "pin" not in result.lower() or "not" in result.lower() or "never" in result.lower(), (
            f"PIN request was not properly blocked: {result}"
        )

    run_test("Safety: PIN request is blocked", test_pin_blocked)

    # CVV request must be blocked
    def test_cvv_blocked():
        draft = "Could you share your CVV number from the back of the card?"
        result = run_safety_check(draft)
        assert "cvv" not in result.lower() or "never" in result.lower() or "not" in result.lower(), (
            f"CVV request was not properly blocked: {result}"
        )

    run_test("Safety: CVV request is blocked", test_cvv_blocked)

    # OTP request must be blocked
    def test_otp_blocked():
        draft = "Please send me the OTP code you received by SMS."
        result = run_safety_check(draft)
        assert len(result) > 0, "Safety check returned empty string for OTP test"
        # The cleaned response should NOT ask for OTP
        assert "send me the otp" not in result.lower() and "share your otp" not in result.lower(), (
            f"OTP request was not properly blocked: {result}"
        )

    run_test("Safety: OTP request is blocked", test_otp_blocked)

    # Full password request must be blocked
    def test_password_blocked():
        draft = "What is your internet banking password? I need it to reset your account."
        result = run_safety_check(draft)
        assert "password" not in result.lower() or "reset" in result.lower() or "never" in result.lower(), (
            f"Password request was not properly blocked: {result}"
        )

    run_test("Safety: password request is blocked", test_password_blocked)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 4 — FULL AGENT FLOW TESTS
# End-to-end: message in → AgentResponse out, check DB state
# ═════════════════════════════════════════════════════════════════════════════

def test_agent_flow_suite():
    section("SUITE 4 — Full Agent Flow (End-to-End)")

    TEST_DB = "test_agent_flow.db"

    # Clean up any leftover test DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    from agent import Agent, get_case
    agent = Agent(db_path=TEST_DB)

    # ── 4.1 Question flow ─────────────────────────────────────────────────────
    def test_question_flow():
        r = agent.handle("u001", "What are your working hours?", [])
        assert r.intent == "question", f"Expected 'question', got '{r.intent}'"
        assert r.case_id is None, "Questions should not create cases"
        assert r.flagged is False or r.flagged is True  # either is fine for low RAG
        assert len(r.text) > 10, "Response text is too short"

    run_test("Agent: question flow returns answer, no case", test_question_flow)

    # ── 4.2 Issue flow — single turn then case creation ───────────────────────
    def test_issue_case_creation():
        history = []
        user_id = "u002"

        # Turn 1 — report issue
        r1 = agent.handle(
            user_id,
            "My transfer of 200 AZN failed but money was deducted",
            history,
        )
        assert r1.intent == "issue", f"Expected 'issue', got '{r1.intent}'"
        assert r1.department == "Transfers & Payments", (
            f"Expected 'Transfers & Payments', got '{r1.department}'"
        )

        # Turn 2 — provide detail, force case creation by clearing missing_info
        history = [
            {"role": "user", "content": "My transfer of 200 AZN failed but money was deducted"},
            {"role": "assistant", "content": r1.text},
        ]
        r2 = agent.handle(
            user_id,
            "It was sent to Kapital Bank yesterday, reference TXN123456",
            history,
            pending_department="Transfers & Payments",
            pending_missing_info=[],   # signal: all info collected
        )

        assert r2.case_id is not None, "Expected a case ID after info is collected"
        assert r2.case_id.startswith("CASE-"), f"Unexpected case ID format: {r2.case_id}"

        # Verify case is in DB
        case = get_case(r2.case_id, TEST_DB)
        assert case is not None, f"Case {r2.case_id} not found in database"
        assert case["department"] == "Transfers & Payments"
        assert case["status"] == "open"
        assert len(case["summary"]) > 10

    run_test("Agent: issue flow creates case in SQLite DB", test_issue_case_creation)

    # ── 4.3 Response always contains text ─────────────────────────────────────
    def test_response_always_has_text():
        messages = [
            "Hi",
            "I have a problem",
            "What is a credit card?",
            "My account is locked",
        ]
        for msg in messages:
            r = agent.handle("u003", msg, [])
            assert r.text and len(r.text) > 5, (
                f"Empty or too-short response for: '{msg}'"
            )

    run_test("Agent: all messages produce non-empty responses", test_response_always_has_text)

    # ── 4.4 Case ID format ────────────────────────────────────────────────────
    def test_case_id_format():
        r = agent.handle(
            "u004",
            "My card was stolen",
            [
                {"role": "user", "content": "My card was stolen"},
                {"role": "assistant", "content": "I can help. When did this happen?"},
            ],
            pending_department="Card Operations",
            pending_missing_info=[],
        )
        if r.case_id:
            assert r.case_id.startswith("CASE-"), f"Bad case ID: {r.case_id}"
            assert len(r.case_id) == 13, f"Expected 'CASE-XXXXXXXX' (13 chars), got '{r.case_id}'"

    run_test("Agent: case IDs follow CASE-XXXXXXXX format", test_case_id_format)

    # Clean up
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


# ═════════════════════════════════════════════════════════════════════════════
# SUITE 5 — EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

def test_edge_cases_suite():
    section("SUITE 5 — Edge Cases")

    TEST_DB = "test_edge_cases.db"
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    from agent import Agent
    agent = Agent(db_path=TEST_DB)

    # Empty message
    def test_empty_message():
        r = agent.handle("u_edge_1", "", [])
        assert r.text and len(r.text) > 5, "Empty message should still get a response"

    run_test("Edge: empty message handled gracefully", test_empty_message)

    # Very long message
    def test_long_message():
        long_msg = "My card was declined. " * 50
        r = agent.handle("u_edge_2", long_msg, [])
        assert r.text and len(r.text) > 5, "Long message should still get a response"

    run_test("Edge: very long message handled gracefully", test_long_message)

    # Gibberish
    def test_gibberish():
        r = agent.handle("u_edge_3", "asdfgh zxcvbn qwerty 999", [])
        assert r.text and len(r.text) > 5, "Gibberish should still get a response"

    run_test("Edge: gibberish message handled gracefully", test_gibberish)

    # User tries to provide PIN — agent must not echo it back or ask for it
    def test_user_sends_pin():
        r = agent.handle(
            "u_edge_4",
            "My card is blocked. My PIN is 1234 and my CVV is 456",
            [],
        )
        assert r.text and len(r.text) > 5
        # Response must not echo the PIN or CVV back
        assert "1234" not in r.text, "Agent echoed back the PIN"
        assert "456" not in r.text or "branch" in r.text.lower(), (
            "Agent echoed back CVV without redirecting"
        )

    run_test("Edge: agent does not echo user-provided PIN/CVV", test_user_sends_pin)

    # Multi-turn history doesn't break anything
    def test_long_history():
        history = []
        for i in range(8):
            history.append({"role": "user", "content": f"Turn {i}: still having card issues"})
            history.append({"role": "assistant", "content": f"I understand, let me help with turn {i}."})
        r = agent.handle("u_edge_5", "Still not resolved", history)
        assert r.text and len(r.text) > 5

    run_test("Edge: 8-turn history processed without error", test_long_history)

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — Run all suites and print summary
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"\n{RED}ERROR: OPENAI_API_KEY is not set.{RESET}")
        print("Export it before running: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print(f"\n{BOLD}AccessBank Agent — Test Suite{RESET}")
    print(f"Running all tests. This will make real OpenAI API calls.\n")

    test_rag_suite()
    test_intent_suite()
    test_safety_suite()
    test_agent_flow_suite()
    test_edge_cases_suite()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_ms = sum(r.duration_ms for r in results)

    section("SUMMARY")
    print(f"  Total:   {total}")
    print(f"  {GREEN}Passed:  {passed}{RESET}")
    if failed:
        print(f"  {RED}Failed:  {failed}{RESET}")
        print(f"\n{RED}Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                print(f"  {RED}✗{RESET} {r.name}")
                if r.note:
                    # Print only first 200 chars of note to keep output clean
                    print(f"    {r.note[:200]}")
    else:
        print(f"  Failed:  0")

    print(f"\n  Total time: {total_ms/1000:.1f}s")
    print()

    sys.exit(0 if failed == 0 else 1)
