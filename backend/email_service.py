import os
import base64
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

_SCOPES = ["https://mail.google.com/"]
_BASE_DIR = Path(__file__).parent
_CREDS_PATH = _BASE_DIR / "credentials.json"
_TOKEN_PATH = _BASE_DIR / "token.json"

SENDER = os.getenv("GMAIL_SENDER")

DEPARTMENT_EMAILS: dict[str, str] = {
    "Digital Banking":          os.getenv("EMAIL_DIGITAL_BANKING", ""),
    "Card Operations":          os.getenv("EMAIL_CARD_OPERATIONS", ""),
    "Transfers & Payments":     os.getenv("EMAIL_TRANSFERS", ""),
    "Loans & Applications":     os.getenv("EMAIL_LOANS", ""),
    "Customer Service":         os.getenv("EMAIL_CUSTOMER_SERVICE", ""),
}


def _get_service():
    creds = None

    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDS_PATH.exists():
                raise FileNotFoundError(
                    "credentials.json not found in backend/. "
                    "Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), _SCOPES)
            creds = flow.run_local_server(port=0)

        with open(_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(to: str, subject: str, html_body: str) -> str:
    """Send an email and return the Gmail message ID."""
    message = MIMEText(html_body, "html")
    message["to"] = to
    message["from"] = SENDER
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    result = _get_service().users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return result["id"]


def send_escalation_email(
    department: str,
    case_id: int,
    user_name: str,
    user_contact: str,
    issue_summary: str,
) -> str:
    """Send escalation email to the department and persist the Gmail message ID on the case."""
    from models import Case
    from database import SessionLocal

    to = DEPARTMENT_EMAILS.get(department)
    if not to:
        raise ValueError(f"Unknown department: {department!r}")

    subject = f"[Case #{case_id}] New Support Request — {department}"
    body = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#0f1f35;padding:20px 24px;border-radius:8px 8px 0 0;">
        <img src="https://upload.wikimedia.org/wikipedia/en/4/46/AccessBank_Azerbaijan_logo.svg"
             alt="AccessBank" style="height:28px;filter:brightness(0) invert(1);">
      </div>
      <div style="border:1px solid #e2e8f0;border-top:none;padding:28px 24px;border-radius:0 0 8px 8px;">
        <h2 style="margin:0 0 4px;color:#0f172a;font-size:18px;">New Support Case</h2>
        <p style="margin:0 0 24px;color:#64748b;font-size:13px;">Case #{case_id} · {department}</p>

        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;width:140px;">Customer Name</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;font-weight:500;">{user_name}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#64748b;">Contact</td>
            <td style="padding:10px 0;border-bottom:1px solid #f1f5f9;color:#0f172a;font-weight:500;">{user_contact}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;color:#64748b;vertical-align:top;">Issue Summary</td>
            <td style="padding:10px 0;color:#0f172a;">{issue_summary}</td>
          </tr>
        </table>

        <p style="margin:28px 0 0;font-size:12px;color:#94a3b8;">
          This is an automated message from AccessBank Customer Support.
        </p>
      </div>
    </div>
    """

    gmail_id = send_email(to, subject, body)

    db = SessionLocal()
    try:
        case = db.query(Case).filter(Case.id == case_id).first()
        if case:
            case.email_ref = gmail_id
            db.commit()
    finally:
        db.close()

    return gmail_id
