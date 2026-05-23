import unittest
from database import SessionLocal
from models import Case
from email_service import send_escalation_email


class TestEmailService(unittest.TestCase):

    def setUp(self):
        self.db = SessionLocal()
        # Insert a real case row so send_escalation_email can find it
        self.case = Case(
            user_name="Test User",
            user_contact="testuser@example.com",
            issue_summary="Card was blocked after three failed PIN attempts.",
            department="Card Operations",
            status="open",
        )
        self.db.add(self.case)
        self.db.commit()
        self.db.refresh(self.case)

    def tearDown(self):
        self.db.delete(self.case)
        self.db.commit()
        self.db.close()

    def test_send_escalation_email(self):
        gmail_id = send_escalation_email(
            department=self.case.department,
            case_id=self.case.id,
            user_name=self.case.user_name,
            user_contact=self.case.user_contact,
            issue_summary=self.case.issue_summary,
        )

        # Email was sent and returned a Gmail message ID
        self.assertIsNotNone(gmail_id)
        self.assertIsInstance(gmail_id, str)
        self.assertTrue(len(gmail_id) > 0)

        # email_ref was persisted on the case row
        self.db.refresh(self.case)
        self.assertEqual(self.case.email_ref, gmail_id)

        print(f"\n  Gmail message ID : {gmail_id}")
        print(f"  Case ID          : {self.case.id}")
        print(f"  email_ref in DB  : {self.case.email_ref}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
