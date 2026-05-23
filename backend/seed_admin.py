from database import SessionLocal
from models import User
from auth.service import hash_password

db = SessionLocal()
try:
    if db.query(User).filter(User.email == "admin@accessbank.az").first():
        print("Admin already exists.")
    else:
        db.add(User(
            email="admin@accessbank.az",
            username="admin",
            hashed_password=hash_password("password"),
            is_admin=True,
        ))
        db.commit()
        print("Admin created — email: admin@accessbank.az  password: password")
finally:
    db.close()
