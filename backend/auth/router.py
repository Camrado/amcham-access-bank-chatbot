import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import User
from auth.schemas import RegisterRequest, LoginRequest, TokenResponse
from auth.service import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("auth")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info("register: user_id=%d username=%s", user.id, user.username)
    return TokenResponse(access_token=create_access_token(user.id, user.email, user.username, user.is_admin))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        logger.warning("login_failed: invalid credentials supplied")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    logger.info("login: user_id=%d username=%s", user.id, user.username)
    return TokenResponse(access_token=create_access_token(user.id, user.email, user.username, user.is_admin))
