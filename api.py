"""
ProofLink FastAPI Backend

A minimal REST API that wraps the ProofLink reconciliation engine.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
import jwt
from jwt import PyJWTError
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from main import EngineConfig, run_prooflink_engine, EngineResult
from db import init_db, insert_run, get_run, list_runs, get_db
from models import User

app = FastAPI(title="ProofLink API", version="0.1")

# Initialize database on startup
init_db()

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using pbkdf2_sha256."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# JWT token configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_TO_A_SECURE_RANDOM_VALUE")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """
    Dependency to get the current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception

    return user


# Pydantic models for auth
class UserCreate(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"


@app.post("/api/v1/runs")
async def create_run(
    payroll_file: UploadFile = File(...),
    rk_file: UploadFile = File(...),
    plan_name: str = Form("Untitled Plan"),
    plan_rules: Optional[str] = Form(None),
):
    """
    Create a new reconciliation run.
    
    Accepts two CSV files (payroll and recordkeeper) and runs the full
    ProofLink pipeline: reconciliation, timing analysis, Secure 2.0 checks,
    and evidence pack generation.
    
    Args:
        plan_rules: Optional JSON string containing eligibility rules configuration.
            Expected format: {"eligibility_rule": "...", "service_days_required": ..., etc.}
    
    Returns:
        JSON with run_id, summary, and evidence_pack_available flag
    """
    run_id = str(uuid4())
    uploads_dir = os.getenv("UPLOADS_DIR", "api_uploads")
    run_dir = Path(uploads_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    payroll_path = run_dir / (payroll_file.filename or "payroll.csv")
    rk_path = run_dir / (rk_file.filename or "rk.csv")
    
    try:
        # Save payroll file
        with open(payroll_path, "wb") as f:
            content = await payroll_file.read()
            f.write(content)
        
        # Save recordkeeper file
        with open(rk_path, "wb") as f:
            content = await rk_file.read()
            f.write(content)
        
        # Parse plan_rules if provided
        plan_rules_dict: Optional[Dict[str, Any]] = None
        if plan_rules:
            try:
                plan_rules_dict = json.loads(plan_rules)
                if not isinstance(plan_rules_dict, dict):
                    plan_rules_dict = None
            except (json.JSONDecodeError, TypeError):
                # Invalid JSON, use None (default behavior)
                plan_rules_dict = None
        
        # Create engine config
        config = EngineConfig(
            plan_name=plan_name,
            output_dir=str(run_dir / "output"),
            proofs_dir=str(run_dir / "proofs"),
        )
        
        # Run the engine
        result = run_prooflink_engine(
            payroll_path=str(payroll_path),
            rk_path=str(rk_path),
            config=config,
            run_id=run_id,
            plan_rules=plan_rules_dict,
        )
        
        # Persist run in DB
        insert_run(
            run_id=run_id,
            status="completed",  # queue/async will add other statuses later
            plan_name=plan_name,
            payroll_filename=payroll_file.filename or "payroll.csv",
            rk_filename=rk_file.filename or "rk.csv",
            summary=result.summary,
            manifest=getattr(result, "manifest", None),
            evidence_pack_path=result.evidence_pack_path,
            error_message=None,
            user_id=None,  # auth/multi-tenant later
        )
        
        return {
            "run_id": run_id,
            "summary": result.summary,
            "evidence_pack_available": True,
            "status": "completed",
        }
    
    except Exception as e:
        # Clean up on error
        if run_dir.exists():
            import shutil
            try:
                shutil.rmtree(run_dir)
            except Exception:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {str(e)}"
        )


@app.get("/api/v1/runs")
def get_run_list(limit: Optional[int] = Query(50, ge=1, le=200)):
    """
    Return a list of recent runs for Run History.

    For now this is global (no auth / no org scoping).
    """
    # Basic guardrail
    if limit is None or limit <= 0:
        limit = 50
    if limit > 200:
        limit = 200

    records = list_runs(limit=limit)

    # Shape for API response: lighter wrapper over DB records
    return {
        "count": len(records),
        "items": [
            {
                "run_id": r["id"],
                "status": r["status"],
                "plan_name": r["plan_name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "payroll_filename": r["payroll_filename"],
                "rk_filename": r["rk_filename"],
                "has_evidence_pack": r["evidence_pack_path"] is not None and r["evidence_pack_path"] != "",
                # summary is included for now; we can trim or expand later
                "summary": r["summary"],
            }
            for r in records
        ],
    }


@app.get("/api/v1/runs/{run_id}")
def get_run_details(run_id: str):
    """
    Get the status and summary of a reconciliation run.
    
    Returns:
        JSON with run_id, summary, status, and evidence_pack_available
    """
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": record["id"],
        "status": record["status"],
        "plan_name": record["plan_name"],
        "summary": record["summary"],
        "manifest": record["manifest"],
        "evidence_pack_available": record["evidence_pack_path"] is not None,
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


@app.get("/api/v1/runs/{run_id}/evidence-pack")
def download_evidence(run_id: str):
    """
    Download the evidence pack ZIP file for a completed run.
    
    Returns:
        ZIP file download
    """
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    path = record["evidence_pack_path"]
    if not path:
        raise HTTPException(status_code=404, detail="Evidence pack not available")
    
    evidence_path = Path(path)
    if not evidence_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evidence pack file not found"
        )
    
    filename = f"evidence_{run_id}.zip"
    
    return FileResponse(
        path=str(evidence_path),
        filename=filename,
        media_type="application/zip"
    )


@app.get("/")
def root():
    """API health check endpoint."""
    return {
        "service": "ProofLink API",
        "version": "0.1",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ==============================
# Authentication endpoints
# ==============================

@app.post("/auth/register", response_model=TokenResponse)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user and return an access token.
    """
    # Check if user exists
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists."
        )
    
    # Create new user
    user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Generate token
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login_user(payload: UserLogin, db: Session = Depends(get_db)):
    """
    Login a user and return an access token.
    """
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
    
    # Generate token
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)

