"""
SQLite database helper for ProofLink API runs.

Provides simple, stdlib-only persistence for reconciliation runs.
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Any

DB_PATH = "prooflink_runs.db"


@contextmanager
def get_conn():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database schema if it doesn't exist."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_name TEXT NOT NULL,
                user_id TEXT,
                payroll_filename TEXT NOT NULL,
                rk_filename TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                manifest_json TEXT,
                evidence_pack_path TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.commit()


def insert_run(
    *,
    run_id: str,
    status: str,
    plan_name: str,
    payroll_filename: str,
    rk_filename: str,
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    evidence_pack_path: str,
    error_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Insert a new run record into the database."""
    now = datetime.utcnow().isoformat() + "Z"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                id,
                created_at,
                updated_at,
                status,
                plan_name,
                user_id,
                payroll_filename,
                rk_filename,
                summary_json,
                manifest_json,
                evidence_pack_path,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now,
                now,
                status,
                plan_name,
                user_id,
                payroll_filename,
                rk_filename,
                json.dumps(summary),
                json.dumps(manifest) if manifest is not None else None,
                evidence_pack_path,
                error_message,
            ),
        )
        conn.commit()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a run record by ID."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                created_at,
                updated_at,
                status,
                plan_name,
                user_id,
                payroll_filename,
                rk_filename,
                summary_json,
                manifest_json,
                evidence_pack_path,
                error_message
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    (
        id_,
        created_at,
        updated_at,
        status,
        plan_name,
        user_id,
        payroll_filename,
        rk_filename,
        summary_json,
        manifest_json,
        evidence_pack_path,
        error_message,
    ) = row

    return {
        "id": id_,
        "created_at": created_at,
        "updated_at": updated_at,
        "status": status,
        "plan_name": plan_name,
        "user_id": user_id,
        "payroll_filename": payroll_filename,
        "rk_filename": rk_filename,
        "summary": json.loads(summary_json),
        "manifest": json.loads(manifest_json) if manifest_json is not None else None,
        "evidence_pack_path": evidence_pack_path,
        "error_message": error_message,
    }

