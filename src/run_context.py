"""
Run context for Evidence Pack v2.

Provides RunContext dataclass to track execution metadata for reconciliation runs.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RunContext:
    """
    Execution context for a reconciliation run.
    
    Attributes:
        run_id: ISO-like filesystem-safe identifier (e.g., 2025-01-18T14-22-11Z)
        engine: Engine identifier (e.g., "ProofLink")
        engine_version: Version string (e.g., "2.0.0")
        execution_timestamp_utc: UTC timestamp of execution start
    """
    run_id: str
    engine: str
    engine_version: str
    execution_timestamp_utc: datetime


def create_run_context(
    engine: str = "ProofLink",
    engine_version: str = "2.0.0"
) -> RunContext:
    """
    Create a new RunContext with auto-generated run_id and current UTC timestamp.
    
    Args:
        engine: Engine identifier (default: "ProofLink")
        engine_version: Version string (default: "2.0.0")
        
    Returns:
        Populated RunContext instance
    """
    # Generate ISO-like filesystem-safe run_id
    # Format: YYYY-MM-DDTHH-MM-SSZ (replacing colons with hyphens for filesystem safety)
    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
    
    return RunContext(
        run_id=run_id,
        engine=engine,
        engine_version=engine_version,
        execution_timestamp_utc=now_utc
    )

