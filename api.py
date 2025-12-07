"""
ProofLink FastAPI Backend

A minimal REST API that wraps the ProofLink reconciliation engine.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from uuid import uuid4
from pathlib import Path

from main import EngineConfig, run_prooflink_engine, EngineResult
from db import init_db, insert_run, get_run

app = FastAPI(title="ProofLink API", version="0.1")

# Initialize database on startup
init_db()


@app.post("/api/v1/runs")
async def create_run(
    payroll_file: UploadFile = File(...),
    rk_file: UploadFile = File(...),
    plan_name: str = Form("Untitled Plan"),
):
    """
    Create a new reconciliation run.
    
    Accepts two CSV files (payroll and recordkeeper) and runs the full
    ProofLink pipeline: reconciliation, timing analysis, Secure 2.0 checks,
    and evidence pack generation.
    
    Returns:
        JSON with run_id, summary, and evidence_pack_available flag
    """
    run_id = str(uuid4())
    run_dir = Path("api_uploads") / run_id
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

