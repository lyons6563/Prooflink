"""
Smoke test for Secure 2.0 catch-up analyzer.

Run with: python -m tests.run_secure20_smoketest
(from the src folder)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import EngineConfig, run_prooflink_engine
import json


def main():
    """Run Secure 2.0 smoke test."""
    # Get paths relative to src directory
    src_dir = Path(__file__).parent.parent
    payroll_path = src_dir / "data" / "demo" / "secure20_test_payroll.csv"
    rk_path = src_dir / "data" / "demo" / "demo_clean_rk.csv"
    
    # Verify files exist
    if not payroll_path.exists():
        print(f"ERROR: Payroll file not found: {payroll_path}")
        return 1
    
    if not rk_path.exists():
        print(f"ERROR: RK file not found: {rk_path}")
        return 1
    
    # Create timestamped output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploads_dir = os.getenv("UPLOADS_DIR", "api_uploads")
    output_dir = src_dir / uploads_dir / "test_secure20" / timestamp / "output"
    proofs_dir = src_dir / uploads_dir / "test_secure20" / timestamp / "proofs"
    
    # Create EngineConfig
    config = EngineConfig(
        plan_name="Secure 2.0 Test Plan",
        late_threshold_days=5,
        secure2_enabled=True,
        payroll_vendor_hint=None,  # Auto-detect
        rk_vendor_hint=None,  # Auto-detect
        output_dir=str(output_dir),
        proofs_dir=str(proofs_dir),
    )
    
    print("=" * 70)
    print("Secure 2.0 Catch-Up Analyzer Smoke Test")
    print("=" * 70)
    print(f"Payroll file: {payroll_path}")
    print(f"RK file: {rk_path}")
    print(f"Output dir: {output_dir}")
    print(f"Proofs dir: {proofs_dir}")
    print()
    
    try:
        # Run the engine
        print("Running ProofLink engine...")
        result = run_prooflink_engine(
            payroll_path=str(payroll_path),
            rk_path=str(rk_path),
            config=config,
            run_id=f"secure20_test_{timestamp}",
        )
        
        print("\n" + "=" * 70)
        print("Engine Run Complete")
        print("=" * 70)
        print(f"Run ID: {result.run_id}")
        print(f"Evidence Pack: {result.evidence_pack_path}")
        print()
        
        # Extract and print Secure 2.0 summary
        secure20_summary = result.summary.get("secure20", {})
        
        if secure20_summary:
            print("=" * 70)
            print("Secure 2.0 Summary")
            print("=" * 70)
            print(json.dumps(secure20_summary, indent=2))
            print()
            
            # Print key metrics
            print("Key Metrics:")
            print(f"  Total rows analyzed: {secure20_summary.get('total_rows', 0)}")
            print(f"  Total violations: {secure20_summary.get('total_violations', 0)}")
            print(f"  HCE violation count: {secure20_summary.get('hce_violation_count', 0)}")
            print(f"  Potential catch-up miscode count: {secure20_summary.get('potential_catchup_miscode_count', 0)}")
            print(f"  CSV path: {secure20_summary.get('csv_path', 'N/A')}")
        else:
            print("WARNING: No Secure 2.0 summary found in results")
            print("Full summary keys:", list(result.summary.keys()))
        
        # Plan Health
        plan_health = result.summary.get("plan_health")
        if plan_health is not None:
            print()
            print("=" * 70)
            print("Plan Health")
            print("=" * 70)
            print(json.dumps(plan_health, indent=2))
        else:
            print()
            print("[INFO] plan_health not present in engine_result.")
        
        # Plan Exceptions
        plan_exceptions = result.summary.get("plan_exceptions")
        if plan_exceptions is not None:
            print()
            print("=" * 70)
            print("Plan Exceptions Summary")
            print("=" * 70)
            print(json.dumps(plan_exceptions, indent=2))
        else:
            print()
            print("[INFO] plan_exceptions not present in engine_result.")
        
        # Evidence Index
        evidence_index = result.summary.get("evidence_index") or []
        if evidence_index:
            print()
            print("=" * 70)
            print("Evidence Index (first 10 entries)")
            print("=" * 70)
            for item in evidence_index[:10]:
                print(f"- [{item.get('category', 'Unknown')}] {item.get('name')} -> {item.get('path')}")
        else:
            print()
            print("[INFO] evidence_index is empty or not present.")
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

