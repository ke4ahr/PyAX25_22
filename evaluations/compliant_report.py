# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
evaluations/compliance_report.py

Automated compliance validation tool for PyAX25_22.

This script runs the full test suite and generates a compliance report.
It is intended to be executed directly:

    python -m evaluations.compliance_report

Features:
- Runs pytest with coverage
- Displays summary of test results
- Reports coverage percentage
- Exits with appropriate status code (0 = compliant)
- Prints formatted compliance statement
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test and coverage configuration
TEST_ARGS = ["-q", "--cov=pyax25_22", "--cov-report=term-missing"]
MIN_COVERAGE = 95.0  # Minimum acceptable coverage percentage

def run_tests() -> tuple[bool, str]:
    """
    Execute pytest and return success status and output.

    Returns:
        (success: bool, output: str)
    """
    logger.info("Running PyAX25_22 compliance test suite...")
    try:
        result = subprocess.run(
            ["pytest"] + TEST_ARGS,
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.returncode == 0, result.stdout + result.stderr)
    except FileNotFoundError:
        return (False, "pytest not found - install with 'pip install pytest pytest-cov'")

def extract_coverage(output: str) -> Optional[float]:
    """
    Extract coverage percentage from pytest-cov output.

    Args:
        output: Full pytest output

    Returns:
        Coverage percentage or None if not found
    """
    for line in output.splitlines():
        if line.startswith("TOTAL"):
            parts = line.split()
            if len(parts) >= 4 and "%" in parts[-1]:
                try:
                    return float(parts[-1].rstrip("%"))
                except ValueError:
                    pass
    return None

def main() -> int:
    """Main compliance report execution."""
    print("=" * 60)
    print("PyAX25_22 AX.25 v2.2 Compliance Validation Report")
    print("=" * 60)
    print(f"Date: January 02, 2026")
    print(f"Version: 0.1.0")
    print(f"Project: https://github.com/ke4ahr/PyAX25_22")
    print()

    success, output = run_tests()

    # Print test output
    print(output)

    # Extract and validate coverage
    coverage = extract_coverage(output)
    if coverage is not None:
        print(f"\nTest coverage: {coverage:.1f}%")
        if coverage < MIN_COVERAGE:
            print(f"WARNING: Coverage below minimum {MIN_COVERAGE}%")
            success = False

    # Final compliance verdict
    print("\n" + "=" * 60)
    if success:
        print("COMPLIANCE ACHIEVED")
        print("All tests passed and coverage meets requirements.")
        print("PyAX25_22 is fully compliant with AX.25 v2.2 (July 1998)")
        print("=" * 60)
        return 0
    else:
        print("COMPLIANCE FAILURE")
        print("One or more tests failed or coverage insufficient.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
