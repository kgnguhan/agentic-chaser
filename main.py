"""
Agentic Chaser – application entry point.

Commands:
  dashboard (default)  Start the Streamlit dashboard
  init-db             Create database tables
  seed                Load test data from data/test into the database
  train               Train ML models using data/synthetic_data
  chaser              Run autonomic chaser cycle (tick time + chase active LOAs)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent


def cmd_dashboard() -> int:
    """Run the Streamlit dashboard."""
    app_path = ROOT / "dashboard" / "streamlit_app.py"
    if not app_path.exists():
        print(f"Error: Dashboard not found at {app_path}", file=sys.stderr)
        return 1
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"],
        cwd=str(ROOT),
    ).returncode


def cmd_init_db() -> int:
    """Create database tables."""
    from models.database import init_db
    init_db()
    print("Database tables created.")
    return 0


def cmd_seed() -> int:
    """Load test data from data/test into the database."""
    from scripts.load_test_data import load_test_data
    try:
        load_test_data()
        print("Test data loaded from data/test.")
        return 0
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        return 1


def cmd_train() -> int:
    """Train ML models using data/synthetic_data."""
    from models.ml_models import train_all_models
    try:
        train_all_models()
        print("Models trained and saved.")
        return 0
    except Exception as e:
        print(f"Error training models: {e}", file=sys.stderr)
        return 1


def cmd_chaser() -> int:
    """Run autonomic chaser: tick time then run one chase cycle over active LOAs."""
    from chaser import run_chaser_cycle
    try:
        run_chaser_cycle()
        print("Chaser cycle completed.")
        return 0
    except Exception as e:
        print(f"Chaser failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic Chaser – LOA/document chaser for advisors")
    parser.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        choices=["dashboard", "init-db", "seed", "train", "chaser"],
        help="Command to run (default: dashboard)",
    )
    args = parser.parse_args()

    if args.command == "dashboard":
        return cmd_dashboard()
    if args.command == "init-db":
        return cmd_init_db()
    if args.command == "seed":
        return cmd_seed()
    if args.command == "train":
        return cmd_train()
    if args.command == "chaser":
        return cmd_chaser()
    return 0


if __name__ == "__main__":
    sys.exit(main())
