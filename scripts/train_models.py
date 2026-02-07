"""Train ML models using data/synthetic_data (sentiment and priority scoring)."""

from __future__ import annotations

from models.ml_models import train_all_models

if __name__ == "__main__":
    train_all_models()
    print("Models trained and saved.")
