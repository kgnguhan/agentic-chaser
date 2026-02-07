"""
Machine learning models for sentiment analysis and priority scoring.

Trains on synthetic data and provides prediction functions for agents.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config.config import settings


# ========== SENTIMENT ANALYSIS ==========


def train_sentiment_model(data_path: Path | None = None) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train sentiment analysis model on synthetic communication data.
    
    Args:
        data_path: Path to communication_sentiment_log.csv
    
    Returns:
        Tuple of (trained model, fitted vectorizer)
    """
    if data_path is None:
        data_path = settings.paths.synthetic_data_dir / "10_sentiment_training_data.csv"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Features: message text
    X = df["message_text"].fillna("")
    y = df["sentiment_label"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    train_score = model.score(X_train_vec, y_train)
    test_score = model.score(X_test_vec, y_test)
    
    print(f"✓ Sentiment model trained")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")
    
    # Save
    model_path = settings.ml.sentiment_model_path
    vectorizer_path = settings.ml.vectorizer_path
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"  Saved to: {model_path}")
    
    return model, vectorizer


def load_sentiment_model() -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Load trained sentiment model and vectorizer.
    
    Returns:
        Tuple of (model, vectorizer)
    """
    model_path = settings.ml.sentiment_model_path
    vectorizer_path = settings.ml.vectorizer_path
    
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Sentiment model not found. Run train_sentiment_model() first."
        )
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer


def predict_sentiment(message: str) -> Tuple[str, float]:
    """
    Predict sentiment of a message.
    
    Args:
        message: Client/advisor message text
    
    Returns:
        Tuple of (sentiment_label, confidence_score)
        Labels: Positive, Neutral, Frustrated, Confused
    """
    model, vectorizer = load_sentiment_model()
    
    # Vectorize
    X = vectorizer.transform([message])
    
    # Predict
    sentiment = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    
    return sentiment, confidence


# ========== PRIORITY SCORING ==========


def train_priority_model(data_path: Path | None = None) -> GradientBoostingRegressor:
    """
    Train priority scoring model on synthetic priority data.
    
    Args:
        data_path: Path to priority_scoring_data.csv
    
    Returns:
        Trained GradientBoostingRegressor
    """
    if data_path is None:
        data_path = settings.paths.synthetic_data_dir / "11_priority_scoring_data.csv"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Features
    feature_cols = [
        "days_in_current_state",
        "sla_overdue",
        "client_age_55_plus",
        "document_quality_score",
    ]
    
    X = df[feature_cols]
    y = df["priority_score_calculated"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"✓ Priority model trained")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    
    # Save
    model_path = settings.ml.priority_model_path
    joblib.dump(model, model_path)
    
    print(f"  Saved to: {model_path}")
    
    return model


def load_priority_model() -> GradientBoostingRegressor:
    """
    Load trained priority scoring model.
    
    Returns:
        Trained GradientBoostingRegressor
    """
    model_path = settings.ml.priority_model_path
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Priority model not found. Run train_priority_model() first."
        )
    
    model = joblib.load(model_path)
    return model


def calculate_priority_score(
    days_in_state: int,
    sla_overdue: int,
    client_age_55_plus: bool,
    doc_quality_score: float,
) -> float:
    """
    Calculate priority score for an LOA workflow.
    
    Args:
        days_in_state: Days spent in current state
        sla_overdue: Days past SLA (negative if not overdue)
        client_age_55_plus: Whether client is 55+
        doc_quality_score: Document quality score (0-100)
    
    Returns:
        Priority score (0-10 scale)
    """
    model = load_priority_model()
    
    # Prepare features
    features = [[
        days_in_state,
        max(0, sla_overdue),  # Clamp to 0 minimum
        1 if client_age_55_plus else 0,
        doc_quality_score,
    ]]
    
    # Predict (sklearn returns numpy scalar; convert to Python float for DB/JSON compatibility)
    priority = model.predict(features)[0]
    # Clamp to 0-10 range and ensure native float for SQLAlchemy/psycopg2
    priority = float(max(0.0, min(10.0, priority)))
    return priority


# ========== HELPER FUNCTIONS ==========


def train_all_models() -> None:
    """
    Train all ML models on synthetic data.
    
    Call this once during setup or when retraining.
    """
    print("=" * 60)
    print("Training ML Models")
    print("=" * 60)
    
    # Train sentiment model
    train_sentiment_model()
    print()
    
    # Train priority model
    train_priority_model()
    print()
    
    print("=" * 60)
    print("✓ All models trained successfully")
    print("=" * 60)