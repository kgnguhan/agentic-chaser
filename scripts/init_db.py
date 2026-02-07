"""Initialize the database: create all tables."""

from __future__ import annotations

from models.database import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created.")
