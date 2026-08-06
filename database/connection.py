"""
Database connection handler.
Manages both SQLite (offline/local) and
PostgreSQL (cloud/sync) connections.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base
import os

# ── LOCAL DATABASE (always available offline) ──
SQLITE_URL = "sqlite:///./data/almanac_local.db"

# ── CLOUD DATABASE (when internet available) ──
POSTGRES_URL = os.getenv(
    "DATABASE_URL",
    None
)

# ── Create local engine ──
local_engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False}
)

LocalSession = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=local_engine
)


def init_local_db():
    """
    Initialize local SQLite database.
    Creates all tables if they do not exist.
    Called once on application startup.
    """
    Base.metadata.create_all(bind=local_engine)
    print("Local database initialized successfully.")


def get_local_db():
    """
    Dependency for FastAPI endpoints.
    Yields a database session and ensures
    it is closed after each request.
    """
    db = LocalSession()
    try:
        yield db
    finally:
        db.close()