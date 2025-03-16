"""Database initialization script"""
import os
import time
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
Base = declarative_base()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_db(data_dir: str):
    """Initialize database tables with retry logic"""
    try:
        db_url = os.getenv('DATABASE_URL', 'postgresql://kaleidoscope:kaleidoscope@localhost/kaleidoscope')
        logger.info(f"Initializing database at {db_url}")
        
        # Wait for database to be ready
        engine = sa.create_engine(db_url)
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with engine.connect() as conn:
                    conn.execute(sa.text('SELECT 1'))
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(f"Database not ready (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(5)
        
        # Create tables
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
