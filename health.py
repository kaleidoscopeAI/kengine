from fastapi import APIRouter, HTTPException
import psutil
import redis
import sqlalchemy as sa
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Check system health"""
    status = {"status": "error", "services": {}}
    
    try:
        # System resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        status["services"]["system"] = {
            "status": "ok",
            "cpu": f"{cpu_percent}%",
            "memory": f"{memory.percent}%"
        }
        
        # Redis
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6380/0')
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            status["services"]["redis"] = {"status": "ok"}
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            status["services"]["redis"] = {"status": "error", "message": str(e)}
        
        # Database
        try:
            db_url = os.getenv('DATABASE_URL', 'postgresql://kaleidoscope:kaleidoscope@localhost/kaleidoscope')
            engine = sa.create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(sa.text('SELECT 1'))
            status["services"]["database"] = {"status": "ok"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            status["services"]["database"] = {"status": "error", "message": str(e)}
        
        # Overall status
        if all(svc["status"] == "ok" for svc in status["services"].values()):
            status["status"] = "ok"
        
        return status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
