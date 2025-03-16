import json
from pathlib import Path
import logging
from typing import Dict, Any
import asyncio

def load_config(config_file: str = "stress_test_config.json") -> Dict[str, Any]:
    """Load configuration from file with defaults."""
    defaults = {
        "concurrent_jobs": 10,
        "duration_seconds": 60,
        "fail_fast": True,
        "logging": {
            "level": "INFO",
            "file": "stress_test.log",
            "detailed_file": "stress_test_detailed.log", 
            "max_file_size": 10_000_000,
            "backup_count": 5
        },
        "simulation": {
            "timeout": 30,
            "retries": 3,
            "error_threshold": 5
        }
    }
    
    try:
        config_path = Path(__file__).parent.parent / "config" / config_file
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            # Merge with defaults
            return {**defaults, **config}
    except Exception as e:
        logging.warning(f"Failed to load config: {e}, using defaults")
    
    return defaults

if __name__ == "__main__":
    config = load_config()
    asyncio.run(stress_test(
        concurrent_jobs=config["concurrent_jobs"],
        duration_seconds=config["duration_seconds"],
        fail_fast=config["fail_fast"]
    ))
