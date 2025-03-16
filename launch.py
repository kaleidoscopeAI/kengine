#!/usr/bin/env python3
"""
Kaleidoscope AI System Launcher
==============================
Local development version
"""

import os
import json
import logging
from pathlib import Path
import uvicorn

# Configure paths
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
WORK_DIR = BASE_DIR / "workdir"
DATA_DIR = BASE_DIR / "data"

# Create required directories
WORK_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("KaleidoscopeLauncher")

def main():
    """Main entry point"""
    try:
        # Load configuration
        config_path = CONFIG_DIR / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path) as f:
            config = json.load(f)
        
        # Initialize core system with absolute paths
        from kaleidoscope_core import KaleidoscopeCore
        core = KaleidoscopeCore(
            work_dir=str(WORK_DIR),
            data_dir=str(DATA_DIR)
        )
        
        # Start API server
        from kaleidoscope_core.api import create_app
        app = create_app(core)
        
        logger.info("Starting Kaleidoscope AI system...")
        uvicorn.run(
            app,
            host=config["api"]["host"],
            port=config["api"]["port"],
            log_level="info"
        )
        
        return 0
    except Exception as e:
        logger.error(f"Error launching system: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
