#!/usr/bin/env python3
"""
Kaleidoscope AI Setup Script
Handles dependency installation and initial configuration.
"""

import subprocess
import os
import sys
import json
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup")

REQUIRED_PACKAGES = [
    "numpy",
    "networkx",
    "rdkit",
    "requests",
    "scipy",
    "matplotlib",
    "sqlite3",
    "asyncio",
]

EXTERNAL_TOOLS = {
    "radare2": "r2",
    "ghidra": "ghidra_server",
    "retdec": "retdec-decompiler",
    "js-beautify": "js-beautify"
}

def check_python_version():
    """Verify Python version meets requirements."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)

def install_python_packages():
    """Install required Python packages."""
    logger.info("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        sys.exit(1)

def check_external_tools() -> Dict[str, bool]:
    """Check if required external tools are installed."""
    results = {}
    for tool, cmd in EXTERNAL_TOOLS.items():
        if shutil.which(cmd):
            results[tool] = True
            logger.info(f"Found {tool}")
        else:
            results[tool] = False
            logger.warning(f"Missing {tool} - some features will be limited")
    return results

def create_directory_structure():
    """Create required directories."""
    dirs = [
        "kaleidoscope_data",
        "kaleidoscope_data/software_analysis",
        "kaleidoscope_data/drug_discovery",
        "kaleidoscope_data/pattern_recognition",
        "kaleidoscope_data/visualizations",
        "kaleidoscope_data/logs"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")

def main():
    """Run setup process."""
    logger.info("Starting Kaleidoscope AI setup...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_python_packages()
    
    # Check external tools
    tool_status = check_external_tools()
    
    # Create directories
    create_directory_structure()
    
    logger.info("Setup completed successfully")

if __name__ == "__main__":
    main()
