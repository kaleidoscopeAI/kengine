from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def get_core():
    """Get Kaleidoscope core instance"""
    from kaleidoscope_core import KaleidoscopeCore
    return KaleidoscopeCore()

def get_simulator():
    """Get quantum simulator instance"""
    from kaleidoscope_core import QuantumSimulator
    return QuantumSimulator()

@router.post("/analyze")
async def analyze_software(file_path: str) -> Dict[str, Any]:
    """Analyze software endpoint"""
    try:
        core = get_core()
        result = core.ingest_software(file_path)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quantum/state")
async def get_quantum_state() -> Dict[str, Any]:
    """Get current quantum state"""
    try:
        simulator = get_simulator()
        state = simulator.measure()
        return {"status": "success", "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
