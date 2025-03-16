from fastapi import APIRouter, Depends, HTTPException
from ..types import SimulationConfig, SimulationResult, APIResponse, SystemMetrics
import uuid
from typing import Dict

router = APIRouter()

def get_request_id():
    return str(uuid.uuid4())

@router.post("/simulate", response_model=APIResponse)
async def run_simulation(config: SimulationConfig):
    """Run quantum simulation"""
    request_id = get_request_id()
    try:
        result = await router.app.state.core.run_simulation(config)
        return APIResponse(
            status="success",
            data=result.dict(),
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get system status"""
    return {"status": "operational"}

@router.get("/circuit/{circuit_id}")
async def get_circuit(circuit_id: str):
    """Get quantum circuit details"""
    core = router.app.state.core
    return await core.get_circuit(circuit_id)

@router.post("/visualize")
async def visualize_circuit(circuit: QuantumCircuit, viz_type: VisualizationType):
    """Generate circuit visualization"""
    core = router.app.state.core
    return await core.visualize(circuit, viz_type)

@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics"""
    try:
        return await router.app.state.core.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/circuit", response_model=APIResponse)
async def create_circuit(circuit: QuantumCircuit):
    """Create new quantum circuit"""
    request_id = get_request_id()
    try:
        circuit_id = await router.app.state.core.store_circuit(circuit)
        return APIResponse(
            status="success",
            data={"circuit_id": circuit_id},
            request_id=request_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Detailed health check"""
    core = router.app.state.core
    metrics = await core.get_metrics()
    return {
        "status": "healthy",
        "memory_usage": metrics["classical_memory"],
        "uptime": metrics["uptime_seconds"]
    }
