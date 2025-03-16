"""
Type definitions for the Kaleidoscope AI system.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel

class FileType(Enum):
    """Supported file types"""
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"

class SimulationConfig(BaseModel):
    """Simulation configuration"""
    qubits: int
    shots: int
    noise_model: Optional[Dict] = None
    optimization_level: int = 1

class SimulationResult(BaseModel):
    """Simulation result data"""
    status: str
    results: Dict
    execution_time: float
    metadata: Optional[Dict] = None

class VisualizationType(Enum):
    """Visualization types"""
    CIRCUIT = "circuit"
    STATEVECTOR = "statevector"
    PROBABILITY = "probability"
    BLOCH = "bloch"

class KaleidoscopeError(Exception):
    """Base error class"""
    def __init__(self, message: str, error_code: str):
        self.error_code = error_code
        super().__init__(message)

class QuantumCircuit(BaseModel):
    """Quantum circuit definition"""
    gates: List[Dict[str, Any]]
    qubits: int
    name: Optional[str] = None
    metadata: Optional[Dict] = None

class SystemMetrics(BaseModel):
    """System metrics data"""
    quantum_memory_usage: float
    classical_memory_usage: float
    active_circuits: int
    total_simulations: int
    uptime_seconds: float

class APIResponse(BaseModel):
    """Standard API response"""
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    request_id: str
