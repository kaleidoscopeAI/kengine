import pytest
import numpy as np
from kaleidoscope_core.quantum import QuantumSimulator
from kaleidoscope_core.types import SimulationConfig

@pytest.fixture
def simulator():
    return QuantumSimulator(qubits=2)

def test_initialization(simulator):
    """Test initial state"""
    state = simulator.get_statevector()
    assert np.allclose(state, [1, 0, 0, 0])

def test_hadamard(simulator):
    """Test Hadamard gate"""
    simulator.apply_hadamard(0)
    state = simulator.get_statevector()
    expected = np.array([1, 1, 0, 0]) / np.sqrt(2)
    assert np.allclose(state, expected)
