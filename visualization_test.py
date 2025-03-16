import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from kaleidoscope_core import KaleidoscopeCore
from kaleidoscope_core.types import QuantumCircuit, VisualizationType

async def test_visualizations():
    """Test different visualization types"""
    core = KaleidoscopeCore(work_dir="workdir", data_dir="data")
    
    circuit = QuantumCircuit(
        gates=[
            {"type": "h", "target": 0},
            {"type": "cnot", "control": 0, "target": 1}
        ],
        qubits=2,
        name="Bell State"
    )
    
    for viz_type in VisualizationType:
        result = await core.visualize(circuit, viz_type)
        print(f"Generated {viz_type.value} visualization")

if __name__ == "__main__":
    asyncio.run(test_visualizations())
