import asyncio
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from kaleidoscope_core import KaleidoscopeCore
from kaleidoscope_core.types import SimulationConfig

async def run_benchmarks(num_qubits_range=(2, 10), shots=1000):
    """Run performance benchmarks"""
    core = KaleidoscopeCore(work_dir="workdir", data_dir="data")
    results = []
    
    for n_qubits in range(*num_qubits_range):
        config = SimulationConfig(qubits=n_qubits, shots=shots)
        start = time.time()
        await core.run_simulation(config)
        elapsed = time.time() - start
        results.append((n_qubits, elapsed))
        
    return results

if __name__ == "__main__":
    results = asyncio.run(run_benchmarks())
    for qubits, time in results:
        print(f"{qubits} qubits: {time:.3f} seconds")
