"""
Core implementation of the Kaleidoscope AI system.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any
from .quantum import QuantumSimulator
from .types import FileType, SimulationConfig, QuantumCircuit, VisualizationType
from .exceptions import KaleidoscopeError
from .monitoring import SystemMonitor

logger = logging.getLogger(__name__)

class KaleidoscopeCore:
    """Main system core class"""
    
    def __init__(self, work_dir: str, data_dir: str):
        self.work_dir = Path(work_dir)
        self.data_dir = Path(data_dir)
        self.quantum_sim = QuantumSimulator()
        self._active_circuits = []
        self._simulation_count = 0
        self._start_time = time.time()
        self.visualization = None
        self.monitor = SystemMonitor()
        self._circuit_store = {}  # Store for active circuits
        self._shutdown_hooks = []
        logger.info("Initialized Kaleidoscope Core")
    
    def register_shutdown_hook(self, hook):
        """Register shutdown hook"""
        self._shutdown_hooks.append(hook)

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        for hook in self._shutdown_hooks:
            await hook()
        self.quantum_sim.cleanup_resources()
        self.monitor.stop()

    async def run_simulation(self, config: SimulationConfig):
        """Run quantum simulation"""
        self._simulation_count += 1
        return await self.quantum_sim.run(config)

    async def visualize(self, circuit: QuantumCircuit, viz_type: VisualizationType):
        """Generate visualization"""
        try:
            viz = self.get_visualizer()
            return await viz.generate(circuit, viz_type)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise KaleidoscopeError(str(e), "VIZ_ERROR")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            "quantum_memory": self.quantum_sim.get_memory_usage(),
            "classical_memory": self.monitor.get_memory_usage()["rss"],
            "cpu_usage": self.monitor.get_cpu_usage(),
            "active_circuits": len(self._active_circuits),
            "total_simulations": self._simulation_count,
            "uptime_seconds": self.monitor.get_uptime()
        }

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")

    def detect_file_type(self, file_path: str) -> FileType:
        """Detect the type of a file"""
        ext_map = {".py": FileType.PYTHON, ".cpp": FileType.CPP, ".c": FileType.C, ".js": FileType.JAVASCRIPT}
        return ext_map.get(os.path.splitext(file_path)[1].lower(), FileType.UNKNOWN)

    def ingest_software(self, file_path: str) -> Dict[str, Any]:
        """Analyze and decompile software"""
        file_type = self.detect_file_type(file_path)
        result = {"file_type": file_type.value, "decompiled_files": []}

        if file_type == FileType.BINARY:
            decompiled_path = os.path.join(self.decompiled_dir, os.path.basename(file_path) + "_decompiled.txt")
            with open(decompiled_path, 'w') as f:
                f.write("Decompiled binary content (simulated)\n")
            result["decompiled_files"].append(decompiled_path)

        return result

    def get_visualizer(self):
        """Get or create visualization component"""
        if self.visualization is None:
            from kaleidoscope_visualization import KaleidoscopeVisualizer
            self.visualization = KaleidoscopeVisualizer()
        return self.visualization
