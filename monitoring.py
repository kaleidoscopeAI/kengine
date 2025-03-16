import psutil
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self._metrics_history = []
        self._running = True
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Start background metrics collection"""
        def collect_metrics():
            while self._running:
                metrics = {
                    "timestamp": time.time(),
                    "memory": self.get_memory_usage(),
                    "cpu": self.get_cpu_usage()
                }
                self._metrics_history.append(metrics)
                time.sleep(60)  # Collect every minute
                
        import threading
        self.monitor_thread = threading.Thread(target=collect_metrics, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self._running = False
        self.monitor_thread.join()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss / (1024 * 1024),
            "vms": memory_info.vms / (1024 * 1024)
        }
        
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent()
        
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
