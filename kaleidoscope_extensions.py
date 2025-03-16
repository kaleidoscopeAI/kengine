from kaleidoscope_core import KaleidoscopeCore
import importlib
from typing import Dict

class ExtensionManager:
    def __init__(self, core: KaleidoscopeCore):
        self.core = core
        self.extensions = {}
        self.extension_states = {}
        
    def load_extension(self, name: str, path: str):
        try:
            module = importlib.import_module(path)
            extension = module.initialize(self.core)
            self.extensions[name] = extension 
            self.extension_states[name] = "active"
            return True
        except Exception as e:
            logger.error(f"Failed to load extension {name}: {e}")
            return False