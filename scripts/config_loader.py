"""
Config Loader - Stub module for custom tools
"""

import os
import json
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Simple config loader for custom tools"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {}
        
        # Try to load from file
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        
        # Override with environment variables
        config['debug'] = os.getenv('DEBUG', config.get('debug', False))
        config['log_level'] = os.getenv('LOG_LEVEL', config.get('log_level', 'INFO'))
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self.config.get(key, default)


# Default instance
config = ConfigLoader()