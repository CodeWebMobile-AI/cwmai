"""
Custom JSON Encoder for handling dataclasses and special objects.
"""

import json
from dataclasses import asdict, is_dataclass
from typing import Any


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and other special objects."""
    
    def default(self, obj: Any) -> Any:
        """Convert special objects to JSON-serializable format.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        # Handle dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
            
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            # Create a dict of serializable attributes
            result = {}
            for key, value in obj.__dict__.items():
                # Skip private attributes that might cause issues
                if not key.startswith('_'):
                    try:
                        # Test if the value is serializable
                        json.dumps(value, cls=CustomJSONEncoder)
                        result[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable attributes
                        pass
            return result
            
        # Let the base class handle other types
        return super().default(obj)
        

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely dump object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    # Remove any cls argument to use our encoder
    kwargs.pop('cls', None)
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)


def safe_json_dump(obj: Any, fp, **kwargs) -> None:
    """Safely dump object to JSON file.
    
    Args:
        obj: Object to serialize
        fp: File pointer
        **kwargs: Additional arguments for json.dump
    """
    # Remove any cls argument to use our encoder
    kwargs.pop('cls', None)
    json.dump(obj, fp, cls=CustomJSONEncoder, **kwargs)