"""
Validation - Stub module for custom tools
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class Validator:
    """Simple validator for custom tools"""
    
    @staticmethod
    def validate_required(value: Any, field_name: str) -> None:
        """Validate that a value is not None or empty"""
        if value is None:
            raise ValueError(f"{field_name} is required")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str) -> None:
        """Validate that a value is of expected type"""
        if not isinstance(value, expected_type):
            raise TypeError(f"{field_name} must be of type {expected_type.__name__}")
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://[^\s]+$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_path(path: Union[str, Path], must_exist: bool = False) -> bool:
        """Validate file path"""
        try:
            p = Path(path)
            if must_exist:
                return p.exists()
            return True
        except:
            return False
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None, field_name: str = "Value") -> None:
        """Validate that a numeric value is within range"""
        if min_val is not None and value < min_val:
            raise ValueError(f"{field_name} must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{field_name} must be at most {max_val}")
    
    @staticmethod
    def validate_choice(value: Any, choices: List[Any], field_name: str = "Value") -> None:
        """Validate that a value is one of the allowed choices"""
        if value not in choices:
            raise ValueError(f"{field_name} must be one of: {', '.join(str(c) for c in choices)}")
    
    @staticmethod
    def validate_dict(data: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate a dictionary against a schema"""
        errors = []
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Check required
            if rules.get('required', False) and value is None:
                errors.append(f"{field} is required")
                continue
            
            if value is not None:
                # Check type
                expected_type = rules.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.append(f"{field} must be of type {expected_type.__name__}")
                
                # Check range for numbers
                if isinstance(value, (int, float)):
                    min_val = rules.get('min')
                    max_val = rules.get('max')
                    if min_val is not None and value < min_val:
                        errors.append(f"{field} must be at least {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"{field} must be at most {max_val}")
                
                # Check choices
                choices = rules.get('choices')
                if choices and value not in choices:
                    errors.append(f"{field} must be one of: {', '.join(str(c) for c in choices)}")
        
        return errors


# Default instance
validator = Validator()