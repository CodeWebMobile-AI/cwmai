"""
Security Input Validation and Sanitization Layer
Addresses OWASP Top 10 vulnerabilities for the CWMAI system.
"""

import json
import re
import os
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import html


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SecurityValidator:
    """
    Comprehensive input validation and sanitization.
    Addresses OWASP Top 10 vulnerabilities:
    - A03: Injection attacks
    - A08: Software and Data Integrity Failures
    - A09: Security Logging and Monitoring Failures
    """

    # Maximum input sizes to prevent DoS
    MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_STRING_LENGTH = 100000  # 100KB
    MAX_API_PROMPT_LENGTH = 50000  # 50KB
    MAX_GITHUB_TITLE_LENGTH = 256
    MAX_GITHUB_BODY_LENGTH = 65536

    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'subprocess\.',
        r'os\.system',
        r'os\.popen',
        r'pickle\.loads',
        r'marshal\.loads',
        r'compile\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
        r"(\bOR\b\s+\d+\s*=\s*\d+|\bAND\b\s+\d+\s*=\s*\d+)",
        r"('|\";|--|/\*|\*/)",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.\/",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"..%2f",
        r"..%5c",
    ]

    def __init__(self):
        """Initialize the security validator."""
        self.logger = logging.getLogger(__name__)
        # Compile regex patterns for performance
        self.dangerous_regex = re.compile('|'.join(self.DANGEROUS_PATTERNS), re.IGNORECASE)
        self.sql_injection_regex = re.compile('|'.join(self.SQL_INJECTION_PATTERNS), re.IGNORECASE)
        self.path_traversal_regex = re.compile('|'.join(self.PATH_TRAVERSAL_PATTERNS), re.IGNORECASE)

    def validate_json_input(self, json_data: Union[str, bytes], max_size: Optional[int] = None) -> ValidationResult:
        """
        Safely validate and parse JSON input.
        Prevents JSON injection and DoS attacks.
        """
        max_size = max_size or self.MAX_JSON_SIZE
        
        try:
            # Check size first
            if isinstance(json_data, str):
                data_size = len(json_data.encode('utf-8'))
            else:
                data_size = len(json_data)
                json_data = json_data.decode('utf-8')
            
            if data_size > max_size:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"JSON input too large: {data_size} bytes (max: {max_size})"]
                )

            # Check for dangerous patterns
            if self.dangerous_regex.search(json_data):
                return ValidationResult(
                    is_valid=False,
                    errors=["JSON contains potentially dangerous code patterns"]
                )

            # Parse JSON with strict validation
            parsed_data = json.loads(json_data)
            
            # Additional validation for nested structures
            validation_errors = self._validate_json_structure(parsed_data)
            if validation_errors:
                return ValidationResult(
                    is_valid=False,
                    errors=validation_errors
                )

            return ValidationResult(
                is_valid=True,
                sanitized_value=parsed_data
            )

        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON format: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"JSON validation error: {str(e)}"]
            )

    def validate_api_prompt(self, prompt: str) -> ValidationResult:
        """
        Validate AI API prompts for size and content.
        Prevents prompt injection and DoS attacks.
        """
        if not isinstance(prompt, str):
            return ValidationResult(
                is_valid=False,
                errors=["Prompt must be a string"]
            )

        # Check length
        if len(prompt) > self.MAX_API_PROMPT_LENGTH:
            return ValidationResult(
                is_valid=False,
                errors=[f"Prompt too long: {len(prompt)} chars (max: {self.MAX_API_PROMPT_LENGTH})"]
            )

        # Check for dangerous patterns
        if self.dangerous_regex.search(prompt):
            return ValidationResult(
                is_valid=False,
                errors=["Prompt contains potentially dangerous code patterns"]
            )

        # Sanitize the prompt
        sanitized_prompt = self._sanitize_text(prompt)
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_prompt,
            warnings=["Prompt sanitized"] if sanitized_prompt != prompt else []
        )

    def validate_github_content(self, title: str, body: str) -> ValidationResult:
        """
        Validate GitHub issue/PR content.
        Prevents injection attacks via GitHub API.
        """
        errors = []
        warnings = []

        # Validate title
        if len(title) > self.MAX_GITHUB_TITLE_LENGTH:
            errors.append(f"Title too long: {len(title)} chars (max: {self.MAX_GITHUB_TITLE_LENGTH})")

        # Validate body
        if len(body) > self.MAX_GITHUB_BODY_LENGTH:
            errors.append(f"Body too long: {len(body)} chars (max: {self.MAX_GITHUB_BODY_LENGTH})")

        # Check for dangerous patterns
        if self.dangerous_regex.search(title) or self.dangerous_regex.search(body):
            errors.append("Content contains potentially dangerous patterns")

        # Sanitize content
        sanitized_title = self._sanitize_github_text(title)
        sanitized_body = self._sanitize_github_text(body)

        if sanitized_title != title or sanitized_body != body:
            warnings.append("Content was sanitized")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value={"title": sanitized_title, "body": sanitized_body},
            errors=errors,
            warnings=warnings
        )

    def validate_file_path(self, file_path: str, allowed_dirs: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate file paths to prevent path traversal attacks.
        """
        if not isinstance(file_path, str):
            return ValidationResult(
                is_valid=False,
                errors=["File path must be a string"]
            )

        # Check for path traversal patterns
        if self.path_traversal_regex.search(file_path):
            return ValidationResult(
                is_valid=False,
                errors=["File path contains path traversal patterns"]
            )

        # Normalize and validate path
        try:
            normalized_path = os.path.normpath(file_path)
            
            # Ensure path doesn't escape allowed directories
            if allowed_dirs:
                is_allowed = any(
                    normalized_path.startswith(os.path.normpath(allowed_dir))
                    for allowed_dir in allowed_dirs
                )
                if not is_allowed:
                    return ValidationResult(
                        is_valid=False,
                        errors=["File path outside allowed directories"]
                    )

            return ValidationResult(
                is_valid=True,
                sanitized_value=normalized_path
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid file path: {str(e)}"]
            )

    def validate_environment_variable(self, var_name: str, var_value: str) -> ValidationResult:
        """
        Validate environment variables for security.
        """
        errors = []
        warnings = []

        # Check variable name
        if not re.match(r'^[A-Z_][A-Z0-9_]*$', var_name):
            errors.append("Invalid environment variable name format")

        # Check for sensitive patterns in value
        if var_name.endswith('_KEY') or var_name.endswith('_TOKEN') or var_name.endswith('_SECRET'):
            # API keys should not contain dangerous patterns
            if self.dangerous_regex.search(var_value):
                errors.append("Environment variable contains dangerous patterns")
            
            # Validate API key format (basic check)
            if len(var_value) < 10:
                warnings.append("API key seems too short")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=var_value,
            errors=errors,
            warnings=warnings
        )

    def sanitize_for_logging(self, data: Any) -> str:
        """
        Sanitize data for safe logging (remove sensitive information).
        """
        if isinstance(data, str):
            # Mask API keys and tokens
            data = re.sub(r'(api[_-]?key|token|secret)["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 
                         r'\1=***REDACTED***', data, flags=re.IGNORECASE)
            
            # Mask authentication headers
            data = re.sub(r'(authorization|x-api-key)["\']?\s*:\s*["\']?([^"\'\s]+)', 
                         r'\1: ***REDACTED***', data, flags=re.IGNORECASE)
            
            # Truncate long strings
            if len(data) > 500:
                data = data[:500] + "...[TRUNCATED]"
        
        elif isinstance(data, dict):
            # Recursively sanitize dictionary
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['key', 'token', 'secret', 'password']):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self.sanitize_for_logging(value)
            return str(sanitized)
        
        return str(data)

    def _validate_json_structure(self, data: Any, depth: int = 0) -> List[str]:
        """
        Recursively validate JSON structure for dangerous content.
        """
        errors = []
        
        # Prevent deep nesting DoS
        if depth > 100:
            errors.append("JSON nesting too deep")
            return errors

        if isinstance(data, dict):
            # Check for dangerous keys
            for key in data.keys():
                if isinstance(key, str) and self.dangerous_regex.search(key):
                    errors.append(f"Dangerous pattern in JSON key: {key}")
                
                # Recursively validate values
                errors.extend(self._validate_json_structure(data[key], depth + 1))
        
        elif isinstance(data, list):
            # Check list size
            if len(data) > 10000:
                errors.append("JSON array too large")
            
            for item in data:
                errors.extend(self._validate_json_structure(item, depth + 1))
        
        elif isinstance(data, str):
            # Check string content
            if self.dangerous_regex.search(data):
                errors.append("Dangerous pattern in JSON string value")

        return errors

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text content by removing dangerous patterns.
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove or escape dangerous patterns
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # HTML escape for safety
        text = html.escape(text)
        
        return text

    def _sanitize_github_text(self, text: str) -> str:
        """
        Sanitize text for GitHub content (preserves markdown).
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove dangerous script tags but preserve markdown
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        return text


# Global validator instance
security_validator = SecurityValidator()


def safe_json_load(json_data: Union[str, bytes], max_size: Optional[int] = None) -> Any:
    """
    Safe JSON loading with validation.
    Replacement for json.loads() calls throughout the codebase.
    """
    result = security_validator.validate_json_input(json_data, max_size)
    if not result.is_valid:
        raise ValueError(f"JSON validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_value


def safe_api_request(prompt: str) -> str:
    """
    Safe API prompt validation.
    Use before sending prompts to AI APIs.
    """
    result = security_validator.validate_api_prompt(prompt)
    if not result.is_valid:
        raise ValueError(f"API prompt validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_value


def safe_github_content(title: str, body: str) -> Dict[str, str]:
    """
    Safe GitHub content validation.
    Use before creating GitHub issues/PRs.
    """
    result = security_validator.validate_github_content(title, body)
    if not result.is_valid:
        raise ValueError(f"GitHub content validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_value


def safe_file_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> str:
    """
    Safe file path validation.
    Use before file operations.
    """
    result = security_validator.validate_file_path(file_path, allowed_dirs)
    if not result.is_valid:
        raise ValueError(f"File path validation failed: {', '.join(result.errors)}")
    
    return result.sanitized_value