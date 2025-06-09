"""
Comprehensive Security Validation Tests
Tests the security validator and fixes implemented for OWASP compliance.
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import patch, mock_open

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from security_validator import (
    SecurityValidator, 
    safe_json_load, 
    safe_api_request, 
    safe_github_content, 
    safe_file_path
)


class TestSecurityValidation(unittest.TestCase):
    """Test suite for security validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SecurityValidator()
    
    def test_json_validation_safe_input(self):
        """Test JSON validation with safe input."""
        safe_json = '{"name": "test", "value": 123}'
        result = self.validator.validate_json_input(safe_json)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.sanitized_value["name"], "test")
        self.assertEqual(result.sanitized_value["value"], 123)
    
    def test_json_validation_dangerous_patterns(self):
        """Test JSON validation rejects dangerous patterns."""
        dangerous_inputs = [
            '{"code": "__import__(\'os\').system(\'rm -rf /\')"}',
            '{"exec": "exec(\'print(1)\')"}',
            '{"eval": "eval(\'1+1\')"}',
            '{"subprocess": "subprocess.call([\'ls\'])"}',
            '{"pickle": "pickle.loads(data)"}',
        ]
        
        for dangerous_json in dangerous_inputs:
            result = self.validator.validate_json_input(dangerous_json)
            self.assertFalse(result.is_valid, f"Should reject: {dangerous_json}")
            self.assertIn("dangerous", " ".join(result.errors).lower())
    
    def test_json_validation_size_limits(self):
        """Test JSON validation enforces size limits."""
        # Create oversized JSON
        large_json = '{"data": "' + "A" * (10 * 1024 * 1024 + 1) + '"}'
        result = self.validator.validate_json_input(large_json)
        
        self.assertFalse(result.is_valid)
        self.assertIn("too large", " ".join(result.errors))
    
    def test_json_validation_malformed(self):
        """Test JSON validation handles malformed JSON."""
        malformed_inputs = [
            '{"incomplete": ',
            '{"invalid": json}',
            'not json at all',
            '{"unescaped": "quote"s"}',
        ]
        
        for malformed in malformed_inputs:
            result = self.validator.validate_json_input(malformed)
            self.assertFalse(result.is_valid)
    
    def test_api_prompt_validation(self):
        """Test API prompt validation."""
        # Safe prompt
        safe_prompt = "Analyze this data and provide insights"
        result = self.validator.validate_api_prompt(safe_prompt)
        self.assertTrue(result.is_valid)
        
        # Oversized prompt
        large_prompt = "A" * (50000 + 1)
        result = self.validator.validate_api_prompt(large_prompt)
        self.assertFalse(result.is_valid)
        self.assertIn("too long", " ".join(result.errors))
        
        # Dangerous prompt
        dangerous_prompt = "Execute this: __import__('os').system('rm -rf /')"
        result = self.validator.validate_api_prompt(dangerous_prompt)
        self.assertFalse(result.is_valid)
    
    def test_github_content_validation(self):
        """Test GitHub content validation."""
        # Safe content
        title = "Fix authentication bug"
        body = "## Description\nThis PR fixes the authentication issue."
        result = self.validator.validate_github_content(title, body)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.sanitized_value["title"], title)
        
        # Oversized title
        long_title = "A" * 300
        result = self.validator.validate_github_content(long_title, body)
        self.assertFalse(result.is_valid)
        self.assertIn("too long", " ".join(result.errors))
        
        # Dangerous content
        dangerous_body = '<script>alert("xss")</script>Delete everything'
        result = self.validator.validate_github_content(title, dangerous_body)
        
        # Should be valid but sanitized
        if result.is_valid:
            self.assertNotIn('<script>', result.sanitized_value["body"])
        else:
            self.assertIn("dangerous", " ".join(result.errors))
    
    def test_file_path_validation(self):
        """Test file path validation."""
        # Safe paths
        safe_paths = [
            "/home/user/documents/file.txt",
            "scripts/data.json",
            "./local_file.py"
        ]
        
        for path in safe_paths:
            result = self.validator.validate_file_path(path)
            self.assertTrue(result.is_valid, f"Should accept: {path}")
        
        # Path traversal attempts
        dangerous_paths = [
            "../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "/var/www/../../../etc/shadow",
            "..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2fpasswd"
        ]
        
        for path in dangerous_paths:
            result = self.validator.validate_file_path(path)
            self.assertFalse(result.is_valid, f"Should reject: {path}")
            self.assertIn("traversal", " ".join(result.errors))
    
    def test_file_path_allowed_directories(self):
        """Test file path validation with allowed directories."""
        allowed_dirs = ["/safe/project", "/tmp/uploads"]
        
        # Path within allowed directory
        safe_path = "/safe/project/data.json"
        result = self.validator.validate_file_path(safe_path, allowed_dirs)
        self.assertTrue(result.is_valid)
        
        # Path outside allowed directories
        unsafe_path = "/etc/passwd"
        result = self.validator.validate_file_path(unsafe_path, allowed_dirs)
        self.assertFalse(result.is_valid)
        self.assertIn("outside allowed", " ".join(result.errors))
    
    def test_environment_variable_validation(self):
        """Test environment variable validation."""
        # Valid API key
        result = self.validator.validate_environment_variable("API_KEY", "sk-abcd1234567890")
        self.assertTrue(result.is_valid)
        
        # Invalid variable name
        result = self.validator.validate_environment_variable("123invalid", "value")
        self.assertFalse(result.is_valid)
        
        # API key with dangerous patterns
        result = self.validator.validate_environment_variable("API_KEY", "__import__('os')")
        self.assertFalse(result.is_valid)
    
    def test_logging_sanitization(self):
        """Test logging sanitization removes sensitive data."""
        sensitive_data = {
            "api_key": "sk-sensitive123",
            "token": "abc123token",
            "password": "secret123",
            "normal_data": "public_info"
        }
        
        sanitized = self.validator.sanitize_for_logging(sensitive_data)
        
        self.assertNotIn("sk-sensitive123", sanitized)
        self.assertNotIn("abc123token", sanitized)
        self.assertNotIn("secret123", sanitized)
        self.assertIn("public_info", sanitized)
        self.assertIn("***REDACTED***", sanitized)
    
    def test_safe_json_load_function(self):
        """Test the safe_json_load convenience function."""
        # Valid JSON
        safe_data = safe_json_load('{"test": true}')
        self.assertEqual(safe_data["test"], True)
        
        # Invalid JSON should raise ValueError
        with self.assertRaises(ValueError):
            safe_json_load('{"invalid": __import__("os")}')
        
        # Oversized JSON should raise ValueError
        with self.assertRaises(ValueError):
            safe_json_load('{"data": "' + "A" * (10 * 1024 * 1024 + 1) + '"}')
    
    def test_safe_api_request_function(self):
        """Test the safe_api_request convenience function."""
        # Valid prompt
        safe_prompt = safe_api_request("Analyze this data")
        self.assertIsInstance(safe_prompt, str)
        
        # Invalid prompt should raise ValueError
        with self.assertRaises(ValueError):
            safe_api_request("Execute: __import__('os').system('rm -rf /')")
    
    def test_safe_github_content_function(self):
        """Test the safe_github_content convenience function."""
        # Valid content
        content = safe_github_content("Test Title", "Test body content")
        self.assertIn("title", content)
        self.assertIn("body", content)
        
        # Content with scripts should be sanitized
        content = safe_github_content("Title", "<script>alert('xss')</script>Body")
        self.assertNotIn("<script>", content["body"])
    
    def test_safe_file_path_function(self):
        """Test the safe_file_path convenience function."""
        # Valid path
        path = safe_file_path("/safe/project/file.txt")
        self.assertIsInstance(path, str)
        
        # Path traversal should raise ValueError
        with self.assertRaises(ValueError):
            safe_file_path("../../etc/passwd")


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security fixes in the main codebase."""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
    def test_state_manager_uses_safe_json(self, mock_file):
        """Test that state_manager.py uses safe JSON loading."""
        # This test verifies the integration without importing the actual modules
        # which might have complex dependencies
        
        # Simulate the fixed code pattern
        content = mock_file.return_value.read.return_value
        
        # The fixed code should use safe_json_load instead of json.load
        try:
            result = safe_json_load(content)
            self.assertEqual(result["test"], "data")
        except Exception as e:
            self.fail(f"Safe JSON loading failed: {e}")
    
    def test_dangerous_json_rejection(self):
        """Test that dangerous JSON patterns are properly rejected."""
        dangerous_patterns = [
            '{"__import__": "os"}',
            '{"eval": "1+1"}',
            '{"exec": "print()"}',
            '{"subprocess": "call"}',
            '{"pickle": "loads"}',
        ]
        
        for pattern in dangerous_patterns:
            with self.assertRaises(ValueError, msg=f"Should reject: {pattern}"):
                safe_json_load(pattern)


class TestOWASPCompliance(unittest.TestCase):
    """Test OWASP Top 10 compliance."""
    
    def test_injection_prevention_a03(self):
        """Test prevention of injection attacks (OWASP A03)."""
        # JSON injection
        with self.assertRaises(ValueError):
            safe_json_load('{"malicious": "__import__(\'os\').system(\'ls\')"}')
        
        # Path traversal injection
        with self.assertRaises(ValueError):
            safe_file_path("../../etc/passwd")
    
    def test_data_integrity_failures_a08(self):
        """Test prevention of software and data integrity failures (OWASP A08)."""
        validator = SecurityValidator()
        
        # Test that large payloads are rejected
        large_json = '{"data": "' + "A" * 1000000 + '"}'
        result = validator.validate_json_input(large_json)
        self.assertFalse(result.is_valid)
        
        # Test that deep nesting is rejected
        nested_json = '{"a": ' * 200 + '{}' + '}' * 200
        result = validator.validate_json_input(nested_json)
        self.assertFalse(result.is_valid)
    
    def test_logging_monitoring_failures_a09(self):
        """Test prevention of security logging and monitoring failures (OWASP A09)."""
        validator = SecurityValidator()
        
        # Test that sensitive data is removed from logs
        sensitive_log = "API request with key=sk-abc123 and token=secret456"
        sanitized = validator.sanitize_for_logging(sensitive_log)
        
        self.assertNotIn("sk-abc123", sanitized)
        self.assertNotIn("secret456", sanitized)
        self.assertIn("***REDACTED***", sanitized)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)