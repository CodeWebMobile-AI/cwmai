#!/usr/bin/env python3
"""
Security Tests for CWMAI

Comprehensive security testing for API clients, credential management,
input validation, and secure logging practices.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.security_manager import (
    SecurityManager, SecurityLevel, SecurityViolation,
    SecureCredentialManager, InputValidator, SecureLogger
)


class TestSecureCredentialManager(unittest.TestCase):
    """Test secure credential management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.credential_manager = SecureCredentialManager()
    
    def test_valid_anthropic_key(self):
        """Test validation of valid Anthropic API key."""
        valid_key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz"
        violation = self.credential_manager.validate_api_key(valid_key, 'anthropic')
        self.assertIsNone(violation)
    
    def test_invalid_anthropic_key_format(self):
        """Test validation of invalid Anthropic API key format."""
        invalid_key = "invalid-key-format"
        violation = self.credential_manager.validate_api_key(invalid_key, 'anthropic')
        self.assertIsNotNone(violation)
        self.assertEqual(violation.type, "invalid_format")
        self.assertEqual(violation.level, SecurityLevel.MEDIUM)
    
    def test_short_anthropic_key(self):
        """Test validation of short Anthropic API key."""
        short_key = "sk-ant-short"
        violation = self.credential_manager.validate_api_key(short_key, 'anthropic')
        self.assertIsNotNone(violation)
        self.assertEqual(violation.type, "weak_credential")
    
    def test_valid_openai_key(self):
        """Test validation of valid OpenAI API key."""
        valid_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmn"
        violation = self.credential_manager.validate_api_key(valid_key, 'openai')
        self.assertIsNone(violation)
    
    def test_invalid_openai_key_format(self):
        """Test validation of invalid OpenAI API key format."""
        invalid_key = "invalid-openai-key"
        violation = self.credential_manager.validate_api_key(invalid_key, 'openai')
        self.assertIsNotNone(violation)
        self.assertEqual(violation.type, "invalid_format")
    
    def test_valid_github_token(self):
        """Test validation of valid GitHub token."""
        valid_token = "ghp_abcdefghijklmnopqrstuvwxyz1234567890ab"
        violation = self.credential_manager.validate_api_key(valid_token, 'github')
        self.assertIsNone(violation)
    
    def test_legacy_github_token(self):
        """Test validation of legacy GitHub token."""
        legacy_token = "abcdefghijklmnopqrstuvwxyz1234567890"
        violation = self.credential_manager.validate_api_key(legacy_token, 'github')
        self.assertIsNotNone(violation)
        self.assertEqual(violation.level, SecurityLevel.WARNING)
    
    def test_missing_credential(self):
        """Test validation of missing credential."""
        violation = self.credential_manager.validate_api_key("", 'anthropic')
        self.assertIsNotNone(violation)
        self.assertEqual(violation.type, "missing_credential")
        self.assertEqual(violation.level, SecurityLevel.HIGH)
    
    def test_mask_sensitive_data(self):
        """Test masking of sensitive data."""
        sensitive_text = "API key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890 and token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        masked_text = self.credential_manager.mask_sensitive_data(sensitive_text)
        
        self.assertNotIn("sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890", masked_text)
        self.assertNotIn("ghp_1234567890abcdefghijklmnopqrstuvwxyz", masked_text)
        self.assertIn("sk-a...890", masked_text)
        self.assertIn("ghp_...xyz", masked_text)
    
    @patch.dict(os.environ, {'TEST_API_KEY': 'sk-test123456789'})
    def test_get_secure_credential(self):
        """Test secure credential retrieval."""
        credential = self.credential_manager.get_secure_credential('TEST_API_KEY')
        self.assertEqual(credential, 'sk-test123456789')
    
    def test_get_missing_credential(self):
        """Test retrieval of missing credential."""
        credential = self.credential_manager.get_secure_credential('NON_EXISTENT_KEY')
        self.assertIsNone(credential)


class TestInputValidator(unittest.TestCase):
    """Test input validation and sanitization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator()
    
    def test_valid_prompt(self):
        """Test validation of valid AI prompt."""
        valid_prompt = "What is the weather like today?"
        violations = self.validator.validate_ai_prompt(valid_prompt)
        self.assertEqual(len(violations), 0)
    
    def test_empty_prompt(self):
        """Test validation of empty prompt."""
        violations = self.validator.validate_ai_prompt("")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].type, "empty_input")
    
    def test_excessive_length_prompt(self):
        """Test validation of excessively long prompt."""
        long_prompt = "A" * (self.validator.max_prompt_length + 1)
        violations = self.validator.validate_ai_prompt(long_prompt)
        
        excessive_length_violations = [v for v in violations if v.type == "excessive_length"]
        self.assertEqual(len(excessive_length_violations), 1)
    
    def test_script_injection_detection(self):
        """Test detection of script injection attempts."""
        malicious_prompt = "Tell me about <script>alert('xss')</script> security"
        violations = self.validator.validate_ai_prompt(malicious_prompt)
        
        injection_violations = [v for v in violations if v.type == "injection_attempt"]
        self.assertEqual(len(injection_violations), 1)
        self.assertEqual(injection_violations[0].level, SecurityLevel.HIGH)
    
    def test_javascript_execution_detection(self):
        """Test detection of JavaScript execution attempts."""
        malicious_prompt = "Click here: javascript:alert('hack')"
        violations = self.validator.validate_ai_prompt(malicious_prompt)
        
        injection_violations = [v for v in violations if v.type == "injection_attempt"]
        self.assertEqual(len(injection_violations), 1)
    
    def test_data_uri_detection(self):
        """Test detection of data URI attacks."""
        malicious_prompt = "Check this: data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4="
        violations = self.validator.validate_ai_prompt(malicious_prompt)
        
        injection_violations = [v for v in violations if v.type == "injection_attempt"]
        self.assertEqual(len(injection_violations), 1)
    
    def test_prompt_sanitization(self):
        """Test prompt sanitization."""
        malicious_prompt = "Tell me about <script>alert('xss')</script> security and javascript:void(0)"
        sanitized = self.validator.sanitize_prompt(malicious_prompt)
        
        self.assertNotIn("<script>", sanitized)
        self.assertNotIn("javascript:", sanitized)
        self.assertIn("security", sanitized)
    
    def test_length_truncation(self):
        """Test length truncation during sanitization."""
        long_prompt = "A" * (self.validator.max_prompt_length + 100)
        sanitized = self.validator.sanitize_prompt(long_prompt)
        
        self.assertLessEqual(len(sanitized), self.validator.max_prompt_length)
    
    def test_valid_json_data(self):
        """Test validation of valid JSON data."""
        valid_data = {"request": "test", "metadata": {"type": "ai_request"}}
        violations = self.validator.validate_json_data(valid_data)
        self.assertEqual(len(violations), 0)
    
    def test_excessive_nesting(self):
        """Test detection of excessive JSON nesting."""
        nested_data = {"level1": {"level2": {"level3": {"level4": {"level5": {
            "level6": {"level7": {"level8": {"level9": {"level10": {"level11": {}}}}}}}}}}}}
        violations = self.validator.validate_json_data(nested_data)
        
        nesting_violations = [v for v in violations if v.type == "excessive_nesting"]
        self.assertEqual(len(nesting_violations), 1)
    
    def test_sensitive_key_detection(self):
        """Test detection of sensitive keys in JSON data."""
        sensitive_data = {
            "api_key": "sk-secret123",
            "password": "secret123",
            "user_token": "token123",
            "normal_field": "safe_value"
        }
        violations = self.validator.validate_json_data(sensitive_data)
        
        sensitive_violations = [v for v in violations if v.type == "sensitive_data_exposure"]
        self.assertGreaterEqual(len(sensitive_violations), 3)  # Should detect api_key, password, user_token


class TestSecureLogger(unittest.TestCase):
    """Test secure logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.credential_manager = SecureCredentialManager()
        self.secure_logger = SecureLogger("test_logger", self.credential_manager)
    
    @patch('logging.getLogger')
    def test_info_logging(self, mock_get_logger):
        """Test secure info logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        self.secure_logger.info("Test message with api key sk-ant-12345678901234567890")
        
        # Check that the logger was called with masked data
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertNotIn("sk-ant-12345678901234567890", call_args)
        self.assertIn("sk-a...890", call_args)
    
    @patch('logging.getLogger')
    def test_warning_logging_with_data(self, mock_get_logger):
        """Test secure warning logging with data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        test_data = {"token": "ghp_1234567890abcdefghijk", "safe_field": "safe_value"}
        self.secure_logger.warning("Warning message", test_data)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertNotIn("ghp_1234567890abcdefghijk", call_args)
    
    @patch('logging.getLogger')
    def test_error_logging(self, mock_get_logger):
        """Test secure error logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        self.secure_logger.error("Error with sensitive sk-openai-12345")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertNotIn("sk-openai-12345", call_args)


class TestSecurityManager(unittest.TestCase):
    """Test overall security manager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.security_manager = SecurityManager()
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-validkey12345678901234567890abcdefghijk',
        'CLAUDE_PAT': 'ghp_validtoken1234567890abcdefghijklmn'
    })
    def test_environment_validation_success(self):
        """Test successful environment validation."""
        violations = self.security_manager.validate_environment()
        
        # Should have no critical violations
        critical_violations = [v for v in violations if v.level == SecurityLevel.CRITICAL]
        self.assertEqual(len(critical_violations), 0)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_environment_validation_missing_credentials(self):
        """Test environment validation with missing credentials."""
        violations = self.security_manager.validate_environment()
        
        # Should have critical violations for missing GitHub token and AI providers
        critical_violations = [v for v in violations if v.level == SecurityLevel.CRITICAL]
        high_violations = [v for v in violations if v.level == SecurityLevel.HIGH]
        
        self.assertGreater(len(critical_violations) + len(high_violations), 0)
    
    def test_ai_request_validation_valid(self):
        """Test AI request validation with valid input."""
        valid_prompt = "What is machine learning?"
        violations = self.security_manager.validate_ai_request(valid_prompt)
        self.assertEqual(len(violations), 0)
    
    def test_ai_request_validation_malicious(self):
        """Test AI request validation with malicious input."""
        malicious_prompt = "Tell me about <script>alert('xss')</script>"
        violations = self.security_manager.validate_ai_request(malicious_prompt)
        
        high_violations = [v for v in violations if v.level == SecurityLevel.HIGH]
        self.assertGreater(len(high_violations), 0)
    
    def test_ai_request_sanitization(self):
        """Test AI request sanitization."""
        malicious_prompt = "Normal question with <script>dangerous content</script>"
        sanitized = self.security_manager.sanitize_ai_request(malicious_prompt)
        
        self.assertNotIn("<script>", sanitized)
        self.assertIn("Normal question", sanitized)
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Generate some violations first
        self.security_manager.validate_environment()
        self.security_manager.validate_ai_request("Test <script>alert('test')</script>")
        
        report = self.security_manager.get_security_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('total_violations', report)
        self.assertIn('violations_by_level', report)
        self.assertIn('security_score', report)
        self.assertIn('recommendations', report)
        
        self.assertIsInstance(report['security_score'], int)
        self.assertGreaterEqual(report['security_score'], 0)
        self.assertLessEqual(report['security_score'], 100)
    
    def test_security_score_calculation(self):
        """Test security score calculation."""
        # Start with clean state
        clean_manager = SecurityManager()
        
        # Should start with perfect score
        clean_report = clean_manager.get_security_report()
        self.assertEqual(clean_report['security_score'], 100)
        
        # Add violations and check score decreases
        clean_manager.validate_ai_request("<script>alert('test')</script>")
        violation_report = clean_manager.get_security_report()
        self.assertLess(violation_report['security_score'], 100)


class TestSecurityIntegration(unittest.TestCase):
    """Test security integration with other components."""
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-validkey12345678901234567890abcdefghijk'
    })
    def test_http_client_security_integration(self):
        """Test HTTP client integration with security manager."""
        try:
            from scripts.http_ai_client import HTTPAIClient
            
            client = HTTPAIClient()
            
            # Should have security manager
            self.assertIsNotNone(client.security_manager)
            
            # Should validate credentials on initialization
            # (No exceptions should be raised for valid credentials)
            
        except ImportError:
            self.skipTest("HTTPAIClient not available for testing")
    
    def test_security_violation_data_structure(self):
        """Test security violation data structure."""
        violation = SecurityViolation(
            level=SecurityLevel.HIGH,
            type="test_violation",
            message="Test message",
            data="test_data",
            remediation="Test remediation"
        )
        
        self.assertEqual(violation.level, SecurityLevel.HIGH)
        self.assertEqual(violation.type, "test_violation")
        self.assertEqual(violation.message, "Test message")
        self.assertEqual(violation.data, "test_data")
        self.assertEqual(violation.remediation, "Test remediation")


def run_security_tests():
    """Run all security tests."""
    print("🔒 Running CWMAI Security Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSecureCredentialManager,
        TestInputValidator,
        TestSecureLogger,
        TestSecurityManager,
        TestSecurityIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🔒 Security Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Test Failures:")
        for test, failure in result.failures:
            print(f"  • {test}: {failure.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n🚨 Test Errors:")
        for test, error in result.errors:
            print(f"  • {test}: {error.split('\\n')[-2]}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)