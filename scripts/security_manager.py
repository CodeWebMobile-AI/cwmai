#!/usr/bin/env python3
"""
Security Manager Module

Provides comprehensive security features for the CWMAI AI automation system.
Implements OWASP-compliant security practices for API clients and data handling.
"""

import os
import re
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityViolation:
    """Represents a security violation found during validation."""
    level: SecurityLevel
    type: str
    message: str
    data: Optional[str] = None
    remediation: Optional[str] = None


class SecureCredentialManager:
    """Manages API credentials and sensitive data securely."""
    
    def __init__(self):
        """Initialize the credential manager."""
        self.logger = logging.getLogger(f"{__name__}.SecureCredentialManager")
        self._sensitive_patterns = [
            r'sk-[a-zA-Z0-9\-_]{40,}',  # Anthropic/OpenAI API keys
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
            r'github_pat_[a-zA-Z0-9_]{82}',  # GitHub fine-grained tokens
            r'AIza[a-zA-Z0-9\-_]{35}',  # Google API keys
            r'[a-zA-Z0-9\-_]{32,}',  # Generic long tokens
        ]
    
    def validate_api_key(self, key: str, provider: str) -> Optional[SecurityViolation]:
        """Validate API key format and security.
        
        Args:
            key: The API key to validate
            provider: Provider name (anthropic, openai, google, etc.)
        
        Returns:
            SecurityViolation if issues found, None if valid
        """
        if not key:
            return SecurityViolation(
                level=SecurityLevel.HIGH,
                type="missing_credential",
                message=f"Missing API key for {provider}",
                remediation="Set the required environment variable"
            )
        
        if len(key) < 20:
            return SecurityViolation(
                level=SecurityLevel.HIGH,
                type="weak_credential",
                message=f"API key for {provider} appears too short",
                remediation="Verify the API key is complete and valid"
            )
        
        # Provider-specific validations
        validation_rules = {
            'anthropic': {'prefix': 'sk-ant-', 'min_length': 50},
            'openai': {'prefix': 'sk-', 'min_length': 40},
            'github': {'prefixes': ['ghp_', 'github_pat_'], 'min_length': 30},
            'google': {'min_length': 30},
            'deepseek': {'min_length': 30}
        }
        
        if provider in validation_rules:
            rules = validation_rules[provider]
            
            if 'prefix' in rules and not key.startswith(rules['prefix']):
                return SecurityViolation(
                    level=SecurityLevel.MEDIUM,
                    type="invalid_format",
                    message=f"API key for {provider} has invalid format",
                    remediation=f"Key should start with '{rules['prefix']}'"
                )
            
            if 'prefixes' in rules and not any(key.startswith(p) for p in rules['prefixes']):
                return SecurityViolation(
                    level=SecurityLevel.MEDIUM,
                    type="invalid_format",
                    message=f"API key for {provider} has invalid format",
                    remediation=f"Key should start with one of: {rules['prefixes']}"
                )
            
            if len(key) < rules['min_length']:
                return SecurityViolation(
                    level=SecurityLevel.MEDIUM,
                    type="weak_credential",
                    message=f"API key for {provider} appears too short",
                    remediation="Verify the API key is complete"
                )
        
        return None
    
    def get_secure_credential(self, name: str) -> Optional[str]:
        """Securely retrieve a credential from environment variables.
        
        Args:
            name: Environment variable name
        
        Returns:
            The credential value or None if not found
        """
        value = os.getenv(name)
        if value:
            self.logger.debug(f"Retrieved credential {name} (length: {len(value)})")
        else:
            self.logger.warning(f"Credential {name} not found in environment")
        return value
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text for safe logging.
        
        Args:
            text: Text that may contain sensitive data
        
        Returns:
            Text with sensitive data masked
        """
        masked_text = text
        
        for pattern in self._sensitive_patterns:
            masked_text = re.sub(pattern, lambda m: self._mask_string(m.group()), masked_text)
        
        return masked_text
    
    def _mask_string(self, value: str) -> str:
        """Mask a string showing only first 4 and last 4 characters."""
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"


class InputValidator:
    """Validates and sanitizes inputs for AI interactions."""
    
    def __init__(self):
        """Initialize the input validator."""
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
        self.max_prompt_length = 100000  # Reasonable limit for AI prompts
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript execution
            r'data:.*base64',  # Data URI attacks
            r'vbscript:',  # VBScript execution
        ]
    
    def validate_ai_prompt(self, prompt: str) -> List[SecurityViolation]:
        """Validate AI prompt for security issues.
        
        Args:
            prompt: The AI prompt to validate
        
        Returns:
            List of security violations found
        """
        violations = []
        
        if not prompt:
            violations.append(SecurityViolation(
                level=SecurityLevel.LOW,
                type="empty_input",
                message="Empty prompt provided",
                remediation="Provide a non-empty prompt"
            ))
            return violations
        
        if len(prompt) > self.max_prompt_length:
            violations.append(SecurityViolation(
                level=SecurityLevel.MEDIUM,
                type="excessive_length",
                message=f"Prompt exceeds maximum length ({len(prompt)} > {self.max_prompt_length})",
                remediation="Reduce prompt length to prevent resource exhaustion"
            ))
        
        # Check for injection patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
                violations.append(SecurityViolation(
                    level=SecurityLevel.HIGH,
                    type="injection_attempt",
                    message=f"Potentially dangerous pattern detected: {pattern}",
                    remediation="Remove or escape dangerous content"
                ))
        
        return violations
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize AI prompt by removing dangerous content.
        
        Args:
            prompt: The prompt to sanitize
        
        Returns:
            Sanitized prompt
        """
        sanitized = prompt
        
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Limit length
        if len(sanitized) > self.max_prompt_length:
            sanitized = sanitized[:self.max_prompt_length]
            self.logger.warning(f"Prompt truncated to {self.max_prompt_length} characters")
        
        return sanitized.strip()
    
    def validate_json_data(self, data: Dict[str, Any]) -> List[SecurityViolation]:
        """Validate JSON data for security issues.
        
        Args:
            data: Dictionary to validate
        
        Returns:
            List of security violations found
        """
        violations = []
        
        # Check for excessive nesting
        max_depth = 10
        if self._get_dict_depth(data) > max_depth:
            violations.append(SecurityViolation(
                level=SecurityLevel.MEDIUM,
                type="excessive_nesting",
                message=f"JSON data exceeds maximum nesting depth ({max_depth})",
                remediation="Reduce data complexity"
            ))
        
        # Check for sensitive data leakage
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        for key in self._get_all_keys(data):
            if any(sensitive_word in key.lower() for sensitive_word in sensitive_keys):
                violations.append(SecurityViolation(
                    level=SecurityLevel.HIGH,
                    type="sensitive_data_exposure",
                    message=f"Potentially sensitive key found: {key}",
                    remediation="Remove or encrypt sensitive data"
                ))
        
        return violations
    
    def _get_dict_depth(self, d: Dict[str, Any], depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(d, dict):
            return depth
        
        max_depth = depth
        for value in d.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, self._get_dict_depth(value, depth + 1))
        
        return max_depth
    
    def _get_all_keys(self, d: Dict[str, Any]) -> List[str]:
        """Get all keys from nested dictionary."""
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value))
        return keys


class SecureLogger:
    """Provides secure logging that prevents sensitive data exposure."""
    
    def __init__(self, name: str, credential_manager: SecureCredentialManager):
        """Initialize secure logger.
        
        Args:
            name: Logger name
            credential_manager: Credential manager for data masking
        """
        self.logger = logging.getLogger(name)
        self.credential_manager = credential_manager
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message with secure data handling."""
        safe_message = self.credential_manager.mask_sensitive_data(str(message))
        if data:
            safe_data = self._sanitize_log_data(data)
            self.logger.info(f"{safe_message} | Data: {safe_data}")
        else:
            self.logger.info(safe_message)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message with secure data handling."""
        safe_message = self.credential_manager.mask_sensitive_data(str(message))
        if data:
            safe_data = self._sanitize_log_data(data)
            self.logger.warning(f"{safe_message} | Data: {safe_data}")
        else:
            self.logger.warning(safe_message)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log error message with secure data handling."""
        safe_message = self.credential_manager.mask_sensitive_data(str(message))
        if data:
            safe_data = self._sanitize_log_data(data)
            self.logger.error(f"{safe_message} | Data: {safe_data}")
        else:
            self.logger.error(safe_message)
    
    def _sanitize_log_data(self, data: Dict[str, Any]) -> str:
        """Sanitize data for logging."""
        try:
            safe_str = json.dumps(data, default=str, indent=None)
            return self.credential_manager.mask_sensitive_data(safe_str)
        except Exception:
            return self.credential_manager.mask_sensitive_data(str(data))


class SecurityManager:
    """Main security manager that orchestrates all security features."""
    
    def __init__(self):
        """Initialize the security manager."""
        self.credential_manager = SecureCredentialManager()
        self.input_validator = InputValidator()
        self.logger = SecureLogger(f"{__name__}.SecurityManager", self.credential_manager)
        self._violations_log: List[SecurityViolation] = []
    
    def validate_environment(self) -> List[SecurityViolation]:
        """Validate the entire environment for security issues.
        
        Returns:
            List of all security violations found
        """
        violations = []
        
        # Validate API keys
        api_keys = {
            'ANTHROPIC_API_KEY': 'anthropic',
            'OPENAI_API_KEY': 'openai',
            'CLAUDE_PAT': 'github',
            'GITHUB_TOKEN': 'github',
            'GOOGLE_API_KEY': 'google',
            'GEMINI_API_KEY': 'google',
            'DEEPSEEK_API_KEY': 'deepseek'
        }
        
        for env_var, provider in api_keys.items():
            key = self.credential_manager.get_secure_credential(env_var)
            if key:
                violation = self.credential_manager.validate_api_key(key, provider)
                if violation:
                    violations.append(violation)
        
        # Ensure at least one GitHub token is available
        github_token = (self.credential_manager.get_secure_credential('CLAUDE_PAT') or 
                       self.credential_manager.get_secure_credential('GITHUB_TOKEN'))
        if not github_token:
            violations.append(SecurityViolation(
                level=SecurityLevel.CRITICAL,
                type="missing_required_credential",
                message="No GitHub token available (CLAUDE_PAT or GITHUB_TOKEN required)",
                remediation="Set either CLAUDE_PAT or GITHUB_TOKEN environment variable"
            ))
        
        # Ensure at least one AI provider is available
        ai_providers = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'DEEPSEEK_API_KEY']
        if not any(self.credential_manager.get_secure_credential(key) for key in ai_providers):
            violations.append(SecurityViolation(
                level=SecurityLevel.HIGH,
                type="missing_ai_provider",
                message="No AI provider credentials available",
                remediation="Set at least one AI provider API key"
            ))
        
        self._violations_log.extend(violations)
        return violations
    
    def validate_ai_request(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> List[SecurityViolation]:
        """Validate an AI request for security issues.
        
        Args:
            prompt: The AI prompt
            metadata: Optional request metadata
        
        Returns:
            List of security violations found
        """
        violations = []
        
        # Validate prompt
        violations.extend(self.input_validator.validate_ai_prompt(prompt))
        
        # Validate metadata if provided
        if metadata:
            violations.extend(self.input_validator.validate_json_data(metadata))
        
        self._violations_log.extend(violations)
        return violations
    
    def sanitize_ai_request(self, prompt: str) -> str:
        """Sanitize an AI request.
        
        Args:
            prompt: The AI prompt to sanitize
        
        Returns:
            Sanitized prompt
        """
        return self.input_validator.sanitize_prompt(prompt)
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report.
        
        Returns:
            Security report with violations and recommendations
        """
        violations_by_level = {}
        for violation in self._violations_log:
            level = violation.level.value
            if level not in violations_by_level:
                violations_by_level[level] = []
            violations_by_level[level].append({
                'type': violation.type,
                'message': violation.message,
                'remediation': violation.remediation
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_violations': len(self._violations_log),
            'violations_by_level': violations_by_level,
            'security_score': self._calculate_security_score(),
            'recommendations': self._get_security_recommendations()
        }
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        if not self._violations_log:
            return 100
        
        # Weight violations by severity
        weights = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 3,
            SecurityLevel.HIGH: 7,
            SecurityLevel.CRITICAL: 15
        }
        
        total_penalty = sum(weights.get(v.level, 1) for v in self._violations_log)
        
        # Calculate score (100 is perfect, decreases with violations)
        score = max(0, 100 - total_penalty)
        return score
    
    def _get_security_recommendations(self) -> List[str]:
        """Get security recommendations based on violations."""
        recommendations = []
        
        violation_types = set(v.type for v in self._violations_log)
        
        if 'missing_credential' in violation_types:
            recommendations.append("Configure all required API credentials")
        
        if 'weak_credential' in violation_types:
            recommendations.append("Verify all API keys are complete and valid")
        
        if 'injection_attempt' in violation_types:
            recommendations.append("Review and sanitize user inputs")
        
        if 'excessive_length' in violation_types:
            recommendations.append("Implement input length limits")
        
        if 'sensitive_data_exposure' in violation_types:
            recommendations.append("Remove sensitive data from logs and outputs")
        
        if not recommendations:
            recommendations.append("Continue monitoring for security threats")
        
        return recommendations


# Global security manager instance
security_manager = SecurityManager()


def validate_environment_security() -> bool:
    """Quick function to validate environment security.
    
    Returns:
        True if no critical violations found
    """
    violations = security_manager.validate_environment()
    critical_violations = [v for v in violations if v.level == SecurityLevel.CRITICAL]
    
    if critical_violations:
        print("❌ Critical security violations found:")
        for violation in critical_violations:
            print(f"   • {violation.message}")
            if violation.remediation:
                print(f"     → {violation.remediation}")
        return False
    
    return True


if __name__ == "__main__":
    # Run security validation
    print("🔒 CWMAI Security Validation")
    print("=" * 50)
    
    violations = security_manager.validate_environment()
    
    if violations:
        print(f"Found {len(violations)} security issues:")
        for violation in violations:
            level_emoji = {
                SecurityLevel.LOW: "ℹ️",
                SecurityLevel.MEDIUM: "⚠️",
                SecurityLevel.HIGH: "❌",
                SecurityLevel.CRITICAL: "🚨"
            }
            print(f"{level_emoji.get(violation.level, '?')} {violation.message}")
            if violation.remediation:
                print(f"   → {violation.remediation}")
        print()
    
    report = security_manager.get_security_report()
    print(f"Security Score: {report['security_score']}/100")
    
    if report['recommendations']:
        print("\n📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"• {rec}")