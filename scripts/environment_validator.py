#!/usr/bin/env python3
"""
Environment Validator for CWMAI Development
Validates all required API keys and environment variables are properly configured.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ValidationStatus(Enum):
    AVAILABLE = "✅"
    MISSING = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"

@dataclass
class SecretConfig:
    """Configuration for a secret/environment variable."""
    name: str
    description: str
    required: bool = True
    validation_func: Optional[callable] = None
    documentation_url: Optional[str] = None

class EnvironmentValidator:
    """Validates environment configuration for CWMAI development."""
    
    def __init__(self):
        self.secrets_config = [
            SecretConfig(
                name="ANTHROPIC_API_KEY",
                description="Anthropic API key for Claude AI access",
                required=True,
                validation_func=self._validate_anthropic_key,
                documentation_url="https://console.anthropic.com/"
            ),
            SecretConfig(
                name="CLAUDE_PAT",
                description="GitHub Personal Access Token with repo permissions",
                required=True,
                validation_func=self._validate_github_token,
                documentation_url="https://github.com/settings/tokens"
            ),
            SecretConfig(
                name="GITHUB_TOKEN",
                description="Alternative GitHub token (fallback for CLAUDE_PAT)",
                required=False,
                validation_func=self._validate_github_token,
                documentation_url="https://github.com/settings/tokens"
            ),
            SecretConfig(
                name="OPENAI_API_KEY",
                description="OpenAI API key for GPT models",
                required=False,
                validation_func=self._validate_openai_key,
                documentation_url="https://platform.openai.com/api-keys"
            ),
            SecretConfig(
                name="GOOGLE_API_KEY",
                description="Google API key for Gemini models",
                required=False,
                validation_func=self._validate_google_key,
                documentation_url="https://aistudio.google.com/app/apikey"
            ),
            SecretConfig(
                name="GEMINI_API_KEY",
                description="Google Gemini API key (alternative to GOOGLE_API_KEY)",
                required=False,
                validation_func=self._validate_google_key,
                documentation_url="https://aistudio.google.com/app/apikey"
            ),
            SecretConfig(
                name="DEEPSEEK_API_KEY",
                description="DeepSeek API key",
                required=False,
                validation_func=self._validate_deepseek_key,
                documentation_url="https://platform.deepseek.com/"
            ),
        ]
        
        self.validation_results: List[Tuple[SecretConfig, ValidationStatus, str]] = []
    
    def _validate_anthropic_key(self, key: str) -> Tuple[ValidationStatus, str]:
        """Validate Anthropic API key format."""
        if not key:
            return ValidationStatus.MISSING, "Not set"
        
        if not key.startswith('sk-ant-'):
            return ValidationStatus.WARNING, "Invalid format (should start with 'sk-ant-')"
        
        if len(key) < 20:
            return ValidationStatus.WARNING, "Key appears too short"
        
        return ValidationStatus.AVAILABLE, "Format looks correct"
    
    def _validate_github_token(self, token: str) -> Tuple[ValidationStatus, str]:
        """Validate GitHub token format."""
        if not token:
            return ValidationStatus.MISSING, "Not set"
        
        # GitHub Personal Access Tokens start with 'ghp_' (classic) or 'github_pat_' (fine-grained)
        if token.startswith(('ghp_', 'github_pat_')):
            return ValidationStatus.AVAILABLE, "Format looks correct"
        
        # Legacy tokens might not have prefix
        if len(token) >= 20:
            return ValidationStatus.WARNING, "Legacy token format (consider updating)"
        
        return ValidationStatus.WARNING, "Unexpected token format"
    
    def _validate_openai_key(self, key: str) -> Tuple[ValidationStatus, str]:
        """Validate OpenAI API key format."""
        if not key:
            return ValidationStatus.MISSING, "Not set"
        
        if not key.startswith('sk-'):
            return ValidationStatus.WARNING, "Invalid format (should start with 'sk-')"
        
        if len(key) < 20:
            return ValidationStatus.WARNING, "Key appears too short"
        
        return ValidationStatus.AVAILABLE, "Format looks correct"
    
    def _validate_google_key(self, key: str) -> Tuple[ValidationStatus, str]:
        """Validate Google API key format."""
        if not key:
            return ValidationStatus.MISSING, "Not set"
        
        # Google API keys are typically 39 characters
        if len(key) < 20:
            return ValidationStatus.WARNING, "Key appears too short"
        
        return ValidationStatus.AVAILABLE, "Format looks correct"
    
    def _validate_deepseek_key(self, key: str) -> Tuple[ValidationStatus, str]:
        """Validate DeepSeek API key format."""
        if not key:
            return ValidationStatus.MISSING, "Not set"
        
        if len(key) < 20:
            return ValidationStatus.WARNING, "Key appears too short"
        
        return ValidationStatus.AVAILABLE, "Format looks correct"
    
    def validate_all(self) -> bool:
        """Validate all environment variables."""
        print("🔍 CWMAI Environment Validation")
        print("=" * 50)
        
        all_valid = True
        required_missing = []
        
        for config in self.secrets_config:
            value = os.getenv(config.name)
            
            if config.validation_func:
                status, message = config.validation_func(value)
            else:
                if value:
                    status, message = ValidationStatus.AVAILABLE, "Set"
                else:
                    status, message = ValidationStatus.MISSING, "Not set"
            
            self.validation_results.append((config, status, message))
            
            # Print result
            print(f"{status.value} {config.name}")
            print(f"   Description: {config.description}")
            print(f"   Status: {message}")
            
            if config.documentation_url:
                print(f"   Documentation: {config.documentation_url}")
            
            print()
            
            # Track issues
            if config.required and status == ValidationStatus.MISSING:
                required_missing.append(config.name)
                all_valid = False
        
        # Summary
        print("📋 Validation Summary")
        print("-" * 30)
        
        if required_missing:
            print(f"❌ Missing required secrets: {', '.join(required_missing)}")
            all_valid = False
        
        # Check GitHub token fallback
        claude_pat = os.getenv('CLAUDE_PAT')
        github_token = os.getenv('GITHUB_TOKEN')
        if not claude_pat and not github_token:
            print("❌ No GitHub token available (need either CLAUDE_PAT or GITHUB_TOKEN)")
            all_valid = False
        elif claude_pat:
            print("✅ GitHub access via CLAUDE_PAT")
        else:
            print("✅ GitHub access via GITHUB_TOKEN")
        
        # Check AI providers
        ai_providers = []
        if os.getenv('ANTHROPIC_API_KEY'):
            ai_providers.append('Anthropic (Claude)')
        if os.getenv('OPENAI_API_KEY'):
            ai_providers.append('OpenAI (GPT)')
        if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
            ai_providers.append('Google (Gemini)')
        if os.getenv('DEEPSEEK_API_KEY'):
            ai_providers.append('DeepSeek')
        
        if ai_providers:
            print(f"🤖 Available AI providers: {', '.join(ai_providers)}")
        else:
            print("⚠️  No AI providers configured")
        
        print()
        
        if all_valid:
            print("✨ Environment validation passed! All required secrets are configured.")
            return True
        else:
            print("❌ Environment validation failed. Please configure missing secrets.")
            print()
            print("💡 To fix:")
            print("   1. Add secrets to GitHub repository: Settings → Secrets and variables → Codespaces")
            print("   2. Or use GitHub CLI: gh secret set SECRET_NAME")
            print("   3. Or run: ./scripts/fetch_secrets.sh")
            return False
    
    def print_setup_help(self):
        """Print help for setting up the environment."""
        print("🔧 Environment Setup Help")
        print("=" * 50)
        print()
        print("To configure secrets for Codespaces:")
        print("1. Go to your GitHub repository")
        print("2. Settings → Secrets and variables → Codespaces")
        print("3. Add the following secrets:")
        print()
        
        for config in self.secrets_config:
            status_marker = "🔴" if config.required else "🟡"
            print(f"{status_marker} {config.name}")
            print(f"   {config.description}")
            if config.documentation_url:
                print(f"   Get key: {config.documentation_url}")
            print()
        
        print("🔴 = Required")
        print("🟡 = Optional")
        print()
        print("Alternative methods:")
        print("• Use GitHub CLI: gh secret set SECRET_NAME")
        print("• Run fetch script: ./scripts/fetch_secrets.sh")
        print("• Set environment variables manually in terminal")

def validate_environment() -> bool:
    """Main validation function that can be imported."""
    validator = EnvironmentValidator()
    return validator.validate_all()

def main():
    """Main entry point for command line usage."""
    validator = EnvironmentValidator()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        validator.print_setup_help()
        sys.exit(0)
    
    success = validator.validate_all()
    
    if not success:
        print()
        validator.print_setup_help()
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()