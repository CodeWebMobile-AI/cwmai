#!/bin/bash

# CWMAI Development Environment Setup Script
# This script runs every time the Codespace starts

set -e

echo "🔧 Setting up CWMAI development environment..."

# Function to check if environment variable is set
check_env_var() {
    local var_name=$1
    local var_value=${!var_name}
    
    if [[ -n "$var_value" ]]; then
        echo "✅ $var_name: Available"
        return 0
    else
        echo "❌ $var_name: Missing"
        return 1
    fi
}

# Function to validate GitHub CLI authentication
validate_github_auth() {
    echo "🔍 Validating GitHub CLI authentication..."
    
    if command -v gh >/dev/null 2>&1; then
        if gh auth status >/dev/null 2>&1; then
            echo "✅ GitHub CLI: Authenticated"
            return 0
        else
            echo "⚠️  GitHub CLI: Not authenticated, attempting auto-login..."
            # Try to authenticate using available token
            if [[ -n "$CLAUDE_PAT" ]]; then
                echo "$CLAUDE_PAT" | gh auth login --with-token
                echo "✅ GitHub CLI: Authenticated with CLAUDE_PAT"
            elif [[ -n "$GITHUB_TOKEN" ]]; then
                echo "$GITHUB_TOKEN" | gh auth login --with-token
                echo "✅ GitHub CLI: Authenticated with GITHUB_TOKEN"
            else
                echo "❌ GitHub CLI: No token available for authentication"
                return 1
            fi
        fi
    else
        echo "❌ GitHub CLI: Not installed"
        return 1
    fi
}

# Function to fetch secrets from GitHub if missing locally
fetch_missing_secrets() {
    echo "🔐 Checking for missing secrets..."
    
    local missing_secrets=()
    local required_secrets=("ANTHROPIC_API_KEY" "CLAUDE_PAT")
    local optional_secrets=("OPENAI_API_KEY" "GOOGLE_API_KEY" "GEMINI_API_KEY" "DEEPSEEK_API_KEY")
    
    # Check required secrets
    for secret in "${required_secrets[@]}"; do
        if [[ -z "${!secret}" ]]; then
            missing_secrets+=("$secret")
        fi
    done
    
    # Check optional secrets
    for secret in "${optional_secrets[@]}"; do
        if [[ -z "${!secret}" ]]; then
            echo "⚠️  Optional secret $secret is missing"
        fi
    done
    
    if [[ ${#missing_secrets[@]} -gt 0 ]]; then
        echo "❌ Missing required secrets: ${missing_secrets[*]}"
        echo "📝 Please ensure these secrets are configured in your GitHub repository:"
        echo "   Settings → Secrets and variables → Codespaces"
        
        # Try to help with GitHub CLI if authenticated
        if gh auth status >/dev/null 2>&1; then
            echo "💡 You can also run: ./scripts/fetch_secrets.sh"
        fi
        
        return 1
    else
        echo "✅ All required secrets are available"
        return 0
    fi
}

# Function to set up Python environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install requirements if they exist
    if [[ -f "requirements.txt" ]]; then
        echo "📦 Installing Python dependencies..."
        pip install -r requirements.txt
        echo "✅ Python dependencies installed"
    fi
    
    # Run environment validation
    if [[ -f "scripts/environment_validator.py" ]]; then
        echo "🔍 Running environment validation..."
        python scripts/environment_validator.py
    fi
}

# Main setup sequence
main() {
    echo "🚀 Starting CWMAI development environment setup..."
    
    # Check environment variables
    echo ""
    echo "🔐 Checking environment variables..."
    check_env_var "ANTHROPIC_API_KEY"
    check_env_var "CLAUDE_PAT"
    check_env_var "GITHUB_TOKEN"
    check_env_var "OPENAI_API_KEY"
    check_env_var "GOOGLE_API_KEY"
    check_env_var "GEMINI_API_KEY"
    check_env_var "DEEPSEEK_API_KEY"
    
    echo ""
    validate_github_auth
    
    echo ""
    fetch_missing_secrets
    
    echo ""
    setup_python_env
    
    echo ""
    echo "✨ CWMAI development environment setup complete!"
    echo "🔧 You can now run your AI scripts with access to all configured secrets."
    echo ""
    echo "Quick start commands:"
    echo "  python run_dynamic_ai.py              # Run the dynamic AI system"
    echo "  python scripts/environment_validator.py # Validate environment"
    echo "  ./scripts/fetch_secrets.sh             # Fetch secrets via GitHub CLI"
    echo ""
}

# Run main function
main "$@"