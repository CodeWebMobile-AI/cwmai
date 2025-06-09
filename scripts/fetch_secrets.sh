#!/bin/bash

# CWMAI Secret Fetching Script
# Fetches secrets from GitHub repository for local development

set -e

REPO_NAME="CodeWebMobile-AI/cwmai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if GitHub CLI is available and authenticated
check_github_cli() {
    print_status $BLUE "🔍 Checking GitHub CLI..."
    
    if ! command -v gh >/dev/null 2>&1; then
        print_status $RED "❌ GitHub CLI (gh) is not installed"
        print_status $YELLOW "💡 Install it from: https://cli.github.com/"
        return 1
    fi
    
    if ! gh auth status >/dev/null 2>&1; then
        print_status $YELLOW "⚠️  GitHub CLI is not authenticated"
        print_status $BLUE "🔧 Run: gh auth login"
        return 1
    fi
    
    print_status $GREEN "✅ GitHub CLI is ready"
    return 0
}

# Function to fetch a single secret
fetch_secret() {
    local secret_name=$1
    local required=$2
    
    print_status $BLUE "🔐 Fetching $secret_name..."
    
    # Try to get the secret value
    local secret_value
    if secret_value=$(gh secret list --repo "$REPO_NAME" --json name,visibility | jq -r ".[] | select(.name==\"$secret_name\") | .name" 2>/dev/null); then
        if [[ "$secret_value" == "$secret_name" ]]; then
            print_status $GREEN "✅ $secret_name: Found in repository"
            
            # Note: GitHub CLI cannot retrieve secret values for security reasons
            # We can only confirm the secret exists
            print_status $YELLOW "⚠️  Cannot retrieve secret value via CLI (GitHub security restriction)"
            print_status $BLUE "💡 The secret exists and will be available in Codespaces automatically"
            return 0
        fi
    fi
    
    if [[ "$required" == "true" ]]; then
        print_status $RED "❌ $secret_name: Not found (REQUIRED)"
        return 1
    else
        print_status $YELLOW "⚠️  $secret_name: Not found (optional)"
        return 0
    fi
}

# Function to set up local environment file
create_env_template() {
    local env_file="$ROOT_DIR/.env.local"
    
    print_status $BLUE "📝 Creating environment template..."
    
    cat > "$env_file" << 'EOF'
# CWMAI Local Development Environment
# Copy this file to .env and fill in your API keys
# This file is ignored by git for security

# Required secrets
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_PAT=your_github_personal_access_token_here

# Optional AI provider keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# GitHub repository (usually auto-detected)
GITHUB_REPOSITORY=CodeWebMobile-AI/cwmai

# To use this file, run:
# source .env.local
# or
# export $(cat .env.local | grep -v '^#' | xargs)
EOF
    
    print_status $GREEN "✅ Created $env_file"
    print_status $YELLOW "💡 Edit this file with your actual API keys for local testing"
}

# Function to validate current environment
validate_current_env() {
    print_status $BLUE "🔍 Validating current environment..."
    
    if [[ -f "$SCRIPT_DIR/environment_validator.py" ]]; then
        python "$SCRIPT_DIR/environment_validator.py"
    else
        print_status $YELLOW "⚠️  Environment validator not found"
    fi
}

# Function to provide setup instructions
show_setup_instructions() {
    print_status $BLUE "📋 Setup Instructions"
    echo ""
    echo "For Codespaces (Recommended):"
    echo "1. Go to your GitHub repository: https://github.com/$REPO_NAME"
    echo "2. Settings → Secrets and variables → Codespaces"
    echo "3. Add the following secrets:"
    echo "   • ANTHROPIC_API_KEY (required)"
    echo "   • CLAUDE_PAT (required)"
    echo "   • OPENAI_API_KEY (optional)"
    echo "   • GOOGLE_API_KEY (optional)"
    echo "   • GEMINI_API_KEY (optional)"
    echo "   • DEEPSEEK_API_KEY (optional)"
    echo ""
    echo "For Local Development:"
    echo "1. Edit .env.local with your API keys"
    echo "2. Run: source .env.local"
    echo "3. Or: export \$(cat .env.local | grep -v '^#' | xargs)"
    echo ""
    echo "To add secrets to GitHub repository:"
    echo "• gh secret set ANTHROPIC_API_KEY"
    echo "• gh secret set CLAUDE_PAT"
    echo "• etc."
}

# Main function
main() {
    print_status $GREEN "🚀 CWMAI Secret Management Tool"
    echo ""
    
    # Parse command line arguments
    case "${1:-fetch}" in
        "fetch")
            print_status $BLUE "🔐 Fetching secrets from GitHub..."
            echo ""
            
            if ! check_github_cli; then
                exit 1
            fi
            
            # List of secrets to check
            declare -A secrets=(
                ["ANTHROPIC_API_KEY"]="true"
                ["CLAUDE_PAT"]="true"
                ["OPENAI_API_KEY"]="false"
                ["GOOGLE_API_KEY"]="false"
                ["GEMINI_API_KEY"]="false"
                ["DEEPSEEK_API_KEY"]="false"
            )
            
            local all_found=true
            for secret in "${!secrets[@]}"; do
                if ! fetch_secret "$secret" "${secrets[$secret]}"; then
                    if [[ "${secrets[$secret]}" == "true" ]]; then
                        all_found=false
                    fi
                fi
            done
            
            echo ""
            if [[ "$all_found" == "true" ]]; then
                print_status $GREEN "✅ All required secrets are configured in GitHub"
                print_status $BLUE "💡 Secrets will be automatically available in Codespaces"
            else
                print_status $RED "❌ Some required secrets are missing"
                show_setup_instructions
                exit 1
            fi
            ;;
            
        "template")
            create_env_template
            ;;
            
        "validate")
            validate_current_env
            ;;
            
        "help"|"--help"|"-h")
            echo "CWMAI Secret Management Tool"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  fetch     - Check which secrets exist in GitHub repository (default)"
            echo "  template  - Create .env.local template file"
            echo "  validate  - Validate current environment"
            echo "  help      - Show this help message"
            echo ""
            show_setup_instructions
            ;;
            
        *)
            print_status $RED "❌ Unknown command: $1"
            print_status $BLUE "💡 Run: $0 help"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"