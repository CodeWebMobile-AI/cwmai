# CWMAI Development Environment

This directory contains the Codespaces configuration for the CWMAI (CodeWebMobile AI) project, providing a consistent development environment with automatic secret management.

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended)
1. Open this repository in GitHub Codespaces
2. Secrets will be automatically synchronized from your repository settings
3. The environment will be ready to use immediately

### Option 2: Local Development Container
1. Install VS Code and the Dev Containers extension
2. Open this repository in VS Code
3. Run "Dev Containers: Reopen in Container" from the command palette
4. Configure secrets manually (see below)

## 🔐 Secret Management

### Required Secrets
- `ANTHROPIC_API_KEY` - For Claude AI access
- `CLAUDE_PAT` - GitHub Personal Access Token with repo permissions

### Optional Secrets
- `OPENAI_API_KEY` - For GPT models
- `GOOGLE_API_KEY` - For Gemini models
- `GEMINI_API_KEY` - Alternative Google API key
- `DEEPSEEK_API_KEY` - For DeepSeek models

### Setting Up Secrets

#### For Codespaces
1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Codespaces**
3. Add each secret with the "New repository secret" button

#### For Local Development
1. Run the setup script: `./scripts/fetch_secrets.sh template`
2. Edit the generated `.env.local` file with your API keys
3. Load the environment: `source .env.local`

#### Using GitHub CLI
```bash
gh secret set ANTHROPIC_API_KEY
gh secret set CLAUDE_PAT
# ... etc for other secrets
```

## 🔧 Development Tools

### Environment Validation
Check if all required secrets are configured:
```bash
python scripts/environment_validator.py
```

### Secret Management
```bash
# Check which secrets exist in GitHub
./scripts/fetch_secrets.sh fetch

# Create local environment template
./scripts/fetch_secrets.sh template

# Validate current environment
./scripts/fetch_secrets.sh validate

# Show help
./scripts/fetch_secrets.sh help
```

### Quick Start Commands
```bash
# Run the AI system
python run_dynamic_ai.py

# Validate environment
python scripts/environment_validator.py

# Test specific components
python test_dynamic_system.py
python test_full_integration.py
```

## 📁 Container Configuration

### Features Included
- **Python 3.11** - Main development environment
- **GitHub CLI** - For repository operations
- **Git** - Version control
- **Zsh + Oh My Zsh** - Enhanced shell experience

### VS Code Extensions
- Python development tools (Python, Pylint, Black, isort)
- Jupyter notebooks support
- GitHub Copilot integration
- JSON and YAML support

### Port Forwarding
- **8000** - AI Server
- **8080** - Dashboard
- **3000** - Development Server

## 🔍 Troubleshooting

### Secrets Not Available
1. Verify secrets are set in GitHub repository settings
2. Restart the Codespace/container
3. Check secret names match exactly (case-sensitive)
4. Run environment validator: `python scripts/environment_validator.py`

### GitHub CLI Authentication Issues
```bash
# Check authentication status
gh auth status

# Re-authenticate if needed
gh auth login
```

### Missing Dependencies
```bash
# Reinstall Python dependencies
pip install -r requirements.txt

# Check container logs
# VS Code: View → Output → Select "Dev Containers"
```

### Container Won't Start
1. Check `.devcontainer/devcontainer.json` syntax
2. Verify all referenced files exist
3. Check VS Code Dev Containers extension is installed
4. Try rebuilding: "Dev Containers: Rebuild Container"

## 🏗️ Container Lifecycle

### Startup Sequence
1. **Container Creation** - Base image setup with features
2. **Post-Create** - Install Python dependencies and validate environment
3. **Post-Start** - Run setup script and environment checks

### Setup Scripts
- `.devcontainer/setup-environment.sh` - Main environment setup
- `scripts/environment_validator.py` - Secret validation
- `scripts/fetch_secrets.sh` - Secret management utilities

## 📚 Additional Resources

### API Key Documentation
- [Anthropic Console](https://console.anthropic.com/) - Get Claude API keys
- [OpenAI Platform](https://platform.openai.com/api-keys) - Get GPT API keys
- [Google AI Studio](https://aistudio.google.com/app/apikey) - Get Gemini API keys
- [GitHub Tokens](https://github.com/settings/tokens) - Create Personal Access Tokens
- [DeepSeek Platform](https://platform.deepseek.com/) - Get DeepSeek API keys

### Development Guides
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [GitHub CLI Manual](https://cli.github.com/manual/)

### CWMAI Project
- [Main README](../README.md) - Project overview and usage
- [Debug Guide](../SWARM_DEBUG_GUIDE.md) - Troubleshooting AI systems

## 🛠️ Customization

### Modifying the Container
1. Edit `.devcontainer/devcontainer.json`
2. Add additional features or VS Code extensions
3. Rebuild the container: "Dev Containers: Rebuild Container"

### Adding New Secrets
1. Add to `scripts/environment_validator.py`
2. Update `.devcontainer/devcontainer.json` containerEnv section
3. Document in this README

### Custom Setup Steps
Add custom initialization to `.devcontainer/setup-environment.sh`

---

**Need Help?** Check the troubleshooting section above or run `./scripts/fetch_secrets.sh help` for secret management assistance.