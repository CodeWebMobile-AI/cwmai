#!/bin/bash
# Setup script for MCP (Model Context Protocol) servers

echo "🚀 Setting up MCP servers for CWMAI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Install MCP servers globally
echo ""
echo "📦 Installing MCP servers..."

# Core MCP servers
MCP_SERVERS=(
    "@modelcontextprotocol/server-github"
    "@modelcontextprotocol/server-filesystem" 
    "@modelcontextprotocol/server-memory"
    "@modelcontextprotocol/server-git"
    "@modelcontextprotocol/server-fetch"
)

# Optional MCP servers (commented out)
# MCP_SERVERS+=("@modelcontextprotocol/server-mysql")
# MCP_SERVERS+=("@modelcontextprotocol/server-postgres")
# MCP_SERVERS+=("@modelcontextprotocol/server-slack")

for server in "${MCP_SERVERS[@]}"; do
    echo "Installing $server..."
    npm install -g "$server" || {
        echo "⚠️  Failed to install $server, trying with npx instead..."
    }
done

echo ""
echo "✅ MCP servers installation complete!"

# Create default MCP configuration
echo ""
echo "📝 Creating default MCP configuration..."

cat > mcp_config.json << 'EOF'
{
  "servers": {
    "github": {
      "name": "github",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": ""
      }
    },
    "filesystem": {
      "name": "filesystem",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
      "args": ["--allowed-directories", "/workspaces/cwmai,/tmp"]
    },
    "memory": {
      "name": "memory",
      "command": ["npx", "-y", "@modelcontextprotocol/server-memory"]
    },
    "git": {
      "name": "git",
      "command": ["npx", "-y", "@modelcontextprotocol/server-git"],
      "args": ["--repository", "/workspaces/cwmai"]
    },
    "fetch": {
      "name": "fetch",
      "command": ["npx", "-y", "@modelcontextprotocol/server-fetch"],
      "args": ["--max-redirects", "5"]
    }
  },
  "default_timeout": 30,
  "max_retries": 3
}
EOF

echo "✅ Created mcp_config.json"

# Check for required environment variables
echo ""
echo "🔍 Checking environment variables..."

if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN not set. GitHub MCP will not work without it."
    echo "   Set it with: export GITHUB_TOKEN='your-github-token'"
else
    echo "✅ GITHUB_TOKEN is set"
fi

if [ -z "$GITHUB_REPOSITORY" ]; then
    echo "⚠️  GITHUB_REPOSITORY not set. Using default: CodeWebMobile-AI/cwmai"
    echo "   Set it with: export GITHUB_REPOSITORY='owner/repo'"
else
    echo "✅ GITHUB_REPOSITORY is set to: $GITHUB_REPOSITORY"
fi

# Create .env.example if it doesn't exist
if [ ! -f .env.example ]; then
    echo ""
    echo "📝 Creating .env.example..."
    cat > .env.example << 'EOF'
# GitHub Configuration
GITHUB_TOKEN=your-github-personal-access-token
GITHUB_REPOSITORY=CodeWebMobile-AI/cwmai

# MySQL Configuration (optional)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=cwmai
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=cwmai

# AI Provider Tokens
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Redis Configuration
REDIS_URL=redis://localhost:6379
EOF
    echo "✅ Created .env.example"
fi

# Test MCP servers
echo ""
echo "🧪 Testing MCP servers..."

# Test filesystem MCP
echo -n "Testing filesystem MCP... "
if npx -y @modelcontextprotocol/server-filesystem --version &> /dev/null; then
    echo "✅"
else
    echo "❌"
fi

# Test memory MCP
echo -n "Testing memory MCP... "
if npx -y @modelcontextprotocol/server-memory --version &> /dev/null; then
    echo "✅"
else
    echo "❌"
fi

echo ""
echo "✨ MCP setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your GITHUB_TOKEN environment variable"
echo "2. Run: python test_mcp_integration.py"
echo "3. Start using MCPs in your code!"
echo ""
echo "For MySQL MCP support, install and configure MySQL, then run:"
echo "   npm install -g @modelcontextprotocol/server-mysql"