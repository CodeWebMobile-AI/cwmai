# CWMAI Conversational AI Assistant 🤖

A Claude-like natural language interface for CWMAI that makes interaction feel like chatting with an intelligent assistant.

## Features

### 🧠 Natural Conversation
- Chat naturally without memorizing commands
- Ask questions and get intelligent responses
- Execute CWMAI operations through conversation
- Context-aware responses that remember your conversation

### 🎯 Smart Command Understanding
- Interprets your intent even with typos or variations
- Suggests alternatives when unsure
- Learns from your patterns over time
- Handles complex multi-step requests

### 🎨 Multiple Response Styles
- **Friendly Professional** (default): Balanced, helpful responses
- **Concise**: Short, to-the-point answers
- **Detailed**: Comprehensive explanations
- **Technical**: Developer-focused language
- **Casual**: Relaxed, conversational tone

### 🔧 Automatic Environment Setup
- Auto-loads `.env.local` and `.env` files
- No manual environment variable exports needed
- Clear error messages for missing configuration

## Installation

1. Make sure you're in the CWMAI directory:
```bash
cd /path/to/cwmai
```

2. The scripts are already executable. You can run directly:
```bash
./cwmai
```

3. (Optional) Add to your PATH for global access:
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$PATH:/path/to/cwmai"

# Then reload your shell
source ~/.bashrc  # or ~/.zshrc
```

## Usage

### Interactive Mode (Recommended)
Start a conversation session:
```bash
cwmai
```

Then chat naturally:
```
You > Hi! Can you help me create an issue for the auth-api repo about slow login times?
Assistant > I'll help you create an issue for the auth-api repository about slow login times...

You > Also search for performance monitoring tools we could use
Assistant > Looking for performance monitoring tools...

You > What's the status of my projects?
Assistant > Let me check the status of your projects...
```

### Single Command Mode
Execute one command and exit:
```bash
cwmai create an issue for frontend-app about responsive design bugs
cwmai show me the system status
cwmai search for AI code review tools
```

### Response Styles
Choose how the assistant responds:
```bash
cwmai --concise          # Brief responses
cwmai --detailed         # Comprehensive responses
cwmai --technical        # Developer-focused language
cwmai --casual           # Relaxed conversation
```

### Command Line Options
```bash
cwmai --help             # Show all options
cwmai --no-env           # Skip .env file loading
cwmai --no-banner        # Skip welcome banner
```

## Example Conversations

### Creating Issues
```
You > I need to create an issue about authentication problems users are having
Assistant > I'll help you create an issue about authentication problems. Could you tell me which repository this is for?

You > It's for the auth-service repo
Assistant > Got it! Let me create an issue for auth-service about authentication problems...
✓ Done! Created issue #42 in auth-service
URL: https://github.com/yourorg/auth-service/issues/42
```

### Complex Requests
```
You > Find all repos with security issues and create tasks to fix them
Assistant > Let me search for repositories with security issues...
Found 3 repositories with security vulnerabilities. I'll create fix tasks for each:
✓ Created task to fix SQL injection in user-api
✓ Created task to update dependencies in payment-service  
✓ Created task to add input validation in admin-portal
```

### Getting Help
```
You > What can you help me with?
Assistant > I'm your CWMAI assistant! I can help you with:

📋 Task Management
• Create issues and tasks for any repository
• List and track active tasks
• Update task status and priorities

🔍 Discovery & Search
• Search through repositories
• Find specific code or documentation
• Analyze repository health

🏗️ Architecture & Design
• Generate system architectures
• Create project structures
• Design database schemas

...and much more! Just tell me what you need in natural language.
```

## Tips for Best Results

1. **Be Specific**: "Create an issue for auth-api about login performance" works better than "make issue"

2. **Use Context**: The assistant remembers your conversation, so you can refer to previous topics

3. **Ask for Clarification**: If unsure, ask "What can you do?" or "Can you help with X?"

4. **Natural Language**: Write as you would speak - the AI understands variations and context

5. **Learn Patterns**: The system learns your common tasks and preferences over time

## Environment Setup

Create a `.env.local` file in the CWMAI directory:
```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional AI providers
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
GITHUB_TOKEN=your_github_token

# Optional services
BRAVE_SEARCH_API_KEY=your_brave_key
REDIS_URL=redis://localhost:6379/0
```

## Troubleshooting

### Missing Environment Variables
```
❌ Error: Missing required environment variables: ANTHROPIC_API_KEY
ℹ️  Please set them in .env.local or .env file
```
**Solution**: Create `.env.local` with your API keys

### Import Errors
```
❌ Error: Cannot find CWMAI scripts
```
**Solution**: Make sure you're running from the CWMAI directory

### Connection Issues
```
⚠️  MCP integration failed to initialize
```
**Solution**: This is usually fine - the system will work with reduced functionality

## Comparison: Old vs New

### Old Way
```bash
# Manually load environment
export $(cat .env.local | grep -v '^#' | xargs)

# Run with specific script
python run_cwmai_cli.py

# Type formal commands
CWMAI> create issue for repo auth-api title "Fix login" body "Users report slow login"
```

### New Way (Smart & Natural)
```bash
# Just run
cwmai

# Chat naturally
You > Hey, users are reporting slow login times in the auth API. Can you create an issue for this?
Assistant > I'll create an issue for the auth-api repository about slow login times...
```

## Architecture

The conversational AI assistant consists of:

1. **Smart Wrapper (`cwmai`)**: Handles environment setup and launches the assistant
2. **Conversational AI Assistant**: Manages natural dialogue and context
3. **Smart NLI**: Enhanced natural language understanding with learning
4. **CWMAI Core**: Executes actual operations (issues, tasks, etc.)

## Contributing

To enhance the conversational assistant:

1. **Add Response Templates**: Edit `_load_conversation_templates()` in `conversational_ai_assistant.py`
2. **Improve Intent Detection**: Enhance `_analyze_input_type()` 
3. **Add Command Patterns**: Update `smart_natural_language_interface.py`
4. **Create Plugins**: Add to `smart_cli_plugins.py`

## Future Enhancements

- Voice input/output support
- Multi-language support  
- Custom personality configuration
- Integration with more development tools
- Slack/Discord bot versions

---

Enjoy chatting with your new AI development assistant! 🚀