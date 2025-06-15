# CWMAI Natural Language Interface

## Overview

The CWMAI Natural Language Interface provides an intuitive way to interact with the CWMAI system using conversational commands. No need to remember specific API calls or complex syntax - just describe what you want to do in plain English.

## Features

- **Natural Language Understanding**: Interprets commands using pattern matching and AI
- **Context Awareness**: Remembers previous commands and repositories
- **Interactive REPL**: Tab completion, command history, and colored output
- **Flexible Commands**: Multiple ways to express the same intent
- **AI-Powered Interpretation**: Falls back to AI for ambiguous commands

## Installation

1. Ensure you have CWMAI installed and configured
2. Install the additional dependency:
   ```bash
   pip install colorama
   ```

## Usage

### Interactive Mode

Start the interactive CLI:
```bash
python run_cwmai_cli.py
```

You'll see a prompt where you can type natural language commands:
```
CWMAI> create an issue for myrepo about adding dark mode
✓ Issue created successfully!
```

### Single Command Mode

Execute a single command directly:
```bash
python run_cwmai_cli.py "search repositories for python AI"
```

## Available Commands

### Creating GitHub Issues
```
create an issue for [repository] about [topic]
make a new issue in [repository] for [topic]
open issue for [repository] about [topic]
```

Examples:
- `create an issue for auth-service about implementing OAuth2`
- `make a new issue in frontend-app for responsive design`

### Searching Repositories
```
search repositories for [query]
find repos about [topic]
look for repositories containing [keyword]
```

Examples:
- `search repositories for machine learning`
- `find me some repos about react components`

### Generating Architecture
```
create architecture for [project]
design system architecture for [application]
generate architecture for [system]
```

Examples:
- `create architecture for e-commerce platform`
- `design system architecture for real-time chat application`

### System Status
```
show status
what's the system status?
how are things doing?
status
```

### Task Management
```
list tasks
show active tasks
what tasks are pending
create task [description]
```

Examples:
- `list active tasks`
- `create task to review and update API documentation`

### Performance Analysis
```
analyze performance
show performance metrics
how is the system performing?
```

### Getting Help
```
help
help [topic]
what commands do you support?
```

Examples:
- `help`
- `help issue`
- `help search`

## Command Shortcuts

The CLI supports shortcuts for common commands:
- `q`, `exit` → `quit`
- `?`, `h` → `help`
- `st` → `show status`
- `lt` → `list tasks`
- `ap` → `analyze performance`

## Special Commands

- `clear` - Clear the screen
- `history` - Show command history
- `quit` or `q` - Exit the CLI

## Context Awareness

The interface remembers context from previous commands:
```
CWMAI> search repositories for python
Found 5 repositories...

CWMAI> create an issue for awesome-python about adding tests
✓ Issue created in repository 'awesome-python'
```

## AI Interpretation

When a command doesn't match known patterns, the system uses AI to interpret your intent:
```
CWMAI> I need to make a bug report for the login system
[AI interprets this as: create issue for login-system about bug report]
✓ Issue created successfully!
```

## Environment Requirements

For full functionality, ensure these environment variables are set:
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` - For AI capabilities
- `GITHUB_TOKEN` or `CLAUDE_PAT` - For GitHub integration
- MCP configuration file (`mcp_config.json`) - For advanced integrations

## Examples

### Creating a Comprehensive Issue
```
CWMAI> create an issue for payment-service about implementing Stripe webhook handling
✓ Issue created successfully!
  Title: Implement Stripe webhook handling
  Labels: enhancement, backend
  Issue #: 42
```

### Searching and Creating Architecture
```
CWMAI> search repositories for microservices
Found 3 repositories:
1. user-service
2. order-service
3. notification-service

CWMAI> create architecture for notification service with email and SMS support
✓ Architecture generated successfully
[Displays detailed architecture document]
```

### Task and Performance Management
```
CWMAI> show status
System Status:
  Total Tasks: 45
  Active Tasks: 12
  Completed Tasks: 33
  AI Providers: Anthropic, OpenAI
  Repositories: 8

CWMAI> analyze performance
Performance Analysis:
  Completion Rate: 73.3%
  Avg Completion Time: 2.5 days
  Success Rate: 91%
  Last 24h Tasks: 5
```

## Tips

1. **Be Specific**: More details lead to better results
   - Good: "create issue for auth-api about adding rate limiting with Redis"
   - Less Good: "create issue about rate limiting"

2. **Use Natural Language**: The system understands variations
   - "make an issue", "create an issue", "open an issue" all work

3. **Leverage Context**: The system remembers your last repository
   - First: "search repositories for payment"
   - Then: "create issue about refund processing" (uses last repo)

4. **Get Help**: Use `help [topic]` for specific guidance
   - `help issue` - Learn about issue creation
   - `help architecture` - Learn about architecture generation

## Troubleshooting

### "No AI provider available"
Set at least one API key:
```bash
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

### "GitHub integration not available"
Set your GitHub token:
```bash
export GITHUB_TOKEN="your-github-token"
```

### "Command not recognized"
- Check spelling and try variations
- Use `help` to see available commands
- The AI will try to interpret unclear commands

## Architecture

The Natural Language Interface consists of:

1. **Pattern Matcher**: Regular expressions for common commands
2. **AI Interpreter**: Falls back to AI for ambiguous input
3. **Command Executor**: Integrates with CWMAI subsystems
4. **Context Manager**: Maintains conversation state
5. **Result Formatter**: Presents data in a readable format

## Contributing

To add new commands:

1. Add patterns to `command_patterns` in `natural_language_interface.py`
2. Implement the executor method (`_command_name`)
3. Update help text and documentation
4. Add tests to `test_natural_language_interface.py`

## Future Enhancements

- Voice input support
- Multi-turn conversations
- Batch command execution
- Command macros and aliases
- Integration with more CWMAI features
- Export command results
- Command suggestions based on history