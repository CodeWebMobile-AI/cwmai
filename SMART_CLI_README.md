# Smart CWMAI CLI - Intelligent Natural Language Interface

## Overview

The Smart CWMAI CLI is an advanced natural language interface that makes interacting with the CWMAI system as intuitive as having a conversation. It features:

- **Multi-Model AI Understanding**: Uses Claude, GPT-4, and Gemini for consensus-based interpretation
- **Learning Capabilities**: Learns from your usage patterns and preferences
- **Context Awareness**: Maintains conversation context and project state
- **Real-Time Intelligence**: Integrates with Brave Search for current market data
- **Smart Suggestions**: Provides intelligent next-step recommendations
- **Error Recovery**: Offers helpful suggestions when things go wrong

## Quick Start

```bash
# Interactive mode (recommended)
python run_smart_cli.py

# Single command mode
python run_smart_cli.py "create an issue for auth-api about adding OAuth2 support"

# With options
python run_smart_cli.py --no-learning --single-model
```

## Features

### 1. Natural Language Understanding

The system understands various ways of expressing the same intent:

- "Create an issue for auth-api about slow login"
- "Make a bug report in auth-api saying users can't log in quickly"
- "File a ticket for the authentication service regarding performance"

All these commands will be understood as creating a performance-related issue.

### 2. Multi-Model Consensus

When enabled (default), the system consults multiple AI models and uses consensus to ensure accurate interpretation:

```
High Confidence (80%+ agreement): Executes immediately
Medium Confidence (60-80%): Asks for confirmation
Low Confidence (<60%): Provides clarification options
```

### 3. Learning System

The CLI learns from your patterns:

- **Command Patterns**: Remembers frequently used commands
- **Project Context**: Tracks your current working project
- **Preferences**: Learns your preferred phrasings and shortcuts
- **Custom Routines**: Understands phrases like "do the usual morning check"

### 4. Context-Aware Intelligence

The system maintains context across commands:

```
> create issue for auth-api about login bug
✅ Created issue #42 in auth-api

> create another issue about the same problem but for signup
✅ Created issue #43 in auth-api (using context from previous command)
```

### 5. Smart Command Processing

#### Simple Commands
```
"show status" → Display system statistics with AI summary
"search for react hooks" → Multi-source search with ranking
"create task to update docs" → Intelligent task generation
```

#### Complex Multi-Step Operations
```
"find all repos with security issues and create a plan to fix them"
→ Searches repositories
→ Identifies security issues
→ Creates prioritized fix plan
→ Generates tasks for each issue
```

#### Market-Aware Features
```
"analyze market for AI code review tools"
→ Real-time market research via Brave Search
→ Competitive analysis
→ Opportunity identification
→ Project recommendations
```

### 6. Enhanced Issue Creation

Issues are automatically enhanced with:
- Better titles and descriptions
- Suggested labels based on content
- Related issue detection
- Technical detail inference

### 7. Intelligent Architecture Generation

```
"design architecture for a real-time chat app like Slack"
→ Market research on competitors
→ Technology stack recommendations
→ Scalability considerations
→ Cost optimization strategies
```

## Command Examples

### Basic Operations

```bash
# Issue Management
"create issue for project-x about memory leak in worker threads"
"make a feature request to add dark mode to the dashboard"

# Search Operations
"search for python machine learning libraries"
"find all my projects using tensorflow"
"look for repositories with recent security updates"

# Architecture & Design
"generate architecture for an e-commerce platform"
"design a microservices system for video streaming"
"create blueprint for mobile-first web app"

# Task Management
"create task to refactor authentication module"
"what tasks are currently active?"
"show tasks for project-x"

# Analysis
"analyze market demand for no-code platforms"
"what's trending in DevOps tools?"
"show performance metrics for last week"
```

### Advanced Operations

```bash
# Conditional Operations
"if there are any critical bugs, create issues for them"

# Batch Operations
"create issues for all the problems found in the security audit"

# Research-Driven Development
"research best practices for API design and create a style guide"

# Competitive Analysis
"analyze competitors of Vercel and suggest how we can differentiate"
```

## Interactive Mode Features

### Tab Completion
- Press Tab to complete commands
- Works with learned patterns
- Suggests based on context

### Command History
- Up/Down arrows search through history
- Intelligent history search
- Pattern-based suggestions

### Shortcuts
- `q` - Quit
- `h` - Help
- `st` - Show status
- `s` - Search
- `ci` - Create issue
- `ga` - Generate architecture
- `ex` - Show examples

### Smart Prompts
The prompt shows context:
- `cwmai>` - Default
- `cwmai:project-name>` - When in project context
- `cwmai:project[🧠25]>` - With learning indicator (25 patterns learned)

## Configuration

### Environment Variables

Required:
- `ANTHROPIC_API_KEY` - For Claude AI

Optional but recommended:
- `OPENAI_API_KEY` - For GPT-4 consensus
- `GEMINI_API_KEY` - For Gemini consensus
- `BRAVE_API_KEY` - For real-time search (already configured)
- `GITHUB_TOKEN` - For GitHub operations

### Options

- `--no-learning` - Disable pattern learning
- `--single-model` - Use only one AI model (faster but less accurate)
- `--skip-check` - Skip environment variable validation

## Tips for Best Results

1. **Be Specific**: "create issue for auth-api about OAuth2" is better than "make issue"

2. **Use Context**: After mentioning a project once, you can refer to it as "it" or "the project"

3. **Chain Commands**: Use "and then" or "after that" for multi-step operations

4. **Ask Questions**: The system can answer questions about your projects and suggest actions

5. **Learn Shortcuts**: The system learns your patterns - use consistent phrasing for common tasks

## Troubleshooting

### "I don't understand" Responses
- Try rephrasing with more specific terms
- Use the `examples` command to see similar requests
- Break complex requests into steps

### Slow Response
- Multi-model consensus takes time but is more accurate
- Use `--single-model` for faster responses
- Ensure good internet connection for web searches

### Learning Issues
- Learning data is saved in `~/.cwmai/user_model.pkl`
- Delete this file to reset learned patterns
- Use `--no-learning` to disable

## Architecture

The Smart CLI uses a sophisticated architecture:

```
User Input
    ↓
Pattern Matching (High Confidence)
    ↓ (if no match)
Multi-Model AI Consensus
    ↓
Intent Classification
    ↓
Context Enhancement
    ↓
Execution with Recovery
    ↓
Learning & Feedback
```

## Privacy & Security

- All learning data is stored locally
- API keys are never logged or transmitted
- Command history is local only
- Web searches use configured API keys

## Future Enhancements

Planned features:
- Voice input support
- Slack/Discord integration
- Custom command aliases
- Team sharing of patterns
- Automated workflow recording