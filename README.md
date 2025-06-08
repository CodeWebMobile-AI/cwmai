# CWMAI - 24/7 Autonomous AI Task Management System

An intelligent task orchestration system that operates continuously, creating and managing development tasks for @claude integration.

## 🚀 Overview

CWMAI transforms AI from a developer into a **24/7 Technical Project Manager** that:
- 🤖 Generates specific, actionable tasks every 30 minutes
- 📋 Creates GitHub issues with @claude mentions for implementation  
- 📊 Tracks progress and reviews completed work
- 🎯 Prioritizes tasks based on business value and dependencies
- 📈 Maintains a real-time dashboard of all activities

## 🏗️ Architecture

### Core Components

1. **Task Manager** (`task_manager.py`)
   - Creates and tracks tasks with full lifecycle management
   - Generates GitHub issues with @claude mentions
   - Manages dependencies and blocking relationships
   - Reviews completed work for quality assurance

2. **AI Brain** (`ai_brain.py`)
   - Intelligent decision engine for task orchestration
   - Chooses optimal actions based on system state
   - Integrates with multiple AI providers (Anthropic, OpenAI, Gemini)
   - Learns from historical performance

3. **Task Analyzer** (`task_analyzer.py`)
   - Analyzes GitHub issues and @claude interactions
   - Provides insights on task completion rates
   - Identifies bottlenecks and stale tasks
   - Generates actionable recommendations

4. **Dashboard Updater** (`update_task_dashboard.py`)
   - Maintains a live dashboard as a pinned GitHub issue
   - Shows real-time metrics and visualizations
   - Tracks performance over time
   - Provides system health monitoring

## 🔄 How It Works

### Continuous Operation (Every 30 Minutes)

1. **Analysis Phase**
   - System analyzes current tasks and @claude interactions
   - Identifies gaps, bottlenecks, and opportunities
   - Reviews completed work quality

2. **Decision Phase**
   - AI Brain decides the best action to take:
     - Generate new tasks
     - Review completed tasks
     - Prioritize backlog
     - Analyze performance
     - Update dashboard

3. **Execution Phase**
   - Creates detailed GitHub issues with @claude mentions
   - Updates task priorities and dependencies
   - Reviews and approves/rejects completed work
   - Updates the live dashboard

### Task Types Generated

- **New Projects**: Full applications using Laravel React starter kit
- **Features**: New functionality with detailed specifications
- **Bug Fixes**: Issues with reproduction steps and fix requirements  
- **Documentation**: API docs, guides, and technical writing
- **Testing**: Unit, integration, and E2E test suites
- **Security**: Vulnerability assessments and fixes
- **Performance**: Optimization tasks with benchmarks
- **Code Reviews**: Quality assessments of existing code

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- GitHub repository with appropriate permissions
- API Keys:
  - `CLAUDE_PAT`: GitHub Personal Access Token (required)
  - `ANTHROPIC_API_KEY`: For Claude AI (recommended)
  - `OPENAI_API_KEY`: For GPT fallback (optional)
  - `GEMINI_API_KEY`: For research tasks (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/CodeWebMobile-AI/cwmai.git
cd cwmai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export CLAUDE_PAT="your-github-token"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 📊 Workflows

### Main Workflow (`main.yml`)
- Runs every 4 hours automatically
- Can be triggered manually with specific actions
- Executes the main AI decision cycle

### Task Manager Workflow (`continuous-task-manager.yml`)
- Runs every 30 minutes for continuous operation
- Generates, reviews, and manages tasks
- Updates the task dashboard

### Monitoring Workflow (`monitoring.yml`)
- Daily health checks
- Weekly performance reports
- System status monitoring

## 📈 Dashboard

The system maintains a live dashboard as a pinned GitHub issue showing:
- Task status distribution
- Performance metrics
- @claude interaction effectiveness
- Insights and recommendations
- Recent activity

Access the dashboard by looking for the "📊 AI Task Management Dashboard" issue.

## 🤝 @claude Integration

The system leverages the `base-claude.yml` workflow to enable @claude to:
- Respond to task assignments in issues
- Create pull requests with implementations
- Provide status updates
- Request clarification when needed

## 📚 Task Templates

Each task type includes specific templates optimized for @claude:
- Clear requirements and acceptance criteria
- Technical specifications
- Laravel React starter kit integration (for new projects)
- Testing and documentation requirements

## 🔍 Monitoring & Analytics

- **Success Rate**: Track task completion effectiveness
- **Velocity**: Measure tasks completed per day
- **Quality Metrics**: Review approval rates
- **Response Time**: @claude interaction speed
- **Bottleneck Detection**: Identify process improvements

## 🚦 Getting Started

1. Fork this repository
2. Set up the required secrets in GitHub
3. Enable GitHub Actions
4. The system will start operating automatically
5. Check the dashboard issue for real-time status

## 🤖 AI Providers

- **Primary**: Anthropic Claude (task orchestration)
- **Secondary**: OpenAI GPT (fallback)
- **Research**: Google Gemini & DeepSeek (analysis)

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built for seamless @claude integration
- Leverages Laravel React starter kit for projects
- Inspired by autonomous development principles