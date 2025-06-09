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

5. **Rate Limiter** (`rate_limiter.py`)
   - Sophisticated Redis-based API rate limiting
   - Multiple strategies: Token Bucket, Sliding Window, Fixed Window, Adaptive
   - Four-tier system: Basic, Premium, Admin, System
   - Real-time monitoring with fallback support

6. **Rate Limit Monitor** (`rate_limit_monitor.py`)
   - Real-time admin dashboard for rate limiting
   - Client usage statistics and analytics
   - Performance metrics and alerting
   - Usage reports and metrics export

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
- Redis server (optional, for rate limiting - fallback mode available)
- API Keys:
  - `CLAUDE_PAT`: GitHub Personal Access Token (required)
  - `ANTHROPIC_API_KEY`: For Claude AI (recommended)
  - `OPENAI_API_KEY`: For GPT fallback (optional)
  - `GEMINI_API_KEY`: For research tasks (optional)
  - `REDIS_URL`: Redis connection URL (optional, for rate limiting)

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

## 🚧 API Rate Limiting

### Overview

CWMAI includes sophisticated Redis-based rate limiting to ensure fair resource usage and system stability. The rate limiting system provides multiple strategies, real-time monitoring, and adaptive capabilities.

### Features

- **Multiple Strategies**: Token Bucket, Sliding Window, Fixed Window, and Adaptive
- **Four-Tier System**: Basic, Premium, Admin, and System tiers with different limits
- **Real-Time Monitoring**: Live dashboard with usage statistics and alerts
- **Fallback Support**: Continues operation even when Redis is unavailable
- **Backward Compatible**: Existing code works without modification

### Configuration

Set up rate limiting with environment variables:

```bash
# Rate limiting configuration
export RATE_LIMIT_ENABLED="true"              # Enable/disable rate limiting
export REDIS_URL="redis://localhost:6379/0"   # Redis connection URL
export RATE_LIMIT_TIER="basic"                # Default tier (basic|premium|admin|system)
export RATE_LIMIT_CLIENT_ID="my_client"       # Unique client identifier
```

### Rate Limit Tiers

| Tier | Requests/Min | Requests/Hour | Requests/Day | Burst | Strategy |
|------|--------------|---------------|--------------|-------|----------|
| **Basic** | 10 | 300 | 1,000 | 5 | Sliding Window |
| **Premium** | 30 | 1,000 | 5,000 | 15 | Token Bucket |
| **Admin** | 100 | 5,000 | 20,000 | 50 | Adaptive |
| **System** | 1,000 | 50,000 | 100,000 | 200 | Token Bucket |

### Usage Examples

#### Basic Usage with HTTP AI Client

```python
from scripts.http_ai_client import HTTPAIClient

# Initialize with rate limiting
client = HTTPAIClient(
    client_id="my_application",
    rate_limit_tier="premium"
)

# Make rate-limited requests
response = await client.generate_enhanced_response("Hello, AI!")

# Check rate limit status
status = client.get_rate_limit_status()
print(f"Remaining requests: {status['stats']['remaining_requests']}")
```

#### Direct Rate Limiter Usage

```python
from scripts.rate_limiter import RateLimiter, RateLimitTier

# Initialize rate limiter
limiter = RateLimiter()

# Check if request is allowed
result = limiter.check_rate_limit("client_123", RateLimitTier.BASIC, "api_endpoint")

if result.allowed:
    print("Request allowed")
    print(f"Remaining: {result.remaining_requests}")
else:
    print(f"Rate limited. Try again in {result.retry_after} seconds")
```

#### Monitoring and Administration

```python
from scripts.rate_limit_monitor import RateLimitMonitor

# Initialize monitor
monitor = RateLimitMonitor()

# Get real-time dashboard
dashboard = monitor.get_real_time_dashboard()
print(f"System status: {dashboard['system_status']}")

# Get client details
client_stats = monitor.get_client_details("client_123")

# Update client tier
result = monitor.update_client_tier("client_123", "premium")

# Generate usage report
report = monitor.generate_usage_report(hours=24)
```

### Command Line Tools

#### Rate Limit Monitor

```bash
# View real-time dashboard
python scripts/rate_limit_monitor.py dashboard

# Get client details
python scripts/rate_limit_monitor.py client --client-id my_client

# Update client tier
python scripts/rate_limit_monitor.py client --client-id my_client --tier premium

# Generate usage report
python scripts/rate_limit_monitor.py report --hours 24

# Export metrics
python scripts/rate_limit_monitor.py export --format json
```

#### Performance Benchmarks

```bash
# Run full benchmark suite
python scripts/rate_limit_benchmarks.py

# Quick benchmarks
python scripts/rate_limit_benchmarks.py --quick

# Save results to file
python scripts/rate_limit_benchmarks.py --output benchmark_results.json
```

### Testing

Run the comprehensive test suite:

```bash
# Run all rate limiting tests
python test_rate_limiting.py

# Run specific test categories
python -m unittest test_rate_limiting.TestRateLimiter
python -m unittest test_rate_limiting.TestHTTPAIClientWithRateLimiting
python -m unittest test_rate_limiting.TestRateLimitMonitor
```

### Architecture

The rate limiting system consists of:

1. **RateLimiter Core** (`rate_limiter.py`): Core rate limiting logic with multiple strategies
2. **HTTP Client Integration** (`http_ai_client.py`): Seamless integration with AI API calls
3. **Monitor Dashboard** (`rate_limit_monitor.py`): Real-time monitoring and administration
4. **Fallback System**: Continues operation when Redis is unavailable

### Performance

- **Throughput**: >1000 requests/second in Redis mode
- **Latency**: <5ms average response time
- **Memory**: <1KB per client in fallback mode
- **Scalability**: Tested with 200+ concurrent clients

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