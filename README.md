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

## 📊 Data Export Feature

CWMAI includes a comprehensive data export system supporting multiple formats and data types for analysis and reporting.

### Supported Export Formats

- **CSV**: Tabular data with full pandas integration for easy analysis
- **JSON**: Structured data with complete metadata and relationships
- **PDF**: Professional reports with tables, charts, and formatted layouts

### Data Types Available for Export

1. **Task Data** (`DataType.TASKS`)
   - Individual task details with status, priority, and progress
   - Task dependencies and relationships
   - Time tracking and completion metrics
   - Filterable by status, priority, date range

2. **Performance Metrics** (`DataType.PERFORMANCE`)
   - System performance indicators and success rates
   - Task completion velocity and quality metrics
   - Repository health scores and activity levels
   - Learning algorithm effectiveness measurements

3. **Repository Analytics** (`DataType.REPOSITORIES`)
   - Repository health scores and activity metrics
   - Contributor statistics and code quality indicators
   - Issue and pull request analytics
   - Filterable by health score, language, activity

4. **System Analytics** (`DataType.ANALYTICS`)
   - AI decision-making accuracy and learning metrics
   - Goal achievement and resource efficiency tracking
   - External context integration and trend analysis
   - System charter and constraint compliance

5. **Complete System State** (`DataType.SYSTEM_STATE`)
   - Full system snapshot including all components
   - Historical state progression and change tracking
   - Configuration and environment details

### Usage Examples

#### Command Line Interface

```bash
# Export all task data as JSON
python scripts/data_export_service.py --data-type tasks --format json

# Export performance metrics as CSV
python scripts/data_export_service.py --data-type performance --format csv

# Export repository data as PDF with custom filename
python scripts/data_export_service.py --data-type repositories --format pdf --filename repo_report.pdf

# Run performance benchmark
python scripts/data_export_service.py --data-type all --format json --benchmark
```

#### Programmatic Usage

```python
from scripts.data_export_service import DataExportService, DataType, ExportFormat

# Initialize export service
export_service = DataExportService(output_dir="exports")

# Export filtered task data
filepath = export_service.export_data(
    data_type=DataType.TASKS,
    export_format=ExportFormat.CSV,
    filters={"status": "completed", "priority": "high"}
)

# Export repository data with health score filter
filepath = export_service.export_data(
    data_type=DataType.REPOSITORIES,
    export_format=ExportFormat.JSON,
    filters={"min_health_score": 85}
)

# Run performance benchmark
benchmark = export_service.get_export_performance_benchmark(
    DataType.PERFORMANCE, 
    ExportFormat.PDF
)
print(f"Export took {benchmark['execution_time_seconds']} seconds")
```

### Advanced Filtering Options

- **Task Filters**: status, priority, type, start_date, end_date
- **Repository Filters**: min_health_score, language, status
- **Date Range Filters**: ISO 8601 formatted date strings
- **Custom Filters**: Extensible filtering system for specific use cases

### Performance Monitoring

The export service includes built-in performance benchmarking:

- **Execution Time**: Precise timing for export operations
- **Memory Usage**: Memory consumption tracking during export
- **File Size**: Output file size metrics
- **Success Rate**: Export operation reliability tracking
- **Format Comparison**: Performance comparison across export formats

### API Integration

Export functionality integrates seamlessly with existing CWMAI components:

- **Real-time Updates**: Exports reflect current system state
- **Backward Compatibility**: Maintains compatibility with existing data structures
- **Automated Scheduling**: Can be integrated with workflow automation
- **Error Handling**: Comprehensive error handling and recovery

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