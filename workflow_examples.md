# Workflow System Examples & Questions

This document provides example questions and tasks that users can ask to trigger the various system workflows.

## 1. System Health Check & Auto-Fix Workflow

**Trigger Questions:**
- "Check the health of all system tools and fix any broken ones"
- "Are all tools working properly? Fix any issues automatically"
- "Run a complete system health check with auto-repair"
- "Diagnose and fix any broken tools in the system"
- "Perform system diagnostics and repair any failures"

**Command:**
```bash
python scripts/workflow_manager.py run system_health_check
```

## 2. Repository Analysis & Issue Creation Workflow

**Trigger Questions:**
- "Analyze all our repositories and create issues for any problems"
- "Scan our GitHub repos for security issues and create tasks"
- "Check all repositories for code quality problems"
- "Run a comprehensive analysis on our codebase and create GitHub issues"
- "Find and document all technical debt in our repositories"

**Command:**
```bash
python scripts/workflow_manager.py run repo_analysis_pipeline --params '{"organization_name": "CodeWebMobile-AI"}'
```

## 3. Daily Maintenance Workflow

**Trigger Questions:**
- "Run the daily maintenance tasks"
- "Clean up logs and temporary files from the system"
- "Perform routine system maintenance"
- "Check API keys and clean up old data"
- "Execute daily housekeeping tasks"

**Command:**
```bash
python scripts/workflow_manager.py run daily_maintenance
```

## 4. Research & Knowledge Update Workflow

**Trigger Questions:**
- "Search for the latest AI research papers and update our knowledge base"
- "Find new machine learning papers and create summaries"
- "Update our research database with recent publications"
- "What are the latest developments in LLM research?"
- "Gather and summarize recent AI/ML research findings"

**Command:**
```bash
python scripts/workflow_manager.py run research_knowledge_update
```

## 5. Code Quality Pipeline Workflow

**Trigger Questions:**
- "Run a complete code quality check on our Python files"
- "Check for security vulnerabilities and code smells"
- "Analyze code complexity and create improvement tasks"
- "Perform linting and security scans on the codebase"
- "Find and fix code quality issues"

**Command:**
```bash
python scripts/workflow_manager.py run code_quality_pipeline
```

## 6. Multi-Tool Data Processing Workflow

**Trigger Questions:**
- "Fetch data from our APIs and analyze patterns"
- "Process external API data and store insights in Redis"
- "Run the data pipeline to analyze trends"
- "Collect and analyze data from multiple sources"
- "Execute the data processing workflow with visualization recommendations"

**Command:**
```bash
python scripts/workflow_manager.py run data_processing_pipeline --params '{"api1_endpoint": "https://api.example.com/data", "api1_key": "key1"}'
```

## 7. Intelligent Task Distribution Workflow

**Trigger Questions:**
- "Assign all unassigned GitHub issues to team members based on their skills"
- "Distribute pending tasks intelligently among the team"
- "Match open issues with the best team members"
- "Create personalized task lists for each developer"
- "Optimize task assignment based on skills and availability"

**Command:**
```bash
python scripts/workflow_manager.py run task_distribution
```

## 8. System Performance Monitoring Workflow

**Trigger Questions:**
- "Monitor system performance and alert on anomalies"
- "Check tool execution times and Redis performance"
- "Analyze system performance metrics"
- "Are there any performance bottlenecks in the system?"
- "Generate performance optimization recommendations"

**Command:**
```bash
python scripts/workflow_manager.py run performance_monitoring
```

## 9. Documentation Auto-Generation Workflow

**Trigger Questions:**
- "Update all documentation based on current code"
- "Generate documentation for all custom tools"
- "Check if documentation is outdated and update it"
- "Create API documentation from code"
- "Automatically refresh project documentation"

**Command:**
```bash
python scripts/workflow_manager.py run documentation_generator
```

## 10. Emergency Response Workflow

**Trigger Questions:**
- "System failure detected! Initiate emergency response"
- "Critical error occurred - start recovery procedures"
- "Handle the current system emergency"
- "Escalate and fix critical failures"
- "Execute disaster recovery workflow"

**Command:**
```bash
python scripts/workflow_manager.py run emergency_response
```

## Custom Workflow Creation

**Using AI to Create Custom Workflows:**
```bash
# Create a custom workflow using AI
python scripts/workflow_manager.py ai-create "Create a workflow that backs up all important files to S3 every night"

# Create a custom monitoring workflow
python scripts/workflow_manager.py ai-create "Monitor GitHub PR reviews and remind reviewers after 24 hours"
```

## Workflow Management Commands

### List All Available Workflows
```bash
python scripts/workflow_manager.py list
```

### Check Workflow Status
```bash
python scripts/workflow_manager.py status <workflow_id>
```

### Execute Existing Workflow
```bash
python scripts/workflow_manager.py execute <workflow_id> --input '{"param": "value"}'
```

## Integration with CWMAI CLI

When using the `cwmai` CLI, you can trigger workflows naturally:

```bash
./cwmai
> Check system health and fix any broken tools
> Analyze our repositories for security issues
> Run daily maintenance
> Find the latest AI research papers
> Check code quality across all Python files
```

The system will recognize these requests and execute the appropriate workflows automatically.