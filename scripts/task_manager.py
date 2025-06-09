"""
Task Manager Module

Manages the creation, tracking, and orchestration of tasks for @claude.
Handles task state, dependencies, and progress monitoring for 24/7 operation.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from github import Github
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state_manager import StateManager
from ai_brain import IntelligentAIBrain
from context_gatherer import ContextGatherer


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Types of tasks that can be created."""
    NEW_PROJECT = "new_project"
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_REVIEW = "code_review"
    DEPENDENCY_UPDATE = "dependency_update"


class TaskManager:
    """Manages task lifecycle and orchestration."""
    
    def __init__(self, github_token: str = None):
        """Initialize the task manager.
        
        Args:
            github_token: GitHub personal access token
        """
        self.github_token = github_token or os.getenv('CLAUDE_PAT')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        self.github = Github(self.github_token) if self.github_token else None
        self.repo = self.github.get_repo(self.repo_name) if self.github else None
        
        # Load or initialize task state
        self.state_file = "task_state.json"
        self.history_file = "task_history.json"
        self.state = self._load_state()
        self.history = self._load_history()
        
        # Initialize AI components
        self.state_manager = StateManager()
        self.system_state = self.state_manager.load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load task state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading task state: {e}")
        
        return {
            "tasks": {},
            "task_counter": 1000,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "active_tasks": 0,
            "completed_today": 0,
            "success_rate": 0.0
        }
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load task history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading task history: {e}")
        return []
    
    def _save_state(self) -> None:
        """Save task state to file."""
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _save_history(self) -> None:
        """Save task history to file."""
        # Keep only last 1000 history entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        self.state["task_counter"] += 1
        self._save_state()
        return f"TASK-{self.state['task_counter']}"
    
    def create_task(self, task_type: TaskType, title: str, description: str,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   dependencies: List[str] = None,
                   estimated_hours: float = 4.0,
                   labels: List[str] = None) -> Dict[str, Any]:
        """Create a new task.
        
        Args:
            task_type: Type of task
            title: Task title
            description: Detailed task description
            priority: Task priority
            dependencies: List of task IDs this depends on
            estimated_hours: Estimated hours to complete
            labels: GitHub labels to apply
            
        Returns:
            Created task dictionary
        """
        task_id = self.generate_task_id()
        
        task = {
            "id": task_id,
            "type": task_type.value,
            "title": title,
            "description": description,
            "priority": priority.value,
            "status": TaskStatus.PENDING.value,
            "dependencies": dependencies or [],
            "blocks": [],
            "estimated_hours": estimated_hours,
            "actual_hours": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "assigned_at": None,
            "completed_at": None,
            "github_issue_number": None,
            "github_pr_numbers": [],
            "iterations": 0,
            "labels": labels or [],
            "metrics": {
                "comments": 0,
                "reactions": 0,
                "code_changes": 0
            }
        }
        
        # Update dependencies to include this as a blocker
        for dep_id in task["dependencies"]:
            if dep_id in self.state["tasks"]:
                self.state["tasks"][dep_id]["blocks"].append(task_id)
        
        self.state["tasks"][task_id] = task
        self._save_state()
        
        # Add to history
        self._add_history_entry("task_created", task_id, {"title": title, "type": task_type.value})
        
        return task
    
    def create_github_issue(self, task: Dict[str, Any]) -> Optional[int]:
        """Create a GitHub issue for a task.
        
        Args:
            task: Task dictionary
            
        Returns:
            GitHub issue number or None if failed
        """
        if not self.repo:
            print("GitHub repository not available")
            return None
        
        try:
            # Format the issue body with @claude mention
            body = self._format_issue_body(task)
            
            # Determine labels
            labels = task["labels"].copy()
            labels.append(f"priority:{task['priority']}")
            labels.append(f"type:{task['type']}")
            labels.append("ai-managed")
            
            # Create the issue
            issue = self.repo.create_issue(
                title=task["title"],
                body=body,
                labels=labels
            )
            
            # Update task with issue number
            task["github_issue_number"] = issue.number
            task["status"] = TaskStatus.ASSIGNED.value
            task["assigned_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
            
            print(f"Created GitHub issue #{issue.number} for {task['id']}")
            return issue.number
            
        except Exception as e:
            print(f"Error creating GitHub issue: {e}")
            return None
    
    def create_ai_task_issue(self, title: str, description: str, labels: List[str] = None, 
                           priority: str = "medium", task_type: str = "task") -> Optional[int]:
        """Create a GitHub issue for AI tasks with @claude mention.
        
        This is the centralized method that all AI-generated task issues should use
        to ensure consistent formatting and @claude mentions.
        
        Args:
            title: Issue title
            description: Issue description/body content
            labels: Additional labels for the issue
            priority: Task priority (low, medium, high)
            task_type: Type of task (task, setup, bug, feature, etc.)
            
        Returns:
            GitHub issue number or None if failed
        """
        if not self.repo:
            print("GitHub repository not available")
            return None
        
        try:
            # Validate @claude mention is present
            if "@claude" not in description:
                print("Warning: AI task issue created without @claude mention, adding it")
                description = f"@claude {description}"
            
            # Create labels list
            issue_labels = labels or []
            issue_labels.extend([f"priority:{priority}", f"type:{task_type}", "ai-managed"])
            
            # Create the issue directly with proper formatting
            issue = self.repo.create_issue(
                title=title,
                body=description,
                labels=issue_labels
            )
            
            print(f"Created AI task issue #{issue.number}: {title}")
            return issue.number
            
        except Exception as e:
            print(f"Error creating AI task issue: {e}")
            return None
    
    def _format_issue_body(self, task: Dict[str, Any]) -> str:
        """Format issue body with @claude mention.
        
        Args:
            task: Task dictionary
            
        Returns:
            Formatted issue body
        """
        # Get task type specific template
        template = self._get_task_template(TaskType(task["type"]))
        
        # Add dependency information
        dep_info = ""
        if task["dependencies"]:
            dep_tasks = [self.state["tasks"].get(dep_id) for dep_id in task["dependencies"]]
            dep_issues = [f"#{t['github_issue_number']}" for t in dep_tasks if t and t.get("github_issue_number")]
            if dep_issues:
                dep_info = f"\n## Dependencies\nThis task depends on: {', '.join(dep_issues)}\n"
        
        # Add blocking information
        block_info = ""
        if task["blocks"]:
            block_tasks = [self.state["tasks"].get(block_id) for block_id in task["blocks"]]
            block_issues = [f"#{t['github_issue_number']}" for t in block_tasks if t and t.get("github_issue_number")]
            if block_issues:
                block_info = f"\n## Blocks\nThis task blocks: {', '.join(block_issues)}\n"
        
        body = f"""@claude {task["description"]}

{template}

## Task Information
- **Task ID**: {task["id"]}
- **Priority**: {task["priority"]}
- **Estimated Hours**: {task["estimated_hours"]}
- **Type**: {task["type"]}
{dep_info}{block_info}

## Acceptance Criteria
Please ensure all requirements are met and include appropriate tests.

---
*This task was automatically generated by the AI Task Manager*
"""
        return body
    
    def _get_task_template(self, task_type: TaskType) -> str:
        """Get task-specific template content.
        
        Args:
            task_type: Type of task
            
        Returns:
            Template string
        """
        templates = {
            TaskType.NEW_PROJECT: """
## Project Setup
1. Fork https://github.com/laravel/react-starter-kit.git
2. Configure for the specified requirements
3. Update README with project-specific information

## Laravel React Starter Benefits
- Full-stack Laravel + React SPA structure
- Authentication with Sanctum pre-configured
- Modern tooling: Vite, TypeScript, Tailwind CSS
- Testing setup with PHPUnit and Jest
- Docker development environment

## Required Customizations
Please implement the features described above while maintaining the starter kit's structure.
""",
            TaskType.FEATURE: """
## Feature Requirements
Please implement this feature following the project's established patterns.

## Implementation Guidelines
- Follow existing code style and conventions
- Include unit and integration tests
- Update documentation as needed
- Ensure backward compatibility
""",
            TaskType.BUG_FIX: """
## Bug Details
Please investigate and fix the described issue.

## Fix Requirements
- Identify root cause
- Implement fix with minimal side effects
- Add regression tests
- Verify fix in multiple scenarios
""",
            TaskType.REFACTOR: """
## Refactoring Goals
Please refactor the code while maintaining functionality.

## Guidelines
- Preserve all existing functionality
- Improve code structure and readability
- Update tests as needed
- Document significant changes
""",
            TaskType.DOCUMENTATION: """
## Documentation Requirements
Please create or update documentation as specified.

## Documentation Standards
- Clear and concise writing
- Include code examples where relevant
- Follow project documentation format
- Ensure accuracy and completeness
""",
            TaskType.TESTING: """
## Testing Requirements
Please implement comprehensive tests.

## Test Coverage Goals
- Unit tests for individual components
- Integration tests for workflows
- Edge case coverage
- Performance benchmarks if applicable
""",
            TaskType.SECURITY: """
## Security Requirements
Please address the security concerns described.

## Security Guidelines
- Follow OWASP best practices
- Include security tests
- Document any security implications
- Consider performance impact
""",
            TaskType.PERFORMANCE: """
## Performance Goals
Please optimize performance as specified.

## Optimization Guidelines
- Benchmark before and after changes
- Document performance improvements
- Ensure no functionality regression
- Consider scalability implications
""",
            TaskType.CODE_REVIEW: """
## Review Requirements
Please review the specified code/PR.

## Review Focus Areas
- Code quality and best practices
- Security vulnerabilities
- Performance implications
- Test coverage
- Documentation completeness
""",
            TaskType.DEPENDENCY_UPDATE: """
## Update Requirements
Please update the specified dependencies.

## Update Guidelines
- Check for breaking changes
- Update related code as needed
- Run full test suite
- Document any migration steps
"""
        }
        
        return templates.get(task_type, "")
    
    def analyze_existing_tasks(self) -> Dict[str, Any]:
        """Analyze current state of tasks in GitHub.
        
        Returns:
            Analysis results
        """
        if not self.repo:
            return {"error": "GitHub repository not available"}
        
        analysis = {
            "open_issues": 0,
            "claude_assigned": 0,
            "in_progress": 0,
            "awaiting_review": 0,
            "stale_tasks": [],
            "completed_recently": 0,
            "by_type": {},
            "by_priority": {}
        }
        
        try:
            # Get all open issues
            issues = self.repo.get_issues(state="open", labels=["ai-managed"])
            
            for issue in issues:
                analysis["open_issues"] += 1
                
                # Check if assigned to @claude
                if "@claude" in issue.body:
                    analysis["claude_assigned"] += 1
                
                # Check status by labels
                labels = [label.name for label in issue.labels]
                if "in-progress" in labels:
                    analysis["in_progress"] += 1
                elif "awaiting-review" in labels:
                    analysis["awaiting_review"] += 1
                
                # Check if stale (no updates in 7 days)
                if (datetime.now(timezone.utc) - issue.updated_at.replace(tzinfo=timezone.utc)).days > 7:
                    analysis["stale_tasks"].append({
                        "number": issue.number,
                        "title": issue.title,
                        "days_stale": (datetime.now(timezone.utc) - issue.updated_at.replace(tzinfo=timezone.utc)).days
                    })
                
                # Count by type and priority
                for label in labels:
                    if label.startswith("type:"):
                        task_type = label.split(":")[1]
                        analysis["by_type"][task_type] = analysis["by_type"].get(task_type, 0) + 1
                    elif label.startswith("priority:"):
                        priority = label.split(":")[1]
                        analysis["by_priority"][priority] = analysis["by_priority"].get(priority, 0) + 1
            
            # Check recently closed issues
            closed_issues = self.repo.get_issues(state="closed", labels=["ai-managed"], sort="updated")
            for issue in closed_issues:
                if (datetime.now(timezone.utc) - issue.closed_at.replace(tzinfo=timezone.utc)).days <= 1:
                    analysis["completed_recently"] += 1
                else:
                    break
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def generate_tasks(self, focus: str = "auto", max_tasks: int = 5) -> List[Dict[str, Any]]:
        """Generate new tasks based on current state and focus area.
        
        Args:
            focus: Focus area for task generation
            max_tasks: Maximum number of tasks to generate
            
        Returns:
            List of generated tasks
        """
        # Analyze current state
        analysis = self.analyze_existing_tasks()
        
        # Don't generate too many tasks if many are open
        if analysis.get("open_issues", 0) > 20:
            max_tasks = min(max_tasks, 2)
        
        # Initialize AI brain for intelligent task generation
        context_gatherer = ContextGatherer()
        context = context_gatherer.gather_context(self.system_state.get("charter", {}))
        ai_brain = IntelligentAIBrain(self.system_state, context)
        
        generated_tasks = []
        
        # Determine task generation strategy
        if focus == "auto":
            # Intelligent task generation based on current state
            task_strategies = self._determine_task_strategies(analysis, ai_brain)
        else:
            # Focused task generation
            task_strategies = self._get_focused_strategies(focus)
        
        # Generate tasks based on strategies
        for strategy in task_strategies[:max_tasks]:
            task = self._generate_task_from_strategy(strategy, ai_brain)
            if task:
                generated_tasks.append(task)
        
        # Save generated tasks to file for reference
        with open("generated_tasks.json", "w") as f:
            json.dump(generated_tasks, f, indent=2)
        
        return generated_tasks
    
    def _determine_task_strategies(self, analysis: Dict[str, Any], ai_brain) -> List[Dict[str, Any]]:
        """Determine task generation strategies based on analysis.
        
        Args:
            analysis: Current task analysis
            ai_brain: AI brain instance
            
        Returns:
            List of task strategies
        """
        strategies = []
        
        # Check if we need a new project
        if len(self.system_state.get("projects", {})) < 3:
            strategies.append({
                "type": TaskType.NEW_PROJECT,
                "priority": TaskPriority.HIGH,
                "reason": "Portfolio expansion needed"
            })
        
        # Check for stale tasks that need attention
        if analysis.get("stale_tasks", []):
            strategies.append({
                "type": TaskType.CODE_REVIEW,
                "priority": TaskPriority.HIGH,
                "reason": "Review stale tasks",
                "target": analysis["stale_tasks"][0]
            })
        
        # Balance task types
        by_type = analysis.get("by_type", {})
        
        # Ensure documentation tasks
        if by_type.get("documentation", 0) < 2:
            strategies.append({
                "type": TaskType.DOCUMENTATION,
                "priority": TaskPriority.MEDIUM,
                "reason": "Documentation coverage needed"
            })
        
        # Ensure testing tasks
        if by_type.get("testing", 0) < 3:
            strategies.append({
                "type": TaskType.TESTING,
                "priority": TaskPriority.HIGH,
                "reason": "Test coverage improvement"
            })
        
        # Regular feature development
        strategies.append({
            "type": TaskType.FEATURE,
            "priority": TaskPriority.MEDIUM,
            "reason": "Feature development"
        })
        
        # Security review if none active
        if by_type.get("security", 0) == 0:
            strategies.append({
                "type": TaskType.SECURITY,
                "priority": TaskPriority.HIGH,
                "reason": "Security review needed"
            })
        
        return strategies
    
    def _get_focused_strategies(self, focus: str) -> List[Dict[str, Any]]:
        """Get task strategies for focused generation.
        
        Args:
            focus: Focus area
            
        Returns:
            List of task strategies
        """
        focus_map = {
            "new_features": TaskType.FEATURE,
            "bug_fixes": TaskType.BUG_FIX,
            "refactoring": TaskType.REFACTOR,
            "documentation": TaskType.DOCUMENTATION,
            "testing": TaskType.TESTING,
            "security": TaskType.SECURITY,
            "performance": TaskType.PERFORMANCE
        }
        
        task_type = focus_map.get(focus, TaskType.FEATURE)
        
        return [
            {
                "type": task_type,
                "priority": TaskPriority.HIGH,
                "reason": f"Focused {focus} generation"
            }
            for _ in range(5)
        ]
    
    def _generate_task_from_strategy(self, strategy: Dict[str, Any], ai_brain) -> Optional[Dict[str, Any]]:
        """Generate a specific task from a strategy.
        
        Args:
            strategy: Task generation strategy
            ai_brain: AI brain instance
            
        Returns:
            Generated task or None
        """
        task_type = strategy["type"]
        
        # Generate task content based on type
        if task_type == TaskType.NEW_PROJECT:
            title, description = self._generate_new_project_task(ai_brain)
        elif task_type == TaskType.FEATURE:
            title, description = self._generate_feature_task(ai_brain)
        elif task_type == TaskType.BUG_FIX:
            title, description = self._generate_bug_fix_task()
        elif task_type == TaskType.DOCUMENTATION:
            title, description = self._generate_documentation_task()
        elif task_type == TaskType.TESTING:
            title, description = self._generate_testing_task()
        elif task_type == TaskType.SECURITY:
            title, description = self._generate_security_task()
        elif task_type == TaskType.CODE_REVIEW:
            title, description = self._generate_review_task(strategy.get("target"))
        else:
            return None
        
        # Create the task
        task = self.create_task(
            task_type=task_type,
            title=title,
            description=description,
            priority=strategy["priority"],
            estimated_hours=self._estimate_hours(task_type)
        )
        
        # Create GitHub issue
        issue_number = self.create_github_issue(task)
        if issue_number:
            task["github_issue_number"] = issue_number
        
        return task
    
    def _generate_new_project_task(self, ai_brain) -> Tuple[str, str]:
        """Generate a new project task.
        
        Returns:
            Title and description tuple
        """
        # Get trending technologies from context
        trends = ai_brain.context.get("github_trending", [])[:3]
        trend_names = [t.get("title", "trending tech") for t in trends]
        
        project_ideas = [
            ("AI-Powered Task Automation Dashboard", "Create a Laravel React dashboard for visualizing and managing AI-generated tasks with real-time updates"),
            ("Smart Code Review Assistant", "Build a tool that analyzes PRs and provides intelligent code review suggestions"),
            ("Automated Documentation Generator", "Develop a system that generates and maintains project documentation automatically"),
            ("Performance Monitoring Platform", "Create a comprehensive performance monitoring solution for web applications"),
            ("Security Vulnerability Scanner", "Build an automated security scanning tool for codebases")
        ]
        
        import random
        title, base_description = random.choice(project_ideas)
        
        description = f"""{base_description}

## Project Requirements

### Core Features
- User authentication and authorization
- Real-time data visualization
- RESTful API with Laravel
- React TypeScript frontend
- Comprehensive test coverage

### Technology Stack
- Laravel 10+ with Sanctum authentication  
- React 18+ with TypeScript
- Tailwind CSS for styling
- PostgreSQL database
- Redis for caching and queues

### Trending Technologies to Consider
{', '.join(trend_names) if trend_names else 'Current best practices'}

### Development Approach
1. Fork the Laravel React starter kit
2. Set up the development environment
3. Implement core features incrementally
4. Add comprehensive testing
5. Deploy with CI/CD pipeline

Please create this project following Laravel and React best practices."""
        
        return title, description
    
    def _generate_feature_task(self, ai_brain) -> Tuple[str, str]:
        """Generate a feature task.
        
        Returns:
            Title and description tuple
        """
        features = [
            ("Add real-time notifications system", "Implement WebSocket-based real-time notifications"),
            ("Create advanced search functionality", "Build Elasticsearch integration for powerful search"),
            ("Implement data export feature", "Add ability to export data in multiple formats (CSV, JSON, PDF)"),
            ("Add multi-language support", "Implement i18n for supporting multiple languages"),
            ("Create API rate limiting", "Implement sophisticated rate limiting with Redis")
        ]
        
        import random
        title, description = random.choice(features)
        
        full_description = f"""{description}

## Feature Specifications

### User Stories
- As a user, I want to receive real-time updates
- As an admin, I need to monitor system activity
- As a developer, I need clear API documentation

### Technical Requirements
- Follow existing architectural patterns
- Maintain backward compatibility
- Include comprehensive tests
- Update API documentation
- Add performance benchmarks

### Implementation Notes
Please ensure this feature integrates seamlessly with the existing codebase."""
        
        return title, full_description
    
    def _generate_bug_fix_task(self) -> Tuple[str, str]:
        """Generate a bug fix task.
        
        Returns:
            Title and description tuple
        """
        bugs = [
            ("Fix memory leak in data processing", "Memory usage grows unbounded during large data imports"),
            ("Resolve race condition in concurrent updates", "Database updates can fail under high concurrency"),
            ("Fix authentication token refresh issue", "JWT tokens not refreshing properly in some scenarios"),
            ("Correct timezone handling errors", "Dates showing incorrectly for users in different timezones"),
            ("Fix file upload validation bypass", "Certain file types bypassing validation checks")
        ]
        
        import random
        title, description = random.choice(bugs)
        
        full_description = f"""Bug Report: {description}

## Steps to Reproduce
1. [Detailed steps will be provided after investigation]
2. Observe the incorrect behavior
3. Check logs for errors

## Expected Behavior
System should handle this scenario gracefully without errors.

## Investigation Required
- Identify root cause
- Check for similar issues elsewhere
- Review recent changes that might have introduced this

## Fix Requirements
- Resolve the core issue
- Add regression tests
- Verify fix doesn't impact other features
- Update documentation if needed"""
        
        return title, full_description
    
    def _generate_documentation_task(self) -> Tuple[str, str]:
        """Generate a documentation task.
        
        Returns:
            Title and description tuple
        """
        docs = [
            ("Create API documentation", "Document all REST API endpoints with examples"),
            ("Write deployment guide", "Create comprehensive deployment documentation"),
            ("Document architecture decisions", "Create ADR (Architecture Decision Records)"),
            ("Create user guide", "Write end-user documentation with screenshots"),
            ("Document testing strategy", "Explain testing approach and how to write tests")
        ]
        
        import random
        title, description = random.choice(docs)
        
        full_description = f"""{description}

## Documentation Requirements

### Content Structure
- Overview and introduction
- Detailed explanations
- Code examples
- Best practices
- Troubleshooting guide

### Format Requirements
- Markdown format
- Clear headings and structure
- Include diagrams where helpful
- Provide real-world examples
- Keep it up-to-date

### Target Audience
- Developers (internal and external)
- System administrators
- End users (where applicable)"""
        
        return title, full_description
    
    def _generate_testing_task(self) -> Tuple[str, str]:
        """Generate a testing task.
        
        Returns:
            Title and description tuple
        """
        tests = [
            ("Add integration tests for API endpoints", "Create comprehensive API integration tests"),
            ("Implement E2E tests for critical user flows", "Add Cypress tests for main user journeys"),
            ("Create performance benchmarks", "Establish performance baselines and tests"),
            ("Add security test suite", "Implement automated security testing"),
            ("Increase unit test coverage to 90%", "Fill gaps in unit test coverage")
        ]
        
        import random
        title, description = random.choice(tests)
        
        full_description = f"""{description}

## Testing Requirements

### Test Scenarios
- Happy path scenarios
- Error conditions
- Edge cases
- Performance under load
- Security vulnerabilities

### Testing Standards
- Follow AAA pattern (Arrange, Act, Assert)
- Use meaningful test names
- Include both positive and negative tests
- Mock external dependencies
- Ensure tests are deterministic

### Coverage Goals
- Minimum 80% code coverage
- All critical paths tested
- Integration with CI/CD pipeline"""
        
        return title, full_description
    
    def _generate_security_task(self) -> Tuple[str, str]:
        """Generate a security task.
        
        Returns:
            Title and description tuple
        """
        security = [
            ("Conduct security audit of authentication", "Review and harden authentication system"),
            ("Implement CSRF protection", "Add comprehensive CSRF protection across the application"),
            ("Add input validation layer", "Implement robust input validation and sanitization"),
            ("Review and update dependencies", "Audit and update dependencies for security patches"),
            ("Implement security headers", "Add and configure security headers for protection")
        ]
        
        import random
        title, description = random.choice(security)
        
        full_description = f"""{description}

## Security Requirements

### Audit Scope
- Review current implementation
- Identify vulnerabilities
- Check against OWASP Top 10
- Review security best practices
- Test attack scenarios

### Implementation Requirements
- Fix identified vulnerabilities
- Add security tests
- Document security measures
- Consider performance impact
- Maintain usability

### Compliance
- Follow security best practices
- Document any trade-offs
- Include security testing in CI/CD"""
        
        return title, full_description
    
    def _generate_review_task(self, target: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """Generate a code review task.
        
        Returns:
            Title and description tuple
        """
        if target:
            title = f"Review stale issue #{target['number']}: {target['title']}"
            description = f"""Please review the stale issue that has been inactive for {target['days_stale']} days.

## Review Requirements
- Check current status and blockers
- Determine if issue is still relevant
- Provide recommendations for moving forward
- Update issue with findings"""
        else:
            title = "Review recent pull requests"
            description = """Please review recent pull requests for code quality and best practices.

## Review Focus Areas
- Code quality and style
- Test coverage
- Security implications
- Performance considerations
- Documentation completeness"""
        
        return title, description
    
    def _estimate_hours(self, task_type: TaskType) -> float:
        """Estimate hours for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Estimated hours
        """
        estimates = {
            TaskType.NEW_PROJECT: 16.0,
            TaskType.FEATURE: 8.0,
            TaskType.BUG_FIX: 4.0,
            TaskType.REFACTOR: 6.0,
            TaskType.DOCUMENTATION: 3.0,
            TaskType.TESTING: 4.0,
            TaskType.SECURITY: 6.0,
            TaskType.PERFORMANCE: 5.0,
            TaskType.CODE_REVIEW: 2.0,
            TaskType.DEPENDENCY_UPDATE: 3.0
        }
        
        return estimates.get(task_type, 4.0)
    
    def review_completed_tasks(self) -> Dict[str, Any]:
        """Review tasks marked as completed by @claude.
        
        Returns:
            Review results
        """
        if not self.repo:
            return {"error": "GitHub repository not available"}
        
        reviews = {
            "reviewed": 0,
            "approved": 0,
            "needs_revision": 0,
            "tasks": []
        }
        
        try:
            # Get issues labeled as awaiting-review
            issues = self.repo.get_issues(state="open", labels=["ai-managed", "awaiting-review"])
            
            for issue in issues:
                reviews["reviewed"] += 1
                
                # Analyze the issue and PR
                review_result = self._review_task_completion(issue)
                
                if review_result["approved"]:
                    reviews["approved"] += 1
                    # Close the issue
                    issue.edit(state="closed")
                    issue.create_comment("✅ Task completed successfully. Great work!")
                else:
                    reviews["needs_revision"] += 1
                    # Request changes
                    issue.remove_from_labels("awaiting-review")
                    issue.add_to_labels("needs-revision")
                    issue.create_comment(f"🔄 Task needs revision:\n\n{review_result['feedback']}")
                
                reviews["tasks"].append({
                    "issue_number": issue.number,
                    "title": issue.title,
                    "result": "approved" if review_result["approved"] else "needs_revision",
                    "feedback": review_result.get("feedback", "")
                })
        
        except Exception as e:
            reviews["error"] = str(e)
        
        return reviews
    
    def _review_task_completion(self, issue) -> Dict[str, Any]:
        """Review a specific task completion.
        
        Args:
            issue: GitHub issue object
            
        Returns:
            Review result
        """
        # Check for linked PRs
        linked_prs = []
        for event in issue.get_events():
            if event.event == "cross-referenced":
                # This is a simple check - in production, you'd want more sophisticated PR detection
                linked_prs.append(event)
        
        # Basic review criteria
        has_pr = len(linked_prs) > 0
        has_tests = any("test" in comment.body.lower() for comment in issue.get_comments())
        has_documentation = any("doc" in comment.body.lower() for comment in issue.get_comments())
        
        # Simple approval logic - in production, this would be more sophisticated
        approved = has_pr
        
        feedback = []
        if not has_pr:
            feedback.append("- No linked pull request found")
        if not has_tests:
            feedback.append("- Consider adding tests")
        if not has_documentation:
            feedback.append("- Consider updating documentation")
        
        return {
            "approved": approved,
            "feedback": "\n".join(feedback) if feedback else "All requirements met"
        }
    
    def prioritize_tasks(self) -> None:
        """Update task priorities based on current state."""
        analysis = self.analyze_existing_tasks()
        
        # Re-prioritize based on various factors
        for task_id, task in self.state["tasks"].items():
            if task["status"] in [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]:
                continue
            
            # Increase priority for blocked tasks
            if task["dependencies"]:
                deps_completed = all(
                    self.state["tasks"].get(dep_id, {}).get("status") == TaskStatus.COMPLETED.value
                    for dep_id in task["dependencies"]
                )
                if deps_completed and task["priority"] != TaskPriority.CRITICAL.value:
                    task["priority"] = TaskPriority.HIGH.value
            
            # Increase priority for old tasks
            created_date = datetime.fromisoformat(task["created_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_date).days
            
            if age_days > 7 and task["priority"] == TaskPriority.LOW.value:
                task["priority"] = TaskPriority.MEDIUM.value
            elif age_days > 14 and task["priority"] == TaskPriority.MEDIUM.value:
                task["priority"] = TaskPriority.HIGH.value
        
        self._save_state()
    
    def generate_report(self) -> str:
        """Generate a task management report.
        
        Returns:
            Markdown formatted report
        """
        analysis = self.analyze_existing_tasks()
        
        report = f"""# Task Management Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview
- **Total Active Tasks**: {analysis.get('open_issues', 0)}
- **Assigned to @claude**: {analysis.get('claude_assigned', 0)}
- **In Progress**: {analysis.get('in_progress', 0)}
- **Awaiting Review**: {analysis.get('awaiting_review', 0)}
- **Completed Today**: {analysis.get('completed_recently', 0)}

## Task Distribution

### By Type
"""
        
        for task_type, count in analysis.get("by_type", {}).items():
            report += f"- **{task_type}**: {count}\n"
        
        report += "\n### By Priority\n"
        for priority, count in analysis.get("by_priority", {}).items():
            report += f"- **{priority}**: {count}\n"
        
        if analysis.get("stale_tasks"):
            report += "\n## ⚠️ Stale Tasks\n"
            for stale in analysis["stale_tasks"][:5]:
                report += f"- #{stale['number']}: {stale['title']} ({stale['days_stale']} days)\n"
        
        report += f"""
## Performance Metrics
- **Success Rate**: {self.state.get('success_rate', 0):.1%}
- **Active Tasks**: {self.state.get('active_tasks', 0)}
- **Completed Today**: {self.state.get('completed_today', 0)}

## Recommendations
"""
        
        if analysis.get('open_issues', 0) > 15:
            report += "- ⚠️ High number of open tasks - focus on completion\n"
        
        if analysis.get('stale_tasks', []):
            report += "- 🔄 Review and update stale tasks\n"
        
        if analysis.get('awaiting_review', 0) > 3:
            report += "- 👀 Multiple tasks awaiting review - prioritize reviews\n"
        
        with open("task_report.md", "w") as f:
            f.write(report)
        
        return report
    
    def _add_history_entry(self, action: str, task_id: str, details: Dict[str, Any]) -> None:
        """Add entry to task history.
        
        Args:
            action: Action type
            task_id: Task ID
            details: Additional details
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "task_id": task_id,
            "details": details
        }
        
        self.history.append(entry)
        self._save_history()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="AI Task Manager")
    parser.add_argument("command", choices=["generate", "review", "prioritize", "report", "analyze"],
                       help="Command to execute")
    parser.add_argument("--focus", default="auto",
                       help="Focus area for task generation")
    parser.add_argument("--max-tasks", type=int, default=5,
                       help="Maximum tasks to generate")
    
    args = parser.parse_args()
    
    # Initialize task manager
    manager = TaskManager()
    
    if args.command == "generate":
        print(f"Generating tasks with focus: {args.focus}")
        tasks = manager.generate_tasks(focus=args.focus, max_tasks=args.max_tasks)
        print(f"Generated {len(tasks)} tasks")
        for task in tasks:
            print(f"- {task['id']}: {task['title']}")
    
    elif args.command == "review":
        print("Reviewing completed tasks...")
        results = manager.review_completed_tasks()
        print(f"Reviewed {results.get('reviewed', 0)} tasks")
        print(f"- Approved: {results.get('approved', 0)}")
        print(f"- Needs revision: {results.get('needs_revision', 0)}")
    
    elif args.command == "prioritize":
        print("Updating task priorities...")
        manager.prioritize_tasks()
        print("Task priorities updated")
    
    elif args.command == "report":
        print("Generating task report...")
        report = manager.generate_report()
        print("Report generated: task_report.md")
    
    elif args.command == "analyze":
        print("Analyzing existing tasks...")
        analysis = manager.analyze_existing_tasks()
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()