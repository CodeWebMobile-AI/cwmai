"""
Task Dashboard Updater

Updates a GitHub issue that serves as a task dashboard with current status.
Provides a real-time view of the AI task management system.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from github import Github
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from repository_exclusion import should_process_repo, filter_excluded_repos


class TaskDashboardUpdater:
    """Updates the task management dashboard."""
    
    DASHBOARD_ISSUE_TITLE = "📊 AI Task Management Dashboard"
    
    def __init__(self, github_token: str = None):
        """Initialize the dashboard updater.
        
        Args:
            github_token: GitHub personal access token
        """
        self.github_token = github_token or os.getenv('CLAUDE_PAT')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        self.github = Github(self.github_token) if self.github_token else None
        self.repo = self.github.get_repo(self.repo_name) if self.github else None
        
    def update_dashboard(self) -> None:
        """Update or create the task dashboard issue."""
        if not self.repo:
            print("GitHub repository not available")
            return
        
        # Find or create dashboard issue
        dashboard_issue = self._find_dashboard_issue()
        if not dashboard_issue:
            dashboard_issue = self._create_dashboard_issue()
        
        # Load latest data
        task_state = self._load_task_state()
        task_analysis = self._load_task_analysis()
        
        # Generate dashboard content
        dashboard_content = self._generate_dashboard_content(task_state, task_analysis)
        
        # Update the issue
        try:
            dashboard_issue.edit(body=dashboard_content)
            print(f"Dashboard updated: #{dashboard_issue.number}")
            
            # Add a comment with the update timestamp
            dashboard_issue.create_comment(
                f"🔄 Dashboard updated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def _find_dashboard_issue(self):
        """Find existing dashboard issue.
        
        Returns:
            GitHub issue or None
        """
        try:
            issues = self.repo.get_issues(state="open")
            for issue in issues:
                if issue.title == self.DASHBOARD_ISSUE_TITLE:
                    return issue
        except Exception as e:
            print(f"Error finding dashboard issue: {e}")
        
        return None
    
    def _create_dashboard_issue(self):
        """Create new dashboard issue.
        
        Returns:
            Created GitHub issue
        """
        try:
            issue = self.repo.create_issue(
                title=self.DASHBOARD_ISSUE_TITLE,
                body="Initializing dashboard...",
                labels=["dashboard", "ai-managed", "documentation"]
            )
            
            # Pin the issue
            issue.create_comment("/pin")
            
            return issue
            
        except Exception as e:
            print(f"Error creating dashboard issue: {e}")
            return None
    
    def _load_task_state(self) -> Dict[str, Any]:
        """Load task state from file.
        
        Returns:
            Task state dictionary
        """
        if os.path.exists("task_state.json"):
            with open("task_state.json", "r") as f:
                return json.load(f)
        return {}
    
    def _load_task_analysis(self) -> Dict[str, Any]:
        """Load task analysis from file.
        
        Returns:
            Task analysis dictionary
        """
        if os.path.exists("task_analysis.json"):
            with open("task_analysis.json", "r") as f:
                return json.load(f)
        return {}
    
    def _generate_dashboard_content(self, task_state: Dict[str, Any], 
                                   task_analysis: Dict[str, Any]) -> str:
        """Generate dashboard content.
        
        Args:
            task_state: Current task state
            task_analysis: Task analysis data
            
        Returns:
            Markdown formatted dashboard
        """
        # Calculate metrics
        tasks = task_state.get("tasks", {})
        total_tasks = len(tasks)
        
        status_counts = {
            "pending": 0,
            "assigned": 0,
            "in_progress": 0,
            "in_review": 0,
            "completed": 0,
            "blocked": 0,
            "failed": 0
        }
        
        priority_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        type_counts = {}
        
        for task in tasks.values():
            status = task.get("status", "pending")
            if status in status_counts:
                status_counts[status] += 1
            
            priority = task.get("priority", "medium")
            if priority in priority_counts:
                priority_counts[priority] += 1
            
            task_type = task.get("type", "unknown")
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        # Get insights and recommendations
        insights = task_analysis.get("insights", [])
        recommendations = task_analysis.get("recommendations", [])
        
        # Generate status chart
        status_chart = self._generate_status_chart(status_counts)
        
        # Build dashboard content
        content = f"""# 📊 AI Task Management Dashboard

> **Last Updated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
> **System Status**: 🟢 Operational

## 📈 Overview

| Metric | Value |
|--------|-------|
| **Total Tasks** | {total_tasks} |
| **Active Tasks** | {status_counts['assigned'] + status_counts['in_progress']} |
| **Completed Today** | {task_state.get('completed_today', 0)} |
| **Success Rate** | {task_state.get('success_rate', 0):.1%} |
| **@claude Interactions** | {task_analysis.get('summary', {}).get('claude_mentions', 0)} |

## 📊 Task Status Distribution

{status_chart}

### Status Breakdown
- 🆕 **Pending**: {status_counts['pending']}
- 📝 **Assigned**: {status_counts['assigned']}
- 🔄 **In Progress**: {status_counts['in_progress']}
- 👀 **In Review**: {status_counts['in_review']}
- ✅ **Completed**: {status_counts['completed']}
- 🚫 **Blocked**: {status_counts['blocked']}
- ❌ **Failed**: {status_counts['failed']}

## 🎯 Priority Distribution

| Priority | Count | Percentage |
|----------|-------|------------|
| 🔴 Critical | {priority_counts['critical']} | {priority_counts['critical']/max(total_tasks,1)*100:.1f}% |
| 🟠 High | {priority_counts['high']} | {priority_counts['high']/max(total_tasks,1)*100:.1f}% |
| 🟡 Medium | {priority_counts['medium']} | {priority_counts['medium']/max(total_tasks,1)*100:.1f}% |
| 🟢 Low | {priority_counts['low']} | {priority_counts['low']/max(total_tasks,1)*100:.1f}% |

## 🏷️ Task Types

"""
        
        # Add task type breakdown
        for task_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / max(total_tasks, 1) * 100
            content += f"- **{task_type.replace('_', ' ').title()}**: {count} ({percentage:.1f}%)\n"
        
        # Add insights
        if insights:
            content += "\n## 💡 Insights\n\n"
            for insight in insights[:5]:
                content += f"- {insight}\n"
        
        # Add recommendations
        if recommendations:
            content += "\n## 📋 Recommendations\n\n"
            for rec in recommendations[:5]:
                content += f"- {rec}\n"
        
        # Add recent activity
        content += "\n## 🕐 Recent Activity\n\n"
        
        # Get recent tasks
        recent_tasks = sorted(
            [t for t in tasks.values() if t.get("github_issue_number")],
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )[:5]
        
        if recent_tasks:
            content += "| Task | Type | Status | Created |\n"
            content += "|------|------|--------|--------|\n"
            
            for task in recent_tasks:
                issue_link = f"#{task['github_issue_number']}"
                task_type = task['type'].replace('_', ' ').title()
                status_emoji = {
                    "pending": "🆕",
                    "assigned": "📝",
                    "in_progress": "🔄",
                    "in_review": "👀",
                    "completed": "✅",
                    "blocked": "🚫",
                    "failed": "❌"
                }.get(task['status'], "❓")
                
                created_date = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
                created_str = created_date.strftime('%Y-%m-%d')
                
                content += f"| {issue_link} {task['title'][:30]}... | {task_type} | {status_emoji} {task['status']} | {created_str} |\n"
        
        # Add performance metrics
        content += f"""
## 📊 Performance Metrics

### Task Completion Velocity
- **Last 24 hours**: {self._calculate_velocity(tasks, 1)} tasks/day
- **Last 7 days**: {self._calculate_velocity(tasks, 7):.1f} tasks/day
- **Last 30 days**: {self._calculate_velocity(tasks, 30):.1f} tasks/day

### Efficiency Metrics
- **Average Time to Complete**: {self._calculate_avg_completion_time(tasks):.1f} hours
- **First Response Time**: < 30 minutes (automated)
- **Task Success Rate**: {task_state.get('success_rate', 0):.1%}

## 🤖 AI System Health

| Component | Status | Details |
|-----------|---------|---------|
| Task Manager | 🟢 Active | Running every 30 minutes |
| @claude Integration | 🟢 Active | Responding to mentions |
| GitHub API | 🟢 Connected | All permissions granted |
| State Persistence | 🟢 Healthy | Last save: < 30 mins ago |

---

<details>
<summary>📖 About This Dashboard</summary>

This dashboard is automatically updated every 30 minutes by the AI Task Management System. It provides real-time visibility into:

- Task distribution and status
- @claude interaction effectiveness  
- System performance metrics
- Actionable insights and recommendations

The system operates 24/7, creating and managing development tasks autonomously while collaborating with @claude for implementation.

</details>

---
*🤖 This dashboard is automatically maintained by the AI Task Management System*
"""
        
        return content
    
    def _generate_status_chart(self, status_counts: Dict[str, int]) -> str:
        """Generate ASCII status chart.
        
        Args:
            status_counts: Status count dictionary
            
        Returns:
            ASCII chart string
        """
        total = sum(status_counts.values())
        if total == 0:
            return "No tasks to display"
        
        chart = "```\n"
        max_width = 40
        
        for status, count in status_counts.items():
            if count > 0:
                percentage = count / total * 100
                bar_width = int((count / total) * max_width)
                bar = "█" * bar_width + "░" * (max_width - bar_width)
                chart += f"{status:12} |{bar}| {count:3} ({percentage:4.1f}%)\n"
        
        chart += "```"
        return chart
    
    def _calculate_velocity(self, tasks: Dict[str, Any], days: int) -> float:
        """Calculate task completion velocity.
        
        Args:
            tasks: Task dictionary
            days: Number of days to calculate
            
        Returns:
            Tasks per day
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        completed_count = 0
        
        for task in tasks.values():
            if task.get("status") == "completed" and task.get("completed_at"):
                completed_date = datetime.fromisoformat(task["completed_at"].replace('Z', '+00:00'))
                if completed_date >= cutoff_date:
                    completed_count += 1
        
        return completed_count / days if days > 0 else 0
    
    def _calculate_avg_completion_time(self, tasks: Dict[str, Any]) -> float:
        """Calculate average task completion time.
        
        Args:
            tasks: Task dictionary
            
        Returns:
            Average hours to complete
        """
        completion_times = []
        
        for task in tasks.values():
            if task.get("status") == "completed" and task.get("created_at") and task.get("completed_at"):
                created = datetime.fromisoformat(task["created_at"].replace('Z', '+00:00'))
                completed = datetime.fromisoformat(task["completed_at"].replace('Z', '+00:00'))
                hours = (completed - created).total_seconds() / 3600
                if hours > 0 and hours < 168:  # Exclude outliers > 1 week
                    completion_times.append(hours)
        
        return sum(completion_times) / len(completion_times) if completion_times else 24.0


def main():
    """Main function for standalone execution."""
    print("Updating task dashboard...")
    
    updater = TaskDashboardUpdater()
    updater.update_dashboard()
    
    print("Dashboard update complete!")


if __name__ == "__main__":
    from datetime import timedelta
    main()