"""
Task Analyzer Module

Analyzes GitHub issues and PRs to understand task state and @claude interactions.
Provides insights for intelligent task generation and management.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from github import Github
import re


class TaskAnalyzer:
    """Analyzes tasks and @claude interactions."""
    
    def __init__(self, github_token: str = None):
        """Initialize the task analyzer.
        
        Args:
            github_token: GitHub personal access token
        """
        self.github_token = github_token or os.getenv('CLAUDE_PAT')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        self.github = Github(self.github_token) if self.github_token else None
        self.repo = self.github.get_repo(self.repo_name) if self.github else None
        
    def analyze_all_tasks(self) -> Dict[str, Any]:
        """Perform comprehensive task analysis.
        
        Returns:
            Analysis results
        """
        if not self.repo:
            return {"error": "GitHub repository not available"}
        
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_issues": 0,
                "open_issues": 0,
                "closed_issues": 0,
                "claude_mentions": 0,
                "active_prs": 0
            },
            "claude_interactions": [],
            "task_states": {
                "pending": [],
                "in_progress": [],
                "awaiting_review": [],
                "completed": [],
                "stale": []
            },
            "insights": [],
            "recommendations": []
        }
        
        # Analyze open issues
        print("Analyzing open issues...")
        open_issues = self.repo.get_issues(state="open")
        for issue in open_issues:
            # Check if issue has ai-managed label
            label_names = [label.name for label in issue.labels]
            if 'ai-managed' not in label_names:
                # Log skipped issues for transparency
                print(f"Skipping issue #{issue.number}: '{issue.title}' (created by: {issue.user.login}) - no ai-managed label")
                continue
                
            analysis["summary"]["open_issues"] += 1
            analysis["summary"]["total_issues"] += 1
            
            # Check for @claude mentions
            if self._has_claude_mention(issue):
                analysis["summary"]["claude_mentions"] += 1
                interaction = self._analyze_claude_interaction(issue)
                if interaction:
                    analysis["claude_interactions"].append(interaction)
            
            # Categorize by state
            state = self._determine_task_state(issue)
            task_info = {
                "number": issue.number,
                "title": issue.title,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
                "comments": issue.comments,
                "state": state
            }
            
            analysis["task_states"][state].append(task_info)
        
        # Analyze closed issues
        print("Analyzing closed issues...")
        closed_issues = self.repo.get_issues(state="closed")
        closed_count = 0
        for issue in closed_issues:
            closed_count += 1
            if closed_count > 100:  # Limit to recent 100
                break
            
            analysis["summary"]["closed_issues"] += 1
            analysis["summary"]["total_issues"] += 1
            
            if self._has_claude_mention(issue):
                analysis["summary"]["claude_mentions"] += 1
        
        # Analyze pull requests
        print("Analyzing pull requests...")
        prs = self.repo.get_pulls(state="open")
        for pr in prs:
            analysis["summary"]["active_prs"] += 1
            
            # Check if PR is related to @claude task
            if self._has_claude_mention(pr):
                analysis["claude_interactions"].append({
                    "type": "pull_request",
                    "number": pr.number,
                    "title": pr.title,
                    "created_at": pr.created_at.isoformat(),
                    "status": "open",
                    "claude_authored": self._is_claude_authored(pr)
                })
        
        # Generate insights
        analysis["insights"] = self._generate_insights(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Save analysis
        with open("task_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _has_claude_mention(self, item) -> bool:
        """Check if an issue/PR has @claude mention.
        
        Args:
            item: GitHub issue or PR
            
        Returns:
            True if @claude is mentioned
        """
        # First check if issue has ai-managed label
        if hasattr(item, 'labels'):
            label_names = [label.name for label in item.labels]
            if 'ai-managed' not in label_names:
                # Skip issues without ai-managed label
                return False
        
        # Check body
        if item.body and "@claude" in item.body.lower():
            return True
        
        # Check comments
        try:
            for comment in item.get_comments():
                if "@claude" in comment.body.lower():
                    return True
        except:
            pass
        
        return False
    
    def _analyze_claude_interaction(self, issue) -> Optional[Dict[str, Any]]:
        """Analyze @claude interaction in an issue.
        
        Args:
            issue: GitHub issue
            
        Returns:
            Interaction analysis or None
        """
        interaction = {
            "issue_number": issue.number,
            "title": issue.title,
            "created_at": issue.created_at.isoformat(),
            "interaction_type": "task_assignment",
            "claude_responses": 0,
            "human_responses": 0,
            "last_interaction": None,
            "task_completed": issue.state == "closed",
            "interaction_quality": "unknown"
        }
        
        # Analyze comments
        for comment in issue.get_comments():
            if "@claude" in comment.body.lower():
                interaction["human_responses"] += 1
            elif comment.user.login == "github-actions[bot]":
                # Likely a Claude response
                interaction["claude_responses"] += 1
                interaction["last_interaction"] = comment.created_at.isoformat()
        
        # Determine interaction quality
        if interaction["claude_responses"] > 0:
            if issue.state == "closed":
                interaction["interaction_quality"] = "successful"
            elif interaction["human_responses"] > interaction["claude_responses"]:
                interaction["interaction_quality"] = "needs_attention"
            else:
                interaction["interaction_quality"] = "in_progress"
        
        return interaction if interaction["claude_responses"] > 0 or interaction["human_responses"] > 0 else None
    
    def _determine_task_state(self, issue) -> str:
        """Determine the state of a task.
        
        Args:
            issue: GitHub issue
            
        Returns:
            Task state
        """
        labels = [label.name for label in issue.labels]
        
        # Check labels first
        if "completed" in labels or "done" in labels:
            return "completed"
        elif "in-progress" in labels or "working" in labels:
            return "in_progress"
        elif "awaiting-review" in labels or "review" in labels:
            return "awaiting_review"
        
        # Check staleness
        days_since_update = (datetime.now(timezone.utc) - issue.updated_at.replace(tzinfo=timezone.utc)).days
        if days_since_update > 7:
            return "stale"
        
        # Check for any activity
        if issue.comments > 0:
            return "in_progress"
        
        return "pending"
    
    def _is_claude_authored(self, pr) -> bool:
        """Check if a PR was authored by Claude.
        
        Args:
            pr: GitHub pull request
            
        Returns:
            True if Claude authored
        """
        # Check PR body for Claude signatures
        if pr.body and ("Generated by Claude" in pr.body or "@claude" in pr.body):
            return True
        
        # Check commit messages
        try:
            commits = pr.get_commits()
            for commit in commits:
                if "claude" in commit.commit.message.lower():
                    return True
        except:
            pass
        
        return False
    
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis.
        
        Args:
            analysis: Analysis data
            
        Returns:
            List of insights
        """
        insights = []
        
        # Task completion rate
        total = analysis["summary"]["total_issues"]
        if total > 0:
            completion_rate = analysis["summary"]["closed_issues"] / total
            insights.append(f"Task completion rate: {completion_rate:.1%}")
        
        # Claude interaction effectiveness
        if analysis["claude_interactions"]:
            successful = sum(1 for i in analysis["claude_interactions"] 
                           if i.get("interaction_quality") == "successful")
            total_interactions = len(analysis["claude_interactions"])
            success_rate = successful / total_interactions if total_interactions > 0 else 0
            insights.append(f"@claude interaction success rate: {success_rate:.1%}")
        
        # Stale task warning
        stale_count = len(analysis["task_states"]["stale"])
        if stale_count > 0:
            insights.append(f"⚠️ {stale_count} tasks are stale and need attention")
        
        # Workload distribution
        in_progress = len(analysis["task_states"]["in_progress"])
        awaiting_review = len(analysis["task_states"]["awaiting_review"])
        if in_progress > 5:
            insights.append(f"High workload: {in_progress} tasks currently in progress")
        if awaiting_review > 3:
            insights.append(f"Review bottleneck: {awaiting_review} tasks awaiting review")
        
        # Claude utilization
        claude_usage = analysis["summary"]["claude_mentions"] / total if total > 0 else 0
        if claude_usage < 0.3:
            insights.append("Low @claude utilization - consider assigning more tasks")
        elif claude_usage > 0.8:
            insights.append("High @claude utilization - system working effectively")
        
        return insights
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations from analysis.
        
        Args:
            analysis: Analysis data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Stale task handling
        if len(analysis["task_states"]["stale"]) > 0:
            recommendations.append("Review and close or update stale tasks")
        
        # Review bottleneck
        if len(analysis["task_states"]["awaiting_review"]) > 3:
            recommendations.append("Prioritize reviewing completed tasks")
        
        # Task generation
        open_count = analysis["summary"]["open_issues"]
        if open_count < 5:
            recommendations.append("Generate more tasks to maintain pipeline")
        elif open_count > 20:
            recommendations.append("Focus on completing existing tasks before generating new ones")
        
        # Claude interaction improvement
        failed_interactions = [i for i in analysis["claude_interactions"] 
                             if i.get("interaction_quality") == "needs_attention"]
        if failed_interactions:
            recommendations.append(f"Review {len(failed_interactions)} @claude interactions that need attention")
        
        # Task diversity
        task_states = analysis["task_states"]
        if not task_states["in_progress"] and task_states["pending"]:
            recommendations.append("Assign pending tasks to @claude to maintain momentum")
        
        return recommendations
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the analysis.
        
        Returns:
            Markdown formatted report
        """
        # Load the analysis
        if os.path.exists("task_analysis.json"):
            with open("task_analysis.json", "r") as f:
                analysis = json.load(f)
        else:
            analysis = self.analyze_all_tasks()
        
        report = f"""# Task Analysis Report

**Generated**: {analysis['timestamp']}

## Summary
- **Total Issues**: {analysis['summary']['total_issues']}
- **Open Issues**: {analysis['summary']['open_issues']}
- **Closed Issues**: {analysis['summary']['closed_issues']}
- **@claude Mentions**: {analysis['summary']['claude_mentions']}
- **Active PRs**: {analysis['summary']['active_prs']}

## Task States
- **Pending**: {len(analysis['task_states']['pending'])}
- **In Progress**: {len(analysis['task_states']['in_progress'])}
- **Awaiting Review**: {len(analysis['task_states']['awaiting_review'])}
- **Completed**: {len(analysis['task_states']['completed'])}
- **Stale**: {len(analysis['task_states']['stale'])}

## Insights
"""
        
        for insight in analysis['insights']:
            report += f"- {insight}\n"
        
        report += "\n## Recommendations\n"
        
        for recommendation in analysis['recommendations']:
            report += f"- {recommendation}\n"
        
        if analysis['claude_interactions']:
            report += "\n## Recent @claude Interactions\n"
            for interaction in analysis['claude_interactions'][:5]:
                report += f"- #{interaction.get('issue_number', 'N/A')}: {interaction.get('title', 'N/A')} "
                report += f"({interaction.get('interaction_quality', 'unknown')})\n"
        
        return report


def main():
    """Main function for standalone execution."""
    print("Starting task analysis...")
    
    analyzer = TaskAnalyzer()
    analysis = analyzer.analyze_all_tasks()
    
    print(f"\nAnalysis complete:")
    print(f"- Total issues analyzed: {analysis['summary']['total_issues']}")
    print(f"- @claude interactions found: {analysis['summary']['claude_mentions']}")
    
    # Generate and print summary report
    report = analyzer.generate_summary_report()
    print("\n" + report)
    
    # Save report
    with open("task_analysis_report.md", "w") as f:
        f.write(report)
    
    print("\nAnalysis saved to task_analysis.json and task_analysis_report.md")


if __name__ == "__main__":
    main()