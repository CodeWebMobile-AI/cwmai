"""
Create Report Module

Generates comprehensive performance reports and analytics for the autonomous AI system.
Tracks system performance, budget usage, project health, and learning metrics.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state_manager import StateManager


class ReportGenerator:
    """Generates comprehensive system reports."""
    
    def __init__(self, output_path: str = "performance_report.md"):
        """Initialize the report generator.
        
        Args:
            output_path: Path to save the generated report
        """
        self.output_path = output_path
        self.state_manager = StateManager()
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report.
        
        Returns:
            Generated report as markdown string
        """
        # Load current state and cycle report
        state = self.state_manager.load_state()
        cycle_report = self._load_cycle_report()
        
        # Generate report sections
        report_sections = [
            self._generate_header(),
            self._generate_executive_summary(state, cycle_report),
            self._generate_current_cycle_summary(cycle_report),
            self._generate_system_performance(state),
            self._generate_budget_analysis(state),
            self._generate_portfolio_health(state),
            self._generate_learning_metrics(state),
            self._generate_recent_activity(state),
            self._generate_recommendations(state),
            self._generate_footer()
        ]
        
        report_content = "\n\n".join(report_sections)
        
        # Save report to file
        self._save_report(report_content)
        
        return report_content
    
    def _load_cycle_report(self) -> Dict[str, Any]:
        """Load the current cycle report data.
        
        Returns:
            Cycle report dictionary or empty dict if not available
        """
        try:
            if os.path.exists("cycle_report.json"):
                with open("cycle_report.json", 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cycle report: {e}")
        
        return {}
    
    def _generate_header(self) -> str:
        """Generate report header."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return f"""# Autonomous AI Software Development System Report

**Generated:** {timestamp}  
**System Version:** 1.0.0  
**Report Type:** Performance Analysis & Metrics

---"""
    
    def _generate_executive_summary(self, state: Dict[str, Any], cycle_report: Dict[str, Any]) -> str:
        """Generate executive summary section.
        
        Args:
            state: System state
            cycle_report: Current cycle report
            
        Returns:
            Executive summary markdown
        """
        performance = state.get("system_performance", {})
        total_cycles = performance.get("total_cycles", 0)
        successful_actions = performance.get("successful_actions", 0)
        failed_actions = performance.get("failed_actions", 0)
        
        success_rate = 0
        if successful_actions + failed_actions > 0:
            success_rate = successful_actions / (successful_actions + failed_actions) * 100
        
        projects = state.get("projects", {})
        avg_health = 0
        if projects:
            avg_health = sum(p.get("health_score", 50) for p in projects.values()) / len(projects)
        
        budget = state.get("api_budget", {})
        budget_used = budget.get("monthly_usage_usd", 0)
        budget_limit = budget.get("monthly_limit_usd", 100)
        budget_percentage = (budget_used / budget_limit) * 100 if budget_limit > 0 else 0
        
        return f"""## Executive Summary

The Autonomous AI Software Development System has completed **{total_cycles}** operational cycles with a **{success_rate:.1f}%** success rate.

### Key Metrics
- **Portfolio Health:** {avg_health:.1f}/100 across {len(projects)} projects
- **Budget Utilization:** ${budget_used:.2f} of ${budget_limit:.2f} ({budget_percentage:.1f}% used)
- **Success Rate:** {successful_actions}/{successful_actions + failed_actions} actions successful
- **System Status:** {"🟢 Operational" if success_rate >= 70 else "🟡 Monitoring" if success_rate >= 50 else "🔴 Attention Required"}"""
    
    def _generate_current_cycle_summary(self, cycle_report: Dict[str, Any]) -> str:
        """Generate current cycle summary.
        
        Args:
            cycle_report: Current cycle report data
            
        Returns:
            Current cycle summary markdown
        """
        if not cycle_report:
            return "## Current Cycle Summary\n\nNo cycle data available."
        
        action = cycle_report.get("action_taken", "Unknown")
        outcome = cycle_report.get("outcome", "Unknown")
        duration = cycle_report.get("duration_seconds", 0)
        cycle_num = cycle_report.get("cycle_number", "Unknown")
        
        # Format outcome with emoji
        outcome_emoji = {
            "success_completed": "✅",
            "success_merged": "✅",
            "success_forced": "✅",
            "success_budget_conservation": "💰",
            "failure_error": "❌",
            "failure_forced": "❌"
        }
        
        outcome_display = f"{outcome_emoji.get(outcome, '❓')} {outcome.replace('_', ' ').title()}"
        
        return f"""## Current Cycle Summary

**Cycle #{cycle_num}** - Completed in {duration:.1f} seconds

- **Action Taken:** `{action}`
- **Outcome:** {outcome_display}
- **Portfolio Health:** {cycle_report.get('portfolio_health', 'N/A')}
- **Budget Impact:** ${cycle_report.get('budget_used', 0):.2f}"""
    
    def _generate_system_performance(self, state: Dict[str, Any]) -> str:
        """Generate system performance analysis.
        
        Args:
            state: System state
            
        Returns:
            System performance markdown
        """
        performance = state.get("system_performance", {})
        learning_metrics = performance.get("learning_metrics", {})
        
        total_cycles = performance.get("total_cycles", 0)
        successful_actions = performance.get("successful_actions", 0)
        failed_actions = performance.get("failed_actions", 0)
        
        decision_accuracy = learning_metrics.get("decision_accuracy", 0)
        resource_efficiency = learning_metrics.get("resource_efficiency", 0)
        goal_achievement = learning_metrics.get("goal_achievement", 0)
        
        # Calculate trends (simplified)
        trend_indicators = {
            "Decision Accuracy": self._get_trend_indicator(decision_accuracy, 0.7),
            "Resource Efficiency": self._get_trend_indicator(resource_efficiency, 1.0),
            "Goal Achievement": self._get_trend_indicator(goal_achievement, 0.8)
        }
        
        return f"""## System Performance Analysis

### Overall Statistics
- **Total Cycles:** {total_cycles}
- **Successful Actions:** {successful_actions}
- **Failed Actions:** {failed_actions}
- **Overall Success Rate:** {(successful_actions / max(successful_actions + failed_actions, 1)) * 100:.1f}%

### Learning Metrics
| Metric | Value | Target | Status |
|--------|-------|---------|--------|
| Decision Accuracy | {decision_accuracy:.3f} | 0.700 | {trend_indicators["Decision Accuracy"]} |
| Resource Efficiency | {resource_efficiency:.3f} | 1.000 | {trend_indicators["Resource Efficiency"]} |
| Goal Achievement | {goal_achievement:.3f} | 0.800 | {trend_indicators["Goal Achievement"]} |"""
    
    def _get_trend_indicator(self, value: float, target: float) -> str:
        """Get trend indicator emoji based on value vs target.
        
        Args:
            value: Current value
            target: Target value
            
        Returns:
            Trend indicator emoji
        """
        if value >= target:
            return "🟢 On Target"
        elif value >= target * 0.8:
            return "🟡 Near Target"
        else:
            return "🔴 Below Target"
    
    def _generate_budget_analysis(self, state: Dict[str, Any]) -> str:
        """Generate budget analysis section.
        
        Args:
            state: System state
            
        Returns:
            Budget analysis markdown
        """
        budget = state.get("api_budget", {})
        monthly_limit = budget.get("monthly_limit_usd", 100)
        monthly_usage = budget.get("monthly_usage_usd", 0)
        remaining = monthly_limit - monthly_usage
        usage_percentage = (monthly_usage / monthly_limit) * 100 if monthly_limit > 0 else 0
        
        # Determine budget status
        if usage_percentage < 50:
            status = "🟢 Healthy"
        elif usage_percentage < 80:
            status = "🟡 Monitoring"
        else:
            status = "🔴 Critical"
        
        # Calculate daily burn rate (simplified)
        performance = state.get("system_performance", {})
        total_cycles = performance.get("total_cycles", 1)
        daily_burn = (monthly_usage / max(total_cycles, 1)) * 6  # Assuming 4-hour cycles
        
        return f"""## Budget Analysis

### Current Status: {status}

- **Monthly Limit:** ${monthly_limit:.2f}
- **Used This Month:** ${monthly_usage:.2f} ({usage_percentage:.1f}%)
- **Remaining Budget:** ${remaining:.2f}
- **Estimated Daily Burn:** ${daily_burn:.2f}

### Budget Breakdown
```
Used:     ${"█" * int(usage_percentage / 5)}{" " * (20 - int(usage_percentage / 5))} {usage_percentage:.1f}%
Remaining: ${"█" * int((100 - usage_percentage) / 5)}{" " * (20 - int((100 - usage_percentage) / 5))} {100 - usage_percentage:.1f}%
```

### Recommendations
{self._get_budget_recommendations(usage_percentage, remaining)}"""
    
    def _get_budget_recommendations(self, usage_percentage: float, remaining: float) -> str:
        """Generate budget recommendations.
        
        Args:
            usage_percentage: Current usage percentage
            remaining: Remaining budget amount
            
        Returns:
            Budget recommendations text
        """
        if usage_percentage < 30:
            return "- Budget utilization is low - consider more aggressive project development"
        elif usage_percentage < 60:
            return "- Budget utilization is healthy - maintain current activity levels"
        elif usage_percentage < 85:
            return "- Budget utilization is high - monitor spending and prioritize essential actions"
        else:
            return "- Budget utilization is critical - activate conservation mode and focus on free actions"
    
    def _generate_portfolio_health(self, state: Dict[str, Any]) -> str:
        """Generate portfolio health analysis.
        
        Args:
            state: System state
            
        Returns:
            Portfolio health markdown
        """
        projects = state.get("projects", {})
        
        if not projects:
            return """## Portfolio Health

No projects in portfolio."""
        
        # Calculate health statistics
        health_scores = [p.get("health_score", 50) for p in projects.values()]
        avg_health = sum(health_scores) / len(health_scores)
        min_health = min(health_scores)
        max_health = max(health_scores)
        
        # Categorize projects
        healthy = len([h for h in health_scores if h >= 80])
        moderate = len([h for h in health_scores if 60 <= h < 80])
        unhealthy = len([h for h in health_scores if h < 60])
        
        # Generate project table
        project_table = "| Project | Health Score | Last Checked | Actions |\n"
        project_table += "|---------|--------------|--------------|----------|\n"
        
        for name, project in projects.items():
            health = project.get("health_score", 50)
            last_checked = project.get("last_checked", "Never")
            action_count = len(project.get("action_history", []))
            
            # Format last checked
            if last_checked != "Never":
                try:
                    dt = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
                    last_checked = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            project_table += f"| {name} | {health}/100 | {last_checked} | {action_count} |\n"
        
        return f"""## Portfolio Health

### Overview
- **Total Projects:** {len(projects)}
- **Average Health:** {avg_health:.1f}/100
- **Health Range:** {min_health} - {max_health}

### Health Distribution
- 🟢 **Healthy (80+):** {healthy} projects
- 🟡 **Moderate (60-79):** {moderate} projects  
- 🔴 **Unhealthy (<60):** {unhealthy} projects

### Project Details
{project_table}"""
    
    def _generate_learning_metrics(self, state: Dict[str, Any]) -> str:
        """Generate learning metrics analysis.
        
        Args:
            state: System state
            
        Returns:
            Learning metrics markdown
        """
        # Analyze decision history if available
        decision_history = state.get("decision_history", [])
        
        if not decision_history:
            return """## Learning & Adaptation

No decision history available for analysis."""
        
        # Analyze recent decisions
        recent_decisions = decision_history[-10:] if len(decision_history) > 10 else decision_history
        
        # Count action types
        action_counts = {}
        for decision in recent_decisions:
            action = decision.get("chosen_action", "Unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Generate action frequency table
        action_table = "| Action Type | Frequency | Percentage |\n"
        action_table += "|-------------|-----------|------------|\n"
        
        total_decisions = len(recent_decisions)
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_decisions) * 100
            action_table += f"| {action} | {count} | {percentage:.1f}% |\n"
        
        return f"""## Learning & Adaptation

### Recent Decision Analysis
Analyzing last {len(recent_decisions)} decisions:

{action_table}

### Learning Insights
{self._generate_learning_insights(recent_decisions)}"""
    
    def _generate_learning_insights(self, decisions: List[Dict[str, Any]]) -> str:
        """Generate learning insights from decision history.
        
        Args:
            decisions: List of recent decisions
            
        Returns:
            Learning insights text
        """
        if not decisions:
            return "No decisions to analyze."
        
        insights = []
        
        # Analyze budget-related decisions
        budget_decisions = [d for d in decisions if "budget" in d.get("context_summary", {}).get("budget_remaining", 0)]
        if budget_decisions:
            insights.append("- System is actively considering budget constraints in decision-making")
        
        # Analyze action diversity
        unique_actions = len(set(d.get("chosen_action") for d in decisions))
        if unique_actions == 1:
            insights.append("- Low action diversity - system may be stuck in a pattern")
        elif unique_actions >= len(decisions) * 0.7:
            insights.append("- High action diversity - system is exploring different strategies")
        else:
            insights.append("- Moderate action diversity - system is balancing exploration and exploitation")
        
        # Analyze external factors
        external_factors = [d.get("context_summary", {}).get("external_factors", 0) for d in decisions]
        avg_external = sum(external_factors) / len(external_factors)
        if avg_external > 3:
            insights.append("- System is actively incorporating external context into decisions")
        
        return "\n".join(insights) if insights else "- No significant patterns identified in recent decisions"
    
    def _generate_recent_activity(self, state: Dict[str, Any]) -> str:
        """Generate recent activity summary.
        
        Args:
            state: System state
            
        Returns:
            Recent activity markdown
        """
        # Collect recent actions from all projects
        all_actions = []
        projects = state.get("projects", {})
        
        for project_name, project in projects.items():
            for action in project.get("action_history", []):
                action_copy = action.copy()
                action_copy["project"] = project_name
                all_actions.append(action_copy)
        
        # Sort by timestamp (most recent first)
        all_actions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Take last 10 actions
        recent_actions = all_actions[:10]
        
        if not recent_actions:
            return """## Recent Activity

No recent activity recorded."""
        
        activity_table = "| Timestamp | Project | Action | Outcome |\n"
        activity_table += "|-----------|---------|--------|---------|\n"
        
        for action in recent_actions:
            timestamp = action.get("timestamp", "Unknown")
            project = action.get("project", "Unknown")
            action_type = action.get("action", "Unknown")
            outcome = action.get("outcome", "Unknown")
            
            # Format timestamp
            if timestamp != "Unknown":
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime("%m-%d %H:%M")
                except:
                    pass
            
            # Format outcome with emoji
            outcome_emoji = "✅" if outcome.startswith("success") else "❌"
            outcome_display = f"{outcome_emoji} {outcome.replace('_', ' ')}"
            
            activity_table += f"| {timestamp} | {project} | {action_type} | {outcome_display} |\n"
        
        return f"""## Recent Activity

### Last {len(recent_actions)} Actions

{activity_table}"""
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> str:
        """Generate system recommendations.
        
        Args:
            state: System state
            
        Returns:
            Recommendations markdown
        """
        recommendations = []
        
        # Budget recommendations
        budget = state.get("api_budget", {})
        usage_pct = (budget.get("monthly_usage_usd", 0) / budget.get("monthly_limit_usd", 100)) * 100
        
        if usage_pct > 90:
            recommendations.append("🔴 **Critical:** Implement immediate budget conservation measures")
        elif usage_pct > 75:
            recommendations.append("🟡 **Warning:** Monitor budget usage closely, consider reducing expensive operations")
        
        # Portfolio health recommendations
        projects = state.get("projects", {})
        if projects:
            health_scores = [p.get("health_score", 50) for p in projects.values()]
            avg_health = sum(health_scores) / len(health_scores)
            
            if avg_health < 60:
                recommendations.append("🔴 **Portfolio:** Focus on fixing critical issues in existing projects")
            elif avg_health > 85:
                recommendations.append("🟢 **Portfolio:** Excellent health - consider expanding with new projects")
        
        # Performance recommendations
        performance = state.get("system_performance", {})
        total_actions = performance.get("successful_actions", 0) + performance.get("failed_actions", 0)
        success_rate = (performance.get("successful_actions", 0) / max(total_actions, 1))
        
        if success_rate < 0.5:
            recommendations.append("🔴 **Performance:** Low success rate - review decision-making algorithms")
        elif success_rate > 0.8:
            recommendations.append("🟢 **Performance:** High success rate - system is performing well")
        
        # Learning recommendations
        learning_metrics = performance.get("learning_metrics", {})
        decision_accuracy = learning_metrics.get("decision_accuracy", 0)
        
        if decision_accuracy < 0.6:
            recommendations.append("🟡 **Learning:** Consider adjusting decision-making weights based on outcomes")
        
        if not recommendations:
            recommendations.append("🟢 **Status:** System is operating within normal parameters")
        
        return f"""## Recommendations

{chr(10).join(recommendations)}

### Next Actions
- Continue monitoring system performance
- Review and adjust charter goals if needed
- Ensure adequate budget allocation for planned activities
- Consider expanding portfolio if resources allow"""
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""---

**Report Generated by:** Autonomous AI Software Development System  
**System Version:** 1.0.0  
**Timestamp:** {datetime.now(timezone.utc).isoformat()}

*This report is automatically generated based on system state and performance metrics.*"""
    
    def _save_report(self, content: str) -> None:
        """Save report to file.
        
        Args:
            content: Report content to save
        """
        try:
            with open(self.output_path, 'w') as f:
                f.write(content)
            print(f"Report saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")


def main():
    """Main function for standalone execution."""
    print("Generating performance report...")
    
    generator = ReportGenerator()
    report_content = generator.generate_report()
    
    print(f"Report generated: {len(report_content)} characters")
    print(f"Saved to: {generator.output_path}")


if __name__ == "__main__":
    main()