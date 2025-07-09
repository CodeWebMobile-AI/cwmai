"""
Research Need Analyzer - Identifies system performance gaps and knowledge needs.

This module continuously monitors CWMAI's performance to identify areas where
research could improve outcomes, focusing on the core needs: task generation,
multi-agent coordination, outcome learning, and portfolio management.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics


class ResearchNeedAnalyzer:
    """Continuously analyze system performance to identify knowledge gaps."""
    
    def __init__(self, state_manager=None):
        self.state_manager = state_manager
        self.analysis_history = []
        self.gap_priorities = {}
        
        # Performance thresholds - made VERY aggressive for proactive research
        self.thresholds = {
            "claude_success_min": 0.95,  # Research when success drops below 95%
            "task_completion_min": 0.85,  # Research when completion drops below 85%
            "task_quality_min": 0.9,  # Research when quality drops below 90%
            "agent_consensus_min": 0.9,  # Research when consensus drops below 90%
            "portfolio_health_min": 0.8,  # Research when health drops below 80%
            # New proactive thresholds - very aggressive
            "learning_opportunity_threshold": 0.3,  # Research on any learning opportunity
            "new_pattern_threshold": 0.4,  # Research when any new pattern emerges
            "improvement_rate_min": 0.2,  # Research if improvement rate is below 20%
            "proactive_trigger": True,  # Always look for research opportunities
            "zero_activity_trigger": True,  # Research when no recent activity
        }
        
        # Gap detection patterns
        self.gap_patterns = {
            "claude_interaction_failure": {
                "indicators": ["claude_success_rate", "task_rejection_rate"],
                "research_areas": ["prompt_engineering", "task_clarity", "acceptance_criteria"]
            },
            "task_quality_issues": {
                "indicators": ["task_completion_rate", "task_quality_score"],
                "research_areas": ["task_decomposition", "complexity_management", "success_criteria"]
            },
            "agent_coordination_problems": {
                "indicators": ["agent_consensus_rate", "decision_time"],
                "research_areas": ["consensus_mechanisms", "agent_specialization", "voting_strategies"]
            },
            "learning_ineffectiveness": {
                "indicators": ["improvement_rate", "pattern_recognition_accuracy"],
                "research_areas": ["outcome_patterns", "feedback_loops", "value_metrics"]
            },
            "portfolio_imbalance": {
                "indicators": ["project_diversity", "resource_allocation_efficiency"],
                "research_areas": ["project_selection", "synergy_identification", "resource_models"]
            }
        }
        
    def analyze_performance_gaps(self) -> Dict:
        """
        Identify what's holding the system back with real performance metrics.
        
        Returns:
            Dictionary of prioritized gaps with research recommendations
        """
        gaps = {}
        
        # Get real performance metrics from system state
        real_metrics = self._get_real_performance_metrics()
        
        # Analyze each core area with real metrics
        gaps["claude_interaction"] = self._analyze_claude_failures(real_metrics)
        gaps["task_generation"] = self._analyze_task_quality(real_metrics)
        gaps["multi_agent"] = self._analyze_coordination_issues(real_metrics)
        gaps["outcome_learning"] = self._analyze_learning_effectiveness(real_metrics)
        gaps["portfolio_health"] = self._analyze_project_selection(real_metrics)
        
        # Prioritize gaps by impact
        prioritized_gaps = self._prioritize_gaps(gaps)
        
        # Store analysis for learning
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "gaps": gaps,
            "priorities": prioritized_gaps,
            "real_metrics": real_metrics
        })
        
        # Keep only recent history
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
        
        return prioritized_gaps
    
    def _get_real_performance_metrics(self) -> Dict:
        """Extract real performance metrics from system state.
        
        Returns:
            Performance metrics dictionary with actual system data
        """
        metrics = {
            'claude_success_rate': 0.0,
            'task_completion_rate': 0.0,
            'ai_provider_success': {},
            'recent_errors': [],
            'system_health': 'unknown',
            'performance_trend': 'unknown'
        }
        
        if not self.state_manager:
            return metrics
            
        try:
            state = self.state_manager.load_state()
            
            # Extract Claude interaction success from performance data
            performance = state.get('performance', {})
            if 'claude_interactions' in performance:
                claude_data = performance['claude_interactions']
                total_attempts = claude_data.get('total_attempts', 0)
                successful = claude_data.get('successful', 0)
                if total_attempts > 0:
                    metrics['claude_success_rate'] = (successful / total_attempts) * 100
                else:
                    # Check for any Claude mention attempts
                    task_state = state.get('task_state', {})
                    tasks = task_state.get('tasks', {})
                    if isinstance(tasks, dict):
                        tasks = list(tasks.values())
                    
                    claude_tasks = [t for t in tasks if '@claude' in str(t.get('description', ''))]
                    metrics['claude_success_rate'] = 0.0 if claude_tasks else None
            
            # Extract task completion rates
            if 'task_completion' in performance:
                task_data = performance['task_completion']
                total_tasks = task_data.get('total_tasks', 0)
                completed_tasks = task_data.get('completed_tasks', 0)
                if total_tasks > 0:
                    metrics['task_completion_rate'] = (completed_tasks / total_tasks) * 100
            else:
                # Fallback: calculate from task state
                task_state = state.get('task_state', {})
                tasks = task_state.get('tasks', {})
                if isinstance(tasks, dict):
                    tasks = list(tasks.values())
                
                if tasks:
                    completed = len([t for t in tasks if t.get('status') == 'completed'])
                    metrics['task_completion_rate'] = (completed / len(tasks)) * 100
            
            # Extract AI provider performance
            if 'ai_providers' in performance:
                metrics['ai_provider_success'] = performance['ai_providers']
            
            # Extract recent errors
            metrics['recent_errors'] = state.get('recent_errors', [])[-10:]
            
            # Calculate system health score
            health_score = self._calculate_system_health_score(metrics)
            metrics['system_health'] = health_score
            
            # Determine performance trend
            metrics['performance_trend'] = self._analyze_performance_trend()
            
        except Exception as e:
            print(f"Error extracting performance metrics: {e}")
            
        return metrics
    
    def _calculate_system_health_score(self, metrics: Dict) -> str:
        """Calculate overall system health based on metrics."""
        score = 100
        
        # Major penalty for Claude failures (most critical)
        claude_rate = metrics.get('claude_success_rate', 0)
        if claude_rate is not None:
            if claude_rate == 0:
                score -= 50  # Severe penalty for 0% Claude success
            elif claude_rate < 20:
                score -= 40  # Major penalty for very low success
            elif claude_rate < 50:
                score -= 25  # Moderate penalty for low success
        
        # Major penalty for task failures
        task_rate = metrics.get('task_completion_rate', 0)
        if task_rate < 10:
            score -= 30  # Major penalty for very low completion
        elif task_rate < 50:
            score -= 20  # Moderate penalty for low completion
        
        # Penalty for frequent errors
        error_count = len(metrics.get('recent_errors', []))
        if error_count > 5:
            score -= 15
        elif error_count > 2:
            score -= 10
        
        # Return health category
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        elif score >= 20:
            return 'poor'
        else:
            return 'critical'
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend from recent analysis history."""
        if len(self.analysis_history) < 2:
            return 'unknown'
        
        # Compare recent metrics
        recent = self.analysis_history[-1].get('real_metrics', {})
        previous = self.analysis_history[-2].get('real_metrics', {})
        
        recent_claude = recent.get('claude_success_rate', 0)
        previous_claude = previous.get('claude_success_rate', 0)
        
        recent_task = recent.get('task_completion_rate', 0)
        previous_task = previous.get('task_completion_rate', 0)
        
        # Simple trend analysis
        claude_improving = recent_claude > previous_claude
        task_improving = recent_task > previous_task
        
        if claude_improving and task_improving:
            return 'improving'
        elif not claude_improving and not task_improving:
            return 'declining'
        else:
            return 'mixed'
    
    def _analyze_task_quality(self, real_metrics: Dict = None) -> Dict:
        """Analyze task generation quality and identify issues."""
        analysis = {
            "severity": "low",
            "issues": [],
            "metrics": {},
            "research_needs": []
        }
        
        if self.state_manager:
            state = self.state_manager.load_state()
            task_state = state.get("task_state", {})
            
            # Calculate metrics
            total_tasks = len(task_state.get("tasks", []))
            completed_tasks = len([t for t in task_state.get("tasks", []) 
                                 if t.get("status") == "completed"])
            failed_tasks = len([t for t in task_state.get("tasks", []) 
                             if t.get("status") == "failed"])
            
            completion_rate = completed_tasks / max(1, total_tasks)
            failure_rate = failed_tasks / max(1, total_tasks)
            
            analysis["metrics"] = {
                "total_tasks": total_tasks,
                "completion_rate": completion_rate,
                "failure_rate": failure_rate,
                "average_task_age": self._calculate_average_task_age(task_state.get("tasks", []))
            }
            
            # Identify issues
            if completion_rate < 0.1:  # Less than 10% completion
                analysis["severity"] = "critical"
                analysis["issues"].append("Extremely low task completion rate")
                analysis["research_needs"].extend([
                    "Task decomposition strategies",
                    "Complexity scoring models",
                    "Task feasibility assessment"
                ])
            
            if failure_rate > 0.5:  # More than 50% failure
                analysis["severity"] = "high"
                analysis["issues"].append("High task failure rate")
                analysis["research_needs"].extend([
                    "Failure pattern analysis",
                    "Task validation techniques",
                    "Success criteria definition"
                ])
            
            # Analyze task descriptions
            task_quality_issues = self._analyze_task_descriptions(task_state.get("tasks", []))
            if task_quality_issues:
                analysis["issues"].extend(task_quality_issues)
                analysis["research_needs"].append("Task description best practices")
        
        return analysis
    
    def _analyze_claude_failures(self, real_metrics: Dict) -> Dict:
        """Analyze why Claude interactions are failing using real performance data."""
        analysis = {
            "severity": "low",
            "issues": [],
            "metrics": {},
            "research_needs": []
        }
        
        # Use real metrics from system state
        claude_success_rate = real_metrics.get('claude_success_rate', 0)
        recent_errors = real_metrics.get('recent_errors', [])
        system_health = real_metrics.get('system_health', 'unknown')
        
        analysis["metrics"] = {
            "success_rate": claude_success_rate,
            "system_health": system_health,
            "recent_errors": len(recent_errors),
            "performance_trend": real_metrics.get('performance_trend', 'unknown')
        }
        
        # CRITICAL: 0% Claude success rate (most severe issue)
        if claude_success_rate == 0:
            analysis["severity"] = "critical"
            analysis["issues"].append("CRITICAL: Zero Claude interaction success rate - System cannot communicate with Claude AI")
            analysis["research_needs"].extend([
                "Claude API authentication and configuration troubleshooting",
                "GitHub issue formatting optimization for Claude",
                "AI task prompt engineering for maximum success",
                "Claude mention and response pattern analysis",
                "Alternative AI communication strategies"
            ])
            
            # Add specific failure patterns based on errors
            claude_error_patterns = self._extract_claude_error_patterns(recent_errors)
            analysis["issues"].extend(claude_error_patterns)
            
        elif claude_success_rate < 10:  # Less than 10% success - Critical
            analysis["severity"] = "critical"
            analysis["issues"].append(f"CRITICAL: Extremely low Claude success rate: {claude_success_rate:.1f}%")
            analysis["research_needs"].extend([
                "Claude interaction debugging and optimization",
                "Prompt structure and formatting improvements",
                "API response handling enhancement"
            ])
            
        elif claude_success_rate < 30:  # Less than 30% success - High priority
            analysis["severity"] = "high"
            analysis["issues"].append(f"HIGH: Low Claude interaction success rate: {claude_success_rate:.1f}%")
            analysis["research_needs"].extend([
                "Prompt clarity and specificity improvements",
                "Task context optimization for Claude",
                "Success criteria specification enhancement"
            ])
            
        elif claude_success_rate < 70:  # Less than 70% success - Medium priority
            analysis["severity"] = "medium"
            analysis["issues"].append(f"MEDIUM: Claude success rate needs improvement: {claude_success_rate:.1f}%")
            analysis["research_needs"].extend([
                "Fine-tuning Claude interaction patterns",
                "Response format optimization"
            ])
        
        # Add system health context
        if system_health == 'critical':
            analysis["issues"].append("Overall system health is critical - impacts Claude performance")
        
        return analysis
    
    def _extract_claude_error_patterns(self, recent_errors: List) -> List[str]:
        """Extract Claude-specific error patterns from recent errors."""
        patterns = []
        
        # Analyze error messages for Claude-related issues
        claude_errors = [error for error in recent_errors 
                        if 'claude' in str(error).lower() or 'anthropic' in str(error).lower()]
        
        if len(claude_errors) > 3:
            patterns.append("Frequent Claude API errors detected")
        
        # Look for authentication errors
        auth_errors = [error for error in recent_errors 
                      if any(keyword in str(error).lower() 
                            for keyword in ['auth', 'token', 'key', 'permission'])]
        
        if auth_errors:
            patterns.append("Authentication/API key issues detected")
        
        # Look for formatting errors
        format_errors = [error for error in recent_errors 
                        if any(keyword in str(error).lower() 
                              for keyword in ['format', 'json', 'parse', 'invalid'])]
        
        if format_errors:
            patterns.append("Request/response formatting issues detected")
        
        return patterns
    
    def _analyze_coordination_issues(self, real_metrics: Dict = None) -> Dict:
        """Analyze multi-agent coordination effectiveness."""
        analysis = {
            "severity": "low",
            "issues": [],
            "metrics": {},
            "research_needs": []
        }
        
        if self.state_manager:
            state = self.state_manager.load_state()
            swarm_state = state.get("swarm_state", {})
            
            # Calculate coordination metrics
            decisions = swarm_state.get("recent_decisions", [])
            if decisions:
                consensus_rates = [d.get("consensus_level", 0) for d in decisions]
                avg_consensus = statistics.mean(consensus_rates) if consensus_rates else 0
                decision_times = [d.get("decision_time", 0) for d in decisions]
                avg_decision_time = statistics.mean(decision_times) if decision_times else 0
                
                analysis["metrics"] = {
                    "average_consensus": avg_consensus,
                    "average_decision_time": avg_decision_time,
                    "total_decisions": len(decisions),
                    "disagreement_rate": len([d for d in decisions if d.get("consensus_level", 0) < 0.7]) / len(decisions)
                }
                
                # Identify coordination issues
                if avg_consensus < 0.7:
                    analysis["severity"] = "medium"
                    analysis["issues"].append("Low agent consensus levels")
                    analysis["research_needs"].extend([
                        "Consensus building mechanisms",
                        "Agent specialization strategies",
                        "Weighted voting systems"
                    ])
                
                if avg_decision_time > 300:  # More than 5 minutes
                    analysis["severity"] = "medium"
                    analysis["issues"].append("Slow decision-making process")
                    analysis["research_needs"].extend([
                        "Efficient consensus algorithms",
                        "Parallel decision strategies",
                        "Agent communication optimization"
                    ])
        
        return analysis
    
    def _analyze_learning_effectiveness(self, real_metrics: Dict = None) -> Dict:
        """Analyze how well the system is learning from outcomes."""
        analysis = {
            "severity": "low",
            "issues": [],
            "metrics": {},
            "research_needs": []
        }
        
        if self.state_manager:
            state = self.state_manager.load_state()
            learning_state = state.get("outcome_learning", {})
            
            # Calculate learning metrics
            learned_patterns = learning_state.get("learned_patterns", [])
            successful_predictions = learning_state.get("successful_predictions", 0)
            total_predictions = learning_state.get("total_predictions", 0)
            
            prediction_accuracy = successful_predictions / max(1, total_predictions)
            
            analysis["metrics"] = {
                "pattern_count": len(learned_patterns),
                "prediction_accuracy": prediction_accuracy,
                "learning_rate": learning_state.get("learning_rate", 0),
                "value_improvement": learning_state.get("value_improvement_rate", 0)
            }
            
            # Identify learning issues
            if len(learned_patterns) < 10:
                analysis["severity"] = "high"
                analysis["issues"].append("Insufficient learned patterns")
                analysis["research_needs"].extend([
                    "Pattern recognition techniques",
                    "Outcome correlation analysis",
                    "Learning algorithm optimization"
                ])
            
            if prediction_accuracy < 0.5:
                analysis["severity"] = "medium"
                analysis["issues"].append("Poor prediction accuracy")
                analysis["research_needs"].extend([
                    "Predictive model improvements",
                    "Feature engineering for outcomes",
                    "Feedback loop optimization"
                ])
        
        return analysis
    
    def _analyze_project_selection(self, real_metrics: Dict = None) -> Dict:
        """Analyze portfolio management and project selection."""
        analysis = {
            "severity": "low",
            "issues": [],
            "metrics": {},
            "research_needs": []
        }
        
        if self.state_manager:
            state = self.state_manager.load_state()
            portfolio = state.get("portfolio", {})
            projects = portfolio.get("projects", [])
            
            # Calculate portfolio metrics
            active_projects = [p for p in projects if p.get("status") == "active"]
            project_diversity = self._calculate_project_diversity(projects)
            resource_efficiency = self._calculate_resource_efficiency(projects)
            
            analysis["metrics"] = {
                "total_projects": len(projects),
                "active_projects": len(active_projects),
                "project_diversity": project_diversity,
                "resource_efficiency": resource_efficiency,
                "average_project_health": self._calculate_average_project_health(projects)
            }
            
            # Identify portfolio issues
            if project_diversity < 0.3:
                analysis["severity"] = "medium"
                analysis["issues"].append("Low project diversity")
                analysis["research_needs"].extend([
                    "Portfolio diversification strategies",
                    "Project selection criteria",
                    "Market opportunity analysis"
                ])
            
            if resource_efficiency < 0.5:
                analysis["severity"] = "high"
                analysis["issues"].append("Poor resource allocation")
                analysis["research_needs"].extend([
                    "Resource optimization models",
                    "Project prioritization frameworks",
                    "Cross-project synergy identification"
                ])
        
        return analysis
    
    def _prioritize_gaps(self, gaps: Dict) -> Dict:
        """
        Prioritize gaps by potential impact on system performance.
        
        Args:
            gaps: Dictionary of analyzed gaps
            
        Returns:
            Prioritized gaps with recommended actions
        """
        prioritized = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Score each gap
        gap_scores = []
        for area, analysis in gaps.items():
            score = self._calculate_gap_score(area, analysis)
            gap_scores.append((score, area, analysis))
        
        # Sort by score (highest first)
        gap_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Categorize by priority
        for score, area, analysis in gap_scores:
            priority_item = {
                "area": area,
                "score": score,
                "severity": analysis["severity"],
                "issues": analysis["issues"],
                "research_needs": analysis["research_needs"],
                "expected_impact": self._estimate_impact(area, analysis)
            }
            
            if analysis["severity"] == "critical" or score > 0.8:
                prioritized["critical"].append(priority_item)
            elif analysis["severity"] == "high" or score > 0.6:
                prioritized["high"].append(priority_item)
            elif analysis["severity"] == "medium" or score > 0.4:
                prioritized["medium"].append(priority_item)
            else:
                prioritized["low"].append(priority_item)
        
        return prioritized
    
    def _calculate_gap_score(self, area: str, analysis: Dict) -> float:
        """Calculate priority score for a gap."""
        base_scores = {
            "claude_interaction": 0.9,  # Critical for task execution
            "task_generation": 0.8,  # Core functionality
            "outcome_learning": 0.7,  # Essential for improvement
            "multi_agent": 0.6,  # Important for quality
            "portfolio_health": 0.5  # Strategic importance
        }
        
        severity_multipliers = {
            "critical": 1.5,
            "high": 1.2,
            "medium": 1.0,
            "low": 0.8
        }
        
        base_score = base_scores.get(area, 0.5)
        severity_mult = severity_multipliers.get(analysis["severity"], 1.0)
        
        # Adjust for issue count
        issue_penalty = min(0.3, len(analysis["issues"]) * 0.1)
        
        return base_score * severity_mult + issue_penalty
    
    def _estimate_impact(self, area: str, analysis: Dict) -> str:
        """Estimate the potential impact of addressing this gap."""
        impact_templates = {
            "claude_interaction": "Improve Claude success rate from {}% to {}%+",
            "task_generation": "Increase task completion from {}% to {}%+",
            "outcome_learning": "Enhance prediction accuracy from {}% to {}%+",
            "multi_agent": "Boost consensus rate from {}% to {}%+",
            "portfolio_health": "Improve resource efficiency from {}% to {}%+"
        }
        
        # Get current metric
        current_value = 0
        if "success_rate" in analysis.get("metrics", {}):
            current_value = analysis["metrics"]["success_rate"] * 100
        elif "completion_rate" in analysis.get("metrics", {}):
            current_value = analysis["metrics"]["completion_rate"] * 100
        elif "prediction_accuracy" in analysis.get("metrics", {}):
            current_value = analysis["metrics"]["prediction_accuracy"] * 100
        elif "average_consensus" in analysis.get("metrics", {}):
            current_value = analysis["metrics"]["average_consensus"] * 100
        elif "resource_efficiency" in analysis.get("metrics", {}):
            current_value = analysis["metrics"]["resource_efficiency"] * 100
        
        # Estimate improvement based on severity
        improvement_targets = {
            "critical": 50,
            "high": 30,
            "medium": 20,
            "low": 10
        }
        
        target_value = current_value + improvement_targets.get(analysis["severity"], 10)
        
        template = impact_templates.get(area, "Improve performance by {}%")
        return template.format(int(current_value), int(target_value))
    
    def _calculate_average_task_age(self, tasks: List[Dict]) -> float:
        """Calculate average age of incomplete tasks."""
        if not tasks:
            return 0
        
        current_time = datetime.now()
        ages = []
        
        for task in tasks:
            if task.get("status") not in ["completed", "failed", "cancelled"]:
                created_time = datetime.fromisoformat(task.get("created_at", current_time.isoformat()))
                age = (current_time - created_time).total_seconds() / 3600  # Hours
                ages.append(age)
        
        return statistics.mean(ages) if ages else 0
    
    def _analyze_task_descriptions(self, tasks: List[Dict]) -> List[str]:
        """Analyze task descriptions for quality issues."""
        issues = []
        
        vague_descriptions = 0
        missing_criteria = 0
        too_complex = 0
        
        for task in tasks:
            description = task.get("description", "")
            
            # Check for vagueness
            if len(description) < 50:
                vague_descriptions += 1
            
            # Check for acceptance criteria
            if "acceptance criteria" not in description.lower() and \
               "success criteria" not in description.lower():
                missing_criteria += 1
            
            # Check complexity (rough heuristic)
            if description.count("\n") > 20 or len(description) > 2000:
                too_complex += 1
        
        if vague_descriptions > len(tasks) * 0.3:
            issues.append("Many tasks have vague descriptions")
        
        if missing_criteria > len(tasks) * 0.5:
            issues.append("Most tasks lack clear acceptance criteria")
        
        if too_complex > len(tasks) * 0.2:
            issues.append("Some tasks are overly complex")
        
        return issues
    
    def _analyze_claude_failure_patterns(self, state: Dict) -> List[str]:
        """Analyze patterns in Claude interaction failures."""
        patterns = []
        
        # Analyze recent task attempts
        task_state = state.get("task_state", {})
        recent_tasks = sorted(
            task_state.get("tasks", []),
            key=lambda t: t.get("created_at", ""),
            reverse=True
        )[:20]  # Last 20 tasks
        
        # Look for patterns
        no_response_count = 0
        rejected_count = 0
        
        for task in recent_tasks:
            if task.get("claude_response_status") == "no_response":
                no_response_count += 1
            elif task.get("claude_response_status") == "rejected":
                rejected_count += 1
        
        if no_response_count > 10:
            patterns.append("Claude frequently not responding to issues")
        
        if rejected_count > 5:
            patterns.append("Claude rejecting many task implementations")
        
        return patterns
    
    def _calculate_project_diversity(self, projects: List[Dict]) -> float:
        """Calculate diversity score for project portfolio."""
        if not projects:
            return 0
        
        # Count unique types/categories
        project_types = set()
        technologies = set()
        
        for project in projects:
            project_types.add(project.get("type", "unknown"))
            technologies.update(project.get("technologies", []))
        
        # Simple diversity score
        type_diversity = len(project_types) / max(3, len(projects))  # Expect at least 3 types
        tech_diversity = len(technologies) / max(5, len(projects) * 2)  # Expect varied tech
        
        return (type_diversity + tech_diversity) / 2
    
    def _calculate_resource_efficiency(self, projects: List[Dict]) -> float:
        """Calculate how efficiently resources are allocated."""
        if not projects:
            return 0
        
        active_projects = [p for p in projects if p.get("status") == "active"]
        if not active_projects:
            return 0
        
        # Calculate efficiency metrics
        total_tasks = sum(p.get("task_count", 0) for p in active_projects)
        completed_tasks = sum(p.get("completed_tasks", 0) for p in active_projects)
        
        completion_rate = completed_tasks / max(1, total_tasks)
        
        # Resource balance (are resources spread too thin?)
        tasks_per_project = [p.get("task_count", 0) for p in active_projects]
        if len(tasks_per_project) > 1:
            balance_score = 1 - (statistics.stdev(tasks_per_project) / 
                               max(1, statistics.mean(tasks_per_project)))
        elif len(tasks_per_project) == 1:
            balance_score = 1.0  # Perfect balance with one project
        else:
            balance_score = 0
        
        return (completion_rate + balance_score) / 2
    
    def _calculate_average_project_health(self, projects: List[Dict]) -> float:
        """Calculate average health score across all projects."""
        if not projects:
            return 0
        
        health_scores = []
        for project in projects:
            # Simple health calculation
            completed_ratio = project.get("completed_tasks", 0) / max(1, project.get("task_count", 0))
            activity_score = 1.0 if project.get("last_activity_days_ago", 30) < 7 else 0.5
            health = (completed_ratio + activity_score) / 2
            health_scores.append(health)
        
        return statistics.mean(health_scores) if len(health_scores) > 0 else 0
    
    def get_immediate_research_needs(self) -> List[Dict]:
        """Get the most urgent research needs."""
        gaps = self.analyze_performance_gaps()
        
        immediate_needs = []
        
        # Extract critical and high priority needs
        for priority in ["critical", "high"]:
            for gap in gaps.get(priority, []):
                for research_need in gap["research_needs"]:
                    immediate_needs.append({
                        "topic": research_need,
                        "area": gap["area"],
                        "severity": gap["severity"],
                        "expected_impact": gap["expected_impact"],
                        "priority": priority
                    })
        
        return immediate_needs[:5]  # Top 5 most urgent needs
    
    def get_proactive_research_opportunities(self) -> List[Dict]:
        """Get proactive research opportunities even when system is performing well."""
        opportunities = []
        
        # 1. New Pattern Discovery
        opportunities.append({
            "topic": "emerging_ai_techniques",
            "area": "innovation",
            "severity": "opportunity",
            "expected_impact": "Discover new capabilities before competitors",
            "priority": "medium",
            "trigger": "scheduled",
            "research_queries": [
                "Latest breakthroughs in autonomous AI systems",
                "New multi-agent coordination strategies",
                "Advanced prompt engineering techniques",
                "Self-improving AI architectures"
            ]
        })
        
        # 2. Best Practices Research
        opportunities.append({
            "topic": "industry_best_practices",
            "area": "continuous_improvement",
            "severity": "enhancement",
            "expected_impact": "Adopt proven methodologies",
            "priority": "medium",
            "trigger": "scheduled",
            "research_queries": [
                "Best practices for AI task generation",
                "Optimal project portfolio management",
                "Effective AI agent communication patterns"
            ]
        })
        
        # 3. Performance Optimization
        opportunities.append({
            "topic": "performance_optimization",
            "area": "efficiency",
            "severity": "improvement",
            "expected_impact": "Increase efficiency by 20-30%",
            "priority": "medium",
            "trigger": "periodic",
            "research_queries": [
                "AI response time optimization techniques",
                "Parallel processing strategies for AI tasks",
                "Resource allocation algorithms"
            ]
        })
        
        # 4. New Domain Exploration
        opportunities.append({
            "topic": "domain_expansion",
            "area": "growth",
            "severity": "opportunity",
            "expected_impact": "Expand capabilities to new domains",
            "priority": "low",
            "trigger": "exploration",
            "research_queries": [
                "AI applications in unexplored domains",
                "Cross-domain knowledge transfer",
                "Domain-specific AI optimizations"
            ]
        })
        
        # 5. Future Technology Trends
        opportunities.append({
            "topic": "future_tech_trends",
            "area": "strategic_planning",
            "severity": "strategic",
            "expected_impact": "Prepare for future technological shifts",
            "priority": "low",
            "trigger": "quarterly",
            "research_queries": [
                "Emerging AI frameworks and tools",
                "Future of autonomous systems",
                "Next-generation AI architectures"
            ]
        })
        
        return opportunities
    
    def should_trigger_learning_research(self, recent_events: List[Dict]) -> bool:
        """Determine if recent events warrant learning-focused research."""
        if not recent_events:
            return False
        
        # Count different event types
        event_types = {}
        for event in recent_events:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Trigger research if:
        # 1. New event types appearing (learning opportunity)
        if len(event_types) > 5:
            return True
        
        # 2. High frequency of failures
        failure_count = event_types.get('failure', 0) + event_types.get('error', 0)
        if failure_count > len(recent_events) * 0.3:  # More than 30% failures
            return True
        
        # 3. Repeated patterns (need to learn from patterns)
        max_frequency = max(event_types.values()) if event_types else 0
        if max_frequency > len(recent_events) * 0.5:  # One type dominates 50%+
            return True
        
        return False