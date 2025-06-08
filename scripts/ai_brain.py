"""
AI Brain Module

The core intelligence engine for the autonomous AI system. Handles strategic decision-making,
adaptive task prioritization, dynamic prompt generation, and meta-learning capabilities.
"""

import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
import openai
import anthropic
import google.generativeai as genai
import requests


class IntelligentAIBrain:
    """Core intelligence engine with adaptive decision-making capabilities."""
    
    # Action types with base scoring weights
    ACTION_TYPES = {
        "CREATE_NEW_PROJECT": {
            "base_score": 50,
            "cost_estimate": 25.0,
            "goal_alignment": {"innovation": 1.5, "growth": 1.2, "experimentation": 1.4}
        },
        "IMPROVE_HEALTHIEST_PROJECT": {
            "base_score": 40,
            "cost_estimate": 15.0,
            "goal_alignment": {"stability": 1.4, "quality": 1.3, "community_engagement": 1.2}
        },
        "FIX_CRITICAL_BUG": {
            "base_score": 70,
            "cost_estimate": 10.0,
            "goal_alignment": {"stability": 1.6, "quality": 1.5, "reliability": 1.4}
        },
        "STRATEGIC_REVIEW": {
            "base_score": 30,
            "cost_estimate": 5.0,
            "goal_alignment": {"optimization": 1.3, "planning": 1.2, "efficiency": 1.1}
        },
        "SECURITY_SWEEP": {
            "base_score": 60,
            "cost_estimate": 12.0,
            "goal_alignment": {"security": 1.8, "stability": 1.3, "compliance": 1.4}
        },
        "DO_NOTHING_SAVE_BUDGET": {
            "base_score": 20,
            "cost_estimate": 0.0,
            "goal_alignment": {"budget_conservation": 2.0, "efficiency": 1.1}
        }
    }
    
    def __init__(self, state: Dict[str, Any], context: Dict[str, Any]):
        """Initialize the AI Brain.
        
        Args:
            state: Current system state
            context: External context information
        """
        self.state = state
        self.context = context
        self.charter = state.get("charter", {})
        self.api_budget = state.get("api_budget", {})
        self.projects = state.get("projects", {})
        self.system_performance = state.get("system_performance", {})
        
        # Initialize AI clients - Anthropic is now primary
        self.anthropic_client = None
        self.openai_client = None
        self.gemini_client = None
        self.deepseek_api_key = None
        
        # Primary AI provider - Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Secondary AI provider - OpenAI
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Research AI providers - Gemini and DeepSeek
        if os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_client = genai.GenerativeModel('gemini-pro')
        
        if os.getenv('DEEPSEEK_API_KEY'):
            self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    
    def decide_next_action(self) -> str:
        """Intelligently decide the next action based on multiple factors.
        
        Returns:
            String identifier of the chosen action
        """
        action_scores = {}
        
        # Calculate scores for each possible action
        for action_type, action_config in self.ACTION_TYPES.items():
            score = self._calculate_action_score(action_type, action_config)
            action_scores[action_type] = score
            print(f"Action {action_type}: score = {score:.2f}")
        
        # Choose action with highest score
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        
        print(f"Selected action: {best_action} (score: {best_score:.2f})")
        
        # Record decision in state for learning
        self._record_decision(best_action, action_scores)
        
        return best_action
    
    def _calculate_action_score(self, action_type: str, action_config: Dict[str, Any]) -> float:
        """Calculate desirability score for an action.
        
        Args:
            action_type: Type of action to score
            action_config: Configuration for the action
            
        Returns:
            Calculated score for the action
        """
        score = action_config["base_score"]
        
        # Goal alignment factor
        goal_factor = self._calculate_goal_alignment(action_config.get("goal_alignment", {}))
        score *= goal_factor
        
        # Portfolio health factor
        health_factor = self._calculate_health_factor(action_type)
        score *= health_factor
        
        # Resource availability factor
        resource_factor = self._calculate_resource_factor(action_config["cost_estimate"])
        score *= resource_factor
        
        # External context factor
        context_factor = self._calculate_context_factor(action_type)
        score *= context_factor
        
        # Historical performance factor
        performance_factor = self._calculate_performance_factor(action_type)
        score *= performance_factor
        
        # Add some randomness to prevent deterministic behavior
        randomness = random.uniform(0.9, 1.1)
        score *= randomness
        
        return max(0, score)  # Ensure non-negative score
    
    def _calculate_goal_alignment(self, goal_alignment: Dict[str, float]) -> float:
        """Calculate how well an action aligns with charter goals.
        
        Args:
            goal_alignment: Action's goal alignment configuration
            
        Returns:
            Goal alignment factor (0.5 to 2.0)
        """
        primary_goal = self.charter.get("primary_goal", "")
        secondary_goal = self.charter.get("secondary_goal", "")
        
        factor = 1.0
        
        # Primary goal alignment
        if primary_goal in goal_alignment:
            factor *= goal_alignment[primary_goal]
        
        # Secondary goal alignment (half weight)
        if secondary_goal in goal_alignment:
            factor *= (1.0 + (goal_alignment[secondary_goal] - 1.0) * 0.5)
        
        return max(0.5, min(2.0, factor))
    
    def _calculate_health_factor(self, action_type: str) -> float:
        """Calculate portfolio health factor.
        
        Args:
            action_type: Type of action being evaluated
            
        Returns:
            Health factor (0.5 to 1.5)
        """
        if not self.projects:
            return 1.0
        
        # Calculate average health score
        health_scores = [proj.get("health_score", 50) for proj in self.projects.values()]
        avg_health = sum(health_scores) / len(health_scores)
        
        if action_type == "FIX_CRITICAL_BUG" and avg_health < 60:
            return 1.5  # Boost bug fixing when health is low
        elif action_type == "CREATE_NEW_PROJECT" and avg_health < 40:
            return 0.5  # Reduce new projects when existing ones are unhealthy
        elif action_type == "IMPROVE_HEALTHIEST_PROJECT" and avg_health > 80:
            return 1.2  # Boost improvements when health is good
        
        return 1.0
    
    def _calculate_resource_factor(self, cost_estimate: float) -> float:
        """Calculate resource availability factor.
        
        Args:
            cost_estimate: Estimated cost of the action
            
        Returns:
            Resource factor (0.1 to 1.5)
        """
        monthly_limit = self.api_budget.get("monthly_limit_usd", 100)
        monthly_usage = self.api_budget.get("monthly_usage_usd", 0)
        remaining_budget = monthly_limit - monthly_usage
        
        if remaining_budget <= 0:
            return 0.1 if cost_estimate > 0 else 1.5  # Heavily favor free actions
        
        budget_ratio = remaining_budget / monthly_limit
        
        if budget_ratio > 0.5:
            return 1.2  # Plenty of budget available
        elif budget_ratio > 0.2:
            cost_ratio = cost_estimate / remaining_budget
            return max(0.3, 1.0 - cost_ratio)  # Scale based on cost
        else:
            # Low budget - heavily penalize expensive actions
            if cost_estimate > remaining_budget * 0.5:
                return 0.1
            else:
                return 0.7
    
    def _calculate_context_factor(self, action_type: str) -> float:
        """Calculate external context factor.
        
        Args:
            action_type: Type of action being evaluated
            
        Returns:
            Context factor (0.8 to 1.3)
        """
        factor = 1.0
        
        # Check for security alerts
        security_alerts = self.context.get("security_alerts", [])
        if security_alerts and action_type == "SECURITY_SWEEP":
            factor *= 1.3
        
        # Check for innovation trends
        market_trends = self.context.get("market_trends", [])
        tech_updates = self.context.get("technology_updates", [])
        if (market_trends or tech_updates) and action_type == "CREATE_NEW_PROJECT":
            factor *= 1.2
        
        # Check GitHub trending for engagement opportunities
        github_trending = self.context.get("github_trending", [])
        if github_trending and action_type == "IMPROVE_HEALTHIEST_PROJECT":
            factor *= 1.1
        
        return max(0.8, min(1.3, factor))
    
    def _calculate_performance_factor(self, action_type: str) -> float:
        """Calculate historical performance factor.
        
        Args:
            action_type: Type of action being evaluated
            
        Returns:
            Performance factor (0.7 to 1.3)
        """
        # Analyze historical success rates for this action type
        total_actions = 0
        successful_actions = 0
        
        for project in self.projects.values():
            for action in project.get("action_history", []):
                if action.get("action") == action_type:
                    total_actions += 1
                    if action.get("outcome", "").startswith("success"):
                        successful_actions += 1
        
        if total_actions == 0:
            return 1.0  # No history, neutral factor
        
        success_rate = successful_actions / total_actions
        
        # Boost actions with high success rates, penalize those with low rates
        if success_rate > 0.7:
            return 1.3
        elif success_rate > 0.5:
            return 1.1
        elif success_rate < 0.3:
            return 0.7
        else:
            return 0.9
    
    def _record_decision(self, chosen_action: str, all_scores: Dict[str, float]) -> None:
        """Record decision for meta-learning.
        
        Args:
            chosen_action: The action that was chosen
            all_scores: Scores for all considered actions
        """
        decision_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chosen_action": chosen_action,
            "action_scores": all_scores,
            "context_summary": {
                "budget_remaining": self.api_budget.get("monthly_limit_usd", 100) - self.api_budget.get("monthly_usage_usd", 0),
                "avg_project_health": self._get_average_project_health(),
                "external_factors": len(self.context.get("security_alerts", [])) + len(self.context.get("market_trends", []))
            }
        }
        
        # Add to system state for learning
        if "decision_history" not in self.state:
            self.state["decision_history"] = []
        
        self.state["decision_history"].append(decision_record)
        
        # Keep only last 50 decisions to prevent unbounded growth
        if len(self.state["decision_history"]) > 50:
            self.state["decision_history"] = self.state["decision_history"][-50:]
    
    def _get_average_project_health(self) -> float:
        """Get average health score across all projects.
        
        Returns:
            Average health score
        """
        if not self.projects:
            return 50.0
        
        health_scores = [proj.get("health_score", 50) for proj in self.projects.values()]
        return sum(health_scores) / len(health_scores)
    
    def generate_dynamic_prompt(self, action_type: str, project_name: Optional[str] = None) -> str:
        """Generate intelligent, context-aware prompts.
        
        Args:
            action_type: Type of action to generate prompt for
            project_name: Target project name (if applicable)
            
        Returns:
            Generated prompt string
        """
        # Get historical context for this action type
        historical_context = self._get_historical_context(action_type, project_name)
        external_context = self._format_external_context()
        
        base_prompts = {
            "CREATE_NEW_PROJECT": self._generate_new_project_prompt(),
            "IMPROVE_HEALTHIEST_PROJECT": self._generate_improvement_prompt(project_name),
            "FIX_CRITICAL_BUG": self._generate_bug_fix_prompt(project_name),
            "STRATEGIC_REVIEW": self._generate_review_prompt(),
            "SECURITY_SWEEP": self._generate_security_prompt(),
            "DO_NOTHING_SAVE_BUDGET": self._generate_conservation_prompt()
        }
        
        base_prompt = base_prompts.get(action_type, "Perform the requested action.")
        
        # Combine with context
        full_prompt = f"{base_prompt}\n\n{historical_context}\n\n{external_context}"
        
        return full_prompt
    
    def _generate_new_project_prompt(self) -> str:
        """Generate prompt for creating new projects."""
        trending_context = ""
        if self.context.get("github_trending"):
            trending_context = "Consider these trending technologies: " + ", ".join([
                trend.get("title", "") for trend in self.context["github_trending"][:3]
            ])
        
        return f"""Create a new innovative software project that aligns with current technology trends.

Requirements:
- Must be technically sound and implementable
- Should demonstrate modern best practices
- Include comprehensive documentation and tests
- Focus on solving real-world problems
- Consider open source community engagement

{trending_context}

Charter Goals: {self.charter.get('primary_goal')} / {self.charter.get('secondary_goal')}"""
    
    def _generate_improvement_prompt(self, project_name: Optional[str]) -> str:
        """Generate prompt for improving projects."""
        if not project_name or project_name not in self.projects:
            project_name = max(self.projects.keys(), key=lambda p: self.projects[p].get("health_score", 0))
        
        project = self.projects[project_name]
        
        return f"""Improve the project '{project_name}' (current health: {project.get('health_score', 'unknown')}).

Focus Areas:
- Code quality and maintainability
- Performance optimizations
- Documentation improvements
- Test coverage enhancement
- Security hardening
- User experience improvements

Recent Activity: {len(project.get('action_history', []))} previous actions
Last Checked: {project.get('last_checked', 'Never')}"""
    
    def _generate_bug_fix_prompt(self, project_name: Optional[str]) -> str:
        """Generate prompt for fixing bugs."""
        return f"""Identify and fix critical bugs in the project '{project_name or 'selected project'}'.

Priority Areas:
- Security vulnerabilities
- Performance bottlenecks
- Functional errors
- Compatibility issues
- Data integrity problems

Ensure all fixes include:
- Comprehensive testing
- Clear documentation
- Regression prevention measures"""
    
    def _generate_review_prompt(self) -> str:
        """Generate prompt for strategic review."""
        return f"""Conduct a strategic review of the project portfolio.

Review Areas:
- Goal alignment assessment
- Resource allocation efficiency
- Performance metrics analysis
- Risk identification
- Opportunity assessment

Current Portfolio: {len(self.projects)} projects
Average Health: {self._get_average_project_health():.1f}
System Performance: {self.system_performance.get('successful_actions', 0)}/{self.system_performance.get('total_cycles', 0)} success rate"""
    
    def _generate_security_prompt(self) -> str:
        """Generate prompt for security sweep."""
        security_context = ""
        if self.context.get("security_alerts"):
            security_context = "Address these security concerns: " + "; ".join([
                alert.get("title", "") for alert in self.context["security_alerts"][:3]
            ])
        
        return f"""Perform a comprehensive security audit and remediation.

Security Focus:
- Vulnerability scanning
- Dependency updates
- Access control review
- Data protection assessment
- Compliance verification

{security_context}"""
    
    def _generate_conservation_prompt(self) -> str:
        """Generate prompt for budget conservation."""
        return f"""Focus on budget conservation and efficiency optimization.

Conservation Strategies:
- Optimize existing processes
- Reduce API usage
- Improve code efficiency
- Automated maintenance tasks
- Resource usage analysis

Current Budget: ${self.api_budget.get('monthly_usage_usd', 0):.2f} / ${self.api_budget.get('monthly_limit_usd', 100):.2f} used"""
    
    def _get_historical_context(self, action_type: str, project_name: Optional[str]) -> str:
        """Get historical context for prompt enhancement."""
        context_parts = []
        
        # Get recent successes and failures for this action type
        successes = []
        failures = []
        
        for project in self.projects.values():
            for action in project.get("action_history", []):
                if action.get("action") == action_type:
                    if action.get("outcome", "").startswith("success"):
                        successes.append(action)
                    else:
                        failures.append(action)
        
        if successes:
            context_parts.append(f"Previous Successes: {len(successes)} successful {action_type} actions")
            recent_success = max(successes, key=lambda x: x.get("timestamp", ""))
            context_parts.append(f"Most Recent Success: {recent_success.get('details', 'No details')}")
        
        if failures:
            context_parts.append(f"Previous Failures: {len(failures)} failed {action_type} actions")
            recent_failure = max(failures, key=lambda x: x.get("timestamp", ""))
            context_parts.append(f"Learn from: {recent_failure.get('details', 'No details')}")
        
        return "\n".join(context_parts) if context_parts else "No historical context available."
    
    def _format_external_context(self) -> str:
        """Format external context for prompt inclusion."""
        context_parts = []
        
        if self.context.get("market_trends"):
            context_parts.append(f"Market Trends: {len(self.context['market_trends'])} current trends identified")
        
        if self.context.get("security_alerts"):
            context_parts.append(f"Security Alerts: {len(self.context['security_alerts'])} active security concerns")
        
        if self.context.get("technology_updates"):
            context_parts.append(f"Technology Updates: {len(self.context['technology_updates'])} recent updates")
        
        return "External Context: " + "; ".join(context_parts) if context_parts else "No external context available."
    
    def run_intelligent_cycle(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run a complete intelligent cycle.
        
        Returns:
            Tuple of (updated_state, report_data)
        """
        cycle_start = datetime.now(timezone.utc)
        
        # Increment cycle counter
        self.state["system_performance"]["total_cycles"] += 1
        
        # Decide next action
        chosen_action = self.decide_next_action()
        
        # Generate dynamic prompt
        prompt = self.generate_dynamic_prompt(chosen_action)
        
        # Execute action (placeholder - in real implementation, this would call AI APIs)
        success = self._execute_action_placeholder(chosen_action, prompt)
        
        # Record outcome
        outcome = "success_completed" if success else "failure_error"
        self._record_action_outcome(chosen_action, outcome, prompt)
        
        # Update performance metrics
        if success:
            self.state["system_performance"]["successful_actions"] += 1
        else:
            self.state["system_performance"]["failed_actions"] += 1
        
        # Calculate cycle duration
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        
        # Create report data
        report_data = {
            "cycle_number": self.state["system_performance"]["total_cycles"],
            "action_taken": chosen_action,
            "outcome": outcome,
            "duration_seconds": cycle_duration,
            "budget_used": self.api_budget.get("monthly_usage_usd", 0),
            "portfolio_health": self._get_average_project_health(),
            "timestamp": cycle_start.isoformat()
        }
        
        return self.state, report_data
    
    def _execute_action_placeholder(self, action_type: str, prompt: str) -> bool:
        """Placeholder for action execution.
        
        In a real implementation, this would:
        1. Call appropriate AI APIs with the generated prompt
        2. Execute the resulting actions (create PRs, files, etc.)
        3. Monitor success/failure
        
        Args:
            action_type: Type of action to execute
            prompt: Generated prompt for the action
            
        Returns:
            Success status
        """
        print(f"Executing {action_type}...")
        print(f"Prompt length: {len(prompt)} characters")
        
        # Simulate success/failure based on action type
        success_rates = {
            "CREATE_NEW_PROJECT": 0.7,
            "IMPROVE_HEALTHIEST_PROJECT": 0.8,
            "FIX_CRITICAL_BUG": 0.6,
            "STRATEGIC_REVIEW": 0.9,
            "SECURITY_SWEEP": 0.75,
            "DO_NOTHING_SAVE_BUDGET": 1.0
        }
        
        success_rate = success_rates.get(action_type, 0.5)
        success = random.random() < success_rate
        
        # Simulate cost
        cost = self.ACTION_TYPES[action_type]["cost_estimate"]
        if success:
            self.state["api_budget"]["monthly_usage_usd"] += cost
        
        print(f"Action {action_type}: {'SUCCESS' if success else 'FAILURE'}")
        
        return success
    
    def _record_action_outcome(self, action_type: str, outcome: str, prompt: str) -> None:
        """Record action outcome for learning.
        
        Args:
            action_type: Type of action that was executed
            outcome: Outcome of the action
            prompt: Prompt that was used
        """
        action_record = {
            "action": action_type,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": f"Executed {action_type} with {len(prompt)} char prompt",
            "prompt_length": len(prompt)
        }
        
        # Add to a sample project or create one if none exist
        if not self.projects:
            self.projects["ai-managed-project"] = {
                "health_score": 75,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "action_history": [],
                "metrics": {"stars": 0, "forks": 0, "issues_open": 0, "pull_requests_open": 0}
            }
        
        # Add to the first project's history
        project_key = list(self.projects.keys())[0]
        if "action_history" not in self.projects[project_key]:
            self.projects[project_key]["action_history"] = []
        
        self.projects[project_key]["action_history"].append(action_record)
        
        # Keep only last 20 actions per project
        if len(self.projects[project_key]["action_history"]) > 20:
            self.projects[project_key]["action_history"] = self.projects[project_key]["action_history"][-20:]
    
    def analyze_with_research_ai(self, content: str, analysis_type: str = "general") -> str:
        """Use research AI providers (Gemini/DeepSeek) to analyze content.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis (general, security, trends, technical)
            
        Returns:
            Analysis result or empty string if failed
        """
        try:
            # Try Gemini first for research tasks
            if self.gemini_client:
                prompt = f"Analyze this {analysis_type} content and provide insights: {content[:1000]}"
                try:
                    response = self.gemini_client.generate_content(prompt)
                    return response.text if response.text else ""
                except Exception as e:
                    print(f"Gemini analysis failed: {e}")
            
            # Fallback to DeepSeek API
            if self.deepseek_api_key:
                return self._call_deepseek_api(content, analysis_type)
                
        except Exception as e:
            print(f"Research AI analysis failed: {e}")
        
        return ""
    
    def _call_deepseek_api(self, content: str, analysis_type: str) -> str:
        """Call DeepSeek API for content analysis.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis
            
        Returns:
            Analysis result or empty string if failed
        """
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze this {analysis_type} content and provide insights: {content[:1000]}"
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
            else:
                print(f"DeepSeek API error: {response.status_code}")
                
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
        
        return ""
    
    def get_primary_ai_client(self):
        """Get the primary AI client (Anthropic).
        
        Returns:
            Primary AI client or None
        """
        return self.anthropic_client
    
    def get_secondary_ai_client(self):
        """Get the secondary AI client (OpenAI).
        
        Returns:
            Secondary AI client or None  
        """
        return self.openai_client
    
    def get_research_ai_status(self) -> Dict[str, bool]:
        """Get status of research AI providers.
        
        Returns:
            Dictionary indicating which research AIs are available
        """
        return {
            "gemini_available": self.gemini_client is not None,
            "deepseek_available": self.deepseek_api_key is not None,
            "anthropic_primary": self.anthropic_client is not None,
            "openai_secondary": self.openai_client is not None
        }