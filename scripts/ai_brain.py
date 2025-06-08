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
        "GENERATE_TASKS": {
            "base_score": 80,
            "goal_alignment": {"innovation": 1.3, "planning": 1.5, "efficiency": 1.4}
        },
        "REVIEW_TASKS": {
            "base_score": 70,
            "goal_alignment": {"quality": 1.5, "optimization": 1.3, "reliability": 1.4}
        },
        "PRIORITIZE_TASKS": {
            "base_score": 60,
            "goal_alignment": {"efficiency": 1.6, "planning": 1.4, "optimization": 1.3}
        },
        "ANALYZE_PERFORMANCE": {
            "base_score": 50,
            "goal_alignment": {"optimization": 1.5, "learning": 1.4, "efficiency": 1.2}
        },
        "UPDATE_DASHBOARD": {
            "base_score": 40,
            "goal_alignment": {"transparency": 1.5, "communication": 1.3, "engagement": 1.2}
        }
    }
    
    def __init__(self, state: Dict[str, Any] = None, context: Dict[str, Any] = None):
        """Initialize the AI Brain.
        
        Args:
            state: Current system state (optional)
            context: External context information
        """
        self.state = state or {}
        self.context = context or {}
        self.charter = self.state.get("charter", {})
        self.projects = self.state.get("projects", {})
        self.system_performance = self.state.get("system_performance", {})
        
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
            "GENERATE_TASKS": self._generate_task_generation_prompt(),
            "REVIEW_TASKS": self._generate_task_review_prompt(),
            "PRIORITIZE_TASKS": self._generate_task_prioritization_prompt(),
            "ANALYZE_PERFORMANCE": self._generate_performance_analysis_prompt(),
            "UPDATE_DASHBOARD": self._generate_dashboard_update_prompt()
        }
        
        base_prompt = base_prompts.get(action_type, "Perform the requested action.")
        
        # Combine with context
        full_prompt = f"{base_prompt}\n\n{historical_context}\n\n{external_context}"
        
        return full_prompt
    
    def _generate_task_generation_prompt(self) -> str:
        """Generate prompt for task generation."""
        trending_context = ""
        if self.context.get("github_trending"):
            trending_context = "Consider these trending technologies: " + ", ".join([
                trend.get("title", "") for trend in self.context["github_trending"][:3]
            ])
        
        return f"""Generate development tasks for @claude based on current project needs.

Considerations:
- Current portfolio health and gaps
- Trending technologies and market demands
- Security vulnerabilities that need addressing
- Documentation and testing coverage
- Feature requests and improvements

{trending_context}

Charter Goals: {self.charter.get('primary_goal')} / {self.charter.get('secondary_goal')}

Create actionable, specific tasks that @claude can implement."""
    
    def _generate_task_review_prompt(self) -> str:
        """Generate prompt for task review."""
        return f"""Review tasks completed by @claude and verify quality.

Review Criteria:
- Code quality and best practices
- Test coverage and reliability
- Documentation completeness
- Security implications
- Performance impact

Provide feedback and determine if tasks meet acceptance criteria."""
    
    def _generate_task_prioritization_prompt(self) -> str:
        """Generate prompt for task prioritization."""
        return f"""Analyze and prioritize the current task backlog.

Prioritization Factors:
- Business value and impact
- Technical dependencies
- Resource availability
- Risk mitigation
- Strategic alignment

Reorganize tasks to maximize efficiency and value delivery."""
    
    def _generate_performance_analysis_prompt(self) -> str:
        """Generate prompt for performance analysis."""
        return f"""Analyze system performance and @claude interaction effectiveness.

Analysis Areas:
- Task completion rates
- Quality metrics
- Time to completion
- Bottleneck identification
- Success patterns

Current Metrics:
- Total Cycles: {self.system_performance.get('total_cycles', 0)}
- Success Rate: {self.system_performance.get('successful_actions', 0)}/{self.system_performance.get('total_cycles', 0)}

Provide insights and optimization recommendations."""
    
    def _generate_dashboard_update_prompt(self) -> str:
        """Generate prompt for dashboard update."""
        return f"""Update the task management dashboard with current metrics.

Dashboard Sections:
- Task status distribution
- Performance metrics
- @claude interaction summary
- Insights and trends
- Recommendations

Ensure the dashboard provides clear visibility into system operations."""
    
    
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
            "portfolio_health": self._get_average_project_health(),
            "timestamp": cycle_start.isoformat()
        }
        
        return self.state, report_data
    
    def _execute_action_placeholder(self, action_type: str, prompt: str) -> bool:
        """Execute task management actions.
        
        Args:
            action_type: Type of action to execute
            prompt: Generated prompt for the action
            
        Returns:
            Success status
        """
        print(f"Executing {action_type}...")
        
        try:
            # Import task management modules
            from task_manager import TaskManager
            from task_analyzer import TaskAnalyzer
            from update_task_dashboard import TaskDashboardUpdater
            
            success = False
            
            if action_type == "GENERATE_TASKS":
                # Generate new tasks for @claude
                manager = TaskManager()
                
                # Determine focus based on context
                focus = "auto"
                if "security" in prompt.lower():
                    focus = "security"
                elif "feature" in prompt.lower():
                    focus = "new_features"
                elif "bug" in prompt.lower():
                    focus = "bug_fixes"
                elif "test" in prompt.lower():
                    focus = "testing"
                
                tasks = manager.generate_tasks(focus=focus, max_tasks=3)
                print(f"Generated {len(tasks)} new tasks")
                success = len(tasks) > 0
                
            elif action_type == "REVIEW_TASKS":
                # Review completed tasks
                manager = TaskManager()
                results = manager.review_completed_tasks()
                print(f"Reviewed {results.get('reviewed', 0)} tasks")
                print(f"- Approved: {results.get('approved', 0)}")
                print(f"- Needs revision: {results.get('needs_revision', 0)}")
                success = results.get('reviewed', 0) > 0 or True  # Success even if no tasks to review
                
            elif action_type == "PRIORITIZE_TASKS":
                # Update task priorities
                manager = TaskManager()
                manager.prioritize_tasks()
                print("Task priorities updated")
                success = True
                
            elif action_type == "ANALYZE_PERFORMANCE":
                # Analyze task performance
                analyzer = TaskAnalyzer()
                analysis = analyzer.analyze_all_tasks()
                report = analyzer.generate_summary_report()
                print(f"Analysis complete - {len(analysis.get('insights', []))} insights generated")
                success = True
                
            elif action_type == "UPDATE_DASHBOARD":
                # Update task dashboard
                updater = TaskDashboardUpdater()
                updater.update_dashboard()
                print("Dashboard updated successfully")
                success = True
                
            else:
                print(f"Unknown action type: {action_type}")
                success = False
            
            return success
            
        except Exception as e:
            print(f"Error executing {action_type}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
    def get_research_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about research AI capabilities.
        
        Returns:
            Dictionary containing detailed information about research AI capabilities,
            availability, and specialized functions
        """
        capabilities = {
            "available_providers": {},
            "research_functions": [],
            "analysis_types": [],
            "total_providers": 0,
            "primary_provider": None,
            "research_ready": False
        }
        
        # Check Anthropic (Primary AI)
        if self.anthropic_client is not None:
            capabilities["available_providers"]["anthropic"] = {
                "status": "available",
                "role": "primary_ai",
                "capabilities": ["strategic_analysis", "code_generation", "task_planning", "decision_making"],
                "api_key_present": True
            }
            capabilities["primary_provider"] = "anthropic"
            capabilities["total_providers"] += 1
        else:
            capabilities["available_providers"]["anthropic"] = {
                "status": "unavailable",
                "role": "primary_ai",
                "reason": "API key not configured"
            }
        
        # Check OpenAI (Secondary AI)
        if self.openai_client is not None:
            capabilities["available_providers"]["openai"] = {
                "status": "available",
                "role": "secondary_ai",
                "capabilities": ["content_generation", "analysis", "coding_assistance"],
                "api_key_present": True
            }
            capabilities["total_providers"] += 1
        else:
            capabilities["available_providers"]["openai"] = {
                "status": "unavailable",
                "role": "secondary_ai",
                "reason": "API key not configured"
            }
        
        # Check Gemini (Research AI)
        if self.gemini_client is not None:
            capabilities["available_providers"]["gemini"] = {
                "status": "available",
                "role": "research_ai",
                "capabilities": ["market_research", "trend_analysis", "content_analysis", "knowledge_synthesis"],
                "api_key_present": True
            }
            capabilities["research_functions"].extend([
                "market_trend_analysis",
                "technology_research",
                "competitive_intelligence"
            ])
            capabilities["total_providers"] += 1
        else:
            capabilities["available_providers"]["gemini"] = {
                "status": "unavailable",
                "role": "research_ai",
                "reason": "API key not configured"
            }
        
        # Check DeepSeek (Research AI)
        if self.deepseek_api_key is not None:
            capabilities["available_providers"]["deepseek"] = {
                "status": "available",
                "role": "research_ai",
                "capabilities": ["deep_analysis", "technical_research", "code_optimization", "pattern_recognition"],
                "api_key_present": True
            }
            capabilities["research_functions"].extend([
                "deep_technical_analysis",
                "optimization_research",
                "algorithmic_insights"
            ])
            capabilities["total_providers"] += 1
        else:
            capabilities["available_providers"]["deepseek"] = {
                "status": "unavailable",
                "role": "research_ai",
                "reason": "API key not configured"
            }
        
        # Define supported analysis types
        capabilities["analysis_types"] = [
            "general",
            "security",
            "trends",
            "technical",
            "market",
            "strategic",
            "performance",
            "competitive"
        ]
        
        # Determine if system is research ready
        research_providers = [
            capabilities["available_providers"]["gemini"]["status"] == "available",
            capabilities["available_providers"]["deepseek"]["status"] == "available"
        ]
        primary_available = capabilities["available_providers"]["anthropic"]["status"] == "available"
        
        capabilities["research_ready"] = primary_available and any(research_providers)
        
        # Add research function availability
        if capabilities["research_ready"]:
            capabilities["research_functions"].extend([
                "analyze_with_research_ai",
                "enhance_context_analysis",
                "multi_provider_synthesis"
            ])
        
        # Remove duplicates from research functions
        capabilities["research_functions"] = list(set(capabilities["research_functions"]))
        
        return capabilities
    
    async def generate_enhanced_response(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Generate enhanced response using primary AI provider.
        
        This is the main AI reasoning method used throughout the dynamic system.
        Uses the best available AI provider to generate comprehensive responses.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini')
            
        Returns:
            Dictionary containing the AI response with content and metadata
        """
        import asyncio
        
        try:
            # Determine which provider to use
            if model == 'claude' or (model is None and self.anthropic_client):
                # Use Anthropic Claude (primary)
                return await self._generate_anthropic_response(prompt)
            elif model == 'gpt' or (model is None and self.openai_client):
                # Use OpenAI GPT (secondary)
                return await self._generate_openai_response(prompt)
            elif model == 'gemini' or (model is None and self.gemini_client):
                # Use Google Gemini (research)
                return await self._generate_gemini_response(prompt)
            else:
                # Fallback: return mock response if no providers available
                return {
                    'content': 'Mock AI response - no providers available',
                    'provider': 'mock',
                    'reasoning': 'No AI providers configured',
                    'confidence': 0.1
                }
                
        except Exception as e:
            # Error fallback
            return {
                'content': f'Error generating response: {str(e)}',
                'provider': 'error',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _generate_anthropic_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Anthropic Claude."""
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'content': message.content[0].text,
                'provider': 'anthropic',
                'model': 'claude-3-sonnet',
                'confidence': 0.9
            }
        except Exception as e:
            return {
                'content': f'Anthropic API error: {str(e)}',
                'provider': 'anthropic',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _generate_openai_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using OpenAI GPT."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            
            return {
                'content': response.choices[0].message.content,
                'provider': 'openai',
                'model': 'gpt-4',
                'confidence': 0.8
            }
        except Exception as e:
            return {
                'content': f'OpenAI API error: {str(e)}',
                'provider': 'openai',
                'error': str(e),
                'confidence': 0.0
            }
    
    async def _generate_gemini_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Google Gemini."""
        try:
            response = self.gemini_client.generate_content(prompt)
            
            return {
                'content': response.text if response.text else 'No response generated',
                'provider': 'gemini',
                'model': 'gemini-pro',
                'confidence': 0.7
            }
        except Exception as e:
            return {
                'content': f'Gemini API error: {str(e)}',
                'provider': 'gemini',
                'error': str(e),
                'confidence': 0.0
            }


# Simple alias for compatibility
AIBrain = IntelligentAIBrain