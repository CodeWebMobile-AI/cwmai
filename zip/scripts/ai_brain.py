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
import requests
from scripts.http_ai_client import HTTPAIClient

# Fix asyncio event loop issues for nested loops (Jupyter, web servers, etc.)
import nest_asyncio
nest_asyncio.apply()
import asyncio


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
    
    def __init__(self, state: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,
                 enable_round_robin: bool = False):
        """Initialize the AI Brain.
        
        Args:
            state: Current system state (optional)
            context: External context information
            enable_round_robin: Whether to enable round-robin AI provider selection
        """
        self.state = state or {}
        self.context = context or {}
        self.charter = self.state.get("charter", {})
        self.projects = self.state.get("projects", {})
        self.system_performance = self.state.get("system_performance", {})
        
        # Initialize HTTP AI client - No SDK dependencies!
        self.http_ai_client = HTTPAIClient(enable_round_robin=enable_round_robin)
        
    def extract_json_from_response(self, text: str) -> Any:
        """Extract JSON from AI response using HTTP client's extractor.
        
        Args:
            text: The AI response text that may contain JSON
            
        Returns:
            Parsed JSON object or None if extraction fails
        """
        return self.http_ai_client.extract_json_from_response(text)
    
    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an AI capability.
        
        Args:
            capability: Name of the capability (e.g., 'problem_analysis')
            params: Parameters for the capability (must include 'prompt')
            
        Returns:
            Dict with 'status' and 'result' keys
        """
        try:
            prompt = params.get('prompt', '')
            if not prompt:
                return {'status': 'error', 'result': 'No prompt provided'}
            
            # Use the HTTP AI client to get response
            response_data = await self.http_ai_client.generate_enhanced_response(prompt)
            
            if response_data and response_data.get('content'):
                return {'status': 'success', 'result': response_data['content']}
            else:
                return {'status': 'error', 'result': 'No response from AI'}
                
        except Exception as e:
            print(f"Error executing capability {capability}: {e}")
            return {'status': 'error', 'result': str(e)}
    
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
            print(f"[DEBUG] Starting action execution: {action_type}")
            
            if action_type == "GENERATE_TASKS":
                print(f"[DEBUG] Executing GENERATE_TASKS action")
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
                
                print(f"[DEBUG] Calling manager.generate_tasks with focus: {focus}")
                tasks = manager.generate_tasks(focus=focus, max_tasks=3)
                print(f"Generated {len(tasks)} new tasks")
                success = len(tasks) > 0
                print(f"[DEBUG] Task generation result: success={success}, task_count={len(tasks)}")
                
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
        """Use research AI providers to analyze content.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis (general, security, trends, technical, strategic, performance)
            
        Returns:
            Analysis result or empty string if failed
        """
        return self.http_ai_client.analyze_with_research_ai(content, analysis_type)
    
    def _call_deepseek_api(self, content: str, analysis_type: str, model: str = "deepseek-chat") -> str:
        """Call DeepSeek API for content analysis (legacy compatibility).
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis
            model: DeepSeek model to use ('deepseek-chat' or 'deepseek-reasoner')
            
        Returns:
            Analysis result or empty string if failed
        """
        prompt = f"Analyze this {analysis_type} content and provide insights: {content[:1000]}"
        result = self.http_ai_client.generate_enhanced_response_sync(prompt, 'deepseek')
        return result.get('content', '')
    
    
    
    def get_primary_ai_client(self):
        """Get the primary AI client (HTTP-based).
        
        Returns:
            HTTP AI client
        """
        return self.http_ai_client
    
    def get_secondary_ai_client(self):
        """Get the secondary AI client (HTTP-based).
        
        Returns:
            HTTP AI client
        """
        return self.http_ai_client
    
    def get_research_ai_status(self) -> Dict[str, bool]:
        """Get status of research AI providers.
        
        Returns:
            Dictionary indicating which research AIs are available
        """
        return self.http_ai_client.get_research_ai_status()
    
    def get_research_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about research AI capabilities.
        
        Returns:
            Dictionary containing detailed information about research AI capabilities,
            availability, and specialized functions
        """
        return self.http_ai_client.get_research_capabilities()
    
    def generate_enhanced_response_sync(self, prompt: str, model: Optional[str] = None, prefill: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for generate_enhanced_response.
        
        This method allows synchronous code to call the async generate_enhanced_response method.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini')
            prefill: Optional assistant message to prefill (for structured outputs with Claude)
            
        Returns:
            Dictionary containing the AI response with content and metadata
        """
        import asyncio
        
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            print(f"[DEBUG] Event loop already running, creating task for AI generation")
            # If we're in a running loop, create a task and handle it properly
            task = loop.create_task(self.generate_enhanced_response(prompt, model, prefill))
            return loop.run_until_complete(task)
        except RuntimeError as e:
            print(f"[DEBUG] No running event loop, using asyncio.run(): {e}")
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self.generate_enhanced_response(prompt, model, prefill))
        except Exception as e:
            print(f"ERROR: Failed to generate AI response: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return {"content": "", "error": str(e), "success": False}
    
    async def generate_enhanced_response(self, prompt: str, model: Optional[str] = None, prefill: str = None) -> Dict[str, Any]:
        """Generate enhanced response using HTTP AI client.
        
        This is the main AI reasoning method used throughout the dynamic system.
        Uses the HTTP AI client for all AI provider communications.
        
        Args:
            prompt: The prompt to send to the AI
            model: Optional model preference ('claude', 'gpt', 'gemini', 'deepseek')
            prefill: Optional assistant message to prefill (for structured outputs with Claude)
            
        Returns:
            Dictionary containing the AI response with content and metadata
        """
        return await self.http_ai_client.generate_enhanced_response(prompt, model, prefill)
    
    # Context Gathering Methods (migrated from ContextGatherer)
    
    async def gather_context(self, charter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gather external context based on system charter.
        
        Args:
            charter: System charter defining goals and constraints
            
        Returns:
            Dictionary containing gathered context
        """
        import os
        from datetime import datetime, timezone
        
        debug = os.getenv('CONTEXT_DEBUG', 'false').lower() == 'true'
        
        if debug:
            print(f"[DEBUG] gather_context called with charter: {charter}")
        
        # Use default charter if none provided
        if charter is None:
            charter = self.context.get('charter', {})
        
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "charter_goals": [charter.get("primary_goal", ""), charter.get("secondary_goal", "")],
            "environment": "ai_brain_context",
            "research_note": "Context gathering now delegated to Research Intelligence System"
        }
        
        try:
            primary_goal = charter.get("primary_goal", "")
            if debug:
                print(f"[DEBUG] Primary goal: '{primary_goal}'")
                print(f"[DEBUG] Secondary goal: '{charter.get('secondary_goal', '')}'")
                print("[DEBUG] Basic trend gathering removed - Research Intelligence System handles research")
            
            # Note: Research Intelligence System will handle specific research based on defined domains:
            # - task_generation: Task decomposition, complexity scoring, validation
            # - claude_interaction: GitHub formatting, prompt engineering, AI context  
            # - multi_agent_coordination: Consensus, swarm intelligence, voting systems
            # - outcome_learning: Pattern recognition, feedback loops, predictive modeling
            # - portfolio_management: Project selection, resource allocation, synergy
            
            context["research_domains"] = [
                "task_generation",
                "claude_interaction", 
                "multi_agent_coordination",
                "outcome_learning",
                "portfolio_management"
            ]
            
        except Exception as e:
            print(f"Error gathering context: {e}")
            if debug:
                import traceback
                print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            context["error"] = str(e)
        
        # Enhance context with AI analysis
        if debug:
            print("[DEBUG] Enhancing context with AI analysis...")
        context = self.enhance_context_with_ai(context)
        
        # Save context to file
        self.save_context(context)
        return context
    
    async def gather_workflow_context(self) -> Dict[str, Any]:
        """Gather minimal context optimized for GitHub Actions workflow.
        
        Returns lightweight context with only essential data for CI performance.
        
        Returns:
            Minimal context dictionary for workflow execution
        """
        import os
        from datetime import datetime, timezone
        
        debug = os.getenv('CONTEXT_DEBUG', 'false').lower() == 'true'
        
        if debug:
            print("[DEBUG] gather_workflow_context called")
        
        # Minimal context for fast CI execution
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "charter_goals": ["workflow_execution", "ci_optimization"],
            "environment": "github_actions",
            "research_note": "Workflow context optimized - Research Intelligence System handles research"
        }
        
        try:
            if debug:
                print("[DEBUG] Workflow context optimized for CI performance")
                print("[DEBUG] Research Intelligence System will handle targeted research when needed")
            
            # Research Intelligence System will handle research based on performance needs
            context["research_domains"] = [
                "task_generation",
                "claude_interaction", 
                "multi_agent_coordination",
                "outcome_learning",
                "portfolio_management"
            ]
            
            if debug:
                print(f"[DEBUG] Workflow context gathering complete")
            
        except Exception as e:
            print(f"Error in workflow context gathering: {e}")
            if debug:
                import traceback
                print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            context["error"] = str(e)
        
        # Minimal enhancement for workflow
        context = self.enhance_context_with_ai(context)
        
        # Save workflow context
        self.save_context(context, "workflow_context.json")
        
        return context
    
    async def gather_production_context(self, charter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gather complete context for production environment.
        
        Loads all available context data including comprehensive market intelligence.
        
        Args:
            charter: Optional system charter
            
        Returns:
            Complete context dictionary for production use
        """
        import os
        from datetime import datetime, timezone
        
        debug = os.getenv('CONTEXT_DEBUG', 'false').lower() == 'true'
        
        if debug:
            print("[DEBUG] gather_production_context called")
        
        # Use default charter if none provided
        if charter is None:
            charter = self.context.get('charter', {})
        
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "charter_goals": [charter.get("primary_goal", ""), charter.get("secondary_goal", "")],
            "environment": "production",
            "research_note": "Production context enhanced with Research Intelligence System"
        }
        
        try:
            if debug:
                print("[DEBUG] Production context enhanced with Research Intelligence System")
                print("[DEBUG] Research domains defined for intelligent research selection")
            
            # Research Intelligence System will handle targeted research based on performance gaps
            context["research_domains"] = [
                "task_generation",
                "claude_interaction", 
                "multi_agent_coordination",
                "outcome_learning",
                "portfolio_management"
            ]
            
            # Production environment gets access to research bridge methods
            context["research_capabilities"] = [
                "conduct_targeted_research",
                "extract_actionable_insights", 
                "evaluate_research_quality",
                "generate_performance_research_query"
            ]
            
            if debug:
                print("[DEBUG] Production context gathering complete")
            
        except Exception as e:
            print(f"Error in production context gathering: {e}")
            if debug:
                import traceback
                print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            context["error"] = str(e)
        
        # Full enhancement for production
        context = self.enhance_context_with_ai(context)
        
        # Save production context
        self.save_context(context, "production_context.json")
        
        return context
    
    def enhance_context_with_ai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance context with AI analysis.
        
        Args:
            context: Raw context data
            
        Returns:
            Enhanced context with AI insights
        """
        try:
            import json
            from datetime import datetime, timezone
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this external context and provide strategic insights.
            
            Context Data:
            {json.dumps(context, indent=2)}
            
            Provide analysis in the following areas:
            1. Key trends and opportunities
            2. Potential risks or concerns
            3. Strategic recommendations
            4. Technology adoption suggestions
            5. Market positioning insights
            
            Return your analysis as a structured summary focusing on actionable insights.
            """
            
            # Use synchronous method for context enhancement
            response = self.generate_enhanced_response_sync(analysis_prompt)
            
            if response and 'content' in response:
                context['ai_analysis'] = {
                    'summary': response['content'],
                    'analyzed_at': datetime.now(timezone.utc).isoformat(),
                    'model_used': response.get('model', 'unknown')
                }
            else:
                context['ai_analysis'] = {
                    'summary': 'AI analysis unavailable',
                    'analyzed_at': datetime.now(timezone.utc).isoformat(),
                    'error': 'No response from AI'
                }
                
        except Exception as e:
            print(f"Error in AI context enhancement: {e}")
            context['ai_analysis'] = {
                'summary': 'AI analysis failed',
                'analyzed_at': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
        
        return context
    
    def save_context(self, context: Dict[str, Any], output_path: str = "context.json") -> None:
        """Save context to file.
        
        Args:
            context: Context dictionary to save
            output_path: File path to save context
        """
        try:
            import json
            import os
            from datetime import datetime, timezone
            
            # Add metadata
            context['saved_at'] = datetime.now(timezone.utc).isoformat()
            context['file_path'] = output_path
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(context, f, indent=2, sort_keys=True)
            
            debug = os.getenv('CONTEXT_DEBUG', 'false').lower() == 'true'
            if debug:
                print(f"[DEBUG] Context saved to {output_path}")
                
        except Exception as e:
            print(f"Error saving context to {output_path}: {e}")
    
    async def select_repository_for_task(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intelligently select which repository should receive the next task.
        
        Args:
            context: System context with repository information
            
        Returns:
            Selected repository info with name and reason, or None
        """
        # Get available repositories
        projects = context.get('projects', context.get('active_projects', []))
        
        if not projects:
            return None
            
        # Convert to list if dict
        if isinstance(projects, dict):
            project_list = list(projects.values())
        else:
            project_list = projects
            
        # Filter for eligible repositories
        from repository_exclusion import should_process_repo
        eligible_repos = []
        for proj in project_list:
            repo_name = proj.get('full_name', proj.get('name', ''))
            if should_process_repo(repo_name) and repo_name:
                eligible_repos.append(proj)
                
        if not eligible_repos:
            return None
            
        # Use AI to select the best repository
        prompt = f"""
        Select which repository should receive the next development task.
        
        Available Repositories:
        {json.dumps(eligible_repos, indent=2)}
        
        Consider:
        1. Repository health scores (lower scores need more attention)
        2. Open issues count (indicates active problems)
        3. Recent activity (prefer active but not overloaded repos)
        4. Task distribution (avoid overloading one repository)
        5. Strategic importance
        
        Return your selection as JSON:
        {{
            "selected_repository": "owner/repo",
            "reason": "Brief explanation of why this repo was selected",
            "priority_score": 0-100,
            "suggested_task_type": "FEATURE/BUG_FIX/ENHANCEMENT/etc"
        }}
        """
        
        response = self.generate_enhanced_response_sync(prompt)
        selection = self.extract_json_from_response(response['content'] if isinstance(response, dict) else response)
        
        if selection and selection.get('selected_repository'):
            # Verify the selection is valid
            selected_name = selection['selected_repository']
            for repo in eligible_repos:
                if repo.get('full_name') == selected_name or repo.get('name') == selected_name:
                    return {
                        'repository': selected_name,
                        'reason': selection.get('reason', 'Strategic selection'),
                        'priority_score': selection.get('priority_score', 50),
                        'suggested_task_type': selection.get('suggested_task_type', 'FEATURE'),
                        'repository_data': repo
                    }
                    
        # Fallback: select repository with lowest health score
        if eligible_repos:
            lowest_health = min(eligible_repos, key=lambda r: r.get('health_score', 100))
            return {
                'repository': lowest_health.get('full_name', lowest_health.get('name')),
                'reason': 'Lowest health score - needs attention',
                'priority_score': 80,
                'suggested_task_type': 'ENHANCEMENT',
                'repository_data': lowest_health
            }
            
        return None


# Simple alias for compatibility
AIBrain = IntelligentAIBrain