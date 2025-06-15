"""
Intelligent Task Generator

Generates tasks based on dynamic goals and learning.
No templates, no hardcoded values - pure AI reasoning.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import uuid
try:
    from repository_exclusion import should_process_repo, filter_excluded_repos
except ImportError:
    # Fallback if module not found
    def should_process_repo(repo_name):
        return 'cwmai' not in repo_name.lower()
    def filter_excluded_repos(repos):
        return [r for r in repos if should_process_repo(r)]

try:
    from smart_context_aggregator import SmartContextAggregator, AggregatedContext
except ImportError:
    SmartContextAggregator = None
    AggregatedContext = None

try:
    from predictive_task_engine import PredictiveTaskEngine
except ImportError:
    PredictiveTaskEngine = None

try:
    from task_persistence import TaskPersistence
except ImportError:
    from scripts.task_persistence import TaskPersistence
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.mcp_redis_integration import MCPRedisIntegration

# Import lifecycle and planning components
try:
    from project_lifecycle_analyzer import ProjectLifecycleAnalyzer, ProjectStage
    from project_planner import ProjectPlanner
    LIFECYCLE_PLANNING_AVAILABLE = True
except ImportError:
    LIFECYCLE_PLANNING_AVAILABLE = False
    print("Warning: Lifecycle planning components not available")

# Import decomposition system components
try:
    from task_decomposition_engine import TaskDecompositionEngine, TaskComplexity
    from complexity_analyzer import ComplexityAnalyzer
    from hierarchical_task_manager import HierarchicalTaskManager
    from progressive_task_generator import ProgressiveTaskGenerator, ProgressionContext
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Decomposition system not available in IntelligentTaskGenerator: {e}")
    DECOMPOSITION_AVAILABLE = False


class IntelligentTaskGenerator:
    """Generate tasks with zero hardcoded logic - all AI-driven."""
    
    def __init__(self, ai_brain, charter_system, learning_system=None,
                 context_aggregator=None, predictive_engine=None):
        """Initialize with AI brain and charter system.
        
        Args:
            ai_brain: AI brain for task generation
            charter_system: Dynamic charter system for guidelines
            learning_system: Optional learning system for value prediction
            context_aggregator: Smart context aggregator for comprehensive awareness
            predictive_engine: Predictive task engine for ML-based predictions
        """
        self.ai_brain = ai_brain
        self.charter_system = charter_system
        self.learning_system = learning_system
        self.context_aggregator = context_aggregator
        self.predictive_engine = predictive_engine
        self.task_history = []
        self.generation_patterns = {}
        self.task_state_file = "task_state.json"
        self.logger = logging.getLogger(__name__)
        
        # Initialize task persistence for duplicate prevention
        self.task_persistence = TaskPersistence()
        
        # Enhanced intelligence features
        self.cross_repo_awareness = True
        self.predictive_generation = True
        self.external_intelligence = True
        self.dynamic_priority_adjustment = True
        
        # Initialize decomposition system components
        self.complexity_analyzer = None
        self.decomposition_engine = None
        self.hierarchical_manager = None
        self.progressive_generator = None
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
        
        if DECOMPOSITION_AVAILABLE:
            try:
                self.complexity_analyzer = ComplexityAnalyzer(ai_brain)
                self.decomposition_engine = TaskDecompositionEngine(ai_brain)
                self.hierarchical_manager = HierarchicalTaskManager()
                self.progressive_generator = ProgressiveTaskGenerator(
                    ai_brain, self.hierarchical_manager, self.complexity_analyzer
                )
                self.logger.info("Decomposition system initialized in IntelligentTaskGenerator")
            except Exception as e:
                self.logger.warning(f"Failed to initialize decomposition system: {e}")
        
        # Initialize lifecycle and planning components
        self.lifecycle_analyzer = None
        self.project_planner = None
        
        if LIFECYCLE_PLANNING_AVAILABLE:
            try:
                self.lifecycle_analyzer = ProjectLifecycleAnalyzer(ai_brain)
                self.project_planner = ProjectPlanner(ai_brain, self.lifecycle_analyzer)
                self.logger.info("Lifecycle planning system initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize lifecycle planning: {e}")
        
    async def generate_task_for_repository(self, repository_name: str, 
                                          repository_analysis: Dict[str, Any],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a task specifically for the analyzed repository.
        
        Args:
            repository_name: Target repository name
            repository_analysis: Deep analysis of the repository
            context: Current system context
            
        Returns:
            Generated task tailored to the repository
        """
        self.logger.info(f"Generating task for repository: {repository_name}")
        
        # Ensure MCP-Redis is initialized
        await self._ensure_mcp_redis()
        
        # Get current charter for guidelines
        charter = await self.charter_system.get_current_charter()
        
        # Analyze repository needs based on deep analysis
        need_analysis = await self._analyze_repository_needs(repository_analysis, charter)
        
        # Generate task for this specific repository
        task = await self._create_repository_specific_task(
            repository_name, 
            repository_analysis, 
            need_analysis,
            charter
        )
        
        # Apply learning if available
        if self.learning_system:
            task = await self._apply_learned_improvements(task)
            
        # Predict value if possible
        if self.learning_system:
            prediction = await self.learning_system.predict_task_value(task)
            if prediction.get('recommendation') == 'modify':
                task = await self._modify_for_higher_value(task, prediction)
                
        # Add generation context
        task['generation_context'] = {
            'repository': repository_name,
            'generation_reason': need_analysis.get('specific_need', 'Unknown'),
            'repository_health': repository_analysis.get('health_metrics', {}).get('health_score', 0),
            'identified_issues': len(repository_analysis.get('specific_needs', [])),
            'ai_confidence_score': need_analysis.get('confidence', 0.0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check if task was skipped due to duplicate
        if task.get('skip'):
            return task
        
        # Perform complexity analysis and decomposition if available
        if DECOMPOSITION_AVAILABLE and self.complexity_analyzer and self.decomposition_engine:
            try:
                enhanced_task = await self._enhance_task_with_decomposition(
                    task, repository_analysis, repository_name
                )
                if enhanced_task:
                    task = enhanced_task
            except Exception as e:
                self.logger.warning(f"Task decomposition enhancement failed: {e}")
        
        # Record generation
        self._record_generation(task, need_analysis)
        
        return task
    
    async def _enhance_task_with_decomposition(self, task: Dict[str, Any], 
                                             repository_analysis: Dict[str, Any],
                                             repository_name: str) -> Optional[Dict[str, Any]]:
        """Enhance task with complexity analysis and potential decomposition.
        
        Args:
            task: Generated task
            repository_analysis: Repository analysis data
            repository_name: Repository name
            
        Returns:
            Enhanced task with decomposition metadata or None
        """
        try:
            # Analyze task complexity
            complexity_analysis = await self.complexity_analyzer.analyze_complexity(
                task, {'repository_analysis': repository_analysis}
            )
            
            self.logger.info(f"Task complexity: {complexity_analysis.overall_level.value} "
                           f"(score: {complexity_analysis.overall_score:.2f})")
            
            # Add complexity metadata to task
            task['complexity_analysis'] = {
                'level': complexity_analysis.overall_level.value,
                'score': complexity_analysis.overall_score,
                'decomposition_recommended': complexity_analysis.decomposition_recommended,
                'estimated_subtasks': complexity_analysis.estimated_subtasks,
                'optimal_chunk_size': complexity_analysis.optimal_chunk_size,
                'risk_factors': complexity_analysis.risk_factors,
                'mitigation_strategies': complexity_analysis.mitigation_strategies,
                'confidence': complexity_analysis.confidence_level
            }
            
            # If decomposition is recommended, generate decomposition plan
            if complexity_analysis.decomposition_recommended:
                self.logger.info(f"Generating decomposition plan for complex task")
                
                # Create repository context
                repository_context = {
                    'repository_analysis': repository_analysis,
                    'repository_name': repository_name
                }
                
                # Generate decomposition
                decomposition_result = await self.decomposition_engine.decompose_task(
                    task, repository_context
                )
                
                # Add decomposition metadata
                task['decomposition'] = {
                    'strategy': decomposition_result.strategy.value,
                    'total_estimated_hours': decomposition_result.total_estimated_hours,
                    'sub_task_count': len(decomposition_result.sub_tasks),
                    'critical_path': decomposition_result.critical_path,
                    'parallel_groups': decomposition_result.parallel_groups,
                    'rationale': decomposition_result.decomposition_rationale,
                    'next_actions': decomposition_result.next_actions
                }
                
                # Add sub-task information
                task['sub_tasks'] = [
                    {
                        'id': st.id,
                        'title': st.title,
                        'description': st.description,
                        'estimated_hours': st.estimated_hours,
                        'sequence_order': st.sequence_order,
                        'can_parallelize': st.can_parallelize,
                        'deliverables': st.deliverables,
                        'acceptance_criteria': st.acceptance_criteria
                    }
                    for st in decomposition_result.sub_tasks
                ]
                
                self.logger.info(f"Task decomposed into {len(decomposition_result.sub_tasks)} sub-tasks")
                
                # Optionally add to hierarchical manager for tracking
                if self.hierarchical_manager:
                    try:
                        hierarchy_id = self.hierarchical_manager.add_task_hierarchy(
                            decomposition_result, task
                        )
                        task['hierarchy_id'] = hierarchy_id
                    except Exception as e:
                        self.logger.warning(f"Failed to add to hierarchical manager: {e}")
                
            return task
            
        except Exception as e:
            self.logger.error(f"Error enhancing task with decomposition: {e}")
            return None
    
    async def generate_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for backward compatibility.
        
        This method now selects a repository first, then generates a task for it.
        """
        self.logger.info("Legacy generate_task called - will select repository first")
        
        # Select repository intelligently
        selected_repo = await self._select_repository_for_task(context)
        
        if not selected_repo:
            # No repository selected, create a NEW_PROJECT task
            return await self._generate_new_project_task(context)
        
        # Analyze the selected repository
        from repository_analyzer import RepositoryAnalyzer
        analyzer = RepositoryAnalyzer()
        repo_analysis = await analyzer.analyze_repository(selected_repo)
        
        # Generate task for the selected repository
        return await self.generate_task_for_repository(selected_repo, repo_analysis, context)
    
    async def _analyze_system_needs(self, context: Dict[str, Any], 
                                   charter: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what the system needs right now with enhanced intelligence.
        
        Args:
            context: System context
            charter: Current charter
            
        Returns:
            Need analysis
        """
        # Get enhanced context if available
        enhanced_context = context
        if self.context_aggregator:
            try:
                aggregated = await self.context_aggregator.gather_comprehensive_context()
                enhanced_context = self._merge_contexts(context, aggregated)
            except Exception as e:
                self.logger.warning(f"Failed to get enhanced context: {e}")
        
        # Get predictive insights if available
        predictions = []
        if self.predictive_engine:
            try:
                predictions = await self.predictive_engine.predict_next_tasks(enhanced_context)
                trends = await self.predictive_engine.analyze_trends(self.task_history)
                warnings = await self.predictive_engine.detect_early_warnings(
                    enhanced_context, self.task_history
                )
            except Exception as e:
                self.logger.warning(f"Failed to get predictions: {e}")
        
        prompt = f"""
        Analyze the current state of the AI development orchestrator system.
        
        System Charter:
        {json.dumps(charter, indent=2)}
        
        Current Context:
        - Active Projects: {json.dumps(context.get('projects', context.get('active_projects', [])), indent=2)}
        - Recent Tasks: {json.dumps(context.get('recent_tasks', [])[-5:], indent=2)}
        - System Capabilities: {json.dumps(context.get('capabilities', []), indent=2)}
        - Market Trends: {json.dumps(context.get('market_trends', [])[:3], indent=2)}
        - Recent Outcomes: {json.dumps(context.get('recent_outcomes', [])[-3:], indent=2)}
        
        Portfolio Analysis:
        - Total Projects: {len(context.get('projects', context.get('active_projects', [])))}
        - Project Types: {self._analyze_project_types(context.get('projects', context.get('active_projects', [])))}
        - Coverage Gaps: {json.dumps(self._identify_portfolio_gaps(context.get('projects', context.get('active_projects', []))), indent=2)}
        
        Determine what the system should focus on next:
        1. Does it need to expand the portfolio with a new project?
        2. Should it enhance existing projects with features?
        3. Does the AI system itself need improvements?
        4. Are there maintenance or quality issues to address?
        
        Consider:
        - Portfolio balance and gaps
        - Market opportunities from trends
        - System performance and capabilities
        - Strategic objectives from charter
        - ML Predictions: {json.dumps([p.__dict__ for p in predictions[:3]], indent=2, default=str) if predictions else 'None'}
        - Early Warnings: {json.dumps([w.__dict__ for w in warnings[:2]], indent=2, default=str) if 'warnings' in locals() else 'None'}
        
        Return analysis as JSON with:
        - need_type: 'portfolio_expansion', 'project_enhancement', 'system_improvement', 'quality_focus'
        - specific_need: Detailed description of what's needed
        - priority: 'critical', 'high', 'medium', 'low'
        - rationale: Why this need exists now
        - confidence: Float 0.0-1.0 indicating confidence in this analysis
        - alternatives: List of 2-3 alternative approaches considered
        - priority_rationale: Why this priority level was chosen
        - opportunity: What opportunity this addresses
        - suggested_approach: How to address this need
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        return self._parse_json_response(response)
    
    def _analyze_project_types(self, projects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of projects in portfolio.
        
        Args:
            projects: List of projects
            
        Returns:
            Project type counts
        """
        types = {}
        for project in projects:
            # Extract type from project name/description
            project_type = self._categorize_project(project)
            types[project_type] = types.get(project_type, 0) + 1
        return types
    
    def _categorize_project(self, project: Dict[str, Any]) -> str:
        """Categorize a project based on its characteristics.
        
        Args:
            project: Project details
            
        Returns:
            Project category
        """
        name = project.get('name', '').lower()
        description = project.get('description', '').lower()
        text = f"{name} {description}"
        
        # Simple categorization
        if any(word in text for word in ['auth', 'login', '2fa', 'oauth']):
            return 'authentication'
        elif any(word in text for word in ['api', 'rest', 'graphql']):
            return 'api_service'
        elif any(word in text for word in ['dashboard', 'admin', 'panel']):
            return 'dashboard'
        elif any(word in text for word in ['analytics', 'report', 'data']):
            return 'analytics'
        else:
            return 'general'
    
    def _merge_contexts(self, basic_context: Dict[str, Any], 
                       aggregated: AggregatedContext) -> Dict[str, Any]:
        """Merge basic and aggregated contexts.
        
        Args:
            basic_context: Basic context
            aggregated: Aggregated context from SmartContextAggregator
            
        Returns:
            Merged context
        """
        merged = basic_context.copy()
        
        # Add enhanced data
        merged['repository_health'] = aggregated.repository_health
        merged['technology_distribution'] = aggregated.technology_distribution
        merged['market_insights'] = aggregated.market_insights
        merged['cross_repo_patterns'] = aggregated.cross_repo_patterns
        merged['external_signals'] = aggregated.external_signals
        merged['strategic_priorities'] = aggregated.strategic_priorities
        
        return merged
    
    def _identify_portfolio_gaps(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify gaps in the project portfolio with repository context.
        
        Args:
            projects: Current projects
            
        Returns:
            List of identified gaps with context about which repositories triggered them
        """
        existing_types = {}
        project_analysis = {}
        
        # Analyze existing projects with repository context
        for project in projects:
            proj_type = self._categorize_project(project)
            repo_name = project.get('full_name', project.get('name', 'unknown'))
            
            if proj_type not in existing_types:
                existing_types[proj_type] = []
            existing_types[proj_type].append(repo_name)
            
            # Store project analysis for gap context
            project_analysis[repo_name] = {
                'type': proj_type,
                'health_score': project.get('health_score', 0),
                'open_issues': project.get('metrics', {}).get('issues_open', 0),
                'language': project.get('language', 'Unknown'),
                'last_activity': project.get('recent_activity', {}).get('last_commit_date', 'Unknown')
            }
        
        # Common project types for a complete portfolio
        desired_types = {
            'authentication', 'api_service', 'dashboard', 'analytics',
            'notification', 'payment', 'cms', 'mobile_backend'
        }
        
        # Identify gaps with context
        gaps = []
        missing_types = desired_types - set(existing_types.keys())
        
        for gap_type in list(missing_types)[:3]:  # Top 3 gaps
            # Find which repositories might benefit from this gap being filled
            related_repos = []
            for repo_name, analysis in project_analysis.items():
                # Repositories with low health scores or many open issues might benefit
                if analysis['health_score'] < 80 or analysis['open_issues'] > 5:
                    related_repos.append(repo_name)
            
            gap_info = {
                'gap_type': gap_type,
                'description': f"Missing {gap_type} capabilities in portfolio",
                'triggered_by_repositories': related_repos[:2],  # Top 2 repos that would benefit
                'existing_similar': existing_types.get(gap_type, []),
                'priority': 'high' if gap_type in ['payment', 'authentication', 'api_service'] else 'medium'
            }
            gaps.append(gap_info)
        
        return gaps
    
    async def _create_task_for_need(self, need: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   charter: Dict[str, Any]) -> Dict[str, Any]:
        """Create specific task for identified need.
        
        Args:
            need: Need analysis
            context: System context
            charter: System charter
            
        Returns:
            Generated task
        """
        # Check if we should skip due to existing similar tasks
        need_description = need.get('specific_need', '')
        if self._check_need_against_persistence(need_description):
            self.logger.info(f"Skipping task creation for need '{need_description}' - similar task exists")
            # Return a placeholder task that will be filtered out
            return {'skip': True, 'reason': 'duplicate_need'}
        # Prepare project information for better task assignment (excluding CWMAI)
        projects_info = []
        # Handle both 'projects' and 'active_projects' keys
        projects = context.get('projects', context.get('active_projects', []))
        
        # Convert list to dict if needed
        if isinstance(projects, list):
            projects_dict = {p.get('name', f'project_{i}'): p for i, p in enumerate(projects)}
        else:
            projects_dict = projects
            
        for proj_name, proj_data in projects_dict.items():
            # Skip excluded repositories (like CWMAI itself)
            repo_name = proj_data.get('full_name', proj_name)
            if not should_process_repo(repo_name):
                continue
                
            projects_info.append({
                'name': proj_name,
                'full_name': proj_data.get('full_name', proj_name),
                'description': proj_data.get('description', 'No description'),
                'language': proj_data.get('language', 'Unknown'),
                'topics': proj_data.get('topics', []),
                'health_score': proj_data.get('health_score', 0),
                'open_issues': proj_data.get('metrics', {}).get('issues_open', 0)
            })
        
        prompt = f"""
        Create a specific, actionable task to address this system need.
        
        Need Analysis:
        {json.dumps(need, indent=2)}
        
        System Charter Guidelines:
        - Task Types: {json.dumps(charter.get('TASK_TYPES', {}), indent=2)}
        - Project Methodology: {charter.get('PROJECT_METHODOLOGY', '')}
        - Decision Principles: {json.dumps(charter.get('DECISION_PRINCIPLES', []), indent=2)}
        
        Available Projects for Task Assignment:
        {json.dumps(projects_info, indent=2)}
        
        IMPORTANT: For FEATURE, ENHANCEMENT, BUG_FIX, or any project-specific task:
        - You MUST set 'target_project' to the repository name (e.g., 'ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations')
        - Match the task to the most appropriate project based on:
          - Project description and purpose
          - Technology stack (language, topics)
          - Current project needs (open issues, health score)
        - For system-wide tasks, leave target_project as null (system tasks are handled separately)
        - For NEW_PROJECT tasks, set target_project to null
        
        Based on the need type '{need.get('need_type', 'unknown')}', generate an appropriate task:
        
        If portfolio_expansion → Create a NEW_PROJECT task that:
          - Solves a real-world problem (DO NOT use generic examples)
          - Has clear monetization potential
          - Can generate revenue 24/7
          - Uses Laravel React starter kit as foundation
          - Research the problem space first
        If project_enhancement → Create a FEATURE task for a specific existing project
        If system_improvement → Create an improvement task for the AI system
        If quality_focus → Create appropriate testing/refactoring task
        
        CRITICAL RULES:
        1. NEW_PROJECT must solve a REAL problem, not a hypothetical one
        2. NEW_PROJECT must explicitly mention using Laravel React starter kit
        3. FEATURE must target a specific existing project by name
        4. improvement must enhance the AI orchestrator system itself
        5. NO hardcoded project ideas - research actual market needs
        
        Generate the task as JSON with:
        - type: The task type (use actual enum values)
        - title: Clear, specific title (max 100 chars)
        - description: Detailed description including:
          * What needs to be built/done
          * Why it's valuable
          * Key requirements
          * Technical approach
          * For NEW_PROJECT: How to fork and customize Laravel React starter kit
        - requirements: List of specific requirements (5-8 items)
        - priority: Task priority based on need
        - estimated_complexity: 'low', 'medium', 'high'
        - success_criteria: How to measure successful completion
        - target_project: For FEATURE tasks, which project to enhance
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        task = self._parse_json_response(response)
        
        # Ensure task has all required fields
        task = self._ensure_task_completeness(task)
        
        return task
    
    async def _analyze_repository_needs(self, repo_analysis: Dict[str, Any], 
                                       charter: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what a specific repository needs based on its analysis.
        
        Args:
            repo_analysis: Deep repository analysis
            charter: System charter
            
        Returns:
            Need analysis for the repository
        """
        # Perform deep repository analysis if available
        deep_analysis = await self._perform_deep_repository_analysis(repo_analysis)
        
        # Extract key information from analysis
        health_score = repo_analysis.get('health_metrics', {}).get('health_score', 0)
        specific_needs = repo_analysis.get('specific_needs', [])
        opportunities = repo_analysis.get('improvement_opportunities', [])
        issues = repo_analysis.get('issues_analysis', {})
        
        # Add deep analysis insights
        if deep_analysis:
            specific_needs.extend(deep_analysis.get('identified_needs', []))
            opportunities.extend(deep_analysis.get('opportunities', []))
        
        # Learn from past outcomes for this repository
        if self.learning_system:
            learned_priorities = await self._get_learned_task_priorities(
                repo_analysis.get('basic_info', {}).get('full_name', '')
            )
            # Adjust priorities based on learning
            specific_needs = self._adjust_needs_from_learning(specific_needs, learned_priorities)
        
        # Incorporate external intelligence
        if self.external_intelligence:
            external_needs = await self._get_external_intelligence_needs(repo_analysis)
            specific_needs.extend(external_needs)
        
        # Prioritize needs with enhanced scoring
        if specific_needs:
            # Use enhanced priority scoring
            highest_priority = max(specific_needs, 
                                 key=lambda x: self._calculate_need_priority_score(x, repo_analysis))
            
            return {
                'need_type': highest_priority['type'],
                'specific_need': highest_priority['description'],
                'priority': highest_priority['priority'],
                'suggested_action': highest_priority['suggested_action'],
                'confidence': highest_priority.get('confidence', 0.9),
                'alternatives': [n['description'] for n in specific_needs if n != highest_priority][:3],
                'deep_analysis_insights': deep_analysis,
                'external_factors': highest_priority.get('external_factors', [])
            }
        
        # If no specific needs, look at opportunities
        if opportunities:
            best_opportunity = max(opportunities, 
                                 key=lambda x: self._calculate_opportunity_score(x, repo_analysis))
            
            return {
                'need_type': best_opportunity['type'],
                'specific_need': best_opportunity['description'],
                'priority': 'medium',
                'suggested_action': f"Implement {best_opportunity['type']} improvement",
                'confidence': 0.7,
                'opportunity_score': best_opportunity.get('score', 0)
            }
        
        # Default: general improvement
        return {
            'need_type': 'maintenance',
            'specific_need': 'General repository maintenance and improvements',
            'priority': 'low',
            'suggested_action': 'Review and update repository',
            'confidence': 0.5
        }
    
    async def _create_repository_specific_task(self, repository_name: str,
                                             repo_analysis: Dict[str, Any],
                                             need_analysis: Dict[str, Any],
                                             charter: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task specifically tailored to the repository.
        
        Args:
            repository_name: Target repository
            repo_analysis: Repository analysis
            need_analysis: Identified needs
            charter: System charter
            
        Returns:
            Repository-specific task
        """
        # Check for existing similar tasks for this repository
        task_type = need_analysis.get('need_type', 'maintenance')
        if self._check_repository_task_against_persistence(repository_name, task_type):
            self.logger.info(f"Skipping task creation for {repository_name}/{task_type} - similar task exists")
            # Return a placeholder task that will be filtered out
            return {'skip': True, 'reason': 'duplicate_repository_task'}
        
        # Get lifecycle context if available
        lifecycle_context = repo_analysis.get('lifecycle_analysis', {})
        current_stage = lifecycle_context.get('current_stage', 'unknown')
        appropriate_tasks = lifecycle_context.get('appropriate_task_types', [])
        transition_plan = lifecycle_context.get('transition_plan', {})
        
        # Check if we're in development mode
        import os
        is_development = os.getenv('NODE_ENV', 'production').lower() == 'development'
        allowed_types_str = ""
        if is_development:
            allowed_types_str = "\nIMPORTANT: Do NOT create SYSTEM_IMPROVEMENT tasks. Only create tasks of type: FEATURE, BUG_FIX, ENHANCEMENT, DOCUMENTATION, TESTING, MAINTENANCE, OPTIMIZATION, RESEARCH, MONITORING, INTEGRATION, REFACTORING, ARCHITECTURE_DOCUMENTATION."
        
        # Extract architecture information if available
        architecture = repo_analysis.get('architecture', {})
        has_architecture = architecture and architecture.get('document_exists', False)
        
        architecture_context = ""
        if has_architecture:
            core_entities = architecture.get('core_entities', [])
            design_colors = architecture.get('design_system', {}).get('colors', {})
            feature_roadmap = architecture.get('feature_roadmap', [])
            
            architecture_context = f"""
        
        Project Architecture (from ARCHITECTURE.md):
        - Core Entities: {', '.join(core_entities) if core_entities else 'Not specified'}
        - Design System: {'Available' if design_colors else 'Not specified'}
        - Feature Roadmap: {len(feature_roadmap)} planned features
        - Planned Features: {json.dumps(feature_roadmap[:3], indent=2) if feature_roadmap else 'None'}
        
        IMPORTANT: Any tasks generated should align with the documented architecture and planned features.
        """
        
        prompt = f"""
        Create a specific task for the repository: {repository_name}
        
        Repository Analysis:
        - Health Score: {repo_analysis.get('health_metrics', {}).get('health_score', 'Unknown')}
        - Primary Language: {repo_analysis.get('basic_info', {}).get('language', 'Unknown')}
        - Open Issues: {repo_analysis.get('basic_info', {}).get('open_issues_count', 0)}
        - Last Updated: {repo_analysis.get('basic_info', {}).get('updated_at', 'Unknown')}
        - Tech Stack: {json.dumps(repo_analysis.get('technical_stack', {}), indent=2)}
        {architecture_context}
        
        Lifecycle Context:
        - Current Stage: {current_stage}
        - Stage-Appropriate Tasks: {json.dumps(appropriate_tasks, indent=2)}
        - Next Stage Requirements: {json.dumps(transition_plan.get('required_tasks', [])[:3], indent=2)}
        - Recommended Focus: {json.dumps(lifecycle_context.get('recommended_focus', []), indent=2)}
        
        Specific Repository Needs:
        {json.dumps(repo_analysis.get('specific_needs', []), indent=2)}
        
        Recent Issues:
        {json.dumps(repo_analysis.get('issues_analysis', {}).get('recent_issues', [])[:3], indent=2)}
        
        Identified Need:
        - Type: {need_analysis.get('need_type')}
        - Description: {need_analysis.get('specific_need')}
        - Suggested Action: {need_analysis.get('suggested_action')}
        - Priority: {need_analysis.get('priority')}
        
        IMPORTANT: Consider the project's lifecycle stage when creating the task.
        - For {current_stage} stage, focus on: {', '.join(appropriate_tasks[:3])}
        - Ensure the task helps progress toward the next stage
        {allowed_types_str}
        
        Create a task that:
        1. Directly addresses the identified need
        2. Is appropriate for the current lifecycle stage
        3. Helps move the project toward the next stage
        4. References specific files, issues, or code from the repository
        5. Is technically appropriate for the repository's stack
        6. Can be processed efficiently by the 24/7 AI system
        
        Special handling for missing architecture:
        - If the project lacks ARCHITECTURE.md and the need is for documentation
        - Consider creating an ARCHITECTURE_DOCUMENTATION task
        - This will analyze the codebase and generate comprehensive architecture docs
        
        Return as JSON with:
        - type: Task type (FEATURE, BUG_FIX, ENHANCEMENT, ARCHITECTURE_DOCUMENTATION, etc.)
        - title: Specific, actionable title mentioning the repository
        - description: Detailed description with context from the analysis
        - requirements: Specific implementation steps
        - priority: Based on repository health and need urgency
        - estimated_complexity: Based on the scope
        - success_criteria: How to measure completion
        - repository: {repository_name}
        - lifecycle_stage: {current_stage}
        - stage_progression_value: How this task helps reach the next stage
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        task = self._parse_json_response(response)
        
        # Check if we're in development mode and filter out SYSTEM_IMPROVEMENT
        if is_development and task.get('type') == 'SYSTEM_IMPROVEMENT':
            self.logger.warning("⚠️ AI generated SYSTEM_IMPROVEMENT task in dev mode, changing to MAINTENANCE")
            task['type'] = 'MAINTENANCE'
            task['title'] = task['title'].replace('Improve system', 'Maintain').replace('Enhance system', 'Update')
        
        # Ensure repository is set
        task['repository'] = repository_name
        
        # Add lifecycle metadata
        task['lifecycle_metadata'] = {
            'current_stage': current_stage,
            'appropriate_for_stage': task.get('type', '').lower() in [t.lower() for t in appropriate_tasks],
            'helps_transition': bool(transition_plan.get('required_tasks'))
        }
        
        # Ensure completeness
        task = self._ensure_task_completeness(task)
        
        return task
    
    async def _select_repository_for_task(self, context: Dict[str, Any]) -> Optional[str]:
        """Select which repository should receive the next task.
        
        Args:
            context: System context with repository information
            
        Returns:
            Selected repository name or None
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
            
        # Filter out excluded repositories
        eligible_repos = []
        for proj in project_list:
            repo_name = proj.get('full_name', proj.get('name', ''))
            if should_process_repo(repo_name):
                eligible_repos.append(proj)
                
        if not eligible_repos:
            return None
            
        # Score repositories for selection
        repo_scores = []
        for repo in eligible_repos:
            score = self._calculate_repository_priority_score(repo)
            repo_scores.append((repo.get('full_name', repo.get('name')), score))
            
        # Sort by score (highest first)
        repo_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return highest scoring repository
        return repo_scores[0][0] if repo_scores else None
    
    def _calculate_repository_priority_score(self, repo: Dict[str, Any]) -> float:
        """Calculate priority score for repository selection with enhanced intelligence.
        
        Args:
            repo: Repository information
            
        Returns:
            Priority score (higher = more likely to be selected)
        """
        score = 0.0
        repo_name = repo.get('full_name', repo.get('name', 'unknown'))
        score_breakdown = {}
        
        # Prefer repositories with lower health scores (need more attention)
        health_score = repo.get('health_score', 50)
        health_contribution = (100 - health_score) * 0.3
        score += health_contribution
        score_breakdown['health_factor'] = {
            'health_score': health_score,
            'contribution': health_contribution,
            'reason': f"Lower health score ({health_score}) indicates need for attention"
        }
        
        # Prefer repositories with open issues (indicates active needs)
        open_issues = repo.get('metrics', {}).get('issues_open', 0)
        if open_issues > 0:
            issue_contribution = min(20, open_issues) * 0.5
            score += issue_contribution
            score_breakdown['open_issues'] = {
                'issue_count': open_issues,
                'contribution': issue_contribution,
                'reason': f"{open_issues} open issues indicate active development needs"
            }
            
        # Prefer recently active repositories
        recent_activity = repo.get('recent_activity', {})
        days_since_commit = recent_activity.get('days_since_last_commit', 999)
        if days_since_commit < 30:
            activity_contribution = 20
            score += activity_contribution
            score_breakdown['activity'] = {
                'days_since_commit': days_since_commit,
                'contribution': activity_contribution,
                'reason': "Recently active repository (< 30 days) indicates ongoing development"
            }
        elif days_since_commit < 90:
            activity_contribution = 10
            score += activity_contribution
            score_breakdown['activity'] = {
                'days_since_commit': days_since_commit,
                'contribution': activity_contribution,
                'reason': "Moderately active repository (< 90 days)"
            }
        
        # Lifecycle stage priority
        lifecycle = repo.get('lifecycle_analysis', {})
        current_stage = lifecycle.get('current_stage')
        if current_stage:
            # Prioritize projects in active development stages
            stage_priority = {
                'inception': 25,  # High priority - needs foundation
                'early_development': 20,  # High priority - needs momentum
                'active_development': 15,  # Medium priority - maintain momentum
                'growth': 10,  # Lower priority - more stable
                'mature': 5,  # Low priority - mostly maintenance
                'maintenance': 3,  # Minimal priority
                'declining': 30  # Very high priority - needs revival
            }
            score += stage_priority.get(current_stage, 10)
            
            # Bonus for projects ready to transition
            transition_readiness = lifecycle.get('transition_plan', {}).get('current_readiness', 0)
            if transition_readiness > 0.7:
                score += 15  # Ready to advance to next stage
        
        # Cross-repository synergy bonus
        repo_name = repo.get('full_name', repo.get('name', ''))
        if self.cross_repo_awareness and hasattr(self, '_cross_repo_synergies'):
            synergy_score = self._cross_repo_synergies.get(repo_name, 0)
            score += synergy_score * 10
            
        # Predictive priority adjustment
        if self.predictive_engine and hasattr(self, '_predicted_priorities'):
            predicted_priority = self._predicted_priorities.get(repo_name, 0)
            score += predicted_priority * 15
            
        # External signal influence
        if self.external_intelligence and hasattr(self, '_external_signals'):
            for signal in self._external_signals:
                if repo_name in signal.get('affected_repos', []):
                    score += signal.get('urgency_modifier', 5)
        
        # Dynamic adjustment based on recent performance
        if self.dynamic_priority_adjustment:
            recent_success_rate = self._get_recent_success_rate(repo_name)
            if recent_success_rate > 0.8:
                score += 10  # Successful repos get more tasks
            elif recent_success_rate < 0.5:
                score -= 5   # Struggling repos need different approach
        
        return max(0, score)
    
    async def _generate_new_project_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a NEW_PROJECT task based on real market research.
        
        Args:
            context: System context
            
        Returns:
            New project task
        """
        charter = await self.charter_system.get_current_charter()
        
        # Research real-world problems first
        research_prompt = f"""
        You are an expert Venture Analyst and Startup Strategist with deep knowledge of global digital markets, technology trends, and business model innovation.

        **Primary Directive:**
        Your task is to identify and propose a list of viable, technology-driven business ideas that solve real-world, everyday problems. Leverage your most current knowledge and connection to Google Search to understand current market gaps and opportunities up to June 2025.

        **Constraints for each idea:**
        * It must be a new or underserved project concept, not a direct clone of an existing major platform.
        * It must be solvable with a web and/or mobile application built on a modern tech stack (like Laravel & React/TypeScript).
        * It must have a clear path to monetization (24/7 revenue potential).
        * It must be scalable, with a design that can grow from a niche audience to a larger market.

        **Task:**
        Generate a list of 3 to 5 distinct project ideas. For each idea, clearly define the target audience and the problem you are solving for them.

        The final output must be a single JSON object containing a single key "project_ideas", which is an array of objects. Each project object must strictly adhere to the following schema:

        **JSON Output Schema:**
        {{
          "project_ideas": [
            {{
              "project_name": "A catchy, brandable name for the business (e.g., 'SyncUp Workspace').",
              "project_goal": "A single, powerful sentence describing the mission of the project.",
              "problem_solved": "A clear description of the specific, everyday problem this business solves for its target audience.",
              "target_audience": "A description of the primary user base (e.g., 'Freelance creatives and small agencies', 'Remote-first tech companies', 'University students studying abroad').",
              "monetization_strategy": "A list of potential revenue streams (e.g., 'Subscription fees for premium features', 'Commission on transactions', 'Pay-per-use service charges').",
              "core_entities": "A list of the primary 'nouns' of the application, which will inform the database schema (e.g., ['Users', 'Teams', 'Projects', 'Documents', 'Permissions']).",
              "key_features": "A list of the essential features needed for a Minimum Viable Product (MVP) (e.g., ['Collaborative document editing', 'Task assignment and tracking', 'Team-based chat channels']).",
              "justification": "A brief rationale explaining why this idea is particularly well-suited for its target audience and has strong market potential."
            }}
          ]
        }}
        """
        
        research_response = await self.ai_brain.generate_enhanced_response(research_prompt, model='gemini')
        research_result = self._parse_json_response(research_response)
        
        # Select the best project idea from the research
        project_ideas = research_result.get('project_ideas', [])
        if not project_ideas:
            self.logger.error("No project ideas generated from research")
            return None
            
        # Select the first project idea (or implement selection logic)
        selected_project = project_ideas[0]
        
        # Generate architecture for the selected project
        architecture_prompt = f"""
        You are a distinguished Chief Technology Officer (CTO), a pragmatic Principal Engineer, and a skilled UI/UX Designer. Your expertise lies in creating scalable, secure, and user-friendly web applications using the Laravel and React/Inertia.js ecosystem with TypeScript.

        **Primary Directive:**
        Leverage your most current knowledge, including best practices, security standards, and package releases up to June 2025. All frontend code must be written in TypeScript. All design choices must adhere to accessibility standards (WCAG AA). Where a specific item is recommended, provide a brief justification and a link to its source.

        **Project Context:**
        * **Starting Point:** "The project will be bootstrapped using the official 'laravel/react-starter-kit', which uses TypeScript. All React components will have a .tsx extension."
        * **Technical Mandates:**
          * **Real-time Requirement:** "For any features that benefit from real-time updates, the architecture MUST use Redis and the 'laravel-echo-server' running locally."
        * **Non-Functional Requirements:**
          * **Testing Strategy:** "The architecture MUST include a comprehensive testing strategy (Pest for backend, Vitest & RTL for frontend)."
          * **Security Posture:** "A 'security-first' approach is mandatory, addressing OWASP Top 10 vulnerabilities."
          * **Observability:** "The plan must include a strategy for logging (e.g., 'stderr' channel) and monitoring key metrics."
        * **Design System Requirements:**
          * **Typography & Colors:** "Generate a simple, professional design system. The color palette must be accessible and logically structured. The typography must be clean and highly readable, sourced from Google Fonts."
        * **Project Name:** "{selected_project.get('project_name', 'New Project')}"
        * **Project Goal:** "{selected_project.get('project_goal', 'Build a web application')}"
        * **Core Entities:** "{', '.join(selected_project.get('core_entities', []))}"
        * **Key Features:** "{json.dumps(selected_project.get('key_features', []), indent=2)}"
        * **Expected Scale:** "Start with a small user base, but design with a clear path for significant scaling."
        * **Team Skills:** "The team is proficient with Laravel but intermediate with React and TypeScript."

        **Task:**
        Generate a **complete, production-grade System Architecture, Design System, and Feature Implementation Roadmap**. The document must provide a foundational engineering architecture AND a foundational design system, plus a detailed plan for implementing each "Key Feature".

        The final output must be a single JSON object that strictly adheres to the following tool schema.

        **Tool Schema (JSON Output):**
        {{
          "title": "Full-Stack Blueprint for {selected_project.get('project_name', 'New Project')}",
          "description": "A comprehensive architecture, design system, and feature roadmap.",
          "design_system": {{
            "suggested_font": {{
              "font_name": "e.g., Inter",
              "google_font_link": "A link to the Google Fonts page for the selected font.",
              "font_stack": "The CSS font-family stack (e.g., 'Inter', sans-serif).",
              "rationale": "A brief explanation of why this font was chosen (e.g., for its excellent readability at various sizes)."
            }},
            "color_palette": {{
              "rationale": "A brief explanation of the color theory behind the palette choice (e.g., 'A complementary palette chosen for its balance and professional feel').",
              "primary": {{ "name": "Primary", "hex": "#RRGGBB", "usage": "Main brand color, used for headers and primary actions." }},
              "secondary": {{ "name": "Secondary", "hex": "#RRGGBB", "usage": "Supporting color for secondary elements." }},
              "accent": {{ "name": "Accent", "hex": "#RRGGBB", "usage": "Used for call-to-action buttons and highlights." }},
              "neutral_text": {{ "name": "Neutral Text", "hex": "#RRGGBB", "usage": "Primary text color for high contrast and readability." }},
              "neutral_background": {{ "name": "Neutral Background", "hex": "#RRGGBB", "usage": "Main background color for content areas." }},
              "neutral_border": {{ "name": "Neutral Border", "hex": "#RRGGBB", "usage": "For card borders, dividers, and form inputs." }},
              "success": {{ "name": "Success", "hex": "#RRGGBB", "usage": "For success messages and confirmation." }},
              "warning": {{ "name": "Warning", "hex": "#RRGGBB", "usage": "For warnings and non-critical alerts." }},
              "danger": {{ "name": "Danger", "hex": "#RRGGBB", "usage": "For error messages and destructive actions." }}
            }}
          }},
          "foundational_architecture": {{
            "core_components": {{ "section_title": "1. Core Components & Rationale", "content": "..." }},
            "database_schema": {{ "section_title": "2. Database Schema Design", "content": "..." }},
            "api_design": {{ "section_title": "3. API Design & Key Endpoints", "content": "..." }},
            "frontend_structure": {{ "section_title": "4. Frontend Structure (TypeScript)", "content": "..." }},
            "real_time_architecture": {{ "section_title": "5. Real-time & Events Architecture", "content": "..." }},
            "auth_flow": {{ "section_title": "6. Authentication & Authorization Flow", "content": "..." }},
            "deployment_plan": {{ "section_title": "7. Deployment & Scalability Plan", "content": "..." }},
            "testing_strategy": {{ "section_title": "8. Testing Strategy", "content": "..." }},
            "security_hardening_plan": {{ "section_title": "9. Security & Hardening Plan", "content": "..." }},
            "logging_and_observability": {{ "section_title": "10. Logging & Observability", "content": "..." }}
          }},
          "feature_implementation_roadmap": [
            {{
              "feature_name": "Example: Real-time User Notifications",
              "description": "A brief summary of the feature's purpose and user-facing behavior.",
              "required_db_changes": [
                "A 'notifications' table with columns: id, user_id (FK), type, data (JSON), read_at (nullable timestamp)."
              ],
              "impacted_backend_components": [
                "A new 'NotificationService' class to handle notification creation.",
                "A new 'NotificationsController' to fetch and mark notifications as read."
              ],
              "impacted_frontend_components": [
                "A new 'NotificationBell.tsx' component in the main layout.",
                "A new '/notifications' page component (`Pages/Notifications/Index.tsx`)."
              ],
              "new_api_endpoints": [
                "GET /api/v1/notifications",
                "POST /api/v1/notifications/{{id}}/mark-as-read"
              ],
              "real_time_events": [
                "A 'UserNotification' event broadcast on a private channel: 'users.{{user_id}}'."
              ],
              "suggested_tests": [
                "Unit test for `NotificationService` to ensure notifications are created correctly.",
                "Component test for `NotificationBell.tsx` to ensure it displays the unread count."
              ]
            }}
          ]
        }}
        """
        
        architecture_response = await self.ai_brain.generate_enhanced_response(architecture_prompt, model='gemini')
        architecture = self._parse_json_response(architecture_response)
        
        # Generate task based on research and architecture
        prompt = f"""
        Create a NEW_PROJECT task based on this market research and architecture.
        
        Selected Project:
        {json.dumps(selected_project, indent=2)}
        
        Architecture Blueprint:
        {json.dumps(architecture, indent=2)}
        
        Create a NEW_PROJECT task for a complete application that:
        1. Directly addresses the researched problem
        2. Uses Laravel React starter kit as the foundation
        3. Has clear path to monetization
        4. Can scale to serve many users
        5. Leverages AI capabilities where appropriate
        
        The task should include:
        - type: 'NEW_PROJECT'
        - title: Specific, descriptive title
        - description: Detailed description including:
          * The real problem being solved
          * Target audience and their pain points
          * How the Laravel React starter kit will be customized
          * Key features that solve the problem
          * Monetization strategy
          * Why this project is valuable
        - requirements: 5-8 specific implementation requirements
        - priority: Based on market opportunity
        - metadata: Including problem_statement, target_audience, monetization_model, architecture
        
        Return as JSON with all required task fields.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        task = self._parse_json_response(response)
        
        # Ensure it's a NEW_PROJECT type
        task['type'] = 'NEW_PROJECT'
        task['repository'] = None  # New projects don't have a repository yet
        
        # Add research metadata and architecture
        if 'metadata' not in task:
            task['metadata'] = {}
        task['metadata']['research_based'] = True
        task['metadata']['research_result'] = research_result
        task['metadata']['selected_project'] = selected_project
        task['metadata']['architecture'] = architecture
        
        # Ensure Laravel React starter kit integration
        task = await self._ensure_starter_kit_integration(task)
        
        return task
    
    def _ensure_task_completeness(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure task has all required fields.
        
        Args:
            task: Generated task
            
        Returns:
            Complete task
        """
        required_fields = ['type', 'title', 'description', 'requirements', 'priority']
        
        for field in required_fields:
            if field not in task:
                self.logger.warning(f"Task missing required field: {field}")
                # Add minimal default
                if field == 'type':
                    task['type'] = 'improvement'
                elif field == 'requirements':
                    task['requirements'] = []
                else:
                    task[field] = 'Not specified'
        
        # Ensure repository field exists
        if 'repository' not in task:
            task['repository'] = None
            
        # For project-specific tasks, log if repository is missing
        project_specific_types = ['FEATURE', 'ENHANCEMENT', 'BUG_FIX', 'REFACTOR', 'TEST']
        if task.get('type', '').upper() in project_specific_types and not task.get('repository'):
            self.logger.warning(f"Task type {task.get('type')} should have a repository but doesn't")
                    
        return task
    
    async def _ensure_starter_kit_integration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure NEW_PROJECT tasks properly use Laravel React starter kit.
        
        Args:
            task: Generated task
            
        Returns:
            Updated task
        """
        description = task.get('description', '')
        
        # Check if starter kit is mentioned
        if 'starter kit' not in description.lower() and 'laravel react' not in description.lower():
            prompt = f"""
            This NEW_PROJECT task needs to properly integrate the Laravel React starter kit.
            
            Current Task:
            {json.dumps(task, indent=2)}
            
            Update the task to:
            1. Explicitly state it will fork https://github.com/laravel/react-starter-kit
            2. Explain how to customize the starter kit for this specific project
            3. List what modifications are needed from the base template
            4. Include setup steps specific to this project type
            5. Maintain all Laravel and React best practices
            
            The starter kit provides:
            - Laravel 10+ backend with Sanctum auth
            - React 18+ with TypeScript frontend
            - Tailwind CSS styling
            - MySQL database setup
            - Redis for caching
            - Docker configuration
            - CI/CD pipeline templates
            
            Return the complete updated task maintaining the same structure.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            updated_task = self._parse_json_response(response)
            
            # Merge updates
            if updated_task:
                task.update(updated_task)
                
        return task
    
    async def _apply_learned_improvements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvements based on learning system insights.
        
        Args:
            task: Generated task
            
        Returns:
            Improved task
        """
        if not self.learning_system:
            return task
            
        # Get recommendations
        recommendations = await self.learning_system.get_recommendations()
        
        if recommendations.get('status') == 'insufficient_data':
            return task
            
        prompt = f"""
        Improve this task based on learned insights about what creates value.
        
        Generated Task:
        {json.dumps(task, indent=2)}
        
        Learning Insights:
        - Success Patterns: {json.dumps(recommendations.get('success_template', {}), indent=2)}
        - Task Priorities: {json.dumps(recommendations.get('task_priorities', []), indent=2)}
        - Things to Avoid: {json.dumps(recommendations.get('avoid_list', []), indent=2)}
        
        Improve the task by:
        1. Incorporating successful patterns
        2. Avoiding known failure patterns
        3. Aligning with recommended priorities
        4. Making it more specific and actionable
        5. Ensuring high value creation potential
        
        Return the improved task with the same structure.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        improved_task = self._parse_json_response(response)
        
        if improved_task:
            # Keep original type and key fields
            improved_task['type'] = task['type']
            return improved_task
            
        return task
    
    async def _modify_for_higher_value(self, task: Dict[str, Any], 
                                      prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Modify task based on value prediction to increase potential value.
        
        Args:
            task: Original task
            prediction: Value prediction with improvements
            
        Returns:
            Modified task
        """
        improvements = prediction.get('value_improvements', [])
        
        if not improvements:
            return task
            
        prompt = f"""
        Modify this task to increase its potential value based on predictions.
        
        Original Task:
        {json.dumps(task, indent=2)}
        
        Predicted Issues:
        - Predicted Value: {prediction.get('predicted_value', 0)}
        - Risks: {json.dumps(prediction.get('risks', []), indent=2)}
        
        Suggested Improvements:
        {json.dumps(improvements, indent=2)}
        
        Modify the task to:
        1. Address identified risks
        2. Incorporate value improvements
        3. Increase success probability
        4. Maintain task feasibility
        
        Return the modified task with the same structure but improved content.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        modified_task = self._parse_json_response(response)
        
        if modified_task:
            # Preserve type
            modified_task['type'] = task['type']
            return modified_task
            
        return task
    
    def _record_generation(self, task: Dict[str, Any], need: Dict[str, Any]) -> None:
        """Record task generation for pattern analysis.
        
        Args:
            task: Generated task
            need: Need analysis that led to task
        """
        # Skip recording placeholder/skipped tasks
        if task.get('skip'):
            return
            
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'need_analysis': need,
            'generation_id': f"gen_{len(self.task_history)}_{datetime.now(timezone.utc).timestamp()}"
        }
        
        self.task_history.append(record)
        
        # Update generation patterns
        need_type = need.get('need_type', 'unknown')
        if need_type not in self.generation_patterns:
            self.generation_patterns[need_type] = []
            
        self.generation_patterns[need_type].append({
            'task_type': task.get('type'),
            'priority': task.get('priority'),
            'timestamp': record['timestamp']
        })
        
        # Also record to persistence system for future duplicate detection
        work_item = self._create_work_item_from_task(task)
        if work_item:
            # Simulate a completion record for duplicate prevention
            execution_result = {
                'status': 'generated',
                'value_created': 0.0,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            self.task_persistence.record_completed_task(work_item, execution_result)
    
    def _get_recent_success_rate(self, repo_name: str) -> float:
        """Get recent success rate for repository.
        
        Args:
            repo_name: Repository name
            
        Returns:
            Success rate (0-1)
        """
        recent_tasks = [t for t in self.task_history[-20:] 
                       if t.get('task', {}).get('repository') == repo_name]
        
        if not recent_tasks:
            return 0.7  # Default neutral rate
        
        successful = sum(1 for t in recent_tasks 
                        if t.get('task', {}).get('status') == 'completed')
        
        return successful / len(recent_tasks)
    
    async def generate_multiple_tasks(self, context: Dict[str, Any], 
                                     count: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple diverse tasks with enhanced anti-duplication logic.
        
        Args:
            context: System context
            count: Number of tasks to generate
            
        Returns:
            List of generated tasks
        """
        # Update intelligence features before generation
        await self._update_intelligence_features(context)
        
        tasks = []
        used_repositories = set()
        used_task_types = set()
        used_gap_types = set()
        
        # Load existing tasks to avoid duplicates
        existing_tasks = self._load_existing_tasks()
        existing_titles = [task.get('title', '').lower() for task in existing_tasks]
        existing_repos = [task.get('repository') for task in existing_tasks if task.get('repository')]
        
        attempts = 0
        max_attempts = count * 5  # More attempts for better diversity
        
        self.logger.info(f"Generating {count} diverse tasks. Existing: {len(existing_tasks)} tasks, {len(set(existing_repos))} repositories")
        
        while len(tasks) < count and attempts < max_attempts:
            attempts += 1
            
            # Update context with already generated tasks and diversity constraints
            generation_context = context.copy()
            generation_context['pending_tasks'] = tasks
            generation_context['existing_tasks'] = existing_tasks
            generation_context['used_repositories'] = list(used_repositories)
            generation_context['used_task_types'] = list(used_task_types)
            generation_context['used_gap_types'] = list(used_gap_types)
            
            # Generate diverse tasks by avoiding same need types
            task = await self.generate_task(generation_context)
            
            if not task:
                self.logger.warning(f"Task generation failed on attempt {attempts}")
                continue
                
            # Skip placeholder tasks that were filtered due to duplicates
            if task.get('skip'):
                self.logger.info(f"Task skipped due to: {task.get('reason', 'unknown')}")
                continue
            
            # Enhanced duplicate detection
            is_duplicate = False
            task_title = task.get('title', '').lower()
            task_repo = task.get('repository')
            task_type = task.get('type')
            
            # Check against task persistence system
            work_item = self._create_work_item_from_task(task)
            if work_item and self.task_persistence.is_duplicate_task(work_item):
                self.logger.info(f"DUPLICATE PREVENTION (Persistence): Skipping duplicate task: {task.get('title')}")
                self.task_persistence.record_skipped_task(task.get('title', ''), "duplicate")
                is_duplicate = True
            
            # Check title duplicates
            elif task_title in existing_titles:
                self.logger.info(f"DUPLICATE PREVENTION: Skipping duplicate title: {task.get('title')}")
                is_duplicate = True
                
            # Check batch duplicates
            elif any(t.get('title', '').lower() == task_title for t in tasks):
                self.logger.info(f"DUPLICATE PREVENTION: Skipping duplicate in current batch: {task.get('title')}")
                is_duplicate = True
                
            # Repository-specific duplicate prevention
            elif task_repo and task_repo in used_repositories and task_type in used_task_types:
                self.logger.info(f"DUPLICATE PREVENTION: Repository {task_repo} already has {task_type} task in this batch")
                is_duplicate = True
                
            # Semantic similarity check
            elif task_repo:
                for existing_task in tasks:
                    if (existing_task.get('repository') == task_repo and 
                        existing_task.get('type') == task_type and
                        self._semantic_similarity_check(task.get('description', ''), existing_task.get('description', '')) > 0.8):
                        self.logger.info(f"DUPLICATE PREVENTION: Semantically similar task already exists for {task_repo}")
                        is_duplicate = True
                        break
            
            if is_duplicate:
                continue
            
            # Ensure diversity within the batch
            if len(tasks) > 0:
                task = await self._ensure_enhanced_task_diversity(task, tasks, generation_context)
                if not task:  # Failed to create diverse task
                    continue
                    
            # Track diversity metrics
            if task_repo:
                used_repositories.add(task_repo)
            if task_type:
                used_task_types.add(task_type)
            
            # Track gap types for NEW_PROJECT tasks
            if task_type == 'NEW_PROJECT':
                gap_type = self._extract_gap_type_from_task(task)
                if gap_type:
                    used_gap_types.add(gap_type)
                
            tasks.append(task)
            existing_titles.append(task_title)
            
            self.logger.info(f"SUCCESS: Generated diverse task {len(tasks)}/{count}: {task.get('title')} (repo: {task_repo}, type: {task_type})")
            
        if len(tasks) < count:
            self.logger.warning(f"Could only generate {len(tasks)} unique tasks out of {count} requested after {attempts} attempts")
        else:
            self.logger.info(f"Successfully generated {len(tasks)} diverse tasks covering {len(used_repositories)} repositories")
            
        return tasks
    
    def _load_existing_tasks(self) -> List[Dict[str, Any]]:
        """Load existing tasks from task state file.
        
        Returns:
            List of existing tasks
        """
        import os
        import json
        
        if not os.path.exists(self.task_state_file):
            return []
            
        try:
            with open(self.task_state_file, 'r') as f:
                state = json.load(f)
                tasks = state.get('tasks', {})
                
                # Convert dict of tasks to list, filtering out completed/cancelled
                active_tasks = []
                for task_id, task in tasks.items():
                    if task.get('status') not in ['completed', 'cancelled']:
                        active_tasks.append(task)
                        
                return active_tasks
                
        except Exception as e:
            self.logger.warning(f"Error loading existing tasks: {e}")
            return []
    
    def _semantic_similarity_check(self, desc1: str, desc2: str) -> float:
        """Enhanced semantic similarity check for duplicate detection.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Similarity score between 0 and 1
        """
        if not desc1 or not desc2:
            return 0.0
            
        # Normalize and tokenize
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Key phrase matching (higher weight for exact technical phrases)
        key_phrases = [
            'payment platform', 'user authentication', 'api integration', 
            'dashboard management', 'real-time notifications', 'cms system',
            'analytics dashboard', 'mobile backend', 'security audit'
        ]
        
        phrase_matches = 0
        for phrase in key_phrases:
            if phrase in desc1.lower() and phrase in desc2.lower():
                phrase_matches += 1
        
        phrase_similarity = phrase_matches / len(key_phrases) if key_phrases else 0.0
        
        # Weighted combination
        return (jaccard_sim * 0.7) + (phrase_similarity * 0.3)
    
    def _extract_gap_type_from_task(self, task: Dict[str, Any]) -> str:
        """Extract the gap type from a NEW_PROJECT task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Gap type string
        """
        title = task.get('title', '').lower()
        description = task.get('description', '').lower()
        text = f"{title} {description}"
        
        gap_keywords = {
            'payment': ['payment', 'billing', 'subscription', 'stripe', 'paypal'],
            'authentication': ['auth', 'login', 'oauth', 'jwt', 'session'],
            'analytics': ['analytics', 'metrics', 'reporting', 'dashboard'],
            'notification': ['notification', 'email', 'sms', 'push'],
            'cms': ['cms', 'content', 'blog', 'article'],
            'api_service': ['api', 'service', 'microservice', 'rest'],
            'mobile_backend': ['mobile', 'app', 'ios', 'android']
        }
        
        for gap_type, keywords in gap_keywords.items():
            if any(keyword in text for keyword in keywords):
                return gap_type
        
        return 'general'
    
    async def _ensure_enhanced_task_diversity(self, task: Dict[str, Any], 
                                            existing_tasks: List[Dict[str, Any]],
                                            context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced task diversity ensuring with context awareness.
        
        Args:
            task: New task to check
            existing_tasks: Already generated tasks in this batch
            context: Generation context with diversity constraints
            
        Returns:
            Diverse task or None if diversity cannot be achieved
        """
        task_type = task.get('type')
        task_repo = task.get('repository')
        
        # Check type diversity within batch
        existing_types = [t.get('type') for t in existing_tasks]
        if existing_types.count(task_type) >= 2:  # Max 2 of same type per batch
            self.logger.info(f"DIVERSITY: Too many {task_type} tasks in batch, generating alternative")
            return await self._generate_alternative_task_type(task, context, existing_types)
        
        # Check repository diversity
        if task_repo:
            existing_repos = [t.get('repository') for t in existing_tasks]
            if existing_repos.count(task_repo) >= 1:  # Max 1 per repository per batch
                self.logger.info(f"DIVERSITY: Repository {task_repo} already has task in batch, finding alternative")
                return await self._generate_alternative_repository_task(task, context, existing_repos)
        
        return task
    
    async def _generate_alternative_task_type(self, original_task: Dict[str, Any], 
                                            context: Dict[str, Any], 
                                            existing_types: List[str]) -> Optional[Dict[str, Any]]:
        """Generate alternative task with different type.
        
        Args:
            original_task: Original task
            context: Generation context
            existing_types: Types already used in batch
            
        Returns:
            Alternative task or None
        """
        alternative_types = ['FEATURE', 'BUG_FIX', 'DOCUMENTATION', 'TESTING', 'SECURITY']
        available_types = [t for t in alternative_types if existing_types.count(t) < 2]
        
        if not available_types:
            return None
            
        # Try to generate alternative using different repository
        return await self._select_repository_for_task(context)
    
    async def _generate_alternative_repository_task(self, original_task: Dict[str, Any],
                                                  context: Dict[str, Any],
                                                  existing_repos: List[str]) -> Optional[Dict[str, Any]]:
        """Generate task for different repository.
        
        Args:
            original_task: Original task
            context: Generation context  
            existing_repos: Repositories already used in batch
            
        Returns:
            Alternative task or None
        """
        # Find unused repository
        all_projects = context.get('projects', context.get('active_projects', []))
        if isinstance(all_projects, dict):
            available_repos = [name for name in all_projects.keys() if name not in existing_repos]
        else:
            available_repos = [p.get('full_name', p.get('name')) for p in all_projects 
                             if p.get('full_name', p.get('name')) not in existing_repos]
        
        if available_repos:
            selected_repo = available_repos[0]  # Select first available
            return await self.generate_task_for_repository(selected_repo, {}, context)
        
        return None
    
    async def _ensure_task_diversity(self, task: Dict[str, Any], 
                                    existing_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure task is diverse from existing tasks.
        
        Args:
            task: New task
            existing_tasks: Already generated tasks
            
        Returns:
            Potentially modified task for diversity
        """
        # Check if task is too similar to existing ones
        task_types = [t.get('type') for t in existing_tasks]
        
        if task.get('type') in task_types:
            # Try to generate a different type
            prompt = f"""
            This task is too similar to already generated tasks.
            
            Current Task:
            {json.dumps(task, indent=2)}
            
            Existing Task Types: {task_types}
            
            Generate a different but equally valuable task that:
            1. Uses a different task type if possible
            2. Addresses a different aspect of the system
            3. Maintains high value potential
            4. Complements the existing tasks
            
            Return a completely different task.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            diverse_task = self._parse_json_response(response)
            
            if diverse_task and diverse_task.get('type') not in task_types:
                return diverse_task
                
        return task
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed JSON or empty dict
        """
        content = response.get('content', '')
        
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        return {}
    
    async def _update_intelligence_features(self, context: Dict[str, Any]) -> None:
        """Update intelligence features before task generation.
        
        Args:
            context: Current context
        """
        # Update cross-repository synergies
        if self.cross_repo_awareness and self.context_aggregator:
            try:
                aggregated = await self.context_aggregator.gather_comprehensive_context()
                self._cross_repo_synergies = {}
                for pattern in aggregated.cross_repo_patterns:
                    for repo in pattern.get('repositories', []):
                        self._cross_repo_synergies[repo] = pattern.get('confidence', 0)
            except Exception as e:
                self.logger.warning(f"Failed to update cross-repo synergies: {e}")
        
        # Update predicted priorities
        if self.predictive_generation and self.predictive_engine:
            try:
                predictions = await self.predictive_engine.predict_next_tasks(context)
                self._predicted_priorities = {}
                for pred in predictions:
                    self._predicted_priorities[pred.repository] = pred.confidence
            except Exception as e:
                self.logger.warning(f"Failed to update predicted priorities: {e}")
        
        # Update external signals
        if self.external_intelligence and self.context_aggregator:
            try:
                aggregated = await self.context_aggregator.gather_comprehensive_context()
                self._external_signals = aggregated.external_signals
            except Exception as e:
                self.logger.warning(f"Failed to update external signals: {e}")
    
    def get_generation_analytics(self) -> Dict[str, Any]:
        """Get analytics about task generation patterns.
        
        Returns:
            Generation analytics
        """
        analytics = {
            'total_generated': len(self.task_history),
            'generation_patterns': self.generation_patterns,
            'recent_tasks': [r['task'] for r in self.task_history[-5:]],
            'need_distribution': self._analyze_need_distribution(),
            'intelligence_features': {
                'cross_repo_awareness': self.cross_repo_awareness,
                'predictive_generation': self.predictive_generation,
                'external_intelligence': self.external_intelligence,
                'dynamic_priority_adjustment': self.dynamic_priority_adjustment
            },
            'learning_metrics': self._get_learning_metrics(),
            'task_quality_metrics': self._calculate_task_quality_metrics(),
            'intelligence_effectiveness': self._measure_intelligence_effectiveness()
        }
        
        # Add predictive analytics if available
        if self.predictive_engine:
            analytics['prediction_confidence'] = self.predictive_engine.get_prediction_confidence()
            analytics['prediction_accuracy'] = self._calculate_prediction_accuracy()
        
        # Add context quality if available
        if self.context_aggregator:
            try:
                analytics['context_quality'] = asyncio.run(
                    self.context_aggregator.get_context_quality_report()
                )
            except:
                pass
        
        # Add continuous improvement metrics
        analytics['improvement_rate'] = self._calculate_improvement_rate()
        analytics['adaptation_score'] = self._calculate_adaptation_score()
        
        return analytics
    
    def _analyze_need_distribution(self) -> Dict[str, int]:
        """Analyze distribution of need types.
        
        Returns:
            Need type counts
        """
        distribution = {}
        for record in self.task_history:
            need_type = record['need_analysis'].get('need_type', 'unknown')
            distribution[need_type] = distribution.get(need_type, 0) + 1
            
        return distribution
    
    def _check_need_against_persistence(self, need_description: str) -> bool:
        """Check if a similar task for this need already exists in persistence.
        
        Args:
            need_description: Description of the need
            
        Returns:
            True if a similar task exists and is in cooldown
        """
        # Get recent task history
        recent_tasks = self.task_persistence.get_task_history(hours_back=168)  # 1 week
        
        # Check for similar needs
        for task in recent_tasks:
            # Check if task description contains key phrases from need
            if self._contains_key_phrases(task.title.lower(), need_description.lower()):
                self.logger.debug(f"Found similar task in history: {task.title}")
                return True
                
        return False
    
    def _check_repository_task_against_persistence(self, repository: str, task_type: str) -> bool:
        """Check if a similar task for this repository/type already exists.
        
        Args:
            repository: Repository name
            task_type: Type of task
            
        Returns:
            True if a similar task exists and is in cooldown
        """
        # Get repository-specific task history
        repo_tasks = self.task_persistence.get_task_history(
            repository=repository, 
            task_type=task_type.upper(),
            hours_back=72  # 3 days for repository-specific tasks
        )
        
        # If any matching tasks found, they're in cooldown
        if repo_tasks:
            self.logger.debug(f"Found {len(repo_tasks)} recent {task_type} tasks for {repository}")
            return True
            
        return False
    
    def _create_work_item_from_task(self, task: Dict[str, Any]) -> Optional[WorkItem]:
        """Create a WorkItem from a generated task for persistence checking.
        
        Args:
            task: Generated task dictionary
            
        Returns:
            WorkItem or None if conversion fails
        """
        try:
            # Map priority strings to TaskPriority enum
            priority_map = {
                'critical': TaskPriority.CRITICAL,
                'high': TaskPriority.HIGH,
                'medium': TaskPriority.MEDIUM,
                'low': TaskPriority.LOW,
                'background': TaskPriority.BACKGROUND
            }
            
            priority = priority_map.get(
                task.get('priority', 'medium').lower(), 
                TaskPriority.MEDIUM
            )
            
            return WorkItem(
                id=f"gen_{datetime.now(timezone.utc).timestamp()}_{uuid.uuid4().hex[:8]}",
                task_type=task.get('type', 'unknown').upper(),
                title=task.get('title', ''),
                description=task.get('description', ''),
                priority=priority,
                repository=task.get('repository'),
                metadata=task.get('metadata', {})
            )
        except Exception as e:
            self.logger.warning(f"Failed to create WorkItem from task: {e}")
            return None
    
    def _contains_key_phrases(self, text1: str, text2: str) -> bool:
        """Check if two texts contain similar key phrases.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts contain similar key phrases
        """
        # Extract key phrases (3+ word sequences)
        def extract_phrases(text):
            words = text.split()
            phrases = set()
            for i in range(len(words) - 2):
                phrases.add(' '.join(words[i:i+3]))
            return phrases
        
        phrases1 = extract_phrases(text1)
        phrases2 = extract_phrases(text2)
        
        # Check for common phrases
        common_phrases = phrases1.intersection(phrases2)
        
        # If significant overlap, consider similar
        if len(common_phrases) >= 2:
            return True
            
        # Also check for specific known duplicate patterns
        duplicate_patterns = [
            ('documentation', 'readme'),
            ('test', 'testing'),
            ('security', 'audit'),
            ('api', 'endpoint'),
            ('dashboard', 'interface'),
            ('authentication', 'auth'),
            ('comprehensive documentation', 'readme'),
            ('update documentation', 'readme'),
            ('create documentation', 'readme'),
            ('write comprehensive documentation', 'documentation'),
            ('create readme', 'documentation'),
            ('update readme', 'documentation')
        ]
        
        for pattern1, pattern2 in duplicate_patterns:
            if (pattern1 in text1 and pattern1 in text2) or \
               (pattern2 in text1 and pattern2 in text2):
                return True
                
        return False
    
    async def _perform_deep_repository_analysis(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep analysis of repository for smarter task generation.
        
        Args:
            repo_analysis: Basic repository analysis
            
        Returns:
            Deep analysis insights
        """
        repo_name = repo_analysis.get('basic_info', {}).get('full_name', 'unknown')
        
        prompt = f"""
        Perform deep analysis of repository {repo_name} to identify hidden needs and opportunities.
        
        Current Analysis:
        - Health Score: {repo_analysis.get('health_metrics', {}).get('health_score', 0)}
        - Tech Stack: {json.dumps(repo_analysis.get('technical_stack', {}), indent=2)}
        - Recent Activity: {json.dumps(repo_analysis.get('recent_activity', {}), indent=2)}
        - Open Issues: {repo_analysis.get('basic_info', {}).get('open_issues_count', 0)}
        
        Analyze for:
        1. Code complexity and technical debt indicators
        2. Missing test coverage areas
        3. Security vulnerabilities based on tech stack
        4. Performance optimization opportunities
        5. Dependency update needs
        6. Architecture improvement possibilities
        7. Documentation gaps
        8. Integration opportunities with other services
        
        Return as JSON with:
        - identified_needs: List of specific needs with priority and confidence
        - opportunities: List of improvement opportunities
        - risk_factors: Potential risks that need attention
        - technical_debt_score: 0-100 score
        - recommended_focus_areas: Top 3 areas to focus on
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            return self._parse_json_response(response) or {}
        except Exception as e:
            self.logger.warning(f"Deep analysis failed: {e}")
            return {}
    
    async def _get_learned_task_priorities(self, repository: str) -> Dict[str, float]:
        """Get learned task priorities for a repository.
        
        Args:
            repository: Repository name
            
        Returns:
            Task type priorities based on past success
        """
        if not self.learning_system:
            return {}
            
        try:
            # Get historical task outcomes for this repository
            history = await self.learning_system.get_repository_history(repository)
            
            # Calculate success rates by task type
            priorities = {}
            for task_type in ['FEATURE', 'BUG_FIX', 'OPTIMIZATION', 'SECURITY', 'DOCUMENTATION']:
                success_rate = await self.learning_system.get_task_type_success_rate(
                    repository, task_type
                )
                value_created = await self.learning_system.get_average_value_created(
                    repository, task_type
                )
                
                # Combined priority score
                priorities[task_type] = (success_rate * 0.6) + (value_created * 0.4)
                
            return priorities
        except Exception as e:
            self.logger.warning(f"Failed to get learned priorities: {e}")
            return {}
    
    def _adjust_needs_from_learning(self, needs: List[Dict[str, Any]], 
                                   learned_priorities: Dict[str, float]) -> List[Dict[str, Any]]:
        """Adjust needs based on learned priorities.
        
        Args:
            needs: List of identified needs
            learned_priorities: Learned task type priorities
            
        Returns:
            Adjusted needs list
        """
        for need in needs:
            task_type = need.get('type', '').upper()
            if task_type in learned_priorities:
                # Boost priority based on learning
                priority_boost = learned_priorities[task_type]
                need['learning_score'] = priority_boost
                need['confidence'] = min(1.0, need.get('confidence', 0.5) * (1 + priority_boost))
                
        return needs
    
    async def _get_external_intelligence_needs(self, repo_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get needs based on external intelligence sources.
        
        Args:
            repo_analysis: Repository analysis
            
        Returns:
            List of externally identified needs
        """
        external_needs = []
        tech_stack = repo_analysis.get('technical_stack', {})
        
        # Check security advisories
        security_needs = await self._check_security_advisories(tech_stack)
        external_needs.extend(security_needs)
        
        # Check framework updates
        update_needs = await self._check_framework_updates(tech_stack)
        external_needs.extend(update_needs)
        
        # Check industry trends
        trend_needs = await self._check_industry_trends(repo_analysis)
        external_needs.extend(trend_needs)
        
        return external_needs
    
    async def _check_security_advisories(self, tech_stack: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for security advisories affecting the tech stack.
        
        Args:
            tech_stack: Repository technology stack
            
        Returns:
            Security-related needs
        """
        needs = []
        
        # Simulate checking security advisories
        frameworks = tech_stack.get('frameworks', [])
        dependencies = tech_stack.get('dependencies', {})
        
        prompt = f"""
        Check for security advisories for these technologies:
        - Frameworks: {json.dumps(frameworks, indent=2)}
        - Key Dependencies: {json.dumps(list(dependencies.keys())[:10], indent=2)}
        
        Identify any critical security updates or vulnerabilities that need attention.
        
        Return as JSON list of security needs, each with:
        - type: 'SECURITY'
        - description: Specific security issue
        - priority: 'critical', 'high', 'medium'
        - suggested_action: What to do
        - external_factors: ['security_advisory']
        - confidence: 0.0-1.0
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            result = self._parse_json_response(response)
            if isinstance(result, list):
                needs.extend(result)
        except Exception as e:
            self.logger.warning(f"Security advisory check failed: {e}")
            
        return needs
    
    async def _check_framework_updates(self, tech_stack: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for framework update opportunities.
        
        Args:
            tech_stack: Repository technology stack
            
        Returns:
            Update-related needs
        """
        needs = []
        frameworks = tech_stack.get('frameworks', [])
        
        if frameworks:
            prompt = f"""
            Check for update opportunities for these frameworks:
            {json.dumps(frameworks, indent=2)}
            
            Identify beneficial updates that would improve the project.
            Focus on updates that provide:
            - Security improvements
            - Performance enhancements
            - New useful features
            - Bug fixes
            
            Return as JSON list of update needs, each with:
            - type: 'UPDATE'
            - description: Specific update recommendation
            - priority: Based on benefits
            - suggested_action: Update steps
            - external_factors: ['framework_update']
            - confidence: 0.0-1.0
            """
            
            try:
                response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
                result = self._parse_json_response(response)
                if isinstance(result, list):
                    needs.extend(result[:2])  # Limit to top 2 updates
            except Exception as e:
                self.logger.warning(f"Framework update check failed: {e}")
                
        return needs
    
    async def _check_industry_trends(self, repo_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check industry trends for improvement opportunities.
        
        Args:
            repo_analysis: Repository analysis
            
        Returns:
            Trend-based needs
        """
        needs = []
        project_type = self._categorize_project(repo_analysis.get('basic_info', {}))
        
        prompt = f"""
        Identify current industry trends and best practices for {project_type} projects.
        
        Consider:
        - AI/ML integration opportunities
        - Modern architectural patterns
        - User experience improvements
        - Performance optimization techniques
        - Security best practices
        
        Return as JSON list of trend-based needs (max 2), each with:
        - type: 'ENHANCEMENT' or 'FEATURE'
        - description: Specific improvement based on trends
        - priority: Based on impact
        - suggested_action: Implementation approach
        - external_factors: ['industry_trend']
        - confidence: 0.0-1.0
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            result = self._parse_json_response(response)
            if isinstance(result, list):
                needs.extend(result[:2])
        except Exception as e:
            self.logger.warning(f"Industry trend check failed: {e}")
            
        return needs
    
    def _calculate_need_priority_score(self, need: Dict[str, Any], 
                                     repo_analysis: Dict[str, Any]) -> float:
        """Calculate enhanced priority score for a need.
        
        Args:
            need: Need to score
            repo_analysis: Repository analysis
            
        Returns:
            Priority score
        """
        base_scores = {'critical': 10, 'high': 7, 'medium': 4, 'low': 2}
        score = base_scores.get(need.get('priority', 'low'), 2)
        
        # Boost for security issues
        if need.get('type') == 'SECURITY':
            score *= 1.5
            
        # Boost for external factors
        if need.get('external_factors'):
            score *= 1.3
            
        # Boost based on confidence
        confidence = need.get('confidence', 0.5)
        score *= (0.5 + confidence)
        
        # Boost based on learning score
        if 'learning_score' in need:
            score *= (1 + need['learning_score'] * 0.5)
            
        # Consider repository health
        health_score = repo_analysis.get('health_metrics', {}).get('health_score', 50)
        if health_score < 50:
            score *= 1.2  # Prioritize unhealthy repos
            
        return score
    
    def _calculate_opportunity_score(self, opportunity: Dict[str, Any], 
                                   repo_analysis: Dict[str, Any]) -> float:
        """Calculate score for an opportunity.
        
        Args:
            opportunity: Opportunity to score
            repo_analysis: Repository analysis
            
        Returns:
            Opportunity score
        """
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        score = impact_scores.get(opportunity.get('impact', 'low'), 1)
        
        # Consider effort vs impact
        effort = opportunity.get('effort', 'medium')
        if effort == 'low' and opportunity.get('impact') == 'high':
            score *= 2  # Quick wins
            
        # Consider alignment with trends
        if 'ai' in opportunity.get('description', '').lower():
            score *= 1.3  # AI improvements are strategic
            
        return score
    
    def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics about learning system performance.
        
        Returns:
            Learning metrics
        """
        if not self.learning_system:
            return {'enabled': False}
            
        metrics = {
            'enabled': True,
            'tasks_learned_from': len([t for t in self.task_history 
                                     if t.get('task', {}).get('learned_from', False)]),
            'value_prediction_accuracy': 0.0,
            'priority_adjustment_rate': 0.0
        }
        
        # Calculate prediction accuracy
        predictions_made = 0
        correct_predictions = 0
        
        for record in self.task_history[-50:]:  # Last 50 tasks
            if 'predicted_value' in record:
                predictions_made += 1
                actual_value = record.get('actual_value', 0)
                predicted_value = record.get('predicted_value', 0)
                if abs(actual_value - predicted_value) < 0.2:
                    correct_predictions += 1
                    
        if predictions_made > 0:
            metrics['value_prediction_accuracy'] = correct_predictions / predictions_made
            
        return metrics
    
    def _calculate_task_quality_metrics(self) -> Dict[str, float]:
        """Calculate metrics about task quality.
        
        Returns:
            Quality metrics
        """
        total_tasks = len(self.task_history)
        if total_tasks == 0:
            return {
                'uniqueness_score': 1.0,
                'context_relevance_score': 1.0,
                'completion_rate': 0.0
            }
            
        # Calculate uniqueness (inverse of duplicates)
        unique_titles = set()
        for record in self.task_history:
            title = record.get('task', {}).get('title', '')
            unique_titles.add(title.lower())
            
        uniqueness = len(unique_titles) / total_tasks
        
        # Calculate context relevance (tasks with high confidence)
        high_confidence_tasks = sum(1 for r in self.task_history 
                                  if r.get('need_analysis', {}).get('confidence', 0) > 0.7)
        relevance = high_confidence_tasks / total_tasks
        
        # Calculate completion rate (if status is tracked)
        completed_tasks = sum(1 for r in self.task_history 
                            if r.get('task', {}).get('status') == 'completed')
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        return {
            'uniqueness_score': uniqueness,
            'context_relevance_score': relevance,
            'completion_rate': completion_rate
        }
    
    def _measure_intelligence_effectiveness(self) -> Dict[str, float]:
        """Measure how effective the intelligence features are.
        
        Returns:
            Effectiveness metrics
        """
        metrics = {
            'cross_repo_impact': 0.0,
            'prediction_usage': 0.0,
            'external_intelligence_impact': 0.0,
            'adaptation_rate': 0.0
        }
        
        recent_tasks = self.task_history[-20:]
        if not recent_tasks:
            return metrics
            
        # Measure cross-repo impact
        cross_repo_tasks = sum(1 for t in recent_tasks 
                             if t.get('task', {}).get('cross_repo_benefit', False))
        metrics['cross_repo_impact'] = cross_repo_tasks / len(recent_tasks)
        
        # Measure prediction usage
        predicted_tasks = sum(1 for t in recent_tasks 
                            if t.get('task', {}).get('from_prediction', False))
        metrics['prediction_usage'] = predicted_tasks / len(recent_tasks)
        
        # Measure external intelligence impact
        external_tasks = sum(1 for t in recent_tasks 
                           if t.get('need_analysis', {}).get('external_factors'))
        metrics['external_intelligence_impact'] = external_tasks / len(recent_tasks)
        
        return metrics
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate accuracy of predictions.
        
        Returns:
            Prediction accuracy (0-1)
        """
        if not hasattr(self, '_prediction_history'):
            return 0.0
            
        correct = sum(1 for p in self._prediction_history if p.get('accurate', False))
        total = len(self._prediction_history)
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of continuous improvement.
        
        Returns:
            Improvement rate
        """
        if len(self.task_history) < 10:
            return 0.0
            
        # Compare quality metrics over time
        early_tasks = self.task_history[:10]
        recent_tasks = self.task_history[-10:]
        
        early_quality = sum(t.get('need_analysis', {}).get('confidence', 0) for t in early_tasks) / 10
        recent_quality = sum(t.get('need_analysis', {}).get('confidence', 0) for t in recent_tasks) / 10
        
        improvement = (recent_quality - early_quality) / early_quality if early_quality > 0 else 0
        
        return max(0, improvement)
    
    def _calculate_adaptation_score(self) -> float:
        """Calculate how well the system adapts to changes.
        
        Returns:
            Adaptation score (0-1)
        """
        # Check diversity of recent task types
        recent_types = [t.get('task', {}).get('type') for t in self.task_history[-20:]]
        unique_types = len(set(recent_types))
        
        # Check repository coverage
        recent_repos = [t.get('task', {}).get('repository') for t in self.task_history[-20:]]
        unique_repos = len(set(r for r in recent_repos if r))
        
        # Combined adaptation score
        type_diversity = min(1.0, unique_types / 5)  # Expect at least 5 different types
        repo_diversity = min(1.0, unique_repos / 3)  # Expect at least 3 different repos
        
        return (type_diversity + repo_diversity) / 2
    
    async def _ensure_mcp_redis(self):
        """Ensure MCP-Redis is initialized."""
        if self._use_mcp and not self.mcp_redis:
            try:
                self.mcp_redis = MCPRedisIntegration()
                await self.mcp_redis.initialize()
                self.logger.info("MCP-Redis integration enabled for task generation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                self._use_mcp = False
    
    # MCP-Redis Enhanced Methods
    async def find_semantically_similar_tasks(self, task_description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find semantically similar tasks using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return []
        
        try:
            similar_tasks = await self.mcp_redis.execute(f"""
                Find tasks semantically similar to:
                "{task_description}"
                
                Search across:
                - All active tasks in the work queue
                - Recently completed tasks (last 30 days)
                - Tasks in progress
                
                Consider:
                - Conceptual similarity, not just keyword matching
                - Different phrasings of the same work
                - Related technologies and approaches
                
                Return up to {limit} tasks with:
                - Task ID and title
                - Similarity score (0-1)
                - Status (active/completed/in_progress)
                - Repository context
                - Why it's similar (explanation)
            """)
            
            return similar_tasks if isinstance(similar_tasks, list) else []
            
        except Exception as e:
            self.logger.error(f"Error finding similar tasks: {e}")
            return []
    
    async def generate_task_with_semantic_awareness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task with semantic duplicate checking."""
        # First generate task normally
        task = await self.generate_task(context)
        
        if not self._use_mcp or not self.mcp_redis or task.get('skip'):
            return task
        
        try:
            # Check for semantic duplicates
            duplicate_check = await self.mcp_redis.execute(f"""
                Check if this task is semantically duplicate:
                Title: {task.get('title', '')}
                Description: {task.get('description', '')}
                Type: {task.get('type', '')}
                
                Compare against:
                - All active tasks
                - Tasks completed in last 7 days
                - Tasks currently in progress
                
                Consider as duplicate if:
                - Same conceptual work (even if worded differently)
                - Would solve the same problem
                - Targets the same component/feature
                
                Return:
                - is_duplicate: boolean
                - similar_task_id: ID of most similar task if duplicate
                - similarity_score: 0-1
                - explanation: why it's considered duplicate
            """)
            
            if isinstance(duplicate_check, dict) and duplicate_check.get('is_duplicate'):
                task['skip'] = True
                task['skip_reason'] = f"Semantic duplicate of task {duplicate_check.get('similar_task_id')}"
                task['similarity_score'] = duplicate_check.get('similarity_score', 0)
                self.logger.info(f"Skipping semantic duplicate: {task.get('title')}")
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error checking semantic duplicates: {e}")
            return task
    
    async def analyze_task_generation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in task generation using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            analysis = await self.mcp_redis.execute(f"""
                Analyze task generation patterns from history:
                Total tasks generated: {len(self.task_history)}
                
                Analyze:
                - Which types of tasks are most frequently generated?
                - What patterns exist in task timing?
                - Which repositories get the most tasks?
                - Are there gaps in task coverage?
                - What task types have highest success rates?
                - Are there repetitive patterns we should avoid?
                - What new task types should we explore?
                
                Provide:
                - Pattern insights
                - Coverage gaps
                - Optimization recommendations
                - Diversity suggestions
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            return {"error": str(e)}
    
    async def get_intelligent_task_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI-powered task suggestions based on system state."""
        if not self._use_mcp or not self.mcp_redis:
            return []
        
        try:
            suggestions = await self.mcp_redis.execute(f"""
                Suggest high-value tasks based on:
                
                System context:
                - Active repositories: {len(context.get('repositories', []))}
                - Recent completions: {len(context.get('completed_tasks', []))}
                - Current focus areas: {context.get('focus_areas', [])}
                
                Consider:
                - Gaps in current task coverage
                - Emerging technology trends
                - System improvement opportunities
                - Cross-repository synergies
                - High-impact low-effort tasks
                
                Generate 5-10 task suggestions with:
                - Task title and description
                - Expected impact (high/medium/low)
                - Effort estimate (hours)
                - Priority reasoning
                - Target repository
                - Dependencies
            """)
            
            return suggestions if isinstance(suggestions, list) else []
            
        except Exception as e:
            self.logger.error(f"Error getting task suggestions: {e}")
            return []
    
    async def optimize_task_diversity(self, pending_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize task list for diversity using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis or not pending_tasks:
            return pending_tasks
        
        try:
            optimized = await self.mcp_redis.execute(f"""
                Optimize this task list for diversity:
                Current tasks: {json.dumps([t.get('title') for t in pending_tasks[:20]])}
                
                Goals:
                - Balance different task types
                - Cover multiple repositories
                - Mix difficulties (easy/medium/hard)
                - Include learning opportunities
                - Avoid repetitive work
                
                Suggest:
                - Which tasks to keep as-is
                - Which to modify for variety
                - New task types to add
                - Tasks to defer or remove
                
                Return optimized task list.
            """)
            
            if isinstance(optimized, list):
                return optimized
            return pending_tasks
            
        except Exception as e:
            self.logger.error(f"Error optimizing diversity: {e}")
            return pending_tasks
    
    async def predict_task_value(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the value of a task using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"predicted_value": 0.5, "confidence": 0}
        
        try:
            prediction = await self.mcp_redis.execute(f"""
                Predict the value of this task:
                Title: {task.get('title')}
                Type: {task.get('type')}
                Description: {task.get('description')}
                Repository: {task.get('repository')}
                
                Based on:
                - Historical task completion data
                - Similar task outcomes
                - Current system priorities
                - Repository importance
                - Task complexity vs impact
                
                Predict:
                - Value score (0-1)
                - Confidence level (0-1)
                - Success probability
                - Expected completion time
                - Potential blockers
                - Value reasoning
            """)
            
            return prediction if isinstance(prediction, dict) else {"predicted_value": 0.5}
            
        except Exception as e:
            self.logger.error(f"Error predicting task value: {e}")
            return {"predicted_value": 0.5, "confidence": 0}