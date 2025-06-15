"""
Dynamic God Mode Controller

Enhanced controller with fully dynamic behavior and no hardcoded values.
Integrates all dynamic components for true AI-driven development orchestration.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# Import base systems
from god_mode_controller import GodModeController, GodModeConfig, IntensityLevel
from swarm_intelligence import AgentRole
from task_manager import TaskManager, TaskType

# Import dynamic systems
from dynamic_charter import DynamicCharter
from outcome_learning import OutcomeLearningSystem
from intelligent_task_generator import IntelligentTaskGenerator
from dynamic_validator import DynamicTaskValidator
from dynamic_swarm import DynamicSwarmIntelligence
from project_creator import ProjectCreator
from repository_exclusion import should_process_repo, RepositoryExclusion


class DynamicGodModeController(GodModeController):
    """Enhanced controller with fully dynamic behavior."""
    
    def __init__(self, config: GodModeConfig, ai_brain=None):
        """Initialize with dynamic components.
        
        Args:
            config: God mode configuration
            ai_brain: AI brain instance (will use factory if None)
        """
        # Create AI brain using factory if not provided
        if ai_brain is None:
            from ai_brain_factory import AIBrainFactory
            ai_brain = AIBrainFactory.create_for_production()
        
        # Initialize parent
        super().__init__(config, ai_brain)
        
        # Initialize repository discovery 
        from state_manager import StateManager
        self.state_manager = StateManager()
        
        # Discover and load repositories
        self.logger.info("Discovering organization repositories...")
        self.current_state = self.state_manager.load_state_with_repository_discovery()
        self.discovered_repositories = self.current_state.get('projects', {})
        self.logger.info(f"Loaded {len(self.discovered_repositories)} repositories from organization")
        
        # Initialize dynamic systems
        self.logger.info("Initializing dynamic AI systems...")
        
        # Core dynamic systems
        self.charter_system = DynamicCharter(self.ai_brain)
        self.learning_system = OutcomeLearningSystem(self.ai_brain)
        self.task_generator = IntelligentTaskGenerator(
            self.ai_brain, 
            self.charter_system,
            self.learning_system
        )
        self.task_validator = DynamicTaskValidator(
            self.ai_brain,
            self.charter_system
        )
        
        # Get GitHub token for project creator
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        if github_token:
            self.project_creator = ProjectCreator(github_token, self.ai_brain)
        else:
            self.logger.warning("No GitHub token found - project creation disabled")
            self.project_creator = None
            
        # Replace basic swarm with dynamic swarm
        self.swarm = DynamicSwarmIntelligence(
            self.ai_brain,
            self.learning_system,
            self.charter_system
        )
        
        # Initialize multi-repo coordinator with discovered repositories
        if github_token and self.discovered_repositories:
            from multi_repo_coordinator import MultiRepoCoordinator
            self.coordinator = MultiRepoCoordinator(github_token, max_concurrent=len(self.discovered_repositories))
            
            # Add discovered repositories to coordinator (excluding CWMAI)
            for project_id, project_data in self.discovered_repositories.items():
                if project_data.get('clone_url'):
                    repo_name = project_data.get('full_name', project_data.get('name', ''))
                    if not should_process_repo(repo_name):
                        self.logger.info(f"Skipping excluded repository: {repo_name}")
                        continue
                    
                    try:
                        self.coordinator.add_repository(project_data['clone_url'])
                        self.logger.info(f"Added repository {project_data['name']} to coordinator")
                    except Exception as e:
                        self.logger.warning(f"Failed to add repository {project_data['name']}: {e}")
        else:
            self.coordinator = None
        
        # Dynamic operation tracking
        self.dynamic_metrics = {
            'charters_generated': 0,
            'tasks_validated': 0,
            'tasks_corrected': 0,
            'outcomes_learned': 0,
            'projects_created': 0,
            'value_created': 0.0
        }
        
        self.logger.info("Dynamic AI systems initialized successfully")
        
    async def run_god_mode_cycle(self) -> Dict[str, Any]:
        """Run fully dynamic god mode cycle.
        
        Returns:
            Cycle results with all operations
        """
        self.logger.info("=== Starting Dynamic God Mode Cycle ===")
        cycle_start = time.time()
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'intensity': self.config.intensity.value,
            'operations': [],
            'metrics': {},
            'learnings': [],
            'errors': [],
            'improvements': []
        }
        
        try:
            # Phase 1: Generate/Update Dynamic Charter
            context = await self._gather_full_context()
            self.current_charter = await self._update_dynamic_charter(context)
            results['charter_version'] = self.current_charter.get('timestamp', 'unknown')
            
            # Phase 2: Enhanced Swarm Analysis with Context
            swarm_analysis = await self._dynamic_swarm_analysis(context)
            
            # Phase 3: Intelligent Task Generation
            tasks = await self._generate_intelligent_tasks(context, swarm_analysis)
            results['tasks_generated'] = len(tasks)
            
            # Phase 4: Validate and Correct Tasks
            validated_tasks = await self._validate_and_correct_tasks(tasks, context)
            results['tasks_validated'] = len(validated_tasks)
            
            # Phase 5: Execute Valid Tasks
            operation_results = await self._execute_validated_operations(
                validated_tasks, 
                swarm_analysis
            )
            results['operations'] = operation_results
            
            # Phase 6: Learn from Outcomes
            learning_insights = await self._learn_from_outcomes(operation_results)
            results['learnings'] = learning_insights
            
            # Phase 7: Generate Recommendations
            recommendations = await self._generate_cycle_recommendations()
            results['recommendations'] = recommendations
            
            # Update metrics
            results['metrics'] = self._calculate_cycle_metrics(operation_results)
            
        except Exception as e:
            self.logger.error(f"Error in dynamic god mode cycle: {e}")
            results['errors'].append({
                'phase': 'cycle_execution',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
        # Calculate cycle duration
        cycle_duration = time.time() - cycle_start
        results['duration'] = cycle_duration
        
        # Save state
        self._save_cycle_state(results)
        
        self.logger.info(f"=== Dynamic God Mode Cycle Complete in {cycle_duration:.2f}s ===")
        
        return results
    
    async def _gather_full_context(self) -> Dict[str, Any]:
        """Gather comprehensive context for decision making.
        
        Returns:
            Full system context
        """
        # Get base context
        context = await self._gather_enhanced_context()
        
        # Add dynamic system states
        context['active_projects'] = self._get_active_projects()
        context['recent_outcomes'] = self.learning_system.outcome_history[-10:]
        context['recent_tasks'] = self.task_generator.task_history[-10:]
        context['capabilities'] = self._get_system_capabilities()
        context['validation_insights'] = await self.task_validator.get_validation_insights()
        
        # Add learning summary
        if self.learning_system.outcome_history:
            context['learning_summary'] = self.learning_system.get_learning_summary()
            
        return context
    
    async def _update_dynamic_charter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update system charter based on current context.
        
        Args:
            context: Current system context
            
        Returns:
            Updated charter
        """
        self.logger.info("Updating dynamic charter based on context")
        
        charter = await self.charter_system.generate_charter(context)
        self.dynamic_metrics['charters_generated'] += 1
        
        # Log charter evolution
        if len(self.charter_system.charter_history) > 1:
            self.logger.info("Charter evolved based on learning")
            
        return charter
    
    async def _dynamic_swarm_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run dynamic swarm analysis with full context.
        
        Args:
            context: System context
            
        Returns:
            Swarm analysis results
        """
        self.logger.info("Running dynamic swarm intelligence analysis")
        
        # Create analysis task for swarm
        analysis_task = {
            'id': f'analysis_{datetime.now(timezone.utc).timestamp()}',
            'type': 'strategic_analysis',
            'title': 'Determine next strategic actions',
            'description': 'Analyze current state and recommend next actions',
            'context': context,
            'requirements': [
                'Identify highest value opportunities',
                'Consider portfolio balance',
                'Learn from past outcomes',
                'Align with system charter'
            ]
        }
        
        # Process with dynamic swarm
        swarm_result = await self.swarm.process_task_swarm(analysis_task, context)
        self.performance_metrics['swarm_decisions'] += 1
        
        return swarm_result
    
    async def _generate_intelligent_tasks(self, context: Dict[str, Any],
                                        swarm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks using repository-first approach.
        
        Args:
            context: System context
            swarm_analysis: Swarm analysis results
            
        Returns:
            Generated tasks
        """
        self.logger.info("Generating intelligent tasks using repository-first approach")
        
        # Add swarm insights to context
        enhanced_context = context.copy()
        enhanced_context['swarm_analysis'] = swarm_analysis
        enhanced_context['swarm_recommendations'] = swarm_analysis.get(
            'collective_review', {}
        ).get('top_suggestions', [])
        
        # Get available repositories
        available_repos = self._get_available_repositories(enhanced_context)
        
        # If no repositories are available, generate NEW_PROJECT tasks
        if not available_repos:
            self.logger.info("No repositories available - generating NEW_PROJECT tasks")
            return await self._generate_new_project_tasks(enhanced_context, min(2, self.config.max_parallel_operations))
        
        # Dynamically adjust task_count based on available repositories
        max_tasks = min(3, self.config.max_parallel_operations)
        task_count = min(max_tasks, len(available_repos))
        
        tasks = []
        used_repositories = enhanced_context.get('selected_repositories', [])
        
        # Repository-first approach: select repos, analyze, then generate tasks
        for i in range(task_count):
            try:
                # Filter out already used repositories
                remaining_repos = []
                for repo in available_repos:
                    repo_name = repo.get('full_name', repo.get('name', ''))
                    if repo_name not in used_repositories:
                        remaining_repos.append(repo)
                
                # Return early if repositories are exhausted
                if not remaining_repos:
                    self.logger.info(f"All available repositories used. Generated {len(tasks)} tasks.")
                    break
                
                # 1. Select repository intelligently
                repo_selection = await self.ai_brain.select_repository_for_task({
                    **enhanced_context,
                    'active_projects': remaining_repos
                })
                
                if not repo_selection:
                    # Generate NEW_PROJECT task as fallback
                    new_project_task = await self._generate_single_new_project_task(enhanced_context)
                    if new_project_task:
                        tasks.append(new_project_task)
                    continue
                
                # 2. Analyze the selected repository
                from repository_analyzer import RepositoryAnalyzer
                analyzer = RepositoryAnalyzer()
                repo_analysis = await analyzer.analyze_repository(repo_selection['repository'])
                
                # 3. Generate task specific to this repository
                task = await self.task_generator.generate_task_for_repository(
                    repo_selection['repository'],
                    repo_analysis,
                    enhanced_context
                )
                
                tasks.append(task)
                
                # Update context to avoid selecting same repo repeatedly
                if 'selected_repositories' not in enhanced_context:
                    enhanced_context['selected_repositories'] = []
                enhanced_context['selected_repositories'].append(repo_selection['repository'])
                used_repositories.append(repo_selection['repository'])
                
            except Exception as e:
                self.logger.warning(f"Error generating task {i+1}: {e}")
                continue
        
        # If we couldn't generate enough tasks, add NEW_PROJECT tasks
        if len(tasks) < task_count:
            new_projects_needed = task_count - len(tasks)
            new_project_tasks = await self._generate_new_project_tasks(enhanced_context, new_projects_needed)
            tasks.extend(new_project_tasks)
        
        self.logger.info(f"Generated {len(tasks)} tasks ({len([t for t in tasks if t.get('repository')])} repository-specific, {len([t for t in tasks if t.get('type') == 'NEW_PROJECT'])} new projects)")
        
        return tasks
    
    async def _validate_and_correct_tasks(self, tasks: List[Dict[str, Any]],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate tasks and correct if needed.
        
        Args:
            tasks: Generated tasks
            context: System context
            
        Returns:
            List of valid/corrected tasks
        """
        self.logger.info(f"Validating {len(tasks)} tasks")
        
        # Add existing tasks to context for deduplication
        validation_context = context.copy()
        validation_context['existing_tasks'] = self._get_existing_tasks()
        
        # Validate as batch for relationship checking
        validations = await self.task_validator.validate_batch(tasks, validation_context)
        
        validated_tasks = []
        
        for i, (task, validation) in enumerate(zip(tasks, validations)):
            self.dynamic_metrics['tasks_validated'] += 1
            
            if validation['valid']:
                validated_tasks.append(task)
                self.logger.info(f"Task {i+1} valid: {task.get('title')}")
            else:
                # Try to use corrected version
                if validation.get('corrected_task'):
                    corrected = validation['corrected_task']
                    self.logger.info(f"Task {i+1} corrected: {corrected.get('title')}")
                    validated_tasks.append(corrected)
                    self.dynamic_metrics['tasks_corrected'] += 1
                else:
                    self.logger.warning(
                        f"Task {i+1} invalid and unfixable: {validation.get('issues')}"
                    )
                    
        return validated_tasks
    
    async def _execute_validated_operations(self, tasks: List[Dict[str, Any]],
                                          swarm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute validated tasks as operations.
        
        Args:
            tasks: Validated tasks
            swarm_analysis: Swarm analysis for context
            
        Returns:
            Operation results
        """
        self.logger.info(f"Executing {len(tasks)} validated operations")
        
        operations = []
        
        # Convert tasks to operations
        for task in tasks:
            operation = {
                'id': f"op_{task.get('type', 'unknown')}_{int(time.time())}",
                'task': task,
                'status': 'pending',
                'result': None,
                'error': None
            }
            operations.append(operation)
            
        # Execute operations with new logic
        futures = []
        
        for operation in operations[:self.config.max_parallel_operations]:
            future = self.executor.submit(self._execute_dynamic_operation, operation)
            futures.append((operation, future))
            
        # Collect results
        for operation, future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                operation['status'] = 'completed'
                operation['result'] = result
                
                # Record outcome for learning
                await self.learning_system.record_outcome(
                    operation['task'],
                    result
                )
                self.dynamic_metrics['outcomes_learned'] += 1
                
            except Exception as e:
                operation['status'] = 'failed'
                operation['error'] = str(e)
                self.logger.error(f"Operation {operation['id']} failed: {e}")
                
        return operations
    
    def _execute_dynamic_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single operation with dynamic handling.
        
        Args:
            operation: Operation to execute
            
        Returns:
            Operation result
        """
        task = operation['task']
        task_type = task.get('type', 'unknown')
        
        self.logger.info(f"Executing dynamic operation: {operation['id']} (Type: {task_type})")
        
        result = {
            'success': False,
            'output': None,
            'metrics': {}
        }
        
        try:
            # Handle based on actual task type (not string comparison)
            if task_type == TaskType.NEW_PROJECT or task_type == 'NEW_PROJECT':
                result = self._handle_new_project(task)
                
            elif task_type == TaskType.FEATURE or task_type == 'FEATURE':
                result = self._handle_feature_task(task)
                
            elif task_type == 'improvement':
                result = self._handle_improvement_task(task)
                
            else:
                # Let parent handle other types
                result = super()._execute_operation(operation)
                
        except Exception as e:
            self.logger.error(f"Error executing operation: {e}")
            result['error'] = str(e)
            
        return result
    
    def _handle_new_project(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle NEW_PROJECT task by creating repository from starter kit.
        
        Args:
            task: NEW_PROJECT task
            
        Returns:
            Execution result
        """
        if not self.project_creator:
            return {
                'success': False,
                'error': 'Project creator not available (no GitHub token)'
            }
            
        try:
            # Create project using Laravel React starter kit
            creation_result = asyncio.run(
                self.project_creator.create_project(task)
            )
            
            if creation_result['success']:
                self.dynamic_metrics['projects_created'] += 1
                self.performance_metrics['repos_managed'] += 1
                
                # Add to multi-repo coordinator if available
                if self.coordinator:
                    self.coordinator.add_repository(
                        creation_result['repo_url']
                    )
                    
            return creation_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Project creation failed: {str(e)}'
            }
    
    def _handle_feature_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FEATURE task by creating issue in appropriate project.
        
        Args:
            task: FEATURE task
            
        Returns:
            Execution result
        """
        target_project = task.get('target_project')
        
        if not target_project:
            return {
                'success': False,
                'error': 'FEATURE task missing target_project'
            }
            
        try:
            # Create issue in target project
            if self.coordinator:
                success = self.coordinator.create_cross_repo_task(
                    repo_url=target_project,
                    task_type='feature',
                    title=task.get('title', 'New Feature'),
                    description=task.get('description', '')
                )
                
                return {
                    'success': success,
                    'output': {'target_project': target_project}
                }
            else:
                return {
                    'success': False,
                    'error': 'Multi-repo coordinator not available'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Feature creation failed: {str(e)}'
            }
    
    def _handle_improvement_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle improvement task for AI system enhancement.
        
        Args:
            task: improvement task
            
        Returns:
            Execution result
        """
        # Use self-improver if available
        if self.config.enable_self_modification and self.improver:
            return super()._execute_operation({'task': task})
        else:
            # Create issue for manual implementation
            # Note: For system improvements, create issue without target project
            # to avoid self-reference loops
            issue = self.task_manager.create_task(
                task_type=TaskType.REFACTOR,  # Use refactor for improvements
                title=task.get('title', 'AI System Improvement'),
                description=task.get('description', ''),
                target_project=task.get('target_project')  # Let task manager handle None properly
            )
            
            return {
                'success': True,
                'output': {'issue_created': issue['id']}
            }
    
    async def _learn_from_outcomes(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Learn from operation outcomes.
        
        Args:
            operations: Completed operations
            
        Returns:
            Learning insights
        """
        if not operations:
            return []
            
        # Get recommendations from learning system
        recommendations = await self.learning_system.get_recommendations()
        
        # Get validation insights
        validation_insights = await self.task_validator.get_validation_insights()
        
        # Combine insights
        insights = []
        
        if recommendations.get('status') != 'insufficient_data':
            insights.append({
                'type': 'outcome_learning',
                'insights': recommendations,
                'impact': 'Improves future task generation and prioritization'
            })
            
        if validation_insights.get('total_validations', 0) > 0:
            insights.append({
                'type': 'validation_learning',
                'validity_rate': validation_insights.get('validity_rate', 0),
                'common_issues': validation_insights.get('patterns_by_type', {}),
                'impact': 'Improves task quality and reduces corrections'
            })
            
        # Calculate value created
        total_value = sum(
            op['result'].get('value_assessment', {}).get('value_score', 0)
            for op in operations
            if op.get('result') and isinstance(op['result'], dict)
        )
        self.dynamic_metrics['value_created'] += total_value
        
        insights.append({
            'type': 'value_creation',
            'cycle_value': total_value,
            'total_value': self.dynamic_metrics['value_created'],
            'average_value': total_value / len(operations) if operations else 0
        })
        
        return insights
    
    async def _generate_cycle_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for next cycle.
        
        Returns:
            Recommendations
        """
        # Get all system recommendations
        learning_rec = await self.learning_system.get_recommendations()
        validation_rec = await self.task_validator.get_validation_insights()
        swarm_analytics = self.swarm.get_swarm_analytics()
        
        prompt = f"""
        Based on this cycle's performance, generate strategic recommendations.
        
        Learning System Insights:
        {json.dumps(learning_rec, indent=2)}
        
        Validation Insights:
        {json.dumps(validation_rec, indent=2)}
        
        Swarm Performance:
        {json.dumps(swarm_analytics, indent=2)}
        
        Dynamic Metrics:
        {json.dumps(self.dynamic_metrics, indent=2)}
        
        Generate recommendations for:
        1. Task generation improvements
        2. Validation criteria adjustments
        3. Strategic focus areas
        4. System capability enhancements
        5. Process optimizations
        
        Format as JSON with specific, actionable recommendations.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.get('content', ''))
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return {
            'status': 'Unable to generate recommendations',
            'fallback': 'Continue with current approach'
        }
    
    def _calculate_cycle_metrics(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive cycle metrics.
        
        Args:
            operations: Completed operations
            
        Returns:
            Cycle metrics
        """
        metrics = self.performance_metrics.copy()
        metrics.update(self.dynamic_metrics)
        
        # Add cycle-specific metrics
        successful_ops = sum(
            1 for op in operations 
            if op.get('result', {}).get('success', False)
        )
        
        metrics['cycle_success_rate'] = successful_ops / len(operations) if operations else 0
        metrics['cycle_operations'] = len(operations)
        metrics['successful_operations'] = successful_ops
        
        return metrics
    
    def _save_cycle_state(self, results: Dict[str, Any]) -> None:
        """Save cycle state for persistence.
        
        Args:
            results: Cycle results
        """
        state = self.state.copy()
        state['last_cycle'] = results
        state['dynamic_metrics'] = self.dynamic_metrics
        state['charter'] = self.current_charter
        
        self.state_manager.save_state_locally(state)
    
    def _get_available_repositories(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of available repositories for task generation.
        
        Args:
            context: System context
            
        Returns:
            List of available repositories
        """
        projects = context.get('active_projects', [])
        
        # Convert dict to list if needed
        if isinstance(projects, dict):
            projects = list(projects.values())
        
        # Filter for eligible repositories
        from repository_exclusion import should_process_repo
        eligible_repos = []
        for proj in projects:
            repo_name = proj.get('full_name', proj.get('name', ''))
            if should_process_repo(repo_name) and repo_name:
                eligible_repos.append(proj)
        
        return eligible_repos
    
    async def _generate_new_project_tasks(self, context: Dict[str, Any], 
                                         count: int) -> List[Dict[str, Any]]:
        """Generate NEW_PROJECT tasks when no repositories exist.
        
        Args:
            context: System context
            count: Number of tasks to generate
            
        Returns:
            List of NEW_PROJECT tasks
        """
        tasks = []
        for i in range(count):
            task = await self._generate_single_new_project_task(context)
            if task:
                tasks.append(task)
        return tasks
    
    async def _generate_single_new_project_task(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a single NEW_PROJECT task.
        
        Args:
            context: System context
            
        Returns:
            NEW_PROJECT task or None
        """
        try:
            # Use the task generator's built-in method for NEW_PROJECT tasks
            task = await self.task_generator._generate_new_project_task(context)
            return task
        except Exception as e:
            self.logger.warning(f"Failed to generate NEW_PROJECT task: {e}")
        
        return None
    
    def _get_existing_tasks(self) -> List[Dict[str, Any]]:
        """Get list of existing tasks from task state.
        
        Returns:
            List of existing tasks
        """
        try:
            task_state_path = 'task_state.json'
            if os.path.exists(task_state_path):
                with open(task_state_path, 'r') as f:
                    task_state = json.load(f)
                    return task_state.get('tasks', [])
        except Exception as e:
            self.logger.warning(f"Could not load existing tasks: {e}")
        
        return []
    
    def _get_active_projects(self) -> List[Dict[str, Any]]:
        """Get list of active projects including discovered repositories.
        
        Returns:
            Active project list with real repositories
        """
        projects = []
        
        # Add discovered repositories from organization (excluding CWMAI)
        for project_id, project_data in self.discovered_repositories.items():
            if project_data.get('status') == 'active':
                repo_name = project_data.get('full_name', project_data.get('name', project_id))
                
                # Skip excluded repositories
                if not should_process_repo(repo_name):
                    continue
                
                projects.append({
                    'name': project_data.get('name', project_id),
                    'full_name': project_data.get('full_name', project_id),
                    'url': project_data.get('url', ''),
                    'health_score': project_data.get('health_score', 0),
                    'language': project_data.get('language', 'Unknown'),
                    'description': project_data.get('description', ''),
                    'metrics': project_data.get('metrics', {}),
                    'recent_activity': project_data.get('recent_activity', {}),
                    'type': project_data.get('type', 'github_repository'),
                    'last_checked': project_data.get('last_checked'),
                    'topics': project_data.get('topics', [])
                })
        
        # Get from multi-repo coordinator if available
        if self.coordinator and hasattr(self.coordinator, 'repositories'):
            for repo_url, repo_state in self.coordinator.repositories.items():
                # Avoid duplicates by checking if already added from discovery
                if not any(p.get('url') == repo_url for p in projects):
                    projects.append({
                        'name': repo_state.name,
                        'url': repo_url,
                        'health_score': repo_state.health_score,
                        'open_issues': repo_state.open_issues,
                        'type': 'coordinator_managed'
                    })
                
        # Add created projects
        if self.project_creator:
            for project in self.project_creator.get_created_projects():
                if project['success']:
                    # Avoid duplicates
                    if not any(p.get('url') == project['repo_url'] for p in projects):
                        projects.append({
                            'name': project['project_name'],
                            'url': project['repo_url'],
                            'created_at': project['created_at'],
                            'type': 'ai_created'
                        })
        
        self.logger.info(f"Found {len(projects)} active projects")
        return projects
    
    def _get_system_capabilities(self) -> List[str]:
        """Get current system capabilities.
        
        Returns:
            List of capabilities
        """
        capabilities = [
            'GitHub API integration',
            'Multi-model AI reasoning',
            'Dynamic charter generation',
            'Intelligent task generation',
            'Task validation and correction',
            'Multi-agent swarm intelligence',
            'Outcome-based learning',
            'Laravel React project creation'
        ]
        
        if self.config.enable_self_modification:
            capabilities.append('Self-modification')
            
        if self.config.enable_predictive:
            capabilities.append('Predictive analytics')
            
        if self.config.enable_quantum:
            capabilities.append('Quantum-inspired optimization')
            
        return capabilities
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status
        """
        status = super().get_status()
        
        # Add dynamic system status
        status['dynamic_systems'] = {
            'charter': {
                'active': True,
                'version_count': len(self.charter_system.charter_history)
            },
            'learning': {
                'active': True,
                'outcomes_recorded': len(self.learning_system.outcome_history),
                'patterns_identified': len(self.learning_system.value_patterns)
            },
            'task_generation': {
                'active': True,
                'tasks_generated': len(self.task_generator.task_history)
            },
            'validation': {
                'active': True,
                'validations_performed': self.dynamic_metrics['tasks_validated'],
                'corrections_made': self.dynamic_metrics['tasks_corrected']
            },
            'swarm': {
                'active': True,
                'analyses_performed': len(self.swarm.swarm_history)
            }
        }
        
        status['projects'] = self._get_active_projects()
        status['value_created'] = self.dynamic_metrics['value_created']
        
        return status