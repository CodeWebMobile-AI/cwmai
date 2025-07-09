"""
NOTE: This version uses SQLite/In-memory instead of Redis.

Hybrid Continuous Orchestrator

Combines the 24/7 operation of Continuous System with the
strategic intelligence of Dynamic God Mode.

Best of both worlds: continuous execution with learning and evolution.

NOTE: This version uses SQLite/In-memory instead of Redis.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import os

# Import base continuous system
from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis as ContinuousOrchestrator, WorkerStatus, WorkerState
from scripts.work_item_types import WorkItem, TaskPriority

# Import Dynamic God Mode components
from scripts.dynamic_charter import DynamicCharter
from scripts.outcome_learning import OutcomeLearningSystem
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.dynamic_validator import DynamicTaskValidator
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.project_creator import ProjectCreator
from scripts.task_manager import TaskManager, TaskType
from scripts.multi_repo_coordinator import MultiRepoCoordinator

# Import utilities
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager
from scripts.repository_exclusion import should_process_repo
from scripts.storage_mode import StorageMode


class HybridContinuousOrchestrator(ContinuousOrchestrator):
    """
    Hybrid orchestrator combining continuous operation with strategic intelligence.
    
    NOTE: This version uses SQLite/In-memory instead of Redis.
    
    Features:
    - 24/7 continuous operation
    - Dynamic charter evolution
    - Outcome-based learning
    - Repository-first task generation
    - Task validation and correction
    - Intelligent work prioritization
    """
    
    def __init__(self, max_workers: int = 5, enable_parallel: bool = True,
                 enable_research: bool = True, enable_round_robin: bool = False):
        """
        Initialize hybrid orchestrator with Dynamic God Mode intelligence.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            enable_research: Whether to enable research evolution
            enable_round_robin: Whether to enable round-robin AI selection
        """
        # Initialize base continuous orchestrator
        super().__init__(max_workers, enable_parallel, enable_research, enable_round_robin)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Dynamic God Mode components
        self._initialize_dynamic_components()
        
        # Metrics for hybrid operation
        self.hybrid_metrics = {
            'charters_generated': 0,
            'tasks_validated': 0,
            'tasks_corrected': 0,
            'outcomes_learned': 0,
            'strategic_decisions': 0,
            'repository_analyses': 0
        }
        
        # Ensure we have metrics from parent (parent uses self.metrics)
        if not hasattr(self, 'metrics'):
            self.metrics = {
                'total_work_completed': 0,
                'total_errors': 0,
                'total_work_created': 0,
                'work_per_hour': 0.0,
                'average_completion_time': 0.0
            }
        
        # Charter update frequency (every N cycles)
        self.charter_update_frequency = 50
        self.cycles_since_charter_update = 0
        
        # Thread-safe tracking of repository assignments
        self._repo_assignment_lock = asyncio.Lock()
        self._assigned_repos = set()  # Repos currently being assigned to workers
        self._processing_issues = set()  # Issue IDs currently being processed
        
    def _initialize_dynamic_components(self):
        """
        Initialize Dynamic God Mode components.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        self.logger.info("Initializing Dynamic God Mode components...")
        
        # Use production AI brain
        self.ai_brain = AIBrainFactory.create_for_production()
        
        # Dynamic systems
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
        
        # Repository analyzer with caching
        self.repository_analyzer = RepositoryAnalyzer(
            ai_brain=self.ai_brain,
            state_manager=self.state_manager,
            use_cache=True
        )
        
        # Project creator for NEW_PROJECT tasks
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        self.project_creator = ProjectCreator(github_token, self.ai_brain) if github_token else None
        
        # Multi-repo coordinator
        if github_token and self.system_state.get('projects'):
            # Count active repositories to set max_concurrent
            active_repos = [p for p in self.system_state.get('projects', {}).values() 
                           if should_process_repo(p.get('full_name', ''))]
            max_concurrent = max(len(active_repos), 50)  # At least 50 or number of repos
            
            # Get storage mode from environment
            storage_mode = os.getenv('STORAGE_MODE', StorageMode.GITHUB)
            self.coordinator = MultiRepoCoordinator(github_token, max_concurrent=max_concurrent, storage_mode=storage_mode)
            
            # Add discovered repositories
            for project_id, project_data in self.system_state.get('projects', {}).items():
                if should_process_repo(project_data.get('full_name', '')):
                    try:
                        if storage_mode == StorageMode.LOCAL:
                            # In local mode, use the project name/path
                            self.coordinator.add_repository(project_data.get('full_name', project_id))
                        else:
                            # In GitHub mode, use clone URL
                            if project_data.get('clone_url'):
                                self.coordinator.add_repository(project_data['clone_url'])
                    except Exception as e:
                        self.logger.warning(f"Failed to add repository {project_id}: {e}")
        else:
            self.coordinator = None
            
        # Load current charter
        self.current_charter = None
        asyncio.create_task(self._load_initial_charter())
        
        self.logger.info("Dynamic God Mode components initialized")
    
    async def _load_initial_charter(self):
        """
        Load or generate initial charter.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        try:
            context = await self._gather_enhanced_context()
            self.current_charter = await self.charter_system.generate_charter(context)
            self.hybrid_metrics['charters_generated'] += 1
        except Exception as e:
            self.logger.error(f"Failed to load initial charter: {e}")
    
    async def find_work(self) -> List[WorkItem]:
        """
        Find work using Dynamic God Mode's intelligent approach.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            List of work items prioritized by importance
        """
        all_work = []
        
        try:
            # 1. First check GitHub issues (real user needs)
            github_work = await self._get_github_issues()
            
            # Enqueue GitHub issues to the work queue for proper distribution
            for work_item in github_work:
                # Convert WorkItem to dict for queue
                work_dict = {
                    'id': work_item.id,
                    'type': work_item.task_type,
                    'title': work_item.title,
                    'description': work_item.description,
                    'repository': work_item.repository,
                    'priority': work_item.priority.value if hasattr(work_item.priority, 'value') else work_item.priority,
                    'metadata': work_item.metadata or {}
                }
                # Enqueue work - the queue will handle deduplication
                if self.hybrid_state.enqueue_work(work_dict):
                    self.logger.info(f"Enqueued issue: {work_item.id} - {work_item.title}")
                else:
                    self.logger.debug(f"Issue already in queue: {work_item.id}")
            
            all_work.extend(github_work)
            self.logger.info(f"Found and enqueued {len(github_work)} GitHub issues")
            
            # 2. If we need more work, generate intelligently
            if len(all_work) < self.max_workers:
                needed = self.max_workers - len(all_work)
                
                # Gather context for intelligent generation
                context = await self._gather_enhanced_context()
                
                # Generate tasks using repository-first approach
                generated_tasks = await self._generate_intelligent_tasks(context, needed)
                
                # Convert to work items and enqueue them
                for task in generated_tasks:
                    work_item = self._task_to_work_item(task)
                    if work_item:
                        # Enqueue generated work too
                        work_dict = {
                            'id': work_item.id,
                            'type': work_item.task_type,
                            'title': work_item.title,
                            'description': work_item.description,
                            'repository': work_item.repository,
                            'priority': work_item.priority.value if hasattr(work_item.priority, 'value') else work_item.priority,
                            'metadata': work_item.metadata or {}
                        }
                        if self.hybrid_state.enqueue_work(work_dict):
                            self.logger.info(f"Enqueued generated task: {work_item.id} - {work_item.title}")
                        all_work.append(work_item)
                
                self.logger.info(f"Generated and enqueued {len(generated_tasks)} intelligent tasks")
            
            # 3. Update charter periodically
            self.cycles_since_charter_update += 1
            if self.cycles_since_charter_update >= self.charter_update_frequency:
                await self._update_charter()
                self.cycles_since_charter_update = 0
            
        except Exception as e:
            self.logger.error(f"Error finding work: {e}")
            traceback.print_exc()
        
        return all_work[:self.max_workers]  # Return up to max_workers items
    
    async def find_single_work_item(self) -> Optional[WorkItem]:
        """
        Find a single work item for a worker, ensuring parallel distribution.
        
        This method overrides the parent to ensure different workers get
        different repositories to work on in parallel.
        
        Returns:
            A single work item or None if no work available
        """
        try:
            # 1. First check the queue (for manually added work)
            work_dict = self.hybrid_state.dequeue_work(timeout=0.1)
            if work_dict:
                # Convert dict to WorkItem
                work_item = WorkItem(
                    id=work_dict.get('id'),
                    task_type=work_dict.get('type', work_dict.get('task_type', 'unknown')),
                    title=work_dict.get('title'),
                    description=work_dict.get('description'),
                    repository=work_dict.get('repository'),
                    priority=TaskPriority(work_dict.get('priority', TaskPriority.MEDIUM.value)),
                    metadata=work_dict.get('metadata', {})
                )
                self.logger.info(f"Dequeued work item from queue: {work_item.id}")
                return work_item
            
            # 2. Generate work intelligently
            # Gather context
            context = await self._gather_enhanced_context()
            
            # Get repositories that are not currently being worked on
            available_repos = self._get_available_repositories(context)
            
            # Use lock to prevent race conditions in repository assignment
            async with self._repo_assignment_lock:
                # Filter out repositories that have workers currently assigned
                busy_repos = set()
                for worker in self.workers.values():
                    if worker.current_work and worker.current_work.repository:
                        busy_repos.add(worker.current_work.repository)
                
                # Also filter out repos currently being assigned
                busy_repos.update(self._assigned_repos)
                
                # Find repositories not being worked on
                free_repos = [r for r in available_repos 
                             if r.get('full_name') not in busy_repos]
                
                if not free_repos and not available_repos:
                    # No repositories at all - try to generate a NEW_PROJECT task
                    self.logger.info("No repositories available - attempting to generate NEW_PROJECT task")
                    task = await self._generate_new_project_task(context)
                    if task:
                        return self._task_to_work_item(task)
                    return None
                
                # Use free repos if available
                if not free_repos:
                    self.logger.info(f"All {len(available_repos)} repositories are busy. Workers: {len(self.workers)}, Busy repos: {busy_repos}")
                    return None  # All repos are busy
                
                # Select a repository for this worker
                # Simple round-robin or random selection to distribute work
                import random
                selected_repo = random.choice(free_repos)
                repo_name = selected_repo['full_name']
                
                # Mark this repo as being assigned
                self._assigned_repos.add(repo_name)
                self.logger.info(f"Assigning repository {repo_name} to a worker (free: {len(free_repos)-1}, busy: {len(busy_repos)+1})")
            
            try:
                # Analyze the repository (outside the lock to avoid blocking)
                repo_analysis = await self.repository_analyzer.analyze_repository(repo_name)
                self.hybrid_metrics['repository_analyses'] += 1
                
                # Generate a task for this repository
                task = await self.task_generator.generate_task_for_repository(
                    repo_name,
                    repo_analysis,
                    context
                )
                
                if task:
                    # Validate the task
                    validation = await self.task_validator.validate_task(task, context)
                    self.hybrid_metrics['tasks_validated'] += 1
                    
                    if validation['valid']:
                        work_item = self._task_to_work_item(task)
                        self.logger.info(f"Generated work item for {repo_name}: {work_item.title}")
                        return work_item
                    elif validation.get('corrected_task'):
                        self.hybrid_metrics['tasks_corrected'] += 1
                        work_item = self._task_to_work_item(validation['corrected_task'])
                        self.logger.info(f"Generated corrected work item for {repo_name}: {work_item.title}")
                        return work_item
                    else:
                        self.logger.warning(f"Generated task was invalid: {validation.get('issues')}")
                
            finally:
                # Always remove from assigned repos when done
                async with self._repo_assignment_lock:
                    self._assigned_repos.discard(repo_name)
            
        except Exception as e:
            self.logger.error(f"Error in find_single_work_item: {e}")
            traceback.print_exc()
        
        return None
    
    async def _generate_intelligent_tasks(self, context: Dict[str, Any], 
                                         count: int) -> List[Dict[str, Any]]:
        """
        Generate tasks using Dynamic God Mode's approach.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            context: System context
            count: Number of tasks to generate
            
        Returns:
            List of validated tasks
        """
        tasks = []
        
        # Get available repositories
        available_repos = self._get_available_repositories(context)
        
        # If no repositories, generate NEW_PROJECT tasks
        if not available_repos:
            self.logger.info("No repositories available - generating NEW_PROJECT tasks")
            for i in range(min(count, 2)):  # Max 2 new projects at once
                task = await self._generate_new_project_task(context)
                if task:
                    tasks.append(task)
            return tasks
        
        # Repository-first task generation
        used_repositories = []
        
        for i in range(count):
            try:
                # Filter out already used repositories
                remaining_repos = [r for r in available_repos 
                                 if r.get('full_name') not in used_repositories]
                
                if not remaining_repos:
                    break
                
                # 1. Select repository intelligently
                repo_selection = await self.ai_brain.select_repository_for_task({
                    **context,
                    'active_projects': remaining_repos
                })
                
                if not repo_selection:
                    continue
                
                # 2. Analyze the selected repository
                repo_name = repo_selection['repository']
                repo_analysis = await self.repository_analyzer.analyze_repository(repo_name)
                self.hybrid_metrics['repository_analyses'] += 1
                
                # 3. Generate task specific to this repository
                task = await self.task_generator.generate_task_for_repository(
                    repo_name,
                    repo_analysis,
                    context
                )
                
                if task:
                    tasks.append(task)
                    used_repositories.append(repo_name)
                
            except Exception as e:
                self.logger.warning(f"Error generating task {i+1}: {e}")
                continue
        
        # Validate all tasks
        if tasks:
            validation_context = context.copy()
            validation_context['existing_tasks'] = self._get_existing_tasks()
            
            validations = await self.task_validator.validate_batch(tasks, validation_context)
            
            validated_tasks = []
            for task, validation in zip(tasks, validations):
                self.hybrid_metrics['tasks_validated'] += 1
                
                if validation['valid']:
                    validated_tasks.append(task)
                elif validation.get('corrected_task'):
                    validated_tasks.append(validation['corrected_task'])
                    self.hybrid_metrics['tasks_corrected'] += 1
                else:
                    self.logger.warning(f"Task invalid and unfixable: {validation.get('issues')}")
            
            return validated_tasks
        
        return []
    
    async def _execute_work_item(self, worker_id: str, work_item: WorkItem) -> Dict[str, Any]:
        """
        Execute a work item and record outcome for learning.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            worker_id: Worker ID
            work_item: Work item to execute
            
        Returns:
            Execution result
        """
        # Track this work item as being processed
        if work_item.task_type == "ISSUE":
            self._processing_issues.add(work_item.id)
        
        try:
            # Execute using parent method
            result = await super()._execute_work_item(worker_id, work_item)
            
            # Record outcome for learning
            try:
                # Convert work item to task format for learning system
                task = {
                    'type': work_item.task_type,
                    'title': work_item.title,
                    'description': work_item.description,
                    'repository': work_item.repository,
                    'priority': work_item.priority.value if isinstance(work_item.priority, TaskPriority) else work_item.priority
                }
                
                await self.learning_system.record_outcome(task, result)
                self.hybrid_metrics['outcomes_learned'] += 1
                
            except Exception as e:
                self.logger.error(f"Error recording outcome: {e}")
            
            return result
            
        finally:
            # Remove from processing set when done
            if work_item.task_type == "ISSUE":
                self._processing_issues.discard(work_item.id)
    
    async def _gather_enhanced_context(self) -> Dict[str, Any]:
        """
        Gather comprehensive context for decision making.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            Enhanced context including learning insights
        """
        # Get base context
        context = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_state': self.system_state,
            'worker_states': self._get_worker_summary(),
            'performance_metrics': self.metrics,  # Parent uses self.metrics
            'hybrid_metrics': self.hybrid_metrics
        }
        
        # Add active projects
        context['active_projects'] = self._get_active_projects()
        
        # Add recent outcomes from learning system
        if hasattr(self.learning_system, 'outcome_history'):
            context['recent_outcomes'] = self.learning_system.outcome_history[-10:]
        
        # Add recent tasks
        if hasattr(self.task_generator, 'task_history'):
            context['recent_tasks'] = self.task_generator.task_history[-10:]
        
        # Add learning summary
        try:
            context['learning_summary'] = await self.learning_system.get_recommendations()
        except:
            pass
        
        # Add current charter
        if self.current_charter:
            context['charter'] = self.current_charter
        
        return context
    
    async def _update_charter(self):
        """
        Update system charter based on current context and learning.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        try:
            self.logger.info("Updating system charter...")
            
            context = await self._gather_enhanced_context()
            self.current_charter = await self.charter_system.generate_charter(context)
            self.hybrid_metrics['charters_generated'] += 1
            
            self.logger.info("Charter updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating charter: {e}")
    
    def _get_available_repositories(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get list of available repositories for task generation.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            context: System context
            
        Returns:
            List of available repositories
        """
        projects = self.system_state.get('projects', {})
        
        # Convert to list format
        repos = []
        for project_id, project_data in projects.items():
            if project_data.get('status') == 'active' and should_process_repo(project_data.get('full_name', '')):
                repos.append({
                    'name': project_data.get('name', project_id),
                    'full_name': project_data.get('full_name', project_id),
                    'url': project_data.get('url', ''),
                    'description': project_data.get('description', ''),
                    'language': project_data.get('language', 'Unknown'),
                    'health_score': project_data.get('health_score', 0)
                })
        
        return repos
    
    def _get_active_projects(self) -> List[Dict[str, Any]]:
        """
        Get list of active projects with details.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        return self._get_available_repositories({})
    
    async def _generate_new_project_task(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a NEW_PROJECT task when no repositories exist.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            context: System context
            
        Returns:
            NEW_PROJECT task or None
        """
        try:
            # Use task generator's method
            task = await self.task_generator._generate_new_project_task(context)
            return task
        except Exception as e:
            self.logger.warning(f"Failed to generate NEW_PROJECT task: {e}")
        return None
    
    def _task_to_work_item(self, task: Dict[str, Any]) -> Optional[WorkItem]:
        """
        Convert a task to a work item.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Args:
            task: Task dictionary
            
        Returns:
            WorkItem or None
        """
        try:
            # Map task priority
            priority_map = {
                'critical': TaskPriority.CRITICAL,
                'high': TaskPriority.HIGH,
                'medium': TaskPriority.MEDIUM,
                'low': TaskPriority.LOW
            }
            
            priority = priority_map.get(
                task.get('priority', 'medium').lower(),
                TaskPriority.MEDIUM
            )
            
            return WorkItem(
                id=task.get('id', f"task_{int(time.time())}"),
                task_type=task.get('type', 'task'),
                title=task.get('title', 'Untitled Task'),
                description=task.get('description', ''),
                repository=task.get('repository', task.get('target_project')),
                priority=priority,
                metadata=task
            )
            
        except Exception as e:
            self.logger.error(f"Error converting task to work item: {e}")
            return None
    
    def _get_existing_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of existing tasks from task state.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
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
    
    def _get_worker_summary(self) -> Dict[str, Any]:
        """
        Get summary of worker states.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            Worker summary
        """
        summary = {
            'total_workers': len(self.workers),
            'idle_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE),
            'busy_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.WORKING),
            'error_workers': sum(1 for w in self.workers.values() if w.status == WorkerStatus.ERROR)
        }
        return summary
    
    async def _get_github_issues(self) -> List[WorkItem]:
        """
        Get GitHub issues and convert them to work items.
        
        Override parent to add deduplication logic.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            List of work items from GitHub issues
        """
        work_items = []
        
        try:
            # In local mode, use intelligent work finder to get existing issues
            if os.getenv('STORAGE_MODE') == 'local':
                from scripts.intelligent_work_finder import IntelligentWorkFinder
                work_finder = IntelligentWorkFinder()
                
                # Get all opportunities (which includes existing issues)
                opportunities = await work_finder.find_opportunities()
                
                # Filter to only ISSUE type work items
                for opportunity in opportunities:
                    if hasattr(opportunity, 'type') and opportunity.type == "ISSUE":
                        # Check if this issue is already being processed
                        issue_id = opportunity.id if hasattr(opportunity, 'id') else None
                        if issue_id and issue_id not in self._processing_issues:
                            work_items.append(opportunity)
                
                return work_items[:self.max_workers]  # Limit to max workers
            
            # For GitHub mode, use parent implementation
            return await super()._get_github_issues()
            
        except Exception as e:
            self.logger.error(f"Error getting issues: {e}")
            traceback.print_exc()
            return []
    
    async def _get_github_issues_original(self) -> List[WorkItem]:
        """
        Get GitHub issues and convert them to work items.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            List of work items from GitHub issues
        """
        work_items = []
        
        try:
            # Check if we have GitHub token
            github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
            if not github_token:
                self.logger.debug("No GitHub token available for fetching issues")
                return work_items
            
            # Try to use GitHub issue queue if available
            try:
                from scripts.github_issue_queue import GitHubIssueQueue
                
                # Check if there's a queue instance we can use for stats
                if hasattr(self, 'github_queue'):
                    stats = await self.github_queue.get_queue_stats()
                    self.logger.info(f"GitHub queue stats: {stats}")
            except ImportError:
                pass
            
            # Check repositories for open issues
            projects = self.system_state.get('projects', {})
            
            for project_id, project_data in projects.items():
                if not project_data.get('status') == 'active':
                    continue
                    
                if not should_process_repo(project_data.get('full_name', '')):
                    continue
                
                repo_name = project_data.get('full_name', project_id)
                
                try:
                    # Use TaskManager to check for issues
                    task_manager = TaskManager(repository=repo_name)
                    
                    # Simple check - if repo has GitHub integration
                    if task_manager.github and task_manager.repo:
                        # Get open AI-managed issues
                        issues = task_manager.repo.get_issues(
                            state='open', 
                            labels=['ai-managed'],
                            sort='created',
                            direction='desc'
                        )
                        
                        # Convert first few issues to work items
                        issue_count = 0
                        for issue in issues:
                            if issue_count >= 3:  # Max 3 issues per repo
                                break
                                
                            # Skip if already being worked on
                            labels = [label.name for label in issue.labels]
                            if 'in-progress' in labels:
                                continue
                            
                            # Convert to work item
                            work_item = self._github_issue_to_work_item(
                                issue, 
                                repo_name,
                                project_data
                            )
                            
                            if work_item:
                                work_items.append(work_item)
                                issue_count += 1
                                
                except Exception as e:
                    self.logger.debug(f"Could not fetch issues for {repo_name}: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error getting GitHub issues: {e}")
            traceback.print_exc()
        
        return work_items
    
    def _github_issue_to_work_item(self, issue: Any, repo_name: str, 
                                   project_data: Dict[str, Any]) -> Optional[WorkItem]:
        """
        Convert a GitHub issue to a work item.
        
        Args:
            issue: GitHub issue object
            repo_name: Repository name
            project_data: Project metadata
            
        Returns:
            WorkItem or None
        """
        try:
            # Extract priority from labels
            priority = TaskPriority.MEDIUM
            labels = [label.name for label in issue.labels]
            
            for label in labels:
                if label.startswith('priority:'):
                    priority_str = label.split(':')[1].upper()
                    priority = TaskPriority[priority_str] if hasattr(TaskPriority, priority_str) else TaskPriority.MEDIUM
                    break
            
            # Extract type from labels
            task_type = 'task'
            for label in labels:
                if label.startswith('type:'):
                    task_type = label.split(':')[1]
                    break
            
            # Create work item
            return WorkItem(
                id=f"gh_issue_{issue.number}_{repo_name.replace('/', '_')}",
                task_type=task_type,
                title=f"[GH-{issue.number}] {issue.title}",
                description=issue.body or issue.title,
                repository=repo_name,
                priority=priority,
                metadata={
                    'github_issue_number': issue.number,
                    'github_issue_url': issue.html_url,
                    'labels': labels,
                    'created_at': issue.created_at.isoformat(),
                    'updated_at': issue.updated_at.isoformat(),
                    'source': 'github_issue'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error converting GitHub issue to work item: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        
        Returns:
            System status including hybrid metrics
        """
        status = await super().get_status()
        
        # Add hybrid-specific status
        status['hybrid_mode'] = {
            'enabled': True,
            'charter_version': len(self.charter_system.charter_history) if hasattr(self.charter_system, 'charter_history') else 0,
            'outcomes_recorded': len(self.learning_system.outcome_history) if hasattr(self.learning_system, 'outcome_history') else 0,
            'metrics': self.hybrid_metrics
        }
        
        # Add learning insights
        try:
            status['learning_insights'] = await self.learning_system.get_recommendations()
        except:
            pass
        
        return status
    
    async def run(self):
        """
        Alias for start() to support both naming conventions.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        await self.start()
    
    async def shutdown(self):
        """
        Graceful shutdown with outcome recording.
        
        NOTE: This version uses SQLite/In-memory instead of Redis.
        """
        self.logger.info("Shutting down Hybrid Continuous Orchestrator...")
        
        # Save final metrics
        try:
            final_state = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'total_work_completed': self.metrics.get('total_work_completed', 0),
                'hybrid_metrics': self.hybrid_metrics,
                'final_charter': self.current_charter
            }
            
            # Save to state
            self.state_manager.update_state({'hybrid_final_state': final_state})
            
        except Exception as e:
            self.logger.error(f"Error saving final state: {e}")
        
        # Call parent shutdown if it exists
        if hasattr(super(), 'shutdown'):
            await super().shutdown()
        else:
            # Manual shutdown
            self.shutdown_requested = True
            if hasattr(self, 'workers'):
                for worker in self.workers.values():
                    worker.status = WorkerStatus.SHUTTING_DOWN


# Example usage
if __name__ == "__main__":
    async def run_hybrid():
        orchestrator = HybridContinuousOrchestrator(
            max_workers=5,
            enable_parallel=True,
            enable_research=True
        )
        
        await orchestrator.run()
    
    asyncio.run(run_hybrid())