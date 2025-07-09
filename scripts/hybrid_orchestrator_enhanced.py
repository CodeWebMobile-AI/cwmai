"""
Enhanced Hybrid Continuous Orchestrator with Dynamic Work Generation

This orchestrator uses dynamic work generation based on:
- ARCHITECTURE.md content analysis
- Completed work history tracking
- Repository-specific context
- No fixed task types or dependencies

NOTE: This version uses SQLite/In-memory instead of Redis.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import traceback
import os
import uuid

# Import base continuous system
from scripts.market_research_engine import MarketResearchEngine
from scripts.portfolio_intelligence import PortfolioIntelligence
from scripts.project_creator import ProjectCreator
from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis as ContinuousOrchestrator, WorkerStatus, WorkerState
from scripts.work_item_types import WorkItem, TaskPriority

# Import Dynamic Work Generator instead of fixed task system
from scripts.dynamic_work_generator import DynamicWorkGenerator

# Import Dynamic God Mode components
from scripts.dynamic_charter import DynamicCharter
from scripts.outcome_learning import OutcomeLearningSystem
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.dynamic_validator import DynamicTaskValidator
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.project_creator import ProjectCreator
from scripts.multi_repo_coordinator import MultiRepoCoordinator

# Import utilities
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager
from scripts.repository_exclusion import should_process_repo
from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.storage_factory import StorageFactory, StorageMode
from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.project_context_model import ProjectContext, ProjectStage, CodeMetrics, RepositoryActivity, TechnicalContext, ArchitectureType

# Import monitoring system
from scripts.monitoring_integration import MonitoringIntegration, integrate_monitoring_into_orchestrator

# Import Claude executor pool
from scripts.dynamic_claude_executor_pool import DynamicClaudeExecutorPool
from scripts.claude_execution_queue import ClaudeExecutionQueue


class EnhancedHybridOrchestrator(ContinuousOrchestrator):
    """
    Enhanced hybrid orchestrator with dynamic work generation.
    
    Features:
    - 24/7 continuous operation
    - Dynamic work generation based on context
    - ARCHITECTURE.md driven development
    - Completed work history tracking
    - Project management analytics
    - No fixed task types or templates
    """
    
    def __init__(self, max_workers: int = 5, enable_parallel: bool = True,
                 enable_round_robin: bool = False,
                 prioritization_strategy: str = "adaptive"):
        """
        Initialize enhanced hybrid orchestrator.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            enable_round_robin: Whether to enable round-robin AI selection
            prioritization_strategy: Task prioritization strategy to use
        """
        # Initialize base continuous orchestrator
        super().__init__(max_workers, enable_parallel, enable_round_robin)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Dynamic Work Generator
        self.work_generator = DynamicWorkGenerator(ai_brain=None)  # Will set AI brain later
        
        # Initialize Dynamic God Mode components
        self._initialize_dynamic_components()
        
        # Project context cache
        self._project_contexts: Dict[str, ProjectContext] = {}
        
        # Metrics for enhanced operation
        self.enhanced_metrics = {
            'smart_tasks_selected': 0,
            'dependency_chains_executed': 0,
            'critical_paths_analyzed': 0,
            'context_updates': 0,
            'strategy_switches': 0,
            'task_assignments': 0
        }
        
        # Ensure we have metrics from parent
        if not hasattr(self, 'metrics'):
            self.metrics = {
                'total_work_completed': 0,
                'total_errors': 0,
                'total_work_created': 0,
                'work_per_hour': 0.0,
                'average_completion_time': 0.0
            }
        
        # Task execution tracking
        self.active_execution_plans: Dict[str, List[List[str]]] = {}
        
        # Track completed tasks to prevent duplication
        self.completed_tasks: Set[str] = set()
        self.processed_issues: Set[str] = set()  # Track issue IDs we've already processed
        self._processed_issues_file = "processed_issues.json"
        self._load_processed_issues()
        
        # Lock for work discovery to prevent multiple workers discovering simultaneously
        self._discovery_lock = asyncio.Lock()
        self._last_discovery_time = 0
        self._discovery_interval = int(os.getenv('DISCOVERY_INTERVAL_SECONDS', '30'))  # Minimum seconds between discoveries
        self._empty_queue_backoff = 0.2  # Start with 200ms
        self._max_backoff = 30.0  # Max 30 seconds
        self._last_queue_check_time = 0
        
        # Track discovered work to prevent duplicates
        self._discovered_work_ids: Set[str] = set()
        self._discovered_issue_ids: Set[str] = set()
        
        # Initialize intelligent work finder
        self.intelligent_work_finder = IntelligentWorkFinder(self.ai_brain, self.system_state)
        
        # Initialize monitoring system
        self.monitoring = MonitoringIntegration()
        self.logger.info("✅ Task monitoring system initialized")
        
        # Initialize Dynamic Claude executor pool
        self.claude_min_executors = int(os.getenv('CLAUDE_MIN_EXECUTORS', 3))
        self.claude_max_executors = int(os.getenv('CLAUDE_MAX_EXECUTORS', 15))
        self.claude_scale_up_threshold = float(os.getenv('CLAUDE_SCALE_UP_THRESHOLD', 5.0))
        self.claude_scale_down_threshold = float(os.getenv('CLAUDE_SCALE_DOWN_THRESHOLD', 0.5))
        self.claude_scale_check_interval = int(os.getenv('CLAUDE_SCALE_CHECK_INTERVAL', 30))
        self.claude_idle_timeout = int(os.getenv('CLAUDE_IDLE_TIMEOUT', 300))
        
        self.claude_queue = ClaudeExecutionQueue()
        self.claude_executor_pool = DynamicClaudeExecutorPool(
            min_executors=self.claude_min_executors,
            max_executors=self.claude_max_executors,
            queue=self.claude_queue,
            scale_up_threshold=self.claude_scale_up_threshold,
            scale_down_threshold=self.claude_scale_down_threshold,
            scale_check_interval=self.claude_scale_check_interval,
            idle_timeout=self.claude_idle_timeout
        )
        self.logger.info(f"✅ Dynamic Claude executor pool initialized (min={self.claude_min_executors}, max={self.claude_max_executors})")
        
    def _initialize_dynamic_components(self):
        """Initialize Dynamic God Mode components."""
        self.logger.info("Initializing Dynamic God Mode components...")
        
        # Use production AI brain
        self.ai_brain = AIBrainFactory.create_for_production()
        
        # Set AI brain in work generator
        self.work_generator.ai_brain = self.ai_brain
        
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
            github_token=os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT'),
            ai_brain=self.ai_brain
        )
        
        # Project creator for NEW_PROJECT tasks
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        self.project_creator = ProjectCreator(github_token, self.ai_brain) if github_token else None
        
        # Multi-repo coordinator
        storage_mode = os.getenv('STORAGE_MODE', StorageMode.GITHUB).lower()
        
        # Only initialize coordinator in GitHub mode or if explicitly needed
        if storage_mode == StorageMode.GITHUB and github_token and self.system_state.get('projects'):
            active_repos = [p for p in self.system_state.get('projects', {}).values() 
                           if should_process_repo(p.get('full_name', ''))]
            max_concurrent = max(len(active_repos), 50)
            
            self.coordinator = MultiRepoCoordinator(github_token, max_concurrent=max_concurrent)
            # Add discovered repositories
            for project_id, project_data in self.system_state.get('projects', {}).items():
                full_name = project_data.get('full_name', '')
                if full_name and should_process_repo(full_name):
                    try:
                        clone_url = project_data.get('clone_url', '')
                        if clone_url:
                            self.coordinator.add_repository(clone_url)
                    except Exception as e:
                        self.logger.warning(f"Failed to add repository {project_id}: {e}")
        elif storage_mode == StorageMode.LOCAL:
            # In local mode, create coordinator but only add repositories that actually exist locally
            self.coordinator = MultiRepoCoordinator(None, max_concurrent=10)
            
            # Get actual local repositories
            local_simulator = LocalGitHubSimulator()
            existing_repos = local_simulator.list_repositories()
            
            for repo in existing_repos:
                if should_process_repo(repo.full_name):
                    try:
                        self.coordinator.add_repository(repo.full_name)
                        self.logger.debug(f"Added existing local repository: {repo.full_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to add local repository {repo.full_name}: {e}")
        else:
            self.coordinator = None
            
        self.logger.info("Dynamic God Mode components initialized")
    
    async def find_work(self) -> List[WorkItem]:
        """
        Find work using enhanced smart task type system.
        
        Returns:
            List of work items prioritized by intelligence
        """
        all_work = []
        
        self.logger.info(f"[FIND_WORK] Starting work discovery (max_workers: {self.max_workers})")
        start_time = time.time()
        
        try:
            # 1. Check storage mode and fetch issues accordingly
            storage_mode = os.getenv('STORAGE_MODE', StorageMode.GITHUB)
            self.logger.info(f"[FIND_WORK] Storage mode: {storage_mode}")
            
            if storage_mode == StorageMode.LOCAL:
                # Fetch issues from local repositories
                self.logger.info(f"[FIND_WORK] Fetching local issues...")
                local_start = time.time()
                local_work = await self._get_local_issues()
                local_duration = time.time() - local_start
                all_work.extend(local_work)
                self.logger.info(f"[FIND_WORK] Found {len(local_work)} local issues in {local_duration:.2f}s")
                
                # Log work item details
                if local_work:
                    self.logger.debug(f"[FIND_WORK] Local work items by repository:")
                    repo_counts = {}
                    for item in local_work:
                        repo_counts[item.repository] = repo_counts.get(item.repository, 0) + 1
                    for repo, count in repo_counts.items():
                        self.logger.debug(f"  - {repo}: {count} items")
            else:
                # Fetch from GitHub (existing logic)
                github_work = []
                # github_work = await self._get_github_issues()  # TODO: implement if needed
                # all_work.extend(github_work)
                self.logger.info(f"[FIND_WORK] Found {len(github_work)} GitHub issues")
            
            self.logger.info(f"[FIND_WORK] Total work items after issue fetch: {len(all_work)}")
            
            # 2. If we need more work, try intelligent work finder with timeout
            if len(all_work) < self.max_workers:
                needed = self.max_workers - len(all_work)
                self.logger.info(f"[FIND_WORK] Need {needed} more work items, trying intelligent work finder...")
                
                # Try intelligent work finder without timeout - let it complete
                try:
                    intelligent_start = time.time()
                    intelligent_work = await self.intelligent_work_finder.discover_work(
                        max_items=needed,
                        current_workload=len(all_work)
                    )
                    intelligent_duration = time.time() - intelligent_start
                    
                    if intelligent_work:
                        all_work.extend(intelligent_work)
                        self.logger.info(f"[FIND_WORK] Intelligent work finder discovered {len(intelligent_work)} work items in {intelligent_duration:.2f}s")
                        for idx, item in enumerate(intelligent_work[:3]):
                            self.logger.debug(f"[FIND_WORK] Intelligent item {idx+1}: {item.repository} - {item.title}")
                except Exception as e:
                    self.logger.error(f"[FIND_WORK] Intelligent work finder error: {e}")
                    self.logger.debug(f"[FIND_WORK] Intelligent work finder traceback: {traceback.format_exc()}")
                    # NO FALLBACK - just continue with what we have
                    self.logger.info("[FIND_WORK] Continuing with existing work items")
            
            self.logger.info(f"[FIND_WORK] Total work items after intelligent generation: {len(all_work)}")
            
            # 3. If we still need more work, use smart task selection
            if len(all_work) < self.max_workers:
                needed = self.max_workers - len(all_work)
                self.logger.info(f"[FIND_WORK] Still need {needed} more work items, using smart task selection...")
                
                # Get project contexts without timeout - let it complete
                try:
                    context_start = time.time()
                    contexts = await self._gather_project_contexts()
                    context_duration = time.time() - context_start
                    self.logger.info(f"[FIND_WORK] Gathered contexts for {len(contexts)} projects in {context_duration:.2f}s")
                except Exception as e:
                    self.logger.error(f"[FIND_WORK] Error gathering project contexts: {e}")
                    contexts = {}  # Empty contexts will skip smart task selection
                
                # Generate work items dynamically for each project
                dynamic_work_added = 0
                for repo_name, context in contexts.items():
                    if len(all_work) >= self.max_workers:
                        break
                    
                    self.logger.debug(f"[FIND_WORK] Generating dynamic work for {repo_name}")
                    
                    # Get completed work for this repository
                    completed_work = self._get_completed_work_for_repo(repo_name)
                    
                    # Generate work items using dynamic work generator
                    try:
                        work_items = await self.work_generator.generate_work_for_repository(
                            repository=repo_name,
                            max_work_items=needed,
                            completed_work=completed_work
                        )
                        
                        self.logger.debug(f"[FIND_WORK] Generated {len(work_items)} work items for {repo_name}")
                        
                        for work_item in work_items:
                            if len(all_work) >= self.max_workers:
                                break
                            all_work.append(work_item)
                            self.enhanced_metrics['smart_tasks_selected'] += 1
                            dynamic_work_added += 1
                            self.logger.debug(f"[FIND_WORK] Added dynamic work: {work_item.task_type} for {repo_name}")
                        
                        # Update context tracking
                        self.enhanced_metrics['context_updates'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"[FIND_WORK] Error generating work for {repo_name}: {e}")
                        continue
                
                self.logger.info(f"[FIND_WORK] Added {dynamic_work_added} dynamically generated work items")
            
            # 3. Skip execution plans - dynamic work doesn't use fixed dependencies
            if all_work:
                self.logger.info(f"[FIND_WORK] Skipping execution plans for dynamic work items")
            
        except Exception as e:
            self.logger.error(f"[FIND_WORK] Critical error in find_work: {e}")
            self.logger.error(f"[FIND_WORK] Full traceback: {traceback.format_exc()}")
        
        total_duration = time.time() - start_time
        final_work = all_work[:self.max_workers]
        
        self.logger.info(f"[FIND_WORK] Work discovery complete:")
        self.logger.info(f"  - Total duration: {total_duration:.2f}s")
        self.logger.info(f"  - Total work items found: {len(all_work)}")
        self.logger.info(f"  - Work items returned: {len(final_work)}")
        
        # Log final work distribution
        if final_work:
            self.logger.info(f"[FIND_WORK] Final work distribution:")
            task_type_counts = {}
            repo_counts = {}
            for item in final_work:
                task_type_counts[item.task_type] = task_type_counts.get(item.task_type, 0) + 1
                repo_counts[item.repository] = repo_counts.get(item.repository, 0) + 1
            
            self.logger.info(f"  By task type:")
            for task_type, count in task_type_counts.items():
                self.logger.info(f"    - {task_type}: {count}")
            
            self.logger.info(f"  By repository:")
            # Filter out None repositories and sort
            valid_repos = [(repo, count) for repo, count in repo_counts.items() if repo is not None]
            for repo, count in sorted(valid_repos)[:5]:  # Top 5 repos
                self.logger.info(f"    - {repo}: {count}")
            if len(repo_counts) > 5:
                self.logger.info(f"    ... and {len(repo_counts) - 5} more repositories")
        
        return final_work
    
    async def _discover_and_queue_work(self):
        """Discover work and add it to the queue.
        
        This method is called when the queue is empty and ensures
        only one worker discovers work at a time.
        """
        async with self._discovery_lock:
            # Check if we've discovered recently
            current_time = time.time()
            if current_time - self._last_discovery_time < self._discovery_interval:
                self.logger.debug(f"[DISCOVERY] Skipping discovery - too recent ({current_time - self._last_discovery_time:.1f}s ago)")
                return
            
            self.logger.info("[DISCOVERY] Starting work discovery...")
            work_items = await self.find_work()
            
            if work_items:
                # Filter out items already in queue or being processed
                new_items = []
                for item in work_items:
                    # Check if item is already in queue or being processed
                    if not await self.hybrid_state.work_queue.is_in_queue(item.id) and not self.is_work_being_processed(item.id):
                        new_items.append(item)
                    else:
                        self.logger.debug(f"[DISCOVERY] Filtering out already queued/processing: {item.id}")
                
                if new_items:
                    self.logger.info(f"[DISCOVERY] Adding {len(new_items)} new work items to queue (filtered from {len(work_items)} total)")
                    added_count = 0
                    
                    for item in new_items:
                        # Add to queue using parent's method
                        if await self.add_work(item):
                            added_count += 1
                            self.logger.debug(f"[DISCOVERY] Queued: {item.id} - {item.title}")
                        else:
                            self.logger.warning(f"[DISCOVERY] Failed to queue: {item.id} - {item.title}")
                    
                    self.logger.info(f"[DISCOVERY] Successfully queued {added_count}/{len(new_items)} work items")
                else:
                    self.logger.info(f"[DISCOVERY] All {len(work_items)} discovered items were duplicates")
                    
                self._last_discovery_time = current_time
            else:
                self.logger.info("[DISCOVERY] No work items discovered")
                self._last_discovery_time = current_time
    
    async def find_single_work_item(self, worker_id: str = None) -> Optional[WorkItem]:
        """Find a single work item for continuous orchestrator compatibility.
        
        This method now properly uses the parent's queue-based approach
        to ensure each worker gets a unique work item.
        """
        # First, check if there's already work in the queue
        work_item = await super().find_single_work_item(worker_id)
        if work_item:
            # Reset backoff on successful work retrieval
            self._empty_queue_backoff = 0.2
            return work_item
            
        # Rate limit queue checks to prevent log spam
        current_time = time.time()
        time_since_last_check = current_time - self._last_queue_check_time
        
        if time_since_last_check < self._empty_queue_backoff:
            # Still in backoff period, return None without logging
            return None
            
        # Update last check time
        self._last_queue_check_time = current_time
        
        # If queue is empty, discover new work and add to queue
        # Only log if we haven't logged recently
        if time_since_last_check > 5.0:  # Only log every 5+ seconds
            self.logger.info(f"[FIND_SINGLE] Queue empty, discovering new work...")
        
        await self._discover_and_queue_work()
        
        # Increase backoff for next time (exponential backoff)
        self._empty_queue_backoff = min(self._empty_queue_backoff * 2, self._max_backoff)
        
        # Now try to get from queue again
        return await super().find_single_work_item(worker_id)
    
    async def _gather_project_contexts(self) -> Dict[str, ProjectContext]:
        """Gather context for all active projects."""
        contexts = {}
        
        for project_id, project_data in self.system_state.get('projects', {}).items():
            repo_name = project_data.get('full_name', '')
            
            # Skip if repo_name is empty or should not be processed
            if not repo_name or not should_process_repo(repo_name):
                if not repo_name:
                    self.logger.debug(f"Skipping project {project_id} - missing full_name")
                continue
            
            # Check cache
            if repo_name in self._project_contexts:
                contexts[repo_name] = self._project_contexts[repo_name]
                continue
            
            try:
                # Create project context
                context = await self._build_project_context(repo_name, project_data)
                contexts[repo_name] = context
                self._project_contexts[repo_name] = context
                
            except Exception as e:
                self.logger.warning(f"Error building context for {repo_name}: {e}")
        
        return contexts
    
    def is_work_being_processed(self, work_id: str) -> bool:
        """Check if a work item is currently being processed by any worker.
        
        Args:
            work_id: ID of the work item to check
            
        Returns:
            True if the work item is being processed
        """
        # Check if any worker is processing this work item
        for worker_id, worker in self.workers.items():
            if hasattr(worker, 'current_work') and worker.current_work:
                if worker.current_work.id == work_id:
                    return True
        
        # Also check processing items in work queue
        with self.hybrid_state.work_queue._processing_lock:
            return work_id in self.hybrid_state.work_queue._processing
    
    async def _build_project_context(
        self, 
        repo_name: str, 
        project_data: Dict[str, Any]
    ) -> ProjectContext:
        """Build a comprehensive project context."""
        # Validate repo_name
        if not repo_name:
            raise ValueError("Cannot build context for empty repository name")
            
        self.logger.debug(f"[BUILD_CONTEXT] Building context for repository: {repo_name}")
        
        # Get repository analysis
        repo_analysis = await self.repository_analyzer.analyze_repository(repo_name)
        
        # Determine project stage
        created_at_str = project_data.get('created_at', datetime.now(timezone.utc).isoformat())
        self.logger.debug(f"[BUILD_CONTEXT] Raw created_at string: {created_at_str}")
        
        # Handle both timezone-aware and naive datetimes
        try:
            if created_at_str.endswith('Z'):
                created_at_str = created_at_str[:-1] + '+00:00'
                self.logger.debug(f"[BUILD_CONTEXT] Converted Z timezone to ISO format: {created_at_str}")
            
            created_at = datetime.fromisoformat(created_at_str)
            self.logger.debug(f"[BUILD_CONTEXT] Parsed datetime: {created_at}")
            self.logger.debug(f"[BUILD_CONTEXT] Datetime tzinfo: {created_at.tzinfo}")
            
            # Ensure timezone awareness
            if created_at.tzinfo is None:
                self.logger.debug(f"[BUILD_CONTEXT] Datetime is naive, adding UTC timezone")
                created_at = created_at.replace(tzinfo=timezone.utc)
            
            self.logger.debug(f"[BUILD_CONTEXT] Final created_at: {created_at}")
        except Exception as dt_error:
            self.logger.error(f"[BUILD_CONTEXT] Error parsing datetime '{created_at_str}': {dt_error}")
            self.logger.debug(f"[BUILD_CONTEXT] Datetime parsing traceback: {traceback.format_exc()}")
            created_at = datetime.now(timezone.utc)
            self.logger.debug(f"[BUILD_CONTEXT] Using current UTC time as fallback: {created_at}")
        
        # Skip stage determination - let task selection be based on actual needs
        # not arbitrary lifecycle stages
        stage = ProjectStage.ACTIVE_DEVELOPMENT  # Default stage for all existing repos
        
        # Extract metrics from analysis
        code_metrics = CodeMetrics(
            lines_of_code=repo_analysis.get('metrics', {}).get('total_lines', 0),
            total_files=repo_analysis.get('metrics', {}).get('total_files', 0),
            language_distribution=repo_analysis.get('languages', {})
        )
        
        # Repository activity
        repo_activity = RepositoryActivity(
            open_issues=project_data.get('open_issues', 0),
            active_contributors=1  # Default, would need GitHub API for accurate count
        )
        
        # Determine architecture type
        architecture = self._detect_architecture(repo_analysis)
        
        # Build technical context
        tech_context = TechnicalContext(
            architecture_type=architecture,
            tech_stack=list(repo_analysis.get('languages', {}).keys())
        )
        
        # Get completed tasks from state
        completed_tasks = set()
        task_history = self.state_manager.get_task_history()
        for task in task_history:
            if (task.get('repository') == repo_name and 
                task.get('status') == 'completed'):
                task_type = task.get('task_type', '')
                if task_type:
                    completed_tasks.add(task_type)
        
        # Build context
        context = ProjectContext(
            project_id=project_data.get('id', repo_name),
            project_name=repo_name,
            created_at=created_at,
            stage=stage,
            code_metrics=code_metrics,
            repository_activity=repo_activity,
            technical_context=tech_context,
            completed_task_types=completed_tasks,
            # team_members removed - AI-driven system
            team_size=0
        )
        
        return context
    
    def _detect_architecture(self, repo_analysis: Dict[str, Any]) -> ArchitectureType:
        """Detect project architecture from repository analysis."""
        # All projects are Laravel/React clones
        return ArchitectureType.LARAVEL_REACT
    
    # Method removed - no longer needed with dynamic work generator
    
    def _get_completed_work_for_repo(self, repository: str) -> List[Dict[str, Any]]:
        """Get completed work for a specific repository."""
        try:
            # Load completed tasks from state
            task_state = self.state_manager.load_task_state()
            completed_tasks = task_state.get('completed_tasks', [])
            
            # Filter for this repository
            repo_completed = [
                {
                    'id': task.get('id'),
                    'title': task.get('title'),
                    'description': task.get('description', ''),
                    'task_type': task.get('task_type'),
                    'completed_at': task.get('completed_at')
                }
                for task in completed_tasks
                if task.get('repository') == repository
            ]
            
            return repo_completed
            
        except Exception as e:
            self.logger.error(f"Error loading completed work for {repository}: {e}")
            return []
    
    async def execute_work(self, work_item: WorkItem, worker_id: str) -> Dict[str, Any]:
        """Execute work with smart task tracking and monitoring."""
        start_time = time.time()
        self.logger.info(f"[EXECUTE_WORK] Worker {worker_id} starting execution of {work_item.id}")
        self.logger.debug(f"[EXECUTE_WORK] Work item details:")
        self.logger.debug(f"  - Type: {work_item.task_type}")
        self.logger.debug(f"  - Title: {work_item.title}")
        self.logger.debug(f"  - Repository: {work_item.repository}")
        self.logger.debug(f"  - Priority: {work_item.priority}")
        self.logger.debug(f"  - Smart task: {work_item.metadata.get('smart_task', False)}")
        
        # Start monitoring for this task
        task_id = work_item.id
        
        # Predict duration and get insights
        predicted_duration = self.monitoring.predict_task_duration(
            work_item.task_type, 
            work_item.repository or ""
        )
        failure_risk = self.monitoring.get_failure_risk(work_item.task_type)
        
        if failure_risk > 0.7:
            self.logger.warning(f"⚠️ High failure risk ({failure_risk:.1%}) for task {task_id}")
        
        # Get best worker recommendation
        best_worker = self.monitoring.get_best_worker(work_item.task_type)
        if best_worker and best_worker != worker_id:
            self.logger.info(f"Note: {best_worker} might be better for {work_item.task_type} tasks")
        
        # Set estimated duration on work item
        work_item.metadata['estimated_duration'] = predicted_duration
        
        # Record lifecycle event
        self.monitoring.track_lifecycle_event(task_id, "assigned", {
            "worker_id": worker_id,
            "predicted_duration": predicted_duration,
            "failure_risk": failure_risk
        })
        
        # Status tracking removed - dynamic work doesn't use fixed task tracking
        
        # Execute using parent method
        self.logger.debug(f"[EXECUTE_WORK] Calling parent execute_work method")
        try:
            result = await super().execute_work(work_item, worker_id)
            execution_time = time.time() - start_time
            
            self.logger.info(f"[EXECUTE_WORK] Worker {worker_id} completed {work_item.id} in {execution_time:.2f}s")
            self.logger.debug(f"[EXECUTE_WORK] Execution result: success={result.get('success')}, error={result.get('error')}")
            
        except Exception as exec_error:
            execution_time = time.time() - start_time
            self.logger.error(f"[EXECUTE_WORK] Worker {worker_id} failed executing {work_item.id} after {execution_time:.2f}s")
            self.logger.error(f"[EXECUTE_WORK] Execution error: {exec_error}")
            self.logger.error(f"[EXECUTE_WORK] Full traceback: {traceback.format_exc()}")
            result = {
                'success': False,
                'error': str(exec_error),
                'traceback': traceback.format_exc()
            }
        
        # Status tracking removed - dynamic work doesn't use fixed task tracking
        
        # Track assignments and performance
        if result.get('success'):
            self.enhanced_metrics['task_assignments'] += 1
            self.logger.info(f"[EXECUTE_WORK] ✅ Successfully executed task (total successful: {self.enhanced_metrics['task_assignments']})")
        else:
            self.logger.warning(f"[EXECUTE_WORK] ❌ Failed to execute task: {result.get('error')}")
        
        # Add execution time to result
        result['execution_time'] = execution_time
        
        return result
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics including smart task metrics."""
        base_analytics = await super().get_analytics()
        
        # Add enhanced metrics
        base_analytics['enhanced_metrics'] = self.enhanced_metrics
        
        # Task system metrics removed - using dynamic work generation
        base_analytics['work_generator_active'] = True
        base_analytics['dynamic_work_enabled'] = True
        
        # Add Claude executor pool statistics
        base_analytics['claude_executor_pool'] = self.claude_executor_pool.get_stats()
        
        return base_analytics
    
    def switch_prioritization_strategy(self, strategy: str):
        """Switch the task prioritization strategy at runtime."""
        # Method kept for compatibility but strategy is now handled by dynamic generator
        self.enhanced_metrics['strategy_switches'] += 1
        self.logger.info(f"Prioritization strategy switch recorded: {strategy} (handled by dynamic generator)")
    
    def _load_processed_issues(self):
        """Load processed issues from persistent storage."""
        if os.path.exists(self._processed_issues_file):
            try:
                with open(self._processed_issues_file, 'r') as f:
                    data = json.load(f)
                    self.processed_issues = set(data.get('processed_issues', []))
                    self.logger.info(f"[PERSISTENCE] Loaded {len(self.processed_issues)} processed issues from disk")
            except Exception as e:
                self.logger.error(f"[PERSISTENCE] Failed to load processed issues: {e}")
                self.processed_issues = set()
        else:
            self.logger.info(f"[PERSISTENCE] No processed issues file found, starting fresh")
            self.processed_issues = set()
    
    def _save_processed_issues(self):
        """Save processed issues to persistent storage."""
        try:
            data = {
                'processed_issues': list(self.processed_issues),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self._processed_issues_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"[PERSISTENCE] Saved {len(self.processed_issues)} processed issues to disk")
        except Exception as e:
            self.logger.error(f"[PERSISTENCE] Failed to save processed issues: {e}")
    
    def mark_issue_processed(self, issue_id: str):
        """Mark an issue as processed and persist to disk."""
        if issue_id not in self.processed_issues:
            self.processed_issues.add(issue_id)
            self._save_processed_issues()
            self.logger.info(f"[PERSISTENCE] Marked issue {issue_id} as processed")
    
    async def _execute_work_item(self, worker_id: str, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a single work item and mark issues as processed.
        
        Override parent to add persistent issue tracking.
        """
        # Execute using parent's method
        result = await super()._execute_work_item(worker_id, work_item)
        
        # If successful and this was an issue, mark it as processed
        if result.get('success', False):
            # Check if this work item has an issue_id in metadata
            issue_id = work_item.metadata.get('issue_id') if work_item.metadata else None
            
            # Also check for issue-based work items
            if not issue_id and work_item.task_type == "ISSUE" and work_item.repository:
                # Construct issue_id from repository and issue number
                issue_number = work_item.metadata.get('issue_number') if work_item.metadata else None
                if issue_number:
                    issue_id = f"{work_item.repository}#{issue_number}"
            
            # Mark as processed if we have an issue_id
            if issue_id:
                self.mark_issue_processed(issue_id)
                self.logger.info(f"[PERSISTENCE] Successfully processed and marked issue: {issue_id}")
        
        return result
    
    async def start(self):
        """Start the orchestrator with Claude queue cleanup."""
        # Clean up stuck Claude tasks before starting
        self.logger.info("🧹 Cleaning up stuck Claude tasks...")
        try:
            recovered = await self.claude_queue.cleanup_stuck_tasks()
            if recovered > 0:
                self.logger.info(f"✅ Recovered {recovered} stuck Claude tasks")
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup stuck tasks: {e}")
        
        # Start Claude executor pool
        self.logger.info("🚀 Starting Claude executor pool...")
        await self.claude_executor_pool.start()
        
        # Call parent's start method
        await super().start()
    
    async def shutdown(self):
        """Shutdown the orchestrator and Claude executor pool."""
        self.logger.info("Shutting down Enhanced Hybrid Orchestrator...")
        
        # Stop Claude executor pool
        try:
            await self.claude_executor_pool.stop()
            self.logger.info("✅ Claude executor pool stopped")
        except Exception as e:
            self.logger.error(f"❌ Error stopping Claude executor pool: {e}")
        
        # Call parent's shutdown if it exists
        if hasattr(super(), 'shutdown'):
            await super().shutdown()


    async def _get_local_issues(self) -> List[WorkItem]:
        """Fetch issues from local repositories."""
        work_items = []
        
        self.logger.info(f"[LOCAL_ISSUES] Starting local issue discovery...")
        self.logger.info(f"[LOCAL_ISSUES] Previously processed issues: {len(self.processed_issues)}")
        
        try:
            # Initialize local simulator
            local_simulator = LocalGitHubSimulator()
            self.logger.info(f"[LOCAL_ISSUES] LocalGitHubSimulator initialized at: {local_simulator.base_path}")
            
            # Get all repositories
            repositories = local_simulator.list_repositories()
            self.logger.info(f"[LOCAL_ISSUES] Found {len(repositories)} local repositories")
            
            total_issues_found = 0
            new_issues_found = 0
            
            for repo_idx, repo in enumerate(repositories):
                self.logger.debug(f"[LOCAL_ISSUES] Processing repository {repo_idx+1}/{len(repositories)}: {repo.full_name}")
                
                try:
                    # Get open issues for this repository
                    issues = local_simulator.get_issues(repo.full_name, state="open")
                    self.logger.debug(f"[LOCAL_ISSUES] Repository '{repo.full_name}' has {len(issues)} open issues")
                    total_issues_found += len(issues)
                    
                    for issue in issues:
                        # Create unique ID for this issue
                        issue_id = f"{repo.full_name}#{issue.number}"
                        
                        # Skip if we've already discovered or processed this issue
                        if issue_id in self._discovered_issue_ids or issue_id in self.processed_issues:
                            self.logger.debug(f"[LOCAL_ISSUES] Skipping already discovered/processed issue: {issue_id}")
                            continue
                        
                        self.logger.debug(f"[LOCAL_ISSUES] Processing new issue: {issue_id} - {issue.title}")
                        
                        # Log issue details for debugging
                        self.logger.debug(f"[LOCAL_ISSUES] Issue details:")
                        self.logger.debug(f"  - Title: {issue.title}")
                        self.logger.debug(f"  - Number: {issue.number}")
                        self.logger.debug(f"  - Labels: {issue.labels}")
                        self.logger.debug(f"  - Created at: {issue.created_at}")
                        self.logger.debug(f"  - Body length: {len(issue.body) if issue.body else 0}")
                        
                        # Convert issue to WorkItem with unique ID
                        unique_suffix = uuid.uuid4().hex[:8]
                        work_item = WorkItem(
                            id=f"issue_{repo.full_name}_{issue.number}_{int(time.time())}_{unique_suffix}",
                            task_type="ISSUE",
                            title=issue.title,
                            description=issue.body or "",
                            priority=self._extract_priority_from_labels(issue.labels),
                            repository=repo.full_name,
                            metadata={
                                "issue_number": issue.number,
                                "labels": issue.labels,
                                "created_at": issue.created_at.isoformat(),
                                "html_url": issue.html_url
                            }
                        )
                        
                        self.logger.debug(f"[LOCAL_ISSUES] Created WorkItem: {work_item.id} with priority: {work_item.priority}")
                        
                        # Don't mark as processed yet - only after successful execution
                        # Store the issue_id in the work item metadata for later tracking
                        work_item.metadata['issue_id'] = issue_id
                        
                        # Mark as discovered to prevent duplicates
                        self._discovered_issue_ids.add(issue_id)
                        self._discovered_work_ids.add(work_item.id)
                        
                        work_items.append(work_item)
                        new_issues_found += 1
                        
                except Exception as repo_error:
                    self.logger.error(f"[LOCAL_ISSUES] Error processing repository '{repo.full_name}': {repo_error}")
                    self.logger.error(f"[LOCAL_ISSUES] Repository error traceback: {traceback.format_exc()}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"[LOCAL_ISSUES] Critical error in _get_local_issues: {e}")
            self.logger.error(f"[LOCAL_ISSUES] Full traceback: {traceback.format_exc()}")
        
        self.logger.info(f"[LOCAL_ISSUES] Issue discovery summary:")
        self.logger.info(f"  - Total issues found: {total_issues_found}")
        self.logger.info(f"  - New issues to process: {new_issues_found}")
        self.logger.info(f"  - Total work items created: {len(work_items)}")
        
        # Shuffle work items to ensure diversity across repositories
        if work_items:
            import random
            random.shuffle(work_items)
            self.logger.info(f"[LOCAL_ISSUES] Work items shuffled for diversity")
            
            # Log first few work items for verification
            self.logger.info(f"[LOCAL_ISSUES] First 3 work items after shuffle:")
            for i, item in enumerate(work_items[:3]):
                self.logger.info(f"  {i+1}. {item.repository}: {item.title}")
        
        return work_items
    
    async def _execute_work_item(self, worker_id: str, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a work item and track issue processing.
        
        Override to add issue tracking only after successful execution.
        """
        # Call parent implementation
        result = await super()._execute_work_item(worker_id, work_item)
        
        # Check if this work item has an associated issue ID
        issue_id = work_item.metadata.get('issue_id')
        if issue_id:
            # Only mark as processed if execution was successful
            if result.get('success', False):
                self.logger.info(f"[ISSUE_TRACKING] Marking issue {issue_id} as processed after successful execution")
                self.processed_issues.add(issue_id)
            else:
                self.logger.warning(f"[ISSUE_TRACKING] Issue {issue_id} execution failed, NOT marking as processed")
        
        # Save completed task to state if successful
        if result.get('success', False):
            # Check if task is actually completed or just queued
            if result.get('completed', True):  # Default to True for backward compatibility
                try:
                    # Update task state using new methods
                    task_state = self.state_manager.load_task_state()
                    
                    # Initialize completed_tasks if not exists
                    if 'completed_tasks' not in task_state:
                        task_state['completed_tasks'] = []
                    
                    # Add completed task
                    completed_task = {
                        'id': work_item.id,
                        'task_type': work_item.task_type,
                        'title': work_item.title,
                        'repository': work_item.repository,
                        'completed_at': datetime.now(timezone.utc).isoformat(),
                        'duration': result.get('duration', 0),
                        'worker_id': worker_id,
                        'issue_id': issue_id
                    }
                    
                    task_state['completed_tasks'].append(completed_task)
                    
                    # Remove from active tasks if present
                    if 'tasks' in task_state and work_item.id in task_state['tasks']:
                        del task_state['tasks'][work_item.id]
                    
                    # Save updated task state to local_repos/state
                    self.state_manager.save_task_state(task_state)
                    self.logger.info(f"✅ [STATE_SAVE] Saved completed task {work_item.id} to local_repos/state/task_state.json")
                    
                    # Also save to separate completed tasks file
                    completed_tasks = self.state_manager.load_completed_tasks()
                    completed_tasks.append(completed_task)
                    self.state_manager.save_completed_tasks(completed_tasks)
                    self.logger.info(f"✅ [STATE_SAVE] Added task to local_repos/state/completed_tasks.json")
                    
                    # Update metrics
                    self.metrics['total_work_completed'] = self.metrics.get('total_work_completed', 0) + 1
                    
                    # Save orchestrator state
                    await self._save_orchestrator_state()
                    
                    # Close the issue if it's an issue-based task
                    if issue_id and work_item.metadata.get('issue_number'):
                        try:
                            # Initialize local simulator to update issue
                            local_simulator = LocalGitHubSimulator()
                            
                            # Update issue to closed state
                            updated_issue = local_simulator.update_issue(
                                work_item.repository,
                                work_item.metadata['issue_number'],
                                state="closed",
                                labels=["completed", "ai-resolved"]
                            )
                            
                            self.logger.info(f"✅ [ISSUE_UPDATE] Closed issue #{work_item.metadata['issue_number']} in {work_item.repository}")
                            
                            # Add a comment to the issue with completion details
                            comment_text = f"""## Task Completed Successfully! 🎉

**Task Type:** {work_item.task_type}
**Completed By:** Worker {worker_id}
**Duration:** {result.get('duration', 0):.2f} seconds
**Completion Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

### Summary
{result.get('output', {}).get('summary', 'Task completed successfully.')}

### Actions Taken
{result.get('output', {}).get('actions_taken', 'See task logs for details.')}

---
*This issue was automatically closed by the AI system after successful task completion.*"""
                        
                            # Add comment to the issue
                            local_simulator.add_issue_comment(
                                work_item.repository,
                                work_item.metadata['issue_number'],
                                comment_text
                            )
                            self.logger.debug(f"[ISSUE_UPDATE] Added completion comment to issue")
                            
                        except Exception as issue_error:
                            self.logger.error(f"[ISSUE_UPDATE] Failed to close issue: {issue_error}")
                            self.logger.error(f"[ISSUE_UPDATE] Will continue despite issue update failure")
                    
                except Exception as e:
                    self.logger.error(f"[STATE_SAVE] Failed to save task completion state: {e}")
                    self.logger.error(f"[STATE_SAVE] Traceback: {traceback.format_exc()}")
            else:
                # Task is successful but not completed (e.g., queued for Claude)
                self.logger.info(f"📋 Task successful but not completed (queued for execution): {work_item.title}")
                # Could track in-progress tasks here if needed
        
        return result
    
    async def _save_orchestrator_state(self):
        """Save orchestrator state to file."""
        try:
            orch_state = {
                'metrics': self.metrics,
                'enhanced_metrics': self.enhanced_metrics,
                'work_queue': [],  # Would need to extract from queue
                'completed_work': list(self.processed_issues),  # Track processed issues
                'workers': {
                    worker_id: {
                        'status': worker.status.value,
                        'total_completed': worker.total_completed,
                        'total_errors': worker.total_errors,
                        'current_work': worker.current_work.id if worker.current_work else None
                    } for worker_id, worker in self.workers.items()
                },
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Save to local_repos/state/continuous_orchestrator_state.json
            self.state_manager.save_orchestrator_state(orch_state)
                
            self.logger.debug("[STATE_SAVE] Orchestrator state saved to local_repos/state/continuous_orchestrator_state.json")
            
        except Exception as e:
            self.logger.error(f"[STATE_SAVE] Failed to save orchestrator state: {e}")
    
    def _extract_priority_from_labels(self, labels: List[str]) -> TaskPriority:
        """Extract priority from issue labels."""
        for label in labels:
            if "priority:critical" in label.lower():
                return TaskPriority.CRITICAL
            elif "priority:high" in label.lower():
                return TaskPriority.HIGH
            elif "priority:medium" in label.lower():
                return TaskPriority.MEDIUM
            elif "priority:low" in label.lower():
                return TaskPriority.LOW
        return TaskPriority.MEDIUM  # Default priority

    async def run(self):
        """Alias for start() to support both naming conventions."""
        # Start Claude executor pool
        self.logger.info("🚀 Starting Claude executor pool...")
        await self.claude_executor_pool.start()
        self.logger.info("✅ Claude executor pool started successfully")
        
        # Start the orchestrator
        await self.start()
    
    async def shutdown(self):
        """Graceful shutdown with outcome recording."""
        self.logger.info("Shutting down Enhanced Hybrid Orchestrator...")
        
        # Stop Claude executor pool
        try:
            self.logger.info("🛑 Stopping Claude executor pool...")
            await self.claude_executor_pool.stop()
            self.logger.info("✅ Claude executor pool stopped")
        except Exception as e:
            self.logger.error(f"Error stopping Claude executor pool: {e}")
        
        # Save final metrics and state
        try:
            # First save orchestrator state
            await self._save_orchestrator_state()
            
            # Then save enhanced metrics
            final_state = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'total_work_completed': self.metrics.get('total_work_completed', 0),
                'enhanced_metrics': self.enhanced_metrics,
                'dynamic_work_generator': True,
                'processed_issues': list(self.processed_issues),
                'completed_tasks_count': len(self.state_manager.load_task_state().get('completed_tasks', []))
            }
            
            # Save to state
            self.state_manager.update_state({'enhanced_final_state': final_state})
            
            # Force save of state manager
            self.state_manager.save_state()
            
            self.logger.info(f"✅ [SHUTDOWN] Final state saved successfully")
            self.logger.info(f"  - Total completed: {self.metrics.get('total_work_completed', 0)}")
            self.logger.info(f"  - Processed issues: {len(self.processed_issues)}")
            
        except Exception as e:
            self.logger.error(f"Error saving final state: {e}")
        
        # Call parent shutdown
        self.shutdown_requested = True
        if hasattr(self, 'workers'):
            for worker in self.workers.values():
                worker.status = WorkerStatus.SHUTTING_DOWN


# Factory function for creating enhanced orchestrator
def create_enhanced_hybrid_orchestrator(**kwargs) -> EnhancedHybridOrchestrator:
    """Create and initialize an enhanced hybrid orchestrator."""
    return EnhancedHybridOrchestrator(**kwargs)