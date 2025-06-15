"""
Background Task Preprocessor - Proactive Task Analysis

This module provides background processing capabilities to pre-analyze repositories
and prepare task suggestions during idle time, reducing latency when tasks are needed.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
from pathlib import Path

from repository_analyzer import RepositoryAnalyzer
from smart_context_aggregator import SmartContextAggregator
from predictive_task_engine import PredictiveTaskEngine


@dataclass
class PreprocessedTask:
    """Pre-generated task ready for quick retrieval."""
    task_id: str
    task_data: Dict[str, Any]
    repository: str
    priority_score: float
    generated_at: datetime
    expires_at: datetime
    context_hash: str
    used: bool = False


@dataclass
class RepositoryQueue:
    """Priority queue for repository processing."""
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0
    
    def add_repository(self, repo: str, priority: float) -> None:
        """Add repository with priority."""
        if repo in self.entry_finder:
            self.remove_repository(repo)
        entry = [priority, self.counter, repo]
        self.counter += 1
        self.entry_finder[repo] = entry
        heapq.heappush(self.heap, entry)
    
    def remove_repository(self, repo: str) -> None:
        """Remove repository from queue."""
        entry = self.entry_finder.pop(repo)
        entry[-1] = None  # Mark as removed
    
    def pop_repository(self) -> Optional[str]:
        """Get highest priority repository."""
        while self.heap:
            priority, count, repo = heapq.heappop(self.heap)
            if repo is not None:
                del self.entry_finder[repo]
                return repo
        return None
    
    def __len__(self) -> int:
        return len(self.entry_finder)


class BackgroundTaskPreprocessor:
    """Preprocesses tasks in background to reduce generation latency."""
    
    def __init__(self, repository_analyzer: RepositoryAnalyzer,
                 context_aggregator: SmartContextAggregator,
                 predictive_engine: Optional[PredictiveTaskEngine] = None,
                 cache_dir: str = "task_cache"):
        """Initialize background preprocessor.
        
        Args:
            repository_analyzer: Repository analyzer
            context_aggregator: Context aggregator
            predictive_engine: Predictive engine for smart preprocessing
            cache_dir: Directory for task cache
        """
        self.repository_analyzer = repository_analyzer
        self.context_aggregator = context_aggregator
        self.predictive_engine = predictive_engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Processing configuration
        self.max_preprocessed_tasks = 100
        self.task_ttl = timedelta(hours=6)
        self.idle_threshold = 0.3  # CPU usage threshold for idle
        self.batch_size = 5
        
        # Task cache
        self.preprocessed_tasks: Dict[str, List[PreprocessedTask]] = defaultdict(list)
        self.task_index: Dict[str, PreprocessedTask] = {}
        
        # Repository processing queue
        self.repo_queue = RepositoryQueue()
        self.processing_history: Dict[str, datetime] = {}
        self.processing_lock = asyncio.Lock()
        
        # Background processing state
        self.is_running = False
        self.processing_task = None
        
        # Metrics
        self.metrics = {
            'tasks_preprocessed': 0,
            'tasks_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'repos_processed': 0,
            'processing_time': 0.0
        }
    
    async def start_background_processing(self) -> None:
        """Start background preprocessing."""
        if self.is_running:
            self.logger.warning("Background processing already running")
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._background_processing_loop())
        self.logger.info("Started background task preprocessing")
    
    async def stop_background_processing(self) -> None:
        """Stop background preprocessing."""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped background task preprocessing")
    
    async def _background_processing_loop(self) -> None:
        """Main background processing loop."""
        while self.is_running:
            try:
                # Check if system is idle
                if await self._is_system_idle():
                    # Process repositories
                    await self._process_repositories_batch()
                    
                    # Clean expired tasks
                    self._clean_expired_tasks()
                    
                    # Save cache periodically
                    if self.metrics['tasks_preprocessed'] % 20 == 0:
                        await self._save_cache()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background processing error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _is_system_idle(self) -> bool:
        """Check if system is idle enough for background processing.
        
        Returns:
            True if system is idle
        """
        # Simplified check - in production would check actual CPU/memory
        # For now, always return True for demo
        return True
    
    async def _process_repositories_batch(self) -> None:
        """Process a batch of repositories."""
        async with self.processing_lock:
            repos_to_process = []
            
            # Get repositories from queue
            for _ in range(self.batch_size):
                repo = self.repo_queue.pop_repository()
                if repo:
                    repos_to_process.append(repo)
                else:
                    # Queue empty, refill it
                    await self._refill_repository_queue()
                    repo = self.repo_queue.pop_repository()
                    if repo:
                        repos_to_process.append(repo)
            
            if not repos_to_process:
                return
            
            self.logger.info(f"Processing {len(repos_to_process)} repositories in background")
            
            # Process each repository
            for repo in repos_to_process:
                try:
                    await self._preprocess_repository_tasks(repo)
                    self.processing_history[repo] = datetime.now(timezone.utc)
                    self.metrics['repos_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to preprocess {repo}: {e}")
    
    async def _preprocess_repository_tasks(self, repository: str) -> None:
        """Preprocess tasks for a repository.
        
        Args:
            repository: Repository to preprocess
        """
        start_time = datetime.now(timezone.utc)
        
        # Analyze repository
        analysis = await self.repository_analyzer.analyze_repository(repository)
        
        # Get current context
        context = await self.context_aggregator.gather_comprehensive_context()
        context_hash = self._hash_context(context)
        
        # Generate predicted tasks
        predicted_needs = []
        
        if self.predictive_engine:
            # Use ML predictions
            predictions = await self.predictive_engine.predict_next_tasks(
                {'repository_health': {repository: analysis.get('health_score', 0.5)}}
            )
            for pred in predictions[:3]:  # Top 3 predictions
                predicted_needs.append({
                    'type': pred.task_type,
                    'urgency': pred.urgency,
                    'trigger': pred.trigger_factors[0] if pred.trigger_factors else 'Predicted need'
                })
        else:
            # Use heuristic predictions
            predicted_needs = self._predict_repository_needs(analysis)
        
        # Generate tasks for predicted needs
        tasks_generated = 0
        for need in predicted_needs:
            task = await self._generate_task_for_need(repository, analysis, need, context)
            if task:
                # Create preprocessed task
                preprocessed = PreprocessedTask(
                    task_id=f"pre_{repository}_{datetime.now(timezone.utc).timestamp()}",
                    task_data=task,
                    repository=repository,
                    priority_score=self._calculate_priority_score(task, need),
                    generated_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + self.task_ttl,
                    context_hash=context_hash
                )
                
                # Store in cache
                self.preprocessed_tasks[repository].append(preprocessed)
                self.task_index[preprocessed.task_id] = preprocessed
                tasks_generated += 1
                self.metrics['tasks_preprocessed'] += 1
        
        # Clean old tasks for this repository
        self._clean_repository_cache(repository)
        
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.metrics['processing_time'] += elapsed
        
        self.logger.info(f"Preprocessed {tasks_generated} tasks for {repository} in {elapsed:.2f}s")
    
    def _predict_repository_needs(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict repository needs using heuristics.
        
        Args:
            analysis: Repository analysis
            
        Returns:
            Predicted needs
        """
        needs = []
        
        health_score = analysis.get('health_metrics', {}).get('health_score', 50)
        
        # Health-based predictions
        if health_score < 30:
            needs.append({
                'type': 'BUG_FIX',
                'urgency': 'high',
                'trigger': 'Critical health score'
            })
        elif health_score < 60:
            needs.append({
                'type': 'MAINTENANCE',
                'urgency': 'medium',
                'trigger': 'Health improvement needed'
            })
        
        # Issue-based predictions
        open_issues = analysis.get('basic_info', {}).get('open_issues_count', 0)
        if open_issues > 10:
            needs.append({
                'type': 'BUG_FIX',
                'urgency': 'medium',
                'trigger': f'{open_issues} open issues'
            })
        
        # Documentation predictions
        has_readme = analysis.get('documentation', {}).get('has_readme', False)
        if not has_readme:
            needs.append({
                'type': 'DOCUMENTATION',
                'urgency': 'low',
                'trigger': 'Missing documentation'
            })
        
        # Testing predictions
        test_coverage = analysis.get('code_quality', {}).get('test_coverage', 0)
        if test_coverage < 50:
            needs.append({
                'type': 'TESTING',
                'urgency': 'medium',
                'trigger': f'Low test coverage ({test_coverage}%)'
            })
        
        return needs[:3]  # Top 3 needs
    
    async def _generate_task_for_need(self, repository: str, 
                                    analysis: Dict[str, Any],
                                    need: Dict[str, Any],
                                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate task for a predicted need.
        
        Args:
            repository: Repository name
            analysis: Repository analysis
            need: Predicted need
            context: Current context
            
        Returns:
            Generated task or None
        """
        # This would use the actual task generator
        # For now, create a mock task
        task = {
            'type': need['type'],
            'title': f"{need['type']} task for {repository}",
            'description': f"Preprocessed task to address: {need['trigger']}",
            'repository': repository,
            'priority': need['urgency'],
            'estimated_hours': 4.0,
            'preprocessed': True,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        return task
    
    def _calculate_priority_score(self, task: Dict[str, Any], 
                                need: Dict[str, Any]) -> float:
        """Calculate priority score for preprocessed task.
        
        Args:
            task: Generated task
            need: Associated need
            
        Returns:
            Priority score (0-1)
        """
        urgency_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }
        
        base_score = urgency_scores.get(need.get('urgency', 'medium'), 0.5)
        
        # Adjust based on task type
        type_modifiers = {
            'BUG_FIX': 0.1,
            'SECURITY': 0.2,
            'PERFORMANCE': 0.05
        }
        
        modifier = type_modifiers.get(task.get('type', ''), 0)
        
        return min(1.0, base_score + modifier)
    
    async def _refill_repository_queue(self) -> None:
        """Refill repository processing queue."""
        # Get all repositories that need processing
        # In production, this would query the actual repository list
        
        # For now, simulate with a list
        all_repos = [
            'api-service',
            'frontend-app', 
            'auth-service',
            'data-processor',
            'admin-dashboard'
        ]
        
        for repo in all_repos:
            # Calculate priority based on last processing time
            last_processed = self.processing_history.get(repo)
            if last_processed:
                hours_since = (datetime.now(timezone.utc) - last_processed).total_seconds() / 3600
                priority = min(1.0, hours_since / 24)  # Higher priority if not processed recently
            else:
                priority = 1.0  # Highest priority if never processed
            
            self.repo_queue.add_repository(repo, -priority)  # Negative for min heap
    
    def _clean_expired_tasks(self) -> None:
        """Remove expired preprocessed tasks."""
        now = datetime.now(timezone.utc)
        expired_count = 0
        
        for repo, tasks in list(self.preprocessed_tasks.items()):
            valid_tasks = []
            for task in tasks:
                if task.expires_at > now and not task.used:
                    valid_tasks.append(task)
                else:
                    del self.task_index[task.task_id]
                    expired_count += 1
            
            if valid_tasks:
                self.preprocessed_tasks[repo] = valid_tasks
            else:
                del self.preprocessed_tasks[repo]
        
        if expired_count > 0:
            self.logger.info(f"Cleaned {expired_count} expired preprocessed tasks")
    
    def _clean_repository_cache(self, repository: str) -> None:
        """Clean old tasks for a repository, keeping only recent ones.
        
        Args:
            repository: Repository to clean
        """
        if repository not in self.preprocessed_tasks:
            return
        
        tasks = self.preprocessed_tasks[repository]
        if len(tasks) <= 10:  # Keep up to 10 tasks per repo
            return
        
        # Sort by priority and keep top 10
        tasks.sort(key=lambda t: t.priority_score, reverse=True)
        removed_tasks = tasks[10:]
        self.preprocessed_tasks[repository] = tasks[:10]
        
        # Remove from index
        for task in removed_tasks:
            del self.task_index[task.task_id]
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create hash of context for cache validation.
        
        Args:
            context: Context to hash
            
        Returns:
            Context hash
        """
        # Simple hash of key metrics
        key_values = [
            str(context.get('repository_health', {})),
            str(context.get('strategic_priorities', [])),
            str(len(context.get('recent_activities', [])))
        ]
        
        import hashlib
        return hashlib.md5('|'.join(key_values).encode()).hexdigest()
    
    async def get_preprocessed_task(self, repository: str, 
                                  task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a preprocessed task if available.
        
        Args:
            repository: Repository name
            task_type: Optional specific task type
            
        Returns:
            Preprocessed task or None
        """
        if repository not in self.preprocessed_tasks:
            self.metrics['cache_misses'] += 1
            return None
        
        # Find matching task
        for task in self.preprocessed_tasks[repository]:
            if task.used or task.expires_at < datetime.now(timezone.utc):
                continue
            
            if task_type and task.task_data.get('type') != task_type:
                continue
            
            # Mark as used
            task.used = True
            self.metrics['cache_hits'] += 1
            self.metrics['tasks_served'] += 1
            
            self.logger.info(f"Serving preprocessed task for {repository}: {task.task_id}")
            return task.task_data
        
        self.metrics['cache_misses'] += 1
        return None
    
    async def _save_cache(self) -> None:
        """Save preprocessed tasks to disk."""
        cache_data = {
            'tasks': {},
            'metrics': self.metrics,
            'processing_history': {
                repo: dt.isoformat() for repo, dt in self.processing_history.items()
            }
        }
        
        # Convert tasks to serializable format
        for repo, tasks in self.preprocessed_tasks.items():
            cache_data['tasks'][repo] = [
                {
                    'task_id': t.task_id,
                    'task_data': t.task_data,
                    'priority_score': t.priority_score,
                    'generated_at': t.generated_at.isoformat(),
                    'expires_at': t.expires_at.isoformat(),
                    'context_hash': t.context_hash,
                    'used': t.used
                }
                for t in tasks
            ]
        
        cache_file = self.cache_dir / 'preprocessed_tasks.json'
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        self.logger.debug(f"Saved {len(self.task_index)} preprocessed tasks to cache")
    
    async def load_cache(self) -> None:
        """Load preprocessed tasks from disk."""
        cache_file = self.cache_dir / 'preprocessed_tasks.json'
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Restore metrics
            self.metrics.update(cache_data.get('metrics', {}))
            
            # Restore processing history
            for repo, dt_str in cache_data.get('processing_history', {}).items():
                self.processing_history[repo] = datetime.fromisoformat(dt_str)
            
            # Restore tasks
            now = datetime.now(timezone.utc)
            loaded_count = 0
            
            for repo, tasks_data in cache_data.get('tasks', {}).items():
                for task_data in tasks_data:
                    expires_at = datetime.fromisoformat(task_data['expires_at'])
                    
                    # Skip expired tasks
                    if expires_at < now:
                        continue
                    
                    task = PreprocessedTask(
                        task_id=task_data['task_id'],
                        task_data=task_data['task_data'],
                        repository=repo,
                        priority_score=task_data['priority_score'],
                        generated_at=datetime.fromisoformat(task_data['generated_at']),
                        expires_at=expires_at,
                        context_hash=task_data['context_hash'],
                        used=task_data['used']
                    )
                    
                    self.preprocessed_tasks[repo].append(task)
                    self.task_index[task.task_id] = task
                    loaded_count += 1
            
            self.logger.info(f"Loaded {loaded_count} preprocessed tasks from cache")
            
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics.
        
        Returns:
            Preprocessing stats
        """
        total_cached = len(self.task_index)
        cache_hit_rate = (self.metrics['cache_hits'] / 
                         (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                         if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0 else 0)
        
        stats = {
            'metrics': self.metrics,
            'cache_status': {
                'total_tasks': total_cached,
                'tasks_by_repo': {repo: len(tasks) 
                                 for repo, tasks in self.preprocessed_tasks.items()},
                'cache_hit_rate': f"{cache_hit_rate:.1%}",
                'average_preprocessing_time': (
                    self.metrics['processing_time'] / self.metrics['repos_processed']
                    if self.metrics['repos_processed'] > 0 else 0
                )
            },
            'queue_status': {
                'repositories_queued': len(self.repo_queue),
                'is_processing': self.is_running
            }
        }
        
        return stats