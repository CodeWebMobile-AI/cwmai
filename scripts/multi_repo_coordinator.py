"""
Multi-Repository Coordinator

Manages multiple GitHub repositories simultaneously with rate limiting,
cross-repository learning, and intelligent task distribution.
"""

import json
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import queue
from github import Github, RateLimitExceededException
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from scripts.task_manager import TaskManager
from scripts.repository_exclusion import RepositoryExclusion, should_process_repo


@dataclass
class RateLimitManager:
    """Manages GitHub API rate limits."""
    requests_per_hour: int = 5000
    requests_made: deque = field(default_factory=lambda: deque())
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding rate limit."""
        with self.lock:
            # Remove requests older than 1 hour
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            while self.requests_made and self.requests_made[0] < cutoff_time:
                self.requests_made.popleft()
            
            return len(self.requests_made) < self.requests_per_hour
    
    def record_request(self) -> None:
        """Record that a request was made."""
        with self.lock:
            self.requests_made.append(datetime.now(timezone.utc))
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        while not self.can_make_request():
            time.sleep(1)  # Wait 1 second and check again
        self.record_request()
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            while self.requests_made and self.requests_made[0] < cutoff_time:
                self.requests_made.popleft()
            return self.requests_per_hour - len(self.requests_made)


@dataclass
class RepositoryState:
    """State information for a single repository."""
    name: str
    url: str
    health_score: float = 0.0
    open_issues: int = 0
    open_prs: int = 0
    last_commit: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    task_queue: List[Dict[str, Any]] = field(default_factory=list)
    patterns_learned: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class CrossRepositoryLearning:
    """Learns patterns across multiple repositories."""
    
    def __init__(self):
        """Initialize cross-repository learning system."""
        self.pattern_database = defaultdict(list)
        self.success_patterns = defaultdict(float)
        self.failure_patterns = defaultdict(float)
        self.repository_similarities = {}
        
    def learn_from_repository(self, repo_state: RepositoryState, 
                            success_rate: float) -> None:
        """Learn patterns from a repository's history."""
        # Extract patterns from repository
        patterns = self._extract_patterns(repo_state)
        
        # Update pattern database
        for pattern in patterns:
            pattern_key = self._pattern_to_key(pattern)
            self.pattern_database[pattern_key].append({
                'repository': repo_state.name,
                'success_rate': success_rate,
                'timestamp': datetime.now(timezone.utc)
            })
            
            # Update success/failure rates
            if success_rate > 0.7:
                self.success_patterns[pattern_key] += 1
            elif success_rate < 0.3:
                self.failure_patterns[pattern_key] += 1
    
    def get_recommendations(self, repo_state: RepositoryState) -> List[Dict[str, Any]]:
        """Get recommendations based on cross-repository patterns."""
        recommendations = []
        
        # Find similar repositories
        similar_repos = self._find_similar_repositories(repo_state)
        
        # Get successful patterns from similar repos
        for similar_repo, similarity_score in similar_repos[:3]:
            patterns = self._get_successful_patterns(similar_repo)
            
            for pattern in patterns:
                recommendations.append({
                    'pattern': pattern,
                    'source_repository': similar_repo,
                    'similarity_score': similarity_score,
                    'success_probability': self._calculate_pattern_success(pattern),
                    'recommendation': self._pattern_to_recommendation(pattern)
                })
        
        # Sort by success probability
        recommendations.sort(key=lambda x: x['success_probability'], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def identify_global_trends(self) -> Dict[str, Any]:
        """Identify trends across all repositories."""
        trends = {
            'most_successful_patterns': [],
            'most_failed_patterns': [],
            'emerging_patterns': [],
            'technology_trends': defaultdict(int)
        }
        
        # Most successful patterns
        sorted_success = sorted(self.success_patterns.items(), 
                              key=lambda x: x[1], reverse=True)
        trends['most_successful_patterns'] = [
            {'pattern': k, 'count': v} for k, v in sorted_success[:5]
        ]
        
        # Most failed patterns
        sorted_failures = sorted(self.failure_patterns.items(), 
                               key=lambda x: x[1], reverse=True)
        trends['most_failed_patterns'] = [
            {'pattern': k, 'count': v} for k, v in sorted_failures[:5]
        ]
        
        # Emerging patterns (recent growth)
        recent_patterns = self._find_emerging_patterns()
        trends['emerging_patterns'] = recent_patterns
        
        # Technology trends
        for pattern_key, occurrences in self.pattern_database.items():
            if 'tech:' in pattern_key:
                tech = pattern_key.split('tech:')[1].split('_')[0]
                trends['technology_trends'][tech] += len(occurrences)
        
        return trends
    
    def _extract_patterns(self, repo_state: RepositoryState) -> List[Dict[str, Any]]:
        """Extract patterns from repository state."""
        patterns = []
        
        # Task type patterns
        task_types = defaultdict(int)
        for task in repo_state.task_queue:
            task_types[task.get('type', 'unknown')] += 1
        
        for task_type, count in task_types.items():
            patterns.append({
                'type': 'task_distribution',
                'task_type': task_type,
                'frequency': count / len(repo_state.task_queue) if repo_state.task_queue else 0
            })
        
        # Performance patterns
        if repo_state.performance_metrics:
            patterns.append({
                'type': 'performance',
                'metrics': repo_state.performance_metrics
            })
        
        # Health patterns
        patterns.append({
            'type': 'health',
            'score': repo_state.health_score,
            'trajectory': 'improving' if repo_state.health_score > 70 else 'declining'
        })
        
        return patterns
    
    def _pattern_to_key(self, pattern: Dict[str, Any]) -> str:
        """Convert pattern to hashable key."""
        if pattern['type'] == 'task_distribution':
            return f"task_{pattern['task_type']}_{int(pattern['frequency']*10)}"
        elif pattern['type'] == 'performance':
            avg_metric = np.mean(list(pattern['metrics'].values()))
            return f"perf_{int(avg_metric*100)}"
        elif pattern['type'] == 'health':
            return f"health_{pattern['trajectory']}_{int(pattern['score'])}"
        else:
            return f"unknown_{hash(str(pattern))}"
    
    def _find_similar_repositories(self, repo_state: RepositoryState) -> List[Tuple[str, float]]:
        """Find repositories similar to the given one."""
        similarities = []
        
        # Calculate similarity with each repository in pattern database
        repo_patterns = set()
        for pattern in self._extract_patterns(repo_state):
            repo_patterns.add(self._pattern_to_key(pattern))
        
        repo_occurrences = defaultdict(int)
        for pattern_key in repo_patterns:
            for occurrence in self.pattern_database.get(pattern_key, []):
                repo_occurrences[occurrence['repository']] += 1
        
        # Calculate similarity scores
        for repo_name, shared_patterns in repo_occurrences.items():
            if repo_name != repo_state.name:
                similarity = shared_patterns / max(len(repo_patterns), 1)
                similarities.append((repo_name, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _get_successful_patterns(self, repository: str) -> List[Dict[str, Any]]:
        """Get successful patterns from a repository."""
        patterns = []
        
        for pattern_key, occurrences in self.pattern_database.items():
            repo_occurrences = [o for o in occurrences if o['repository'] == repository]
            
            if repo_occurrences:
                avg_success = np.mean([o['success_rate'] for o in repo_occurrences])
                
                if avg_success > 0.7:
                    patterns.append({
                        'key': pattern_key,
                        'success_rate': avg_success,
                        'occurrences': len(repo_occurrences)
                    })
        
        return patterns
    
    def _calculate_pattern_success(self, pattern: Dict[str, Any]) -> float:
        """Calculate success probability of a pattern."""
        pattern_key = pattern.get('key', '')
        
        success_count = self.success_patterns.get(pattern_key, 0)
        failure_count = self.failure_patterns.get(pattern_key, 0)
        
        total = success_count + failure_count
        if total == 0:
            return 0.5  # No data, assume 50%
        
        return success_count / total
    
    def _pattern_to_recommendation(self, pattern: Dict[str, Any]) -> str:
        """Convert pattern to actionable recommendation."""
        pattern_key = pattern.get('key', '')
        
        if pattern_key.startswith('task_'):
            task_type = pattern_key.split('_')[1]
            return f"Consider adding more {task_type} tasks - high success rate in similar repos"
        elif pattern_key.startswith('perf_'):
            score = int(pattern_key.split('_')[1])
            if score > 80:
                return "Apply performance optimization strategies from high-performing repos"
            else:
                return "Review and improve performance metrics"
        elif pattern_key.startswith('health_'):
            trajectory = pattern_key.split('_')[1]
            return f"Repository health is {trajectory} - apply successful intervention patterns"
        else:
            return "Apply learned patterns from similar repositories"
    
    def _find_emerging_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns that are growing in frequency."""
        emerging = []
        
        # Look at patterns from last 30 days vs previous 30 days
        cutoff_recent = datetime.now(timezone.utc) - timedelta(days=30)
        cutoff_old = datetime.now(timezone.utc) - timedelta(days=60)
        
        for pattern_key, occurrences in self.pattern_database.items():
            recent = sum(1 for o in occurrences if o['timestamp'] > cutoff_recent)
            old = sum(1 for o in occurrences if cutoff_old < o['timestamp'] <= cutoff_recent)
            
            if old > 0 and recent > old * 1.5:  # 50% growth
                emerging.append({
                    'pattern': pattern_key,
                    'growth_rate': (recent - old) / old,
                    'recent_count': recent
                })
        
        emerging.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return emerging[:5]


class MultiRepoCoordinator:
    """Coordinates work across multiple repositories."""
    
    def __init__(self, github_token: str = None, max_concurrent: int = 10):
        """Initialize multi-repository coordinator.
        
        Args:
            github_token: GitHub personal access token
            max_concurrent: Maximum concurrent repositories to manage
        """
        self.github_token = github_token or os.getenv('CLAUDE_PAT')
        self.max_concurrent = max_concurrent
        
        # Initialize components
        self.github = Github(self.github_token) if self.github_token else None
        self.rate_limiter = RateLimitManager()
        self.learning_system = CrossRepositoryLearning()
        self.task_manager = TaskManager(self.github_token)
        
        # Repository management
        self.repositories: Dict[str, RepositoryState] = {}
        self.active_repos: Set[str] = set()
        self.task_queue = queue.PriorityQueue()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.shutdown = False
        
    def add_repository(self, repo_url: str, create_if_not_exists: bool = False) -> Optional[str]:
        """Add a repository to manage.
        
        Args:
            repo_url: Repository URL or full name (owner/repo)
            create_if_not_exists: Create repository if it doesn't exist
            
        Returns:
            Repository URL if successful, None otherwise
        """
        # Extract repo name from URL if needed
        if repo_url.startswith('https://github.com/'):
            repo_name = repo_url.replace('https://github.com/', '').replace('.git', '')
        else:
            repo_name = repo_url
            repo_url = f"https://github.com/{repo_name}"
        
        # Check if repository should be excluded
        if not should_process_repo(repo_name):
            print(f"Repository {repo_name} is excluded from coordination flows")
            print(f"Reason: {RepositoryExclusion.get_exclusion_reason(repo_name)}")
            return None
        
        if len(self.active_repos) >= self.max_concurrent:
            print(f"Cannot add {repo_name}: Maximum concurrent repos reached")
            return None
        
        if repo_name in self.repositories:
            print(f"Repository {repo_name} already added")
            return repo_url
        
        try:
            # Try to get existing repository
            self.rate_limiter.wait_if_needed()
            try:
                repo = self.github.get_repo(repo_name)
            except Exception as e:
                if create_if_not_exists and '404' in str(e):
                    # Create the repository
                    owner, name = repo_name.split('/')
                    if owner == 'CodeWebMobile-AI':
                        # Double-check exclusion before creating
                        if not should_process_repo(repo_name):
                            raise Exception(f"Cannot create excluded repository: {repo_name}")
                        
                        # Create in organization
                        org = self.github.get_organization('CodeWebMobile-AI')
                        repo = org.create_repo(
                            name=name,
                            description=f"AI-generated project for {name}",
                            private=False,
                            auto_init=True
                        )
                        print(f"Created new repository: {repo_name}")
                    else:
                        raise Exception("Can only create repos in CodeWebMobile-AI org")
                else:
                    raise
            
            # Create repository state
            repo_state = RepositoryState(
                name=repo_name,
                url=repo.html_url,
                last_checked=datetime.now(timezone.utc)
            )
            
            # Initial analysis
            self._analyze_repository(repo_state, repo)
            
            # Add to managed repositories
            self.repositories[repo_name] = repo_state
            self.active_repos.add(repo_name)
            
            print(f"Added repository: {repo_name}")
            return repo_url
            
        except Exception as e:
            if '404' in str(e):
                self.logger.debug(f"Repository {repo_name} not found (404), skipping")
            else:
                self.logger.warning(f"Error adding repository {repo_name}: {e}")
            return None
    
    def create_cross_repo_task(self, repo_url: str, task_type: str, 
                              title: str, description: str) -> bool:
        """Create a task/issue in a specific repository.
        
        Args:
            repo_url: Repository URL
            task_type: Type of task
            title: Issue title
            description: Issue description
            
        Returns:
            Success status
        """
        # Extract repo name from URL
        if repo_url.startswith('https://github.com/'):
            repo_name = repo_url.replace('https://github.com/', '').replace('.git', '')
        else:
            repo_name = repo_url
        
        try:
            # Get repository
            self.rate_limiter.wait_if_needed()
            repo = self.github.get_repo(repo_name)
            
            # Create issue using centralized method to ensure @claude mention
            labels = [task_type, 'ai-generated', 'ai-managed']
            if 'feature' in task_type.lower():
                labels.append('enhancement')
            
            # Use task manager's centralized method with target project
            formatted_description = description + "\n\n---\n*This issue was created by the AI coordination system.*"
            
            self.rate_limiter.wait_if_needed()
            issue_number = self.task_manager.create_ai_task_issue(
                title=title,
                description=formatted_description,
                labels=labels,
                priority="medium",
                task_type=task_type,
                repository=repo_name  # Pass the target repository
            )
            
            if not issue_number:
                print(f"Failed to create issue in {repo_name}")
                return False
            
            issue = repo.get_issue(issue_number)
            
            print(f"Created issue #{issue.number} in {repo_name}: {title}")
            return True
            
        except Exception as e:
            print(f"Error creating issue in {repo_name}: {e}")
            return False
    
    def coordinate_cycle(self) -> Dict[str, Any]:
        """Run one coordination cycle across all repositories."""
        cycle_start = time.time()
        results = {
            'repositories_processed': 0,
            'tasks_created': 0,
            'tasks_distributed': 0,
            'patterns_learned': 0,
            'errors': [],
            'recommendations': {}
        }
        
        # Process each active repository (excluding any that shouldn't be processed)
        futures = []
        for repo_name in list(self.active_repos):
            if should_process_repo(repo_name):
                future = self.executor.submit(self._process_repository, repo_name)
                futures.append((repo_name, future))
            else:
                print(f"Skipping excluded repository in coordination cycle: {repo_name}")
        
        # Collect results
        for repo_name, future in futures:
            try:
                repo_results = future.result(timeout=60)
                results['repositories_processed'] += 1
                results['tasks_created'] += repo_results.get('tasks_created', 0)
                results['patterns_learned'] += repo_results.get('patterns_learned', 0)
                
            except Exception as e:
                results['errors'].append({
                    'repository': repo_name,
                    'error': str(e)
                })
        
        # Cross-repository learning
        self._perform_cross_learning()
        
        # Generate global insights
        results['global_trends'] = self.learning_system.identify_global_trends()
        
        # Distribute tasks intelligently
        distributed = self._distribute_tasks()
        results['tasks_distributed'] = distributed
        
        # Generate recommendations for each repo
        for repo_name, repo_state in self.repositories.items():
            recommendations = self.learning_system.get_recommendations(repo_state)
            results['recommendations'][repo_name] = recommendations
        
        results['cycle_duration'] = time.time() - cycle_start
        
        return results
    
    def _process_repository(self, repo_name: str) -> Dict[str, Any]:
        """Process a single repository."""
        results = {
            'tasks_created': 0,
            'patterns_learned': 0,
            'health_score': 0
        }
        
        try:
            repo_state = self.repositories[repo_name]
            
            # Get repository object
            self.rate_limiter.wait_if_needed()
            repo = self.github.get_repo(repo_name)
            
            # Update repository state
            self._analyze_repository(repo_state, repo)
            results['health_score'] = repo_state.health_score
            
            # Generate tasks based on analysis
            new_tasks = self._generate_repository_tasks(repo_state)
            results['tasks_created'] = len(new_tasks)
            
            # Add tasks to queue
            for task in new_tasks:
                priority = self._calculate_task_priority(task, repo_state)
                self.task_queue.put((priority, task))
            
            # Learn from repository
            success_rate = self._calculate_repository_success(repo_state)
            self.learning_system.learn_from_repository(repo_state, success_rate)
            results['patterns_learned'] = 1
            
        except RateLimitExceededException:
            print(f"Rate limit exceeded for {repo_name}")
            time.sleep(60)  # Wait a minute
        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_repository(self, repo_state: RepositoryState, repo) -> None:
        """Analyze repository and update state."""
        # Get basic metrics
        self.rate_limiter.wait_if_needed()
        repo_state.open_issues = repo.open_issues_count
        
        # Get pull requests
        self.rate_limiter.wait_if_needed()
        open_prs = list(repo.get_pulls(state='open'))
        repo_state.open_prs = len(open_prs)
        
        # Get last commit
        try:
            self.rate_limiter.wait_if_needed()
            commits = list(repo.get_commits()[:1])
            if commits:
                repo_state.last_commit = commits[0].commit.author.date.replace(tzinfo=timezone.utc)
        except:
            pass
        
        # Calculate health score
        repo_state.health_score = self._calculate_health_score(repo_state)
        repo_state.last_checked = datetime.now(timezone.utc)
    
    def _calculate_health_score(self, repo_state: RepositoryState) -> float:
        """Calculate repository health score."""
        score = 100.0
        
        # Penalize for too many open issues
        if repo_state.open_issues > 50:
            score -= 20
        elif repo_state.open_issues > 20:
            score -= 10
        
        # Penalize for too many open PRs
        if repo_state.open_prs > 10:
            score -= 15
        elif repo_state.open_prs > 5:
            score -= 5
        
        # Penalize for stale repository
        if repo_state.last_commit:
            days_since_commit = (datetime.now(timezone.utc) - repo_state.last_commit).days
            if days_since_commit > 30:
                score -= 20
            elif days_since_commit > 14:
                score -= 10
        
        return max(0, min(100, score))
    
    def _generate_repository_tasks(self, repo_state: RepositoryState) -> List[Dict[str, Any]]:
        """Generate tasks for a repository based on its state."""
        tasks = []
        
        # Generate tasks based on health score
        if repo_state.health_score < 50:
            # Critical health - urgent tasks
            tasks.append({
                'repository': repo_state.name,
                'type': 'health_critical',
                'title': f"Urgent: Improve repository health (score: {repo_state.health_score:.0f})",
                'description': "Repository health is critical. Review open issues and PRs.",
                'priority': 'critical'
            })
        
        # Too many open issues
        if repo_state.open_issues > 20:
            tasks.append({
                'repository': repo_state.name,
                'type': 'issue_management',
                'title': f"Triage {repo_state.open_issues} open issues",
                'description': "Review and categorize open issues, close stale ones",
                'priority': 'high'
            })
        
        # Too many open PRs
        if repo_state.open_prs > 5:
            tasks.append({
                'repository': repo_state.name,
                'type': 'pr_review',
                'title': f"Review {repo_state.open_prs} open pull requests",
                'description': "Review and merge/close open PRs",
                'priority': 'high'
            })
        
        # Stale repository
        if repo_state.last_commit:
            days_since = (datetime.now(timezone.utc) - repo_state.last_commit).days
            if days_since > 14:
                tasks.append({
                    'repository': repo_state.name,
                    'type': 'maintenance',
                    'title': "Update dependencies and maintenance",
                    'description': f"Repository hasn't been updated in {days_since} days",
                    'priority': 'medium'
                })
        
        # Add to repository task queue
        repo_state.task_queue.extend(tasks)
        
        return tasks
    
    def _calculate_task_priority(self, task: Dict[str, Any], 
                               repo_state: RepositoryState) -> float:
        """Calculate priority score for task distribution."""
        base_priority = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }.get(task.get('priority', 'medium'), 0.4)
        
        # Adjust based on repository health
        health_factor = 1.0 - (repo_state.health_score / 100.0)
        
        # Adjust based on task type
        type_factor = {
            'health_critical': 1.5,
            'security': 1.3,
            'pr_review': 1.1,
            'issue_management': 1.0,
            'maintenance': 0.8
        }.get(task.get('type', 'general'), 1.0)
        
        return base_priority * health_factor * type_factor
    
    def _calculate_repository_success(self, repo_state: RepositoryState) -> float:
        """Calculate success rate for a repository."""
        if not repo_state.performance_metrics:
            return 0.5  # Default 50%
        
        # Simple success calculation based on metrics
        metrics = repo_state.performance_metrics
        
        success_factors = []
        
        if 'pr_merge_rate' in metrics:
            success_factors.append(metrics['pr_merge_rate'])
        
        if 'issue_close_rate' in metrics:
            success_factors.append(metrics['issue_close_rate'])
        
        if 'build_success_rate' in metrics:
            success_factors.append(metrics['build_success_rate'])
        
        if success_factors:
            return np.mean(success_factors)
        
        # Fallback to health-based estimate
        return repo_state.health_score / 100.0
    
    def _perform_cross_learning(self) -> None:
        """Perform cross-repository learning."""
        # This is called automatically during learn_from_repository
        # Additional global learning could be added here
        pass
    
    def _distribute_tasks(self) -> int:
        """Distribute tasks intelligently across repositories."""
        distributed = 0
        
        # Get tasks from priority queue
        tasks_to_distribute = []
        while not self.task_queue.empty() and len(tasks_to_distribute) < 50:
            try:
                priority, task = self.task_queue.get_nowait()
                tasks_to_distribute.append((priority, task))
            except queue.Empty:
                break
        
        # Sort by priority (higher first)
        tasks_to_distribute.sort(key=lambda x: x[0], reverse=True)
        
        # Distribute tasks (in practice, would create GitHub issues)
        for priority, task in tasks_to_distribute:
            repo_name = task.get('repository')
            if repo_name in self.repositories:
                # Add to repository's task queue
                self.repositories[repo_name].task_queue.append(task)
                distributed += 1
                
                print(f"Distributed task to {repo_name}: {task['title']}")
        
        return distributed
    
    def get_repository_status(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific repository."""
        if repo_name not in self.repositories:
            return None
        
        repo_state = self.repositories[repo_name]
        
        return {
            'name': repo_state.name,
            'url': repo_state.url,
            'health_score': repo_state.health_score,
            'open_issues': repo_state.open_issues,
            'open_prs': repo_state.open_prs,
            'last_commit': repo_state.last_commit.isoformat() if repo_state.last_commit else None,
            'last_checked': repo_state.last_checked.isoformat() if repo_state.last_checked else None,
            'task_count': len(repo_state.task_queue),
            'learned_patterns': len(repo_state.patterns_learned)
        }
    
    def get_global_insights(self) -> Dict[str, Any]:
        """Get insights across all repositories."""
        insights = {
            'total_repositories': len(self.repositories),
            'active_repositories': len(self.active_repos),
            'average_health': 0,
            'total_open_issues': 0,
            'total_open_prs': 0,
            'healthiest_repo': None,
            'most_active_repo': None,
            'cross_repo_patterns': {}
        }
        
        if self.repositories:
            # Calculate averages
            health_scores = [r.health_score for r in self.repositories.values()]
            insights['average_health'] = np.mean(health_scores)
            
            insights['total_open_issues'] = sum(r.open_issues for r in self.repositories.values())
            insights['total_open_prs'] = sum(r.open_prs for r in self.repositories.values())
            
            # Find healthiest repository
            healthiest = max(self.repositories.values(), key=lambda r: r.health_score)
            insights['healthiest_repo'] = {
                'name': healthiest.name,
                'score': healthiest.health_score
            }
            
            # Find most active (most tasks)
            most_active = max(self.repositories.values(), key=lambda r: len(r.task_queue))
            insights['most_active_repo'] = {
                'name': most_active.name,
                'task_count': len(most_active.task_queue)
            }
        
        # Get cross-repository patterns
        insights['cross_repo_patterns'] = self.learning_system.identify_global_trends()
        
        return insights
    
    def save_state(self, filepath: str) -> None:
        """Save coordinator state to file."""
        state = {
            'repositories': {name: {
                'name': repo.name,
                'url': repo.url,
                'health_score': repo.health_score,
                'open_issues': repo.open_issues,
                'open_prs': repo.open_prs,
                'last_commit': repo.last_commit.isoformat() if repo.last_commit else None,
                'last_checked': repo.last_checked.isoformat() if repo.last_checked else None,
                'task_count': len(repo.task_queue)
            } for name, repo in self.repositories.items()},
            'active_repos': list(self.active_repos),
            'patterns_learned': len(self.learning_system.pattern_database)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved state to {filepath}")
    
    def shutdown(self) -> None:
        """Shutdown the coordinator."""
        self.shutdown = True
        self.executor.shutdown(wait=True)


def demonstrate_multi_repo():
    """Demonstrate multi-repository coordination."""
    print("=== Multi-Repository Coordinator Demo ===\n")
    
    # Create coordinator
    coordinator = MultiRepoCoordinator(max_concurrent=10)
    
    # Add some repositories (these would be real repos in practice)
    demo_repos = [
        "microsoft/vscode",
        "facebook/react", 
        "tensorflow/tensorflow"
    ]
    
    print("Adding repositories...")
    for repo in demo_repos:
        # In practice, would actually add these
        # For demo, we'll simulate
        coordinator.repositories[repo] = RepositoryState(
            name=repo,
            url=f"https://github.com/{repo}",
            health_score=np.random.uniform(60, 90),
            open_issues=np.random.randint(10, 100),
            open_prs=np.random.randint(0, 20),
            last_commit=datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 30)),
            last_checked=datetime.now(timezone.utc)
        )
        coordinator.active_repos.add(repo)
        print(f"  Added: {repo}")
    
    print("\nRunning coordination cycle...")
    results = coordinator.coordinate_cycle()
    
    print(f"\nCycle Results:")
    print(f"  Repositories processed: {results['repositories_processed']}")
    print(f"  Tasks created: {results['tasks_created']}")
    print(f"  Tasks distributed: {results['tasks_distributed']}")
    print(f"  Patterns learned: {results['patterns_learned']}")
    
    print("\nGlobal Insights:")
    insights = coordinator.get_global_insights()
    print(f"  Average health: {insights['average_health']:.1f}")
    print(f"  Total open issues: {insights['total_open_issues']}")
    print(f"  Total open PRs: {insights['total_open_prs']}")
    
    if insights['healthiest_repo']:
        print(f"  Healthiest repo: {insights['healthiest_repo']['name']} "
              f"(score: {insights['healthiest_repo']['score']:.1f})")
    
    # Save state
    coordinator.save_state("multi_repo_state.json")
    
    # Cleanup
    coordinator.shutdown()


if __name__ == "__main__":
    demonstrate_multi_repo()