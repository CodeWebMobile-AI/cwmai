"""
Task Manager Module

Manages the creation, tracking, and orchestration of tasks for @claude.
Handles task state, dependencies, and progress monitoring for 24/7 operation.
"""

import json
import os
import sys
import time
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from github import Github
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state_manager import StateManager
from ai_brain import IntelligentAIBrain
# ContextGatherer functionality migrated to IntelligentAIBrain
from repository_exclusion import should_process_repo, RepositoryExclusion

# Import AI-powered content generator
try:
    from ai_task_content_generator import AITaskContentGenerator
    AI_CONTENT_GENERATOR_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"AI content generator not available: {e}")
    AI_CONTENT_GENERATOR_AVAILABLE = False

# Import new decomposition system components
try:
    from task_decomposition_engine import TaskDecompositionEngine, TaskComplexity
    from hierarchical_task_manager import HierarchicalTaskManager
    from complexity_analyzer import ComplexityAnalyzer
    from progressive_task_generator import ProgressiveTaskGenerator, ProgressionContext
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Decomposition system not available: {e}")
    DECOMPOSITION_AVAILABLE = False


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(Enum):
    """Types of tasks that can be created."""
    NEW_PROJECT = "new_project"
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_REVIEW = "code_review"
    DEPENDENCY_UPDATE = "dependency_update"


class TaskManager:
    """Manages task lifecycle and orchestration."""
    
    def __init__(self, github_token: str = None, repository: str = None):
        """Initialize the task manager.
        
        Args:
            github_token: GitHub personal access token
            repository: Target repository for tasks (should never be cwmai)
        """
        self.github_token = github_token or os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
        
        # Set up logger
        import logging
        self.logger = logging.getLogger(__name__)
        
        # IMPORTANT: Never default to cwmai repository
        from scripts.repository_exclusion import is_excluded_repo
        
        # If repository is provided, ensure it's not excluded
        if repository and is_excluded_repo(repository):
            self.logger.warning(f"Attempted to initialize TaskManager with excluded repository: {repository}")
            repository = None
            
        self.repo_name = repository
        
        self.logger.debug(f"TaskManager initialization:")
        self.logger.debug(f"   - GitHub token exists: {bool(self.github_token)}")
        self.logger.debug(f"   - Repository name: {self.repo_name if self.repo_name else 'None (no default)'}")
        
        self.github = Github(self.github_token) if self.github_token else None
        self.repo = None
        
        # Only get repo if a valid repository name is provided
        if self.github and self.repo_name:
            try:
                self.repo = self.github.get_repo(self.repo_name)
            except Exception as e:
                self.logger.warning(f"Could not access repository {self.repo_name}: {e}")
                self.repo = None
        
        self.logger.debug(f"   - GitHub client created: {bool(self.github)}")
        self.logger.debug(f"   - Repository object created: {bool(self.repo)}")
        
        # Load or initialize task state
        self.state_file = "task_state.json"
        self.history_file = "task_history.json"
        self.state = self._load_state()
        self.history = self._load_history()
        
        # Initialize AI components
        self.state_manager = StateManager()
        self.system_state = self.state_manager.load_state()
        
        # Initialize AI brain for all AI-powered features
        self.ai_brain = IntelligentAIBrain(self.system_state, {})
        
        # Initialize AI content generator if available
        self.ai_content_generator = None
        global AI_CONTENT_GENERATOR_AVAILABLE
        
        if AI_CONTENT_GENERATOR_AVAILABLE:
            try:
                self.ai_content_generator = AITaskContentGenerator(self.ai_brain)
                self.logger.debug("AI content generator initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI content generator: {e}")
                AI_CONTENT_GENERATOR_AVAILABLE = False
                self.ai_content_generator = None
        
        # Initialize decomposition system if available
        self.decomposition_engine = None
        self.hierarchical_manager = None
        self.complexity_analyzer = None
        self.progressive_generator = None
        
        global DECOMPOSITION_AVAILABLE  # Declare global at start of method
        
        if DECOMPOSITION_AVAILABLE:
            try:
                # Initialize decomposition components (using same AI brain)
                self.complexity_analyzer = ComplexityAnalyzer(self.ai_brain)
                self.decomposition_engine = TaskDecompositionEngine(self.ai_brain)
                self.hierarchical_manager = HierarchicalTaskManager()
                self.progressive_generator = ProgressiveTaskGenerator(
                    self.ai_brain, self.hierarchical_manager, self.complexity_analyzer
                )
                
                self.logger.debug("Decomposition system initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize decomposition system: {e}")
                DECOMPOSITION_AVAILABLE = False
        
    def _load_state(self) -> Dict[str, Any]:
        """Load task state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading task state: {e}")
        
        return {
            "tasks": {},
            "task_counter": 1000,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "active_tasks": 0,
            "completed_today": 0,
            "success_rate": 0.0
        }
    
    def _find_duplicate_task(self, task_type: TaskType, title: str, description: str, repository: str = None) -> Optional[Dict[str, Any]]:
        """Find if a similar task already exists, enhanced with repository-aware duplicate detection.
        
        Args:
            task_type: Type of task to check
            title: Task title to compare
            description: Task description to compare
            repository: Target repository for context-aware matching
            
        Returns:
            Existing task if found, None otherwise
        """
        # Check all tasks in state
        for task_id, task in self.state.get("tasks", {}).items():
            # Skip completed or cancelled tasks
            if task.get("status") in ["completed", "cancelled"]:
                continue
            
            # Repository-aware duplicate detection
            task_repo = task.get("repository")
            
            # Same repository exact match
            if (repository and task_repo == repository and
                task.get("type") == task_type.value and 
                task.get("title").lower() == title.lower()):
                return task
                
            # Cross-repository duplicate for similar system-wide tasks
            if ((not repository or task_type in [TaskType.NEW_PROJECT]) and
                task.get("type") == task_type.value and
                self._calculate_similarity(task.get("title", ""), title) > 0.9):
                return task
                
            # Enhanced similarity check for repository-specific tasks
            if (repository and task_repo == repository and
                task.get("type") == task_type.value and
                self._calculate_similarity(task.get("title", ""), title) > 0.8):
                return task
                
            # Semantic duplicate detection for same repository
            if (repository and task_repo == repository and
                task.get("type") == task_type.value and
                self._semantic_similarity(description, task.get("description", "")) > 0.85):
                return task
                
        return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings.
        
        Simple character-based similarity for now.
        Returns value between 0 and 1.
        """
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
        
        if str1 == str2:
            return 1.0
            
        # Simple overlap calculation
        shorter = min(len(str1), len(str2))
        longer = max(len(str1), len(str2))
        
        if longer == 0:
            return 0.0
            
        # Count matching characters in same positions
        matches = sum(1 for i in range(shorter) if str1[i] == str2[i])
        
        # Also check if one is substring of other
        if str1 in str2 or str2 in str1:
            return 0.95
            
        return matches / longer
    
    def _calculate_enhanced_similarity(self, title1: str, title2: str) -> float:
        """Calculate enhanced similarity using multiple algorithms.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        # Normalize titles
        title1_lower = title1.lower().strip()
        title2_lower = title2.lower().strip()
        
        # Exact match
        if title1_lower == title2_lower:
            return 1.0
        
        # Calculate basic similarity
        basic_sim = self._calculate_similarity(title1, title2)
        
        # Calculate Jaccard similarity
        words1 = set(title1_lower.split())
        words2 = set(title2_lower.split())
        
        jaccard_sim = 0.0
        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard_sim = len(intersection) / len(union)
        
        # Calculate token overlap with stop words removed
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be'
        }
        
        tokens1 = {word for word in words1 if word not in stop_words and len(word) > 2}
        tokens2 = {word for word in words2 if word not in stop_words and len(word) > 2}
        
        token_sim = 0.0
        if tokens1 and tokens2:
            token_intersection = tokens1.intersection(tokens2)
            token_union = tokens1.union(tokens2)
            token_sim = len(token_intersection) / len(token_union)
        
        # Check for substring containment
        substring_bonus = 0.0
        if title1_lower in title2_lower or title2_lower in title1_lower:
            substring_bonus = 0.15
        
        # Weighted combination
        combined_similarity = (
            basic_sim * 0.2 +
            jaccard_sim * 0.3 +
            token_sim * 0.5 +
            substring_bonus
        )
        
        return min(1.0, combined_similarity)
    
    def _extract_key_terms(self, title: str) -> str:
        """Extract key search terms from title for GitHub search.
        
        Args:
            title: Task title
            
        Returns:
            Space-separated search terms
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'add', 'create', 'implement', 'fix', 'update', 'improve', 'enhance'
        }
        
        # Extract words
        words = title.lower().split()
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Take the most important terms (max 5)
        return ' '.join(key_terms[:5])
    
    def _semantic_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate semantic similarity between two task descriptions.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Similarity score between 0 and 1
        """
        if not desc1 or not desc2:
            return 0.0
            
        # Normalize descriptions
        desc1_words = set(desc1.lower().split())
        desc2_words = set(desc2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(desc1_words.intersection(desc2_words))
        union = len(desc1_words.union(desc2_words))
        
        if union == 0:
            return 0.0
            
        jaccard_sim = intersection / union
        
        # Check for common technical keywords that indicate similar functionality
        tech_keywords = {
            'api', 'authentication', 'database', 'ui', 'frontend', 'backend',
            'testing', 'security', 'performance', 'integration', 'deployment',
            'dashboard', 'user', 'admin', 'payment', 'notification', 'email'
        }
        
        desc1_tech = desc1_words.intersection(tech_keywords)
        desc2_tech = desc2_words.intersection(tech_keywords)
        
        if desc1_tech and desc2_tech:
            tech_overlap = len(desc1_tech.intersection(desc2_tech)) / max(len(desc1_tech), len(desc2_tech))
            # Weight technical keyword overlap higher for semantic similarity
            return min(1.0, jaccard_sim + (tech_overlap * 0.3))
        
        return jaccard_sim
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load task history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading task history: {e}")
        return []
    
    def _save_state(self) -> None:
        """Save task state to file."""
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _save_history(self) -> None:
        """Save task history to file."""
        # Keep only last 1000 history entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def generate_task_id(self) -> str:
        """Generate a unique task ID using timestamp and random component."""
        import uuid
        timestamp = int(time.time() * 1000)  # milliseconds
        random_component = str(uuid.uuid4())[:8]
        return f"TASK-{timestamp}-{random_component}"
    
    def create_task(self, task_type: TaskType, title: str, description: str,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   dependencies: List[str] = None,
                   estimated_hours: float = None,  # Deprecated: will use complexity model
                   labels: List[str] = None,
                   repository: str = None) -> Dict[str, Any]:
        """Create a new task.
        
        Args:
            task_type: Type of task
            title: Task title
            description: Detailed task description
            priority: Task priority
            dependencies: List of task IDs this depends on
            estimated_hours: Estimated hours to complete
            labels: GitHub labels to apply
            
        Returns:
            Created task dictionary
        """
        # Validate repository exists if specified
        if repository:
            # Load system state to check valid repositories
            system_state_file = "system_state.json"
            if os.path.exists(system_state_file):
                with open(system_state_file, 'r') as f:
                    system_state = json.load(f)
                
                valid_repositories = set(system_state.get('projects', {}).keys())
                valid_repositories.update(system_state.get('repositories', {}).keys())
                
                if repository not in valid_repositories:
                    raise ValueError(f"Cannot create task for non-existent repository: {repository}. Valid repositories: {', '.join(sorted(valid_repositories))}")
        
        # Use distributed locking if available, otherwise use file locking
        lock_acquired = False
        lock_key = f"task_creation:{hashlib.md5(f'{title}:{repository}'.encode()).hexdigest()}"
        
        lock_manager = None
        lock_file = None
        
        try:
            # Try Redis distributed lock first
            try:
                from scripts.redis_distributed_locks import RedisDistributedLockManager
                lock_manager = RedisDistributedLockManager()
                # Run async lock acquisition in sync context
                loop = asyncio.new_event_loop()
                # Initialize the lock manager first
                loop.run_until_complete(lock_manager.initialize())
                lock_acquired = loop.run_until_complete(lock_manager.acquire_lock(
                    lock_name=lock_key,
                    timeout_seconds=30,
                    requester_id=f"task_manager_{os.getpid()}"
                ))
                loop.close()
            except Exception as e:
                self.logger.debug(f"Redis locking not available: {e}, using file-based coordination")
                
            # If Redis not available, use file-based coordination
            if not lock_acquired:
                import fcntl
                lock_file_path = f"/tmp/.task_lock_{lock_key}.lock"
                lock_file = open(lock_file_path, 'w')
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                except IOError:
                    lock_file.close()
                    raise ValueError(f"Another process is creating a similar task: {title}")
            
            # Check for duplicate tasks before creating (with lock held)
            existing_task = self._find_duplicate_task(task_type, title, description, repository)
            if existing_task:
                self.logger.warning(f"Found existing task in {repository or 'system'}: {existing_task['id']} - {title}")
                # Don't return existing task, raise an exception instead
                raise ValueError(f"Duplicate task already exists: {existing_task['id']} - {title}")
            
            task_id = self.generate_task_id()
            
            # Calculate task complexity using new dependencies/sequences model
            task_complexity = self._calculate_task_complexity(task_type, description, repository)
            
            # Use legacy estimated_hours if provided, otherwise use complexity-based cycles
            if estimated_hours is None:
                # For backward compatibility, convert AI cycles to approximate hours for external systems
                estimated_hours = task_complexity['estimated_ai_cycles'] * 2.0  # Rough conversion
            
            task = {
            "id": task_id,
            "type": task_type.value,
            "title": title,
            "description": description,
            "priority": priority.value,
            "status": TaskStatus.PENDING.value,
            "dependencies": dependencies or [],
            "blocks": [],
            "estimated_hours": estimated_hours,  # Legacy field for compatibility
            "actual_hours": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "assigned_at": None,
            "completed_at": None,
            "github_issue_number": None,
            "github_pr_numbers": [],
            "iterations": 0,
            "labels": labels or [],
            "repository": repository,
            "metrics": {
                "comments": 0,
                "reactions": 0,
                "code_changes": 0
            },
            "complexity": task_complexity,  # New complexity model
            "relationships": {
                "upstream_impacts": [],  # Tasks this affects
                "downstream_dependencies": dependencies or [],  # Tasks this depends on
                "cross_project_relevance": {},  # Other projects affected
                "knowledge_artifacts": []  # Documentation/patterns created
            },
            "performance_tracking": {
                "performance_baseline": {},  # Metrics before task
                "performance_target": {},  # Expected improvements
                "actual_performance_delta": {},  # Real change achieved
                "reusable_components": []  # What can be extracted
            }
            }
            
            # Update dependencies to include this as a blocker
            for dep_id in task["dependencies"]:
                if dep_id in self.state["tasks"]:
                    self.state["tasks"][dep_id]["blocks"].append(task_id)
            
            self.state["tasks"][task_id] = task
            self._save_state()
            
            # Add to history
            self._add_history_entry("task_created", task_id, {"title": title, "type": task_type.value})
            
            # Analyze relationships with other tasks
            self.analyze_task_relationships(task_id)
            
            # Note: Task decomposition is available but requires async context
            # This can be triggered later via the hierarchical task manager
            if DECOMPOSITION_AVAILABLE and self.complexity_analyzer and self.decomposition_engine:
                self.logger.debug(f"Task {task_id} marked for potential decomposition analysis")
            
            return task
            
        finally:
            # Release locks
            if lock_manager and lock_acquired:
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(lock_manager.release_lock(lock_key, f"task_manager_{os.getpid()}"))
                    loop.close()
                except Exception as e:
                    self.logger.error(f"Error releasing Redis lock: {e}")
            
            if lock_file:
                try:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    # Clean up lock file
                    lock_file_path = f"/tmp/.task_lock_{lock_key}.lock"
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except Exception as e:
                    self.logger.error(f"Error releasing file lock: {e}")
    
    async def _analyze_and_decompose_task(self, task: Dict[str, Any], repository: str = None) -> Optional[Dict[str, Any]]:
        """Analyze task complexity and decompose if needed.
        
        Args:
            task: Task to analyze
            repository: Repository context
            
        Returns:
            Decomposed task or None if no decomposition needed
        """
        if not (self.complexity_analyzer and self.decomposition_engine and self.hierarchical_manager):
            return None
        
        try:
            # Analyze complexity
            complexity_analysis = await self.complexity_analyzer.analyze_complexity(task)
            
            self.logger.info(f"Task complexity analysis: {complexity_analysis.overall_level.value} "
                  f"({complexity_analysis.overall_score:.2f})")
            
            # Store complexity analysis in task metadata
            task['complexity_analysis'] = {
                'level': complexity_analysis.overall_level.value,
                'score': complexity_analysis.overall_score,
                'decomposition_recommended': complexity_analysis.decomposition_recommended,
                'estimated_subtasks': complexity_analysis.estimated_subtasks,
                'confidence': complexity_analysis.confidence_level
            }
            
            # Decompose if recommended
            if complexity_analysis.decomposition_recommended:
                self.logger.info(f"Decomposing task into {complexity_analysis.estimated_subtasks} sub-tasks")
                
                # Get repository context for better decomposition
                repository_context = {}
                if repository:
                    repository_context = await self._get_repository_context(repository)
                
                # Perform decomposition
                decomposition_result = await self.decomposition_engine.decompose_task(task, repository_context)
                
                # Add to hierarchical manager
                hierarchy_id = self.hierarchical_manager.add_task_hierarchy(decomposition_result, task)
                
                # Update task with hierarchy information
                task['hierarchy_id'] = hierarchy_id
                task['is_decomposed'] = True
                task['sub_tasks'] = [st.id for st in decomposition_result.sub_tasks]
                task['decomposition_strategy'] = decomposition_result.strategy.value
                
                # Create GitHub issues for sub-tasks
                if self.github and self.repo:
                    await self._create_subtask_issues(decomposition_result.sub_tasks, task)
                
                self.logger.info(f"Task decomposed successfully: {len(decomposition_result.sub_tasks)} sub-tasks created")
                return task
            else:
                self.logger.debug("Task complexity does not require decomposition")
                
        except Exception as e:
            self.logger.error(f"Error in task decomposition: {e}")
            
        return None
    
    async def _get_repository_context(self, repository: str) -> Dict[str, Any]:
        """Get repository context for decomposition.
        
        Args:
            repository: Repository name
            
        Returns:
            Repository context
        """
        context = {'repository': repository}
        
        if self.github and repository:
            try:
                # Get basic repository information
                if '/' in repository:
                    repo = self.github.get_repo(repository)
                else:
                    # Default to CodeWebMobile-AI organization
                    org_name = "CodeWebMobile-AI"
                    repo = self.github.get_repo(f"{org_name}/{repository}")
                
                context.update({
                    'language': repo.language,
                    'topics': repo.get_topics(),
                    'open_issues': repo.open_issues_count,
                    'size': repo.size,
                    'created_at': repo.created_at.isoformat(),
                    'updated_at': repo.updated_at.isoformat()
                })
                
            except Exception as e:
                self.logger.warning(f"Could not get repository context for {repository}: {e}")
        
        return context
    
    async def _create_subtask_issues(self, sub_tasks: List, parent_task: Dict[str, Any]) -> None:
        """Create GitHub issues for sub-tasks.
        
        Args:
            sub_tasks: List of sub-tasks
            parent_task: Parent task
        """
        if not (self.github and self.repo):
            return
        
        parent_issue_number = parent_task.get('github_issue_number')
        
        for sub_task in sub_tasks:
            try:
                # Create issue title and description
                issue_title = f"[SUB-TASK] {sub_task.title}"
                
                issue_body = f"""@claude {sub_task.description}

## Sub-task Details
- **Parent Task**: #{parent_issue_number if parent_issue_number else 'TBD'}
- **Sequence Order**: {sub_task.sequence_order}
- **Estimated Hours**: {sub_task.estimated_hours}
- **Can Parallelize**: {sub_task.can_parallelize}

## Deliverables
{chr(10).join(f"- {deliverable}" for deliverable in sub_task.deliverables)}

## Acceptance Criteria
{chr(10).join(f"- {criteria}" for criteria in sub_task.acceptance_criteria)}

## Technical Requirements
{chr(10).join(f"- {req}" for req in sub_task.technical_requirements)}

---
*This is an automatically decomposed sub-task. Complete this before moving to dependent tasks.*
"""
                
                # Create labels
                labels = [
                    'sub-task',
                    'ai-managed',
                    f'priority:{sub_task.priority}',
                    f'sequence:{sub_task.sequence_order}'
                ]
                
                if sub_task.can_parallelize:
                    labels.append('parallelizable')
                
                # Create the issue
                issue = self.repo.create_issue(
                    title=issue_title,
                    body=issue_body,
                    labels=labels
                )
                
                self.logger.info(f"Created sub-task issue #{issue.number}: {sub_task.title}")
                
                # Update hierarchical manager with issue number
                if self.hierarchical_manager and sub_task.id in self.hierarchical_manager.task_nodes:
                    self.hierarchical_manager.task_nodes[sub_task.id].github_issue_number = issue.number
                
            except Exception as e:
                self.logger.error(f"Error creating sub-task issue for {sub_task.title}: {e}")
    
    def mark_task_completed(self, task_id: str, actual_hours: float = 0.0) -> bool:
        """Mark a task as completed and trigger progressive task generation.
        
        Args:
            task_id: Task ID to mark as completed
            actual_hours: Actual hours spent on task
            
        Returns:
            Success status
        """
        if task_id not in self.state["tasks"]:
            self.logger.warning(f"Task {task_id} not found")
            return False
        
        task = self.state["tasks"][task_id]
        
        # Update task status
        task["status"] = TaskStatus.COMPLETED.value
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        task["actual_hours"] = actual_hours
        
        # Update hierarchical manager if available
        if DECOMPOSITION_AVAILABLE and self.hierarchical_manager:
            try:
                self.hierarchical_manager.update_task_progress(task_id, 100.0, actual_hours, 'completed')
            except Exception as e:
                self.logger.warning(f"Failed to update hierarchical progress: {e}")
        
        # Generate progressive tasks if available
        if DECOMPOSITION_AVAILABLE and self.progressive_generator:
            try:
                self._generate_progressive_tasks(task)
            except Exception as e:
                self.logger.warning(f"Progressive task generation failed: {e}")
        
        self._save_state()
        self._add_history_entry("task_completed", task_id, {"actual_hours": actual_hours})
        
        self.logger.info(f"Task {task_id} marked as completed")
        return True
    
    async def _generate_progressive_tasks(self, completed_task: Dict[str, Any]) -> None:
        """Generate progressive tasks based on completion.
        
        Args:
            completed_task: Recently completed task
        """
        if not self.progressive_generator:
            return
        
        try:
            # Create progression context
            from progressive_task_generator import ProgressionContext
            
            context = ProgressionContext(
                completed_task=completed_task,
                repository_context=await self._get_repository_context(completed_task.get('repository', '')),
                project_state=self.system_state,
                recent_patterns=[],
                current_priorities=[],
                ai_agent_capacity={
                    'available_processing_cycles': 24,  # 24/7 operation
                    'parallel_task_limit': 3,  # Can handle multiple tasks concurrently
                    'complexity_threshold': 0.8  # Upper limit for task complexity
                },
                processing_constraints={
                    'max_concurrent_tasks': 3,
                    'priority_queue_depth': 10,
                    'context_window_limit': 100000  # AI context limitations
                }
            )
            
            # Generate next task suggestions
            suggestions = await self.progressive_generator.generate_next_tasks(completed_task, context)
            
            if suggestions:
                self.logger.info(f"Generated {len(suggestions)} follow-up task suggestions:")
                for suggestion in suggestions:
                    self.logger.info(f"  - {suggestion.title} ({suggestion.task_type}, {suggestion.confidence:.2f} confidence)")
                    
                    # Optionally auto-create high-confidence suggestions
                    if suggestion.confidence > 0.8:
                        try:
                            # Convert suggestion to task
                            task_type = TaskType(suggestion.task_type.lower())
                            priority = TaskPriority(suggestion.priority.upper())
                            
                            follow_up_task = self.create_task(
                                task_type=task_type,
                                title=suggestion.title,
                                description=suggestion.description,
                                priority=priority,
                                estimated_hours=suggestion.estimated_hours,
                                repository=completed_task.get('repository'),
                                dependencies=[completed_task['id']] if suggestion.prerequisites else None
                            )
                            
                            self.logger.info(f"Auto-created high-confidence follow-up task: {follow_up_task['id']}")
                            
                        except Exception as e:
                            self.logger.info(f"Error creating follow-up task: {e}")
            
        except Exception as e:
            self.logger.info(f"Error generating progressive tasks: {e}")
    
    def get_task_hierarchy(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task hierarchy for a given task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task hierarchy or None
        """
        if not (DECOMPOSITION_AVAILABLE and self.hierarchical_manager):
            return None
        
        task = self.state["tasks"].get(task_id)
        if not task:
            return None
        
        hierarchy_id = task.get('hierarchy_id')
        if hierarchy_id:
            return self.hierarchical_manager.get_task_hierarchy(hierarchy_id)
        
        return None
    
    def get_ready_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tasks that are ready to be worked on.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of ready tasks
        """
        # Use hierarchical manager if available for better task prioritization
        if DECOMPOSITION_AVAILABLE and self.hierarchical_manager:
            try:
                return self.hierarchical_manager.get_ready_tasks(limit)
            except Exception as e:
                self.logger.info(f"Warning: Error getting ready tasks from hierarchical manager: {e}")
        
        # Fallback to original logic
        ready_tasks = []
        
        for task_id, task in self.state["tasks"].items():
            if (task["status"] in ["pending", "assigned"] and 
                self._are_dependencies_satisfied(task_id)):
                
                ready_tasks.append({
                    'id': task_id,
                    'title': task['title'],
                    'description': task['description'],
                    'type': task['type'],
                    'priority': task['priority'],
                    'estimated_hours': task['estimated_hours'],
                    'repository': task.get('repository'),
                    'github_issue_number': task.get('github_issue_number')
                })
        
        # Sort by priority
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        ready_tasks.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return ready_tasks[:limit]
    
    def get_task_queue(self) -> List[Dict[str, Any]]:
        """Get all tasks from the task queue.
        
        Returns:
            List of all tasks with their full information
        """
        task_list = []
        
        for task_id, task in self.state.get("tasks", {}).items():
            task_copy = task.copy()
            task_copy['id'] = task_id
            task_list.append(task_copy)
        
        return task_list
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if dependencies are satisfied
        """
        task = self.state["tasks"].get(task_id)
        if not task:
            return False
        
        for dep_id in task.get("dependencies", []):
            if dep_id in self.state["tasks"]:
                dep_task = self.state["tasks"][dep_id]
                if dep_task["status"] != "completed":
                    return False
        
        return True
    
    def create_github_issue(self, task: Dict[str, Any]) -> Optional[int]:
        """Create a GitHub issue for a task.
        
        Args:
            task: Task dictionary
            
        Returns:
            GitHub issue number or None if failed
        """
        self.logger.info(f"🎯 create_github_issue called for task: {task.get('title', 'Unknown')}")
        self.logger.info(f"   - Task type: {task.get('type', 'Unknown')}")
        self.logger.info(f"   - Repository: {task.get('repository', 'None')}")
        self.logger.info(f"   - GitHub instance: {self.github}")
        self.logger.info(f"   - Repo instance: {self.repo}")
        
        # Determine target repository
        target_repo = self.repo  # Default to self.repo
        repository = task.get('repository')
        
        if repository and self.github:
            # Import exclusion check
            from scripts.repository_exclusion import should_process_repo, get_exclusion_reason
            
            # Check if repository is excluded before proceeding
            if not should_process_repo(repository):
                self.logger.info(f"❌ Cannot create task in excluded repository: {repository}")
                self.logger.info(f"   Reason: {get_exclusion_reason(repository)}")
                return None
            
            try:
                # Try to get the specific repository
                if '/' in repository:
                    # Full repo name provided
                    target_repo = self.github.get_repo(repository)
                else:
                    # Just repo name, assume same organization
                    # Default to CodeWebMobile-AI organization
                    org_name = "CodeWebMobile-AI"
                    full_repo_name = f"{org_name}/{repository}"
                    target_repo = self.github.get_repo(full_repo_name)
                self.logger.info(f"✅ Successfully got repository: {target_repo.full_name}")
            except Exception as e:
                self.logger.info(f"❌ Failed to access target repository {repository}: {e}")
                # No fallback - if we can't access the repository, we shouldn't create issues
                target_repo = None
        
        if not target_repo:
            self.logger.info("❌ No repository available for issue creation")
            self.logger.info(f"   - self.repo: {self.repo}")
            self.logger.info(f"   - self.github: {self.github}")
            return None
        
        # Final check - ensure the target repository is not excluded
        from scripts.repository_exclusion import should_process_repo, get_exclusion_reason
        if target_repo and hasattr(target_repo, 'full_name') and not should_process_repo(target_repo.full_name):
            self.logger.info(f"❌ Cannot create task in excluded repository: {target_repo.full_name}")
            self.logger.info(f"   Reason: {get_exclusion_reason(target_repo.full_name)}")
            return None
        
        try:
            # Check for existing issues before creating a new one
            existing_issue = self._check_existing_github_issue(target_repo, task)
            if existing_issue:
                self.logger.info(f"✅ Issue already exists: #{existing_issue.number} for: {task['title']}")
                # Update task with existing issue number
                task["github_issue_number"] = existing_issue.number
                task["status"] = TaskStatus.ASSIGNED.value
                task["assigned_at"] = datetime.now(timezone.utc).isoformat()
                self._save_state()
                return existing_issue.number
            
            # Format the issue body with @claude mention
            body = self._format_issue_body(task)
            
            # Determine labels
            labels = task["labels"].copy()
            labels.append(f"priority:{task['priority']}")
            labels.append(f"type:{task['type']}")
            labels.append("ai-managed")
            
            # Create the issue in the target repository
            issue = target_repo.create_issue(
                title=task["title"],
                body=body,
                labels=labels
            )
            
            # Update task with issue number
            task["github_issue_number"] = issue.number
            task["status"] = TaskStatus.ASSIGNED.value
            task["assigned_at"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
            
            self.logger.info(f"Created GitHub issue #{issue.number} for {task['id']}")
            return issue.number
            
        except Exception as e:
            self.logger.info(f"Error creating GitHub issue: {e}")
            return None
    
    def create_ai_task_issue(self, title: str, description: str, labels: List[str] = None, 
                           priority: str = "medium", task_type: str = "task", 
                           repository: str = None) -> Optional[int]:
        """Create a GitHub issue for AI tasks with @claude mention.
        
        This is the centralized method that all AI-generated task issues should use
        to ensure consistent formatting and @claude mentions.
        
        Args:
            title: Issue title
            description: Issue description/body content
            labels: Additional labels for the issue
            priority: Task priority (low, medium, high)
            task_type: Type of task (task, setup, bug, feature, etc.)
            
        Returns:
            GitHub issue number or None if failed
        """
        # Determine target repository
        target_repo = self.repo  # Default
        
        if repository and self.github:
            # Check if target project is excluded
            if not should_process_repo(repository):
                self.logger.info(f"Cannot create task in excluded repository: {repository}")
                self.logger.info(f"Reason: {RepositoryExclusion.get_exclusion_reason(repository)}")
                return None
            
            try:
                # Try to get the specific repository
                if '/' in repository:
                    # Full repo name provided
                    target_repo = self.github.get_repo(repository)
                else:
                    # Just repo name, assume same organization
                    # Default to CodeWebMobile-AI organization
                    org_name = "CodeWebMobile-AI"
                    full_repo_name = f"{org_name}/{repository}"
                    target_repo = self.github.get_repo(full_repo_name)
                self.logger.info(f"Creating AI issue in repository: {target_repo.full_name}")
            except Exception as e:
                self.logger.info(f"Failed to access target repository {repository}: {e}")
                self.logger.info(f"Falling back to default repository: {self.repo_name}")
                # Fall back to default repo
                target_repo = self.repo
        
        # Final check if the target repo is excluded
        if target_repo and not should_process_repo(target_repo.full_name):
            self.logger.info(f"Cannot create task in excluded repository: {target_repo.full_name}")
            return None
        
        if not target_repo:
            self.logger.info("No repository available for issue creation")
            return None
        
        try:
            # Check for existing issues before creating
            temp_task = {
                "title": title,
                "type": task_type,
                "id": f"ai_task_{int(time.time())}"  # Temporary ID for checking
            }
            existing_issue = self._check_existing_github_issue(target_repo, temp_task)
            if existing_issue:
                self.logger.info(f"✅ Issue already exists: #{existing_issue.number} for: {title}")
                return existing_issue.number
            
            # Validate @claude mention is present
            if "@claude" not in description:
                self.logger.info("Warning: AI task issue created without @claude mention, adding it")
                description = f"@claude {description}"
            
            # Create labels list
            issue_labels = labels or []
            issue_labels.extend([f"priority:{priority}", f"type:{task_type}", "ai-managed"])
            
            # Create the issue directly with proper formatting
            issue = target_repo.create_issue(
                title=title,
                body=description,
                labels=issue_labels
            )
            
            self.logger.info(f"Created AI task issue #{issue.number}: {title}")
            return issue.number
            
        except Exception as e:
            self.logger.info(f"Error creating AI task issue: {e}")
            return None
    
    def _check_existing_github_issue(self, repo, task: Dict[str, Any]):
        """Check if a similar issue already exists in the repository.
        
        Args:
            repo: GitHub repository object
            task: Task dictionary
            
        Returns:
            GitHub issue object if found, None otherwise
        """
        try:
            self.logger.info(f"Searching for duplicates of: {task['title']}")
            
            # Use GitHub search API for more efficient duplicate detection
            search_queries = [
                f'repo:{repo.full_name} is:issue state:open "{task["title"]}"',  # Exact title
                f'repo:{repo.full_name} is:issue "Task ID: {task["id"]}"',  # Task ID
            ]
            
            # Extract key terms for broader search
            key_terms = self._extract_key_terms(task["title"])
            if key_terms:
                search_queries.append(f'repo:{repo.full_name} is:issue state:open {key_terms}')
            
            # Also check recently closed issues (last 30 days)
            from datetime import datetime, timedelta
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            search_queries.append(
                f'repo:{repo.full_name} is:issue state:closed closed:>{thirty_days_ago} "{task["title"]}"'
            )
            
            checked_issues = set()
            
            # Different thresholds for different task types
            similarity_thresholds = {
                'documentation': 0.85,
                'bug_fix': 0.90,
                'feature': 0.88,
                'testing': 0.87,
                'new_project': 0.95,
                'refactor': 0.86,
                'security': 0.89,
                'performance': 0.88
            }
            
            # Try search API first
            try:
                from github import Github
                g = Github(self.github_token)
                
                for query in search_queries:
                    self.logger.info(f"Searching with query: {query}")
                    search_results = g.search_issues(query=query)
                    
                    for issue in search_results:
                        if issue.number in checked_issues:
                            continue
                        checked_issues.add(issue.number)
                        
                        # Check for exact title match
                        if issue.title.lower() == task["title"].lower():
                            state_info = f" (closed {issue.closed_at})" if issue.state == 'closed' else ""
                            self.logger.info(f"Found exact title match: Issue #{issue.number}{state_info}")
                            return issue
                        
                        # Check for task ID in body
                        if issue.body and f"Task ID: {task['id']}" in issue.body:
                            self.logger.info(f"Found task ID match: Issue #{issue.number}")
                            return issue
                        
                        # Enhanced similarity check
                        similarity = self._calculate_enhanced_similarity(issue.title, task["title"])
                        
                        threshold = similarity_thresholds.get(task.get("type"), 0.88)
                        
                        if similarity > threshold:
                            state_info = f" (closed {issue.closed_at})" if issue.state == 'closed' else ""
                            self.logger.info(
                                f"Found similar {task.get('type', 'task')}: Issue #{issue.number} "
                                f"(similarity: {similarity:.2%}){state_info}"
                            )
                            return issue
                        
                        # Stop after checking 50 issues
                        if len(checked_issues) > 50:
                            break
                            
            except Exception as e:
                self.logger.info(f"Search API error: {e}, falling back to iteration")
            
            # Fallback to traditional iteration if search fails or found nothing
            if len(checked_issues) == 0:
                issues = repo.get_issues(state='open', sort='created', direction='desc')
                
                count = 0
                for issue in issues:
                    if count >= 100:  # Reduced for better performance
                        break
                    count += 1
                    
                    # Check for exact title match
                    if issue.title.lower() == task["title"].lower():
                        self.logger.info(f"Found exact title match: Issue #{issue.number}")
                        return issue
                    
                    # Check for task ID in body
                    if issue.body and f"Task ID: {task['id']}" in issue.body:
                        self.logger.info(f"Found task ID match: Issue #{issue.number}")
                        return issue
                    
                    # Check for high similarity
                    similarity = self._calculate_enhanced_similarity(issue.title, task["title"])
                    threshold = similarity_thresholds.get(task.get("type"), 0.88)
                    
                    if similarity > threshold:
                        self.logger.info(f"Found similar task: Issue #{issue.number} (similarity: {similarity:.2%})")
                        return issue
            
            self.logger.info(f"No duplicates found after checking {len(checked_issues) or count} issues")
            return None
            
        except Exception as e:
            self.logger.info(f"Error checking for existing GitHub issue: {e}")
            return None
    
    def _format_issue_body(self, task: Dict[str, Any]) -> str:
        """Format issue body with @claude mention.
        
        Args:
            task: Task dictionary
            
        Returns:
            Formatted issue body
        """
        # Get task type specific template
        template = self._get_task_template(TaskType(task["type"]))
        
        # Add dependency information
        dep_info = ""
        if task["dependencies"]:
            dep_tasks = [self.state["tasks"].get(dep_id) for dep_id in task["dependencies"]]
            dep_issues = [f"#{t['github_issue_number']}" for t in dep_tasks if t and t.get("github_issue_number")]
            if dep_issues:
                dep_info = f"\n## Dependencies\nThis task depends on: {', '.join(dep_issues)}\n"
        
        # Add blocking information
        block_info = ""
        if task["blocks"]:
            block_tasks = [self.state["tasks"].get(block_id) for block_id in task["blocks"]]
            block_issues = [f"#{t['github_issue_number']}" for t in block_tasks if t and t.get("github_issue_number")]
            if block_issues:
                block_info = f"\n## Blocks\nThis task blocks: {', '.join(block_issues)}\n"
        
        body = f"""@claude {task["description"]}

{template}

## Task Information
- **Task ID**: {task["id"]}
- **Priority**: {task["priority"]}
- **Estimated Hours**: {task["estimated_hours"]}
- **Type**: {task["type"]}
{dep_info}{block_info}

## Acceptance Criteria
Please ensure all requirements are met and include appropriate tests.

---
*This task was automatically generated by the AI Task Manager*
"""
        return body
    
    def _get_task_template(self, task_type: TaskType) -> str:
        """Get task-specific template content.
        
        Args:
            task_type: Type of task
            
        Returns:
            Template string
        """
        templates = {
            TaskType.NEW_PROJECT: """
## Project Setup
1. Fork https://github.com/laravel/react-starter-kit.git
2. Configure for the specified requirements
3. Update README with project-specific information

## Laravel React Starter Benefits
- Full-stack Laravel + React SPA structure
- Authentication with Sanctum pre-configured
- Modern tooling: Vite, TypeScript, Tailwind CSS
- Testing setup with PHPUnit and Jest
- Docker development environment

## Required Customizations
Please implement the features described above while maintaining the starter kit's structure.
""",
            TaskType.FEATURE: """
## Feature Requirements
Please implement this feature following the project's established patterns.

## Implementation Guidelines
- Follow existing code style and conventions
- Include unit and integration tests
- Update documentation as needed
- Ensure backward compatibility
""",
            TaskType.BUG_FIX: """
## Bug Details
Please investigate and fix the described issue.

## Fix Requirements
- Identify root cause
- Implement fix with minimal side effects
- Add regression tests
- Verify fix in multiple scenarios
""",
            TaskType.REFACTOR: """
## Refactoring Goals
Please refactor the code while maintaining functionality.

## Guidelines
- Preserve all existing functionality
- Improve code structure and readability
- Update tests as needed
- Document significant changes
""",
            TaskType.DOCUMENTATION: """
## Documentation Requirements
Please create or update documentation as specified.

## Documentation Standards
- Clear and concise writing
- Include code examples where relevant
- Follow project documentation format
- Ensure accuracy and completeness
""",
            TaskType.TESTING: """
## Testing Requirements
Please implement comprehensive tests.

## Test Coverage Goals
- Unit tests for individual components
- Integration tests for workflows
- Edge case coverage
- Performance benchmarks if applicable
""",
            TaskType.SECURITY: """
## Security Requirements
Please address the security concerns described.

## Security Guidelines
- Follow OWASP best practices
- Include security tests
- Document any security implications
- Consider performance impact
""",
            TaskType.PERFORMANCE: """
## Performance Goals
Please optimize performance as specified.

## Optimization Guidelines
- Benchmark before and after changes
- Document performance improvements
- Ensure no functionality regression
- Consider scalability implications
""",
            TaskType.CODE_REVIEW: """
## Review Requirements
Please review the specified code/PR.

## Review Focus Areas
- Code quality and best practices
- Security vulnerabilities
- Performance implications
- Test coverage
- Documentation completeness
""",
            TaskType.DEPENDENCY_UPDATE: """
## Update Requirements
Please update the specified dependencies.

## Update Guidelines
- Check for breaking changes
- Update related code as needed
- Run full test suite
- Document any migration steps
"""
        }
        
        return templates.get(task_type, "")
    
    def analyze_existing_tasks(self) -> Dict[str, Any]:
        """Analyze current state of tasks in GitHub.
        
        Returns:
            Analysis results
        """
        if not self.repo:
            return {"error": "GitHub repository not available"}
        
        analysis = {
            "open_issues": 0,
            "claude_assigned": 0,
            "in_progress": 0,
            "awaiting_review": 0,
            "stale_tasks": [],
            "completed_recently": 0,
            "by_type": {},
            "by_priority": {}
        }
        
        try:
            # Get all open issues
            issues = self.repo.get_issues(state="open", labels=["ai-managed"])
            
            for issue in issues:
                analysis["open_issues"] += 1
                
                # Check if assigned to @claude
                if "@claude" in issue.body:
                    analysis["claude_assigned"] += 1
                
                # Check status by labels
                labels = [label.name for label in issue.labels]
                if "in-progress" in labels:
                    analysis["in_progress"] += 1
                elif "awaiting-review" in labels:
                    analysis["awaiting_review"] += 1
                
                # Check if stale (no updates in 7 days)
                if (datetime.now(timezone.utc) - issue.updated_at.replace(tzinfo=timezone.utc)).days > 7:
                    analysis["stale_tasks"].append({
                        "number": issue.number,
                        "title": issue.title,
                        "days_stale": (datetime.now(timezone.utc) - issue.updated_at.replace(tzinfo=timezone.utc)).days
                    })
                
                # Count by type and priority
                for label in labels:
                    if label.startswith("type:"):
                        task_type = label.split(":")[1]
                        analysis["by_type"][task_type] = analysis["by_type"].get(task_type, 0) + 1
                    elif label.startswith("priority:"):
                        priority = label.split(":")[1]
                        analysis["by_priority"][priority] = analysis["by_priority"].get(priority, 0) + 1
            
            # Check recently closed issues
            closed_issues = self.repo.get_issues(state="closed", labels=["ai-managed"], sort="updated")
            for issue in closed_issues:
                if (datetime.now(timezone.utc) - issue.closed_at.replace(tzinfo=timezone.utc)).days <= 1:
                    analysis["completed_recently"] += 1
                else:
                    break
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def generate_tasks(self, focus: str = "auto", max_tasks: int = 5) -> List[Dict[str, Any]]:
        """Generate new tasks based on current state and focus area.
        
        Args:
            focus: Focus area for task generation
            max_tasks: Maximum number of tasks to generate
            
        Returns:
            List of generated tasks
        """
        # Analyze current state
        analysis = self.analyze_existing_tasks()
        
        # Don't generate too many tasks if many are open
        if analysis.get("open_issues", 0) > 20:
            max_tasks = min(max_tasks, 2)
        
        # Use the class AI brain instance
        ai_brain = self.ai_brain
        try:
            import asyncio
            import traceback
            
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                self.logger.info(f"[DEBUG] Event loop already running, using create_task approach")
                # If we're in a running loop, create a task and await it
                task = loop.create_task(ai_brain.gather_context(self.system_state.get("charter", {})))
                context = loop.run_until_complete(task)
                self.logger.info(f"[DEBUG] Context gathering succeeded via event loop task")
            except RuntimeError as e:
                self.logger.info(f"[DEBUG] No running event loop, using asyncio.run(): {e}")
                # No running event loop, safe to use asyncio.run()
                context = asyncio.run(ai_brain.gather_context(self.system_state.get("charter", {})))
                self.logger.info(f"[DEBUG] Context gathering succeeded via asyncio.run()")
            
            ai_brain.context.update(context)
            self.logger.info(f"[DEBUG] Context updated successfully: {len(context)} items")
            
        except Exception as e:
            self.logger.info(f"ERROR: Context gathering failed in task generation: {e}")
            self.logger.info(f"ERROR: Exception type: {type(e).__name__}")
            self.logger.info(f"ERROR: Traceback:")
            traceback.print_exc()
            # Continue with existing context
        
        generated_tasks = []
        
        # Determine task generation strategy
        if focus == "auto":
            # Intelligent task generation based on current state
            task_strategies = self._determine_task_strategies(analysis, ai_brain)
        else:
            # Focused task generation
            task_strategies = self._get_focused_strategies(focus)
        
        # Generate tasks based on strategies
        self.logger.info(f"[DEBUG] Generated {len(task_strategies)} task strategies, processing up to {max_tasks}")
        for i, strategy in enumerate(task_strategies[:max_tasks]):
            strategy_repo = strategy.get('target_repository', 'system')
            self.logger.info(f"[DEBUG] Processing strategy {i+1}/{min(len(task_strategies), max_tasks)}: {strategy.get('type', 'unknown')} for repository: {strategy_repo}")
            try:
                task = self._generate_task_from_strategy(strategy, ai_brain)
                if task:
                    task_repo = task.get('repository', 'system')
                    generated_tasks.append(task)
                    self.logger.info(f"[DEBUG] Successfully generated task for {task_repo}: {task.get('title', 'untitled')}")
                else:
                    self.logger.info(f"[DEBUG] Strategy {i+1} for {strategy_repo} did not generate a task")
            except Exception as e:
                self.logger.info(f"ERROR: Failed to generate task from strategy {i+1} for repository {strategy_repo}: {e}")
                self.logger.info(f"ERROR: Strategy details: {strategy}")
                import traceback
                traceback.print_exc()
        
        # Save generated tasks to file for reference
        with open("generated_tasks.json", "w") as f:
            json.dump(generated_tasks, f, indent=2)
        
        return generated_tasks
    
    def _determine_task_strategies(self, analysis: Dict[str, Any], ai_brain) -> List[Dict[str, Any]]:
        """Determine task generation strategies based on analysis.
        
        Args:
            analysis: Current task analysis
            ai_brain: AI brain instance
            
        Returns:
            List of task strategies
        """
        strategies = []
        
        # Check if we need a new project
        if len(self.system_state.get("projects", {})) < 3:
            strategies.append({
                "type": TaskType.NEW_PROJECT,
                "priority": TaskPriority.HIGH,
                "reason": "Portfolio expansion needed",
                "target_repository": None,  # New projects don't have existing repos
                "triggered_by": "portfolio_gap_analysis"
            })
        
        # Check for stale tasks that need attention
        if analysis.get("stale_tasks", []):
            stale_task = analysis["stale_tasks"][0]
            strategies.append({
                "type": TaskType.CODE_REVIEW,
                "priority": TaskPriority.HIGH,
                "reason": f"Review stale task #{stale_task.get('number', 'unknown')}",
                "target": stale_task,
                "target_repository": self.repo_name,  # Stale tasks are from main repo
                "triggered_by": f"stale_task_analysis_{stale_task.get('days_stale', 0)}_days"
            })
        
        # Balance task types
        by_type = analysis.get("by_type", {})
        
        # Ensure documentation tasks
        if by_type.get("documentation", 0) < 2:
            # Find a repository that needs documentation
            best_repo = self._find_repository_needing_documentation()
            strategies.append({
                "type": TaskType.DOCUMENTATION,
                "priority": TaskPriority.MEDIUM,
                "reason": "Documentation coverage needed",
                "target_repository": best_repo,
                "triggered_by": f"documentation_gap_analysis_current_count_{by_type.get('documentation', 0)}"
            })
        
        # Ensure testing tasks
        if by_type.get("testing", 0) < 3:
            # Find a repository that needs testing
            best_repo = self._find_repository_needing_testing()
            strategies.append({
                "type": TaskType.TESTING,
                "priority": TaskPriority.HIGH,
                "reason": "Test coverage improvement",
                "target_repository": best_repo,
                "triggered_by": f"test_coverage_gap_analysis_current_count_{by_type.get('testing', 0)}"
            })
        
        # Regular feature development
        best_repo = self._find_repository_for_feature_development()
        strategies.append({
            "type": TaskType.FEATURE,
            "priority": TaskPriority.MEDIUM,
            "reason": "Feature development",
            "target_repository": best_repo,
            "triggered_by": "regular_feature_development_cycle"
        })
        
        # Security review if none active
        if by_type.get("security", 0) == 0:
            best_repo = self._find_repository_needing_security_review()
            strategies.append({
                "type": TaskType.SECURITY,
                "priority": TaskPriority.HIGH,
                "reason": "Security review needed",
                "target_repository": best_repo,
                "triggered_by": f"security_gap_analysis_no_active_security_tasks"
            })
        
        return strategies
    
    def _find_repository_needing_documentation(self) -> str:
        """Find repository that most needs documentation work.
        
        Returns:
            Repository name or None
        """
        projects = self.system_state.get("projects", {})
        if not projects:
            return None
            
        # Prefer repositories with lower health scores (may indicate poor docs)
        best_repo = None
        lowest_health = 100
        
        for proj_name, proj_data in projects.items():
            if not should_process_repo(proj_data.get('full_name', proj_name)):
                continue
                
            health_score = proj_data.get('health_score', 50)
            if health_score < lowest_health:
                lowest_health = health_score
                best_repo = proj_data.get('full_name', proj_name)
        
        return best_repo
    
    def _find_repository_needing_testing(self) -> str:
        """Find repository that most needs testing work.
        
        Returns:
            Repository name or None
        """
        projects = self.system_state.get("projects", {})
        if not projects:
            return None
            
        # Prefer repositories with open issues (may indicate bugs needing tests)
        best_repo = None
        most_issues = 0
        
        for proj_name, proj_data in projects.items():
            if not should_process_repo(proj_data.get('full_name', proj_name)):
                continue
                
            open_issues = proj_data.get('metrics', {}).get('issues_open', 0)
            if open_issues > most_issues:
                most_issues = open_issues
                best_repo = proj_data.get('full_name', proj_name)
        
        return best_repo
    
    def _find_repository_for_feature_development(self) -> str:
        """Find repository that's best suited for feature development.
        
        Returns:
            Repository name or None
        """
        projects = self.system_state.get("projects", {})
        if not projects:
            return None
            
        # Prefer repositories with recent activity and good health
        best_repo = None
        best_score = 0
        
        for proj_name, proj_data in projects.items():
            if not should_process_repo(proj_data.get('full_name', proj_name)):
                continue
                
            health_score = proj_data.get('health_score', 0)
            recent_commits = proj_data.get('recent_activity', {}).get('recent_commits', 0)
            
            # Score based on health and activity
            score = health_score + (recent_commits * 2)
            
            if score > best_score:
                best_score = score
                best_repo = proj_data.get('full_name', proj_name)
        
        return best_repo
    
    def _identify_portfolio_gaps(self, projects: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify gaps in the project portfolio.
        
        Args:
            projects: Current projects dictionary
            
        Returns:
            List of identified gaps
        """
        existing_types = {}
        
        # Analyze existing projects
        for proj_name, proj_data in projects.items():
            proj_type = self._categorize_project(proj_data)
            if proj_type not in existing_types:
                existing_types[proj_type] = []
            existing_types[proj_type].append(proj_name)
        
        # Common project types for a complete portfolio
        desired_types = {
            'authentication', 'api_service', 'dashboard', 'analytics',
            'notification', 'payment', 'cms', 'mobile_backend'
        }
        
        # Identify gaps
        gaps = []
        missing_types = desired_types - set(existing_types.keys())
        
        for gap_type in list(missing_types)[:3]:  # Top 3 gaps
            gap_info = {
                'gap_type': gap_type,
                'description': f"Missing {gap_type} capabilities in portfolio",
                'priority': 'high' if gap_type in ['payment', 'authentication', 'api_service'] else 'medium'
            }
            gaps.append(gap_info)
        
        return gaps
    
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
    
    def _find_repository_needing_security_review(self) -> str:
        """Find repository that most needs security review.
        
        Returns:
            Repository name or None
        """
        projects = self.system_state.get("projects", {})
        if not projects:
            return None
            
        # Prefer repositories that haven't been reviewed recently or have security concerns
        best_repo = None
        
        for proj_name, proj_data in projects.items():
            if not should_process_repo(proj_data.get('full_name', proj_name)):
                continue
                
            # For now, just return the first available repository
            # In future, could check for security-related issues, age, etc.
            if not best_repo:
                best_repo = proj_data.get('full_name', proj_name)
                break
        
        return best_repo
    
    def _get_focused_strategies(self, focus: str) -> List[Dict[str, Any]]:
        """Get task strategies for focused generation.
        
        Args:
            focus: Focus area
            
        Returns:
            List of task strategies
        """
        focus_map = {
            "new_features": TaskType.FEATURE,
            "bug_fixes": TaskType.BUG_FIX,
            "refactoring": TaskType.REFACTOR,
            "documentation": TaskType.DOCUMENTATION,
            "testing": TaskType.TESTING,
            "security": TaskType.SECURITY,
            "performance": TaskType.PERFORMANCE
        }
        
        task_type = focus_map.get(focus, TaskType.FEATURE)
        
        return [
            {
                "type": task_type,
                "priority": TaskPriority.HIGH,
                "reason": f"Focused {focus} generation"
            }
            for _ in range(5)
        ]
    
    def _generate_task_from_strategy(self, strategy: Dict[str, Any], ai_brain) -> Optional[Dict[str, Any]]:
        """Generate a specific task from a strategy.
        
        Args:
            strategy: Task generation strategy
            ai_brain: AI brain instance
            
        Returns:
            Generated task or None
        """
        try:
            task_type = strategy["type"]
            self.logger.info(f"[DEBUG] Generating task of type: {task_type}")
            
            # Generate task content based on type
            title, description = None, None
            
            if task_type == TaskType.NEW_PROJECT:
                self.logger.info(f"[DEBUG] Calling _generate_new_project_task")
                title, description = self._generate_new_project_task(ai_brain)
            elif task_type == TaskType.FEATURE:
                self.logger.info(f"[DEBUG] Calling _generate_feature_task")
                title, description = self._generate_feature_task(ai_brain)
            elif task_type == TaskType.BUG_FIX:
                self.logger.info(f"[DEBUG] Calling _generate_bug_fix_task")
                title, description = self._generate_bug_fix_task()
            elif task_type == TaskType.DOCUMENTATION:
                self.logger.info(f"[DEBUG] Calling _generate_documentation_task")
                title, description = self._generate_documentation_task()
            elif task_type == TaskType.TESTING:
                self.logger.info(f"[DEBUG] Calling _generate_testing_task")
                title, description = self._generate_testing_task()
            elif task_type == TaskType.SECURITY:
                self.logger.info(f"[DEBUG] Calling _generate_security_task")
                title, description = self._generate_security_task()
            elif task_type == TaskType.CODE_REVIEW:
                self.logger.info(f"[DEBUG] Calling _generate_review_task")
                title, description = self._generate_review_task(strategy.get("target"))
            else:
                self.logger.info(f"[DEBUG] Unknown task type: {task_type}")
                return None
            
            self.logger.info(f"[DEBUG] Generated content - Title: {title[:50] if title else 'None'}...")
            
        except Exception as e:
            self.logger.info(f"ERROR: Exception in task content generation: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Create the task using new complexity model
        task = self.create_task(
            task_type=task_type,
            title=title,
            description=description,
            priority=strategy["priority"],
            repository=strategy.get("target_repository")  # Use repository from strategy if available
        )
        
        # Create GitHub issue
        issue_number = self.create_github_issue(task)
        if issue_number:
            task["github_issue_number"] = issue_number
        
        return task
    
    def _generate_new_project_task(self, ai_brain) -> Tuple[str, str]:
        """Generate a new project task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get context for new project generation
                context = {
                    'github_trending': ai_brain.context.get("github_trending", []),
                    'portfolio_gaps': self._identify_portfolio_gaps(self.system_state.get('projects', {})),
                    'projects': self.system_state.get('projects', {})
                }
                
                # Generate AI content
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                title, description = loop.run_until_complete(
                    self.ai_content_generator.generate_new_project_content(context)
                )
                loop.close()
                
                return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for new project, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        trends = ai_brain.context.get("github_trending", [])[:3]
        trend_names = [t.get("title", "trending tech") for t in trends]
        
        project_ideas = [
            ("AI-Powered Task Automation Dashboard", "Create a Laravel React dashboard for visualizing and managing AI-generated tasks with real-time updates"),
            ("Smart Code Review Assistant", "Build a tool that analyzes PRs and provides intelligent code review suggestions"),
            ("Automated Documentation Generator", "Develop a system that generates and maintains project documentation automatically"),
            ("Performance Monitoring Platform", "Create a comprehensive performance monitoring solution for web applications"),
            ("Security Vulnerability Scanner", "Build an automated security scanning tool for codebases")
        ]
        
        import random
        title, base_description = random.choice(project_ideas)
        
        description = f"""{base_description}

## Project Requirements

### Core Features
- User authentication and authorization
- Real-time data visualization
- RESTful API with Laravel
- React TypeScript frontend
- Comprehensive test coverage

### Technology Stack
- Laravel 11+ with Sanctum authentication  
- React 18+ with TypeScript
- Tailwind CSS for styling
- MySQL database
- Redis for caching, queues, and real-time updates
- Laravel Echo Server (local) for WebSockets

### Trending Technologies to Consider
{', '.join(trend_names) if trend_names else 'Current best practices'}

### Development Approach
1. Fork the Laravel React starter kit
2. Set up the development environment
3. Implement core features incrementally
4. Add comprehensive testing
5. Deploy with CI/CD pipeline

Please create this project following Laravel and React best practices."""
        
        return title, description
    
    def _generate_feature_task(self, ai_brain) -> Tuple[str, str]:
        """Generate a feature task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get a random repository for the feature
                repos = list(self.system_state.get('projects', {}).keys())
                if repos:
                    from repository_exclusion import RepositoryExclusion
                    eligible_repos = RepositoryExclusion.filter_excluded_repos(repos)
                    if eligible_repos:
                        import random
                        repository = random.choice(eligible_repos)
                        
                        # Get repository context
                        repo_context = {
                            'basic_info': {
                                'description': self.system_state.get('projects', {}).get(repository, {}).get('description', ''),
                                'language': self.system_state.get('projects', {}).get(repository, {}).get('language', 'Unknown'),
                                'open_issues_count': self.system_state.get('projects', {}).get(repository, {}).get('metrics', {}).get('issues_open', 0)
                            },
                            'technical_stack': self.system_state.get('projects', {}).get(repository, {}).get('topics', []),
                            'issues_analysis': {
                                'recent_issues': []
                            }
                        }
                        
                        # Generate AI content
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        title, description = loop.run_until_complete(
                            self.ai_content_generator.generate_feature_content(repository, repo_context)
                        )
                        loop.close()
                        
                        return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        features = [
            ("Add real-time notifications system", "Implement WebSocket-based real-time notifications"),
            ("Create advanced search functionality", "Build Elasticsearch integration for powerful search"),
            ("Implement data export feature", "Add ability to export data in multiple formats (CSV, JSON, PDF)"),
            ("Add multi-language support", "Implement i18n for supporting multiple languages"),
            ("Create API rate limiting", "Implement sophisticated rate limiting with Redis")
        ]
        
        import random
        title, description = random.choice(features)
        
        full_description = f"""{description}

## Feature Specifications

### User Stories
- As a user, I want to receive real-time updates
- As an admin, I need to monitor system activity
- As a developer, I need clear API documentation

### Technical Requirements
- Follow existing architectural patterns
- Maintain backward compatibility
- Include comprehensive tests
- Update API documentation
- Add performance benchmarks

### Implementation Notes
Please ensure this feature integrates seamlessly with the existing codebase."""
        
        return title, full_description
    
    def _generate_bug_fix_task(self) -> Tuple[str, str]:
        """Generate a bug fix task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get a repository for bug fix
                repos = list(self.system_state.get('projects', {}).keys())
                if repos:
                    from repository_exclusion import RepositoryExclusion
                    eligible_repos = RepositoryExclusion.filter_excluded_repos(repos)
                    if eligible_repos:
                        import random
                        repository = random.choice(eligible_repos)
                        
                        # Get repository context
                        repo_data = self.system_state.get('projects', {}).get(repository, {})
                        repo_context = {
                            'recent_activity': {
                                'recent_commits': repo_data.get('recent_activity', {}).get('recent_commits', []),
                                'last_commit_date': repo_data.get('recent_activity', {}).get('last_commit_date', 'Unknown')
                            },
                            'issues_analysis': {
                                'bug_issues': []
                            }
                        }
                        
                        # Generate AI content
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        title, description = loop.run_until_complete(
                            self.ai_content_generator.generate_bug_fix_content(repository, repo_context)
                        )
                        loop.close()
                        
                        return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for bug fix, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        bugs = [
            ("Fix memory leak in data processing", "Memory usage grows unbounded during large data imports"),
            ("Resolve race condition in concurrent updates", "Database updates can fail under high concurrency"),
            ("Fix authentication token refresh issue", "JWT tokens not refreshing properly in some scenarios"),
            ("Correct timezone handling errors", "Dates showing incorrectly for users in different timezones"),
            ("Fix file upload validation bypass", "Certain file types bypassing validation checks")
        ]
        
        import random
        title, description = random.choice(bugs)
        
        full_description = f"""Bug Report: {description}

## Steps to Reproduce
1. [Detailed steps will be provided after investigation]
2. Observe the incorrect behavior
3. Check logs for errors

## Expected Behavior
System should handle this scenario gracefully without errors.

## Investigation Required
- Identify root cause
- Check for similar issues elsewhere
- Review recent changes that might have introduced this

## Fix Requirements
- Resolve the core issue
- Add regression tests
- Verify fix doesn't impact other features
- Update documentation if needed"""
        
        return title, full_description
    
    def _generate_documentation_task(self) -> Tuple[str, str]:
        """Generate a documentation task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get a repository that needs documentation
                repository = self._find_repository_needing_documentation()
                if repository:
                    # Get repository context
                    repo_data = self.system_state.get('projects', {}).get(repository, {})
                    repo_context = {
                        'basic_info': {
                            'description': repo_data.get('description', ''),
                            'language': repo_data.get('language', 'Unknown')
                        },
                        'technical_stack': repo_data.get('topics', []),
                        'documentation_status': {
                            'has_readme': True,  # Assume basic README exists
                            'needs_improvement': True
                        }
                    }
                    
                    # Generate AI content
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    title, description = loop.run_until_complete(
                        self.ai_content_generator.generate_documentation_content(repository, repo_context)
                    )
                    loop.close()
                    
                    return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for documentation, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        docs = [
            ("Create API documentation", "Document all REST API endpoints with examples"),
            ("Write deployment guide", "Create comprehensive deployment documentation"),
            ("Document architecture decisions", "Create ADR (Architecture Decision Records)"),
            ("Create user guide", "Write end-user documentation with screenshots"),
            ("Document testing strategy", "Explain testing approach and how to write tests")
        ]
        
        import random
        title, description = random.choice(docs)
        
        full_description = f"""{description}

## Documentation Requirements

### Content Structure
- Overview and introduction
- Detailed explanations
- Code examples
- Best practices
- Troubleshooting guide

### Format Requirements
- Markdown format
- Clear headings and structure
- Include diagrams where helpful
- Provide real-world examples
- Keep it up-to-date

### Target Audience
- Developers (internal and external)
- System administrators
- End users (where applicable)"""
        
        return title, full_description
    
    def _generate_testing_task(self) -> Tuple[str, str]:
        """Generate a testing task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get a repository that needs testing
                repository = self._find_repository_needing_testing()
                if repository:
                    # Get repository context
                    repo_data = self.system_state.get('projects', {}).get(repository, {})
                    repo_context = {
                        'basic_info': {
                            'language': repo_data.get('language', 'Unknown')
                        },
                        'technical_stack': repo_data.get('topics', []),
                        'test_coverage': {
                            'current': 'unknown',
                            'needs_improvement': True
                        }
                    }
                    
                    # Generate AI content
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    title, description = loop.run_until_complete(
                        self.ai_content_generator.generate_testing_content(repository, repo_context)
                    )
                    loop.close()
                    
                    return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for testing, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        tests = [
            ("Add integration tests for API endpoints", "Create comprehensive API integration tests"),
            ("Implement E2E tests for critical user flows", "Add Cypress tests for main user journeys"),
            ("Create performance benchmarks", "Establish performance baselines and tests"),
            ("Add security test suite", "Implement automated security testing"),
            ("Increase unit test coverage to 90%", "Fill gaps in unit test coverage")
        ]
        
        import random
        title, description = random.choice(tests)
        
        full_description = f"""{description}

## Testing Requirements

### Test Scenarios
- Happy path scenarios
- Error conditions
- Edge cases
- Performance under load
- Security vulnerabilities

### Testing Standards
- Follow AAA pattern (Arrange, Act, Assert)
- Use meaningful test names
- Include both positive and negative tests
- Mock external dependencies
- Ensure tests are deterministic

### Coverage Goals
- Minimum 80% code coverage
- All critical paths tested
- Integration with CI/CD pipeline"""
        
        return title, full_description
    
    def _generate_security_task(self) -> Tuple[str, str]:
        """Generate a security task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Get a repository that needs security review
                repository = self._find_repository_needing_security_review()
                if repository:
                    # Get repository context
                    repo_data = self.system_state.get('projects', {}).get(repository, {})
                    repo_context = {
                        'technical_stack': repo_data.get('topics', []),
                        'security_analysis': {
                            'last_review': 'unknown',
                            'known_issues': []
                        }
                    }
                    
                    # Generate AI content
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    title, description = loop.run_until_complete(
                        self.ai_content_generator.generate_security_content(repository, repo_context)
                    )
                    loop.close()
                    
                    return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for security, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        security = [
            ("Conduct security audit of authentication", "Review and harden authentication system"),
            ("Implement CSRF protection", "Add comprehensive CSRF protection across the application"),
            ("Add input validation layer", "Implement robust input validation and sanitization"),
            ("Review and update dependencies", "Audit and update dependencies for security patches"),
            ("Implement security headers", "Add and configure security headers for protection")
        ]
        
        import random
        title, description = random.choice(security)
        
        full_description = f"""{description}

## Security Requirements

### Audit Scope
- Review current implementation
- Identify vulnerabilities
- Check against OWASP Top 10
- Review security best practices
- Test attack scenarios

### Implementation Requirements
- Fix identified vulnerabilities
- Add security tests
- Document security measures
- Consider performance impact
- Maintain usability

### Compliance
- Follow security best practices
- Document any trade-offs
- Include security testing in CI/CD"""
        
        return title, full_description
    
    def _generate_review_task(self, target: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """Generate a code review task using AI content generation.
        
        Returns:
            Title and description tuple
        """
        # If AI content generator is available, use it
        if self.ai_content_generator:
            try:
                import asyncio
                # Generate AI content
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                title, description = loop.run_until_complete(
                    self.ai_content_generator.generate_review_content(target, self.repo_name)
                )
                loop.close()
                
                return title, description
            except Exception as e:
                self.logger.warning(f"AI content generation failed for review, falling back to template: {e}")
        
        # Fallback to template if AI generation fails
        if target:
            title = f"Review stale issue #{target['number']}: {target['title']}"
            description = f"""Please review the stale issue that has been inactive for {target['days_stale']} days.

## Review Requirements
- Check current status and blockers
- Determine if issue is still relevant
- Provide recommendations for moving forward
- Update issue with findings"""
        else:
            title = "Review recent pull requests"
            description = """Please review recent pull requests for code quality and best practices.

## Review Focus Areas
- Code quality and style
- Test coverage
- Security implications
- Performance considerations
- Documentation completeness"""
        
        return title, description
    
    def _calculate_task_complexity(self, task_type: TaskType, description: str = "", repository: str = None) -> Dict[str, Any]:
        """Calculate task complexity using dependencies/sequences model for 24/7 AI operation.
        
        Args:
            task_type: Type of task
            description: Task description for complexity analysis
            repository: Target repository for context
            
        Returns:
            Task complexity with dependencies, sequences, and AI processing cycles
        """
        # Base complexity patterns for different task types
        complexity_patterns = {
            TaskType.NEW_PROJECT: {
                'sequence_steps': [
                    'Repository setup and initialization',
                    'Laravel backend configuration', 
                    'React frontend setup',
                    'Database schema design',
                    'Authentication implementation',
                    'Core feature development',
                    'Testing and validation',
                    'Documentation and deployment'
                ],
                'parallel_opportunities': [
                    ['Database schema design', 'Frontend component planning'],
                    ['Backend API development', 'Frontend UI development'],
                    ['Unit testing', 'Integration testing', 'Documentation']
                ],
                'estimated_ai_cycles': 8,
                'complexity_score': 0.9
            },
            TaskType.FEATURE: {
                'sequence_steps': [
                    'Requirements analysis',
                    'Design and architecture',
                    'Implementation',
                    'Testing',
                    'Integration',
                    'Documentation'
                ],
                'parallel_opportunities': [
                    ['Backend implementation', 'Frontend implementation'],
                    ['Unit tests', 'Integration tests']
                ],
                'estimated_ai_cycles': 4,
                'complexity_score': 0.6
            },
            TaskType.BUG_FIX: {
                'sequence_steps': [
                    'Issue reproduction',
                    'Root cause analysis',
                    'Fix implementation',
                    'Regression testing',
                    'Verification'
                ],
                'parallel_opportunities': [
                    ['Fix implementation', 'Test case creation']
                ],
                'estimated_ai_cycles': 2,
                'complexity_score': 0.4
            },
            TaskType.REFACTOR: {
                'sequence_steps': [
                    'Code analysis',
                    'Refactoring plan',
                    'Implementation',
                    'Testing validation',
                    'Performance verification'
                ],
                'parallel_opportunities': [
                    ['Code refactoring', 'Test updates']
                ],
                'estimated_ai_cycles': 3,
                'complexity_score': 0.5
            },
            TaskType.DOCUMENTATION: {
                'sequence_steps': [
                    'Content analysis',
                    'Documentation structure',
                    'Content creation',
                    'Review and validation'
                ],
                'parallel_opportunities': [
                    ['Content creation', 'Example preparation']
                ],
                'estimated_ai_cycles': 2,
                'complexity_score': 0.3
            },
            TaskType.TESTING: {
                'sequence_steps': [
                    'Test planning',
                    'Test implementation',
                    'Test execution',
                    'Result analysis'
                ],
                'parallel_opportunities': [
                    ['Unit tests', 'Integration tests', 'E2E tests']
                ],
                'estimated_ai_cycles': 3,
                'complexity_score': 0.5
            },
            TaskType.SECURITY: {
                'sequence_steps': [
                    'Security audit',
                    'Vulnerability assessment',
                    'Fix implementation',
                    'Security testing',
                    'Verification'
                ],
                'parallel_opportunities': [
                    ['Code scanning', 'Dependency analysis'],
                    ['Implementation', 'Documentation']
                ],
                'estimated_ai_cycles': 4,
                'complexity_score': 0.7
            },
            TaskType.PERFORMANCE: {
                'sequence_steps': [
                    'Performance baseline',
                    'Bottleneck identification',
                    'Optimization implementation',
                    'Performance testing',
                    'Verification'
                ],
                'parallel_opportunities': [
                    ['Code optimization', 'Database optimization'],
                    ['Performance testing', 'Load testing']
                ],
                'estimated_ai_cycles': 3,
                'complexity_score': 0.6
            },
            TaskType.CODE_REVIEW: {
                'sequence_steps': [
                    'Code analysis',
                    'Review and feedback',
                    'Approval or revision'
                ],
                'parallel_opportunities': [
                    ['Security review', 'Performance review', 'Style review']
                ],
                'estimated_ai_cycles': 1,
                'complexity_score': 0.2
            },
            TaskType.DEPENDENCY_UPDATE: {
                'sequence_steps': [
                    'Dependency analysis',
                    'Update planning',
                    'Implementation',
                    'Testing',
                    'Verification'
                ],
                'parallel_opportunities': [
                    ['Dependency updates', 'Test updates']
                ],
                'estimated_ai_cycles': 2,
                'complexity_score': 0.4
            }
        }
        
        base_complexity = complexity_patterns.get(task_type, {
            'sequence_steps': ['Analysis', 'Implementation', 'Testing'],
            'parallel_opportunities': [['Implementation', 'Documentation']],
            'estimated_ai_cycles': 2,
            'complexity_score': 0.4
        })
        
        # Adjust complexity based on description keywords
        complexity_modifiers = {
            'ai': 0.2, 'machine learning': 0.3, 'complex': 0.2, 'integration': 0.15,
            'multiple': 0.1, 'advanced': 0.15, 'comprehensive': 0.2, 'real-time': 0.2,
            'scalable': 0.15, 'distributed': 0.25, 'microservice': 0.2
        }
        
        description_lower = description.lower()
        complexity_adjustment = sum(
            modifier for keyword, modifier in complexity_modifiers.items()
            if keyword in description_lower
        )
        
        # Repository-specific adjustments
        repo_adjustment = 0.0
        if repository:
            # Larger/more complex repositories increase complexity
            if 'cms' in repository.lower():
                repo_adjustment += 0.1
            if 'ai' in repository.lower():
                repo_adjustment += 0.15
            if 'studio' in repository.lower():
                repo_adjustment += 0.1
        
        final_complexity_score = min(1.0, base_complexity['complexity_score'] + complexity_adjustment + repo_adjustment)
        
        # Adjust AI cycles based on final complexity
        cycle_multiplier = 1.0 + (final_complexity_score - base_complexity['complexity_score'])
        estimated_cycles = max(1, int(base_complexity['estimated_ai_cycles'] * cycle_multiplier))
        
        return {
            'sequence_steps': base_complexity['sequence_steps'],
            'parallel_opportunities': base_complexity['parallel_opportunities'],
            'blocking_dependencies': [],  # Will be populated based on actual dependencies
            'estimated_ai_cycles': estimated_cycles,
            'complexity_score': final_complexity_score,
            'can_parallelize': len(base_complexity['parallel_opportunities']) > 0,
            'repository_context': repository,
            'description_keywords': [kw for kw in complexity_modifiers.keys() if kw in description_lower]
        }
    
    def review_completed_tasks(self) -> Dict[str, Any]:
        """Review tasks marked as completed by @claude.
        
        Returns:
            Review results
        """
        if not self.repo:
            return {"error": "GitHub repository not available"}
        
        reviews = {
            "reviewed": 0,
            "approved": 0,
            "needs_revision": 0,
            "tasks": []
        }
        
        try:
            # Get issues labeled as awaiting-review
            issues = self.repo.get_issues(state="open", labels=["ai-managed", "awaiting-review"])
            
            for issue in issues:
                reviews["reviewed"] += 1
                
                # Analyze the issue and PR
                review_result = self._review_task_completion(issue)
                
                if review_result["approved"]:
                    reviews["approved"] += 1
                    # Close the issue
                    issue.edit(state="closed")
                    issue.create_comment("✅ Task completed successfully. Great work!")
                else:
                    reviews["needs_revision"] += 1
                    # Request changes
                    issue.remove_from_labels("awaiting-review")
                    issue.add_to_labels("needs-revision")
                    issue.create_comment(f"🔄 Task needs revision:\n\n{review_result['feedback']}")
                
                reviews["tasks"].append({
                    "issue_number": issue.number,
                    "title": issue.title,
                    "result": "approved" if review_result["approved"] else "needs_revision",
                    "feedback": review_result.get("feedback", "")
                })
        
        except Exception as e:
            reviews["error"] = str(e)
        
        return reviews
    
    def _review_task_completion(self, issue) -> Dict[str, Any]:
        """Review a specific task completion.
        
        Args:
            issue: GitHub issue object
            
        Returns:
            Review result
        """
        # Check for linked PRs
        linked_prs = []
        for event in issue.get_events():
            if event.event == "cross-referenced":
                # This is a simple check - in production, you'd want more sophisticated PR detection
                linked_prs.append(event)
        
        # Basic review criteria
        has_pr = len(linked_prs) > 0
        has_tests = any("test" in comment.body.lower() for comment in issue.get_comments())
        has_documentation = any("doc" in comment.body.lower() for comment in issue.get_comments())
        
        # Simple approval logic - in production, this would be more sophisticated
        approved = has_pr
        
        feedback = []
        if not has_pr:
            feedback.append("- No linked pull request found")
        if not has_tests:
            feedback.append("- Consider adding tests")
        if not has_documentation:
            feedback.append("- Consider updating documentation")
        
        return {
            "approved": approved,
            "feedback": "\n".join(feedback) if feedback else "All requirements met"
        }
    
    def prioritize_tasks(self) -> None:
        """Update task priorities based on current state."""
        analysis = self.analyze_existing_tasks()
        
        # Re-prioritize based on various factors
        for task_id, task in self.state["tasks"].items():
            if task["status"] in [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]:
                continue
            
            # Increase priority for blocked tasks
            if task["dependencies"]:
                deps_completed = all(
                    self.state["tasks"].get(dep_id, {}).get("status") == TaskStatus.COMPLETED.value
                    for dep_id in task["dependencies"]
                )
                if deps_completed and task["priority"] != TaskPriority.CRITICAL.value:
                    task["priority"] = TaskPriority.HIGH.value
            
            # Increase priority for old tasks
            created_date = datetime.fromisoformat(task["created_at"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_date).days
            
            if age_days > 7 and task["priority"] == TaskPriority.LOW.value:
                task["priority"] = TaskPriority.MEDIUM.value
            elif age_days > 14 and task["priority"] == TaskPriority.MEDIUM.value:
                task["priority"] = TaskPriority.HIGH.value
        
        self._save_state()
    
    def generate_report(self) -> str:
        """Generate a task management report.
        
        Returns:
            Markdown formatted report
        """
        analysis = self.analyze_existing_tasks()
        
        report = f"""# Task Management Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview
- **Total Active Tasks**: {analysis.get('open_issues', 0)}
- **Assigned to @claude**: {analysis.get('claude_assigned', 0)}
- **In Progress**: {analysis.get('in_progress', 0)}
- **Awaiting Review**: {analysis.get('awaiting_review', 0)}
- **Completed Today**: {analysis.get('completed_recently', 0)}

## Task Distribution

### By Type
"""
        
        for task_type, count in analysis.get("by_type", {}).items():
            report += f"- **{task_type}**: {count}\n"
        
        report += "\n### By Priority\n"
        for priority, count in analysis.get("by_priority", {}).items():
            report += f"- **{priority}**: {count}\n"
        
        if analysis.get("stale_tasks"):
            report += "\n## ⚠️ Stale Tasks\n"
            for stale in analysis["stale_tasks"][:5]:
                report += f"- #{stale['number']}: {stale['title']} ({stale['days_stale']} days)\n"
        
        report += f"""
## Performance Metrics
- **Success Rate**: {self.state.get('success_rate', 0):.1%}
- **Active Tasks**: {self.state.get('active_tasks', 0)}
- **Completed Today**: {self.state.get('completed_today', 0)}

## Recommendations
"""
        
        if analysis.get('open_issues', 0) > 15:
            report += "- ⚠️ High number of open tasks - focus on completion\n"
        
        if analysis.get('stale_tasks', []):
            report += "- 🔄 Review and update stale tasks\n"
        
        if analysis.get('awaiting_review', 0) > 3:
            report += "- 👀 Multiple tasks awaiting review - prioritize reviews\n"
        
        with open("task_report.md", "w") as f:
            f.write(report)
        
        return report
    
    def _add_history_entry(self, action: str, task_id: str, details: Dict[str, Any]) -> None:
        """Add entry to task history.
        
        Args:
            action: Action type
            task_id: Task ID
            details: Additional details
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "task_id": task_id,
            "details": details
        }
        
        self.history.append(entry)
        self._save_history()
    
    def analyze_task_relationships(self, task_id: str) -> None:
        """Analyze and update relationships for a task.
        
        Args:
            task_id: Task ID to analyze
        """
        if task_id not in self.state["tasks"]:
            return
            
        task = self.state["tasks"][task_id]
        
        # Find tasks that might be impacted
        upstream_impacts = []
        cross_project_relevance = {}
        
        for other_id, other_task in self.state["tasks"].items():
            if other_id == task_id:
                continue
                
            # Skip completed tasks
            if other_task.get("status") in ["completed", "cancelled"]:
                continue
            
            # Check if this task impacts others
            if task.get("repository") == other_task.get("repository"):
                # Same project - check for feature overlap
                if self._tasks_overlap(task, other_task):
                    upstream_impacts.append(other_id)
            else:
                # Different projects - check cross-project relevance
                if task.get("type") in ["INFRASTRUCTURE", "ENHANCEMENT"]:
                    other_project = other_task.get("repository")
                    if other_project and other_project not in cross_project_relevance:
                        cross_project_relevance[other_project] = f"May benefit from {task['type'].lower()}"
        
        # Update relationships
        if "relationships" not in task:
            task["relationships"] = {}
            
        task["relationships"]["upstream_impacts"] = upstream_impacts
        task["relationships"]["cross_project_relevance"] = cross_project_relevance
        
        # Identify potential knowledge artifacts
        if task.get("type") in ["FEATURE", "NEW_PROJECT", "ENHANCEMENT"]:
            task["relationships"]["knowledge_artifacts"] = [
                f"{task.get('type', 'task').lower()}-pattern.md",
                f"implementation-guide-{task_id}.md"
            ]
        
        self._save_state()
    
    def _tasks_overlap(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> bool:
        """Check if two tasks have overlapping scope.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            True if tasks overlap
        """
        # Check title similarity
        if self._calculate_similarity(task1.get("title", ""), task2.get("title", "")) > 0.6:
            return True
            
        # Check if they modify same area (based on keywords)
        keywords1 = set(task1.get("title", "").lower().split() + 
                       task1.get("description", "").lower().split()[:20])
        keywords2 = set(task2.get("title", "").lower().split() + 
                       task2.get("description", "").lower().split()[:20])
        
        common_keywords = keywords1.intersection(keywords2)
        if len(common_keywords) > 5:
            return True
            
        return False
    
    def set_performance_baseline(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """Set performance baseline metrics for a task.
        
        Args:
            task_id: Task ID
            metrics: Baseline metrics dict
        """
        if task_id not in self.state["tasks"]:
            return
            
        task = self.state["tasks"][task_id]
        if "performance_tracking" not in task:
            task["performance_tracking"] = {}
            
        task["performance_tracking"]["performance_baseline"] = metrics
        task["performance_tracking"]["baseline_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Auto-generate targets based on task type
        if task.get("type") == "ENHANCEMENT":
            task["performance_tracking"]["performance_target"] = {
                "improvement_percentage": 20,
                "metric_targets": self._generate_performance_targets(task, metrics)
            }
        
        self._save_state()
    
    def record_performance_outcome(self, task_id: str, final_metrics: Dict[str, Any]) -> None:
        """Record actual performance outcome after task completion.
        
        Args:
            task_id: Task ID
            final_metrics: Final metrics after task
        """
        if task_id not in self.state["tasks"]:
            return
            
        task = self.state["tasks"][task_id]
        if "performance_tracking" not in task:
            task["performance_tracking"] = {}
            
        baseline = task["performance_tracking"].get("performance_baseline", {})
        
        # Calculate deltas
        delta = {}
        for key, final_value in final_metrics.items():
            if key in baseline:
                baseline_value = baseline[key]
                if isinstance(baseline_value, (int, float)) and isinstance(final_value, (int, float)):
                    delta[key] = {
                        "absolute_change": final_value - baseline_value,
                        "percentage_change": ((final_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0
                    }
                else:
                    delta[key] = {"changed": baseline_value != final_value}
        
        task["performance_tracking"]["actual_performance_delta"] = delta
        task["performance_tracking"]["outcome_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Identify reusable components based on success
        if self._is_performance_successful(task):
            task["performance_tracking"]["reusable_components"] = self._extract_reusable_components(task)
        
        self._save_state()
    
    def _generate_performance_targets(self, task: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance targets based on task type and baseline.
        
        Args:
            task: Task dict
            baseline: Baseline metrics
            
        Returns:
            Target metrics
        """
        targets = {}
        
        # Generate targets based on common metrics
        for key, value in baseline.items():
            if isinstance(value, (int, float)):
                if "time" in key.lower() or "duration" in key.lower():
                    targets[key] = value * 0.8  # 20% faster
                elif "error" in key.lower() or "bug" in key.lower():
                    targets[key] = value * 0.5  # 50% fewer errors
                elif "coverage" in key.lower() or "score" in key.lower():
                    targets[key] = min(value * 1.2, 100)  # 20% better, max 100
                else:
                    targets[key] = value * 1.1  # 10% improvement
        
        return targets
    
    def _is_performance_successful(self, task: Dict[str, Any]) -> bool:
        """Check if task achieved performance targets.
        
        Args:
            task: Task dict
            
        Returns:
            True if successful
        """
        perf = task.get("performance_tracking", {})
        delta = perf.get("actual_performance_delta", {})
        
        # Consider successful if any metric improved significantly
        for metric, change in delta.items():
            if isinstance(change, dict) and change.get("percentage_change", 0) > 10:
                return True
                
        return False
    
    def _extract_reusable_components(self, task: Dict[str, Any]) -> List[str]:
        """Extract reusable components from successful task.
        
        Args:
            task: Task dict
            
        Returns:
            List of reusable components
        """
        components = []
        
        # Based on task type
        if task.get("type") == "FEATURE":
            components.extend([
                f"{task['title'].replace(' ', '_').lower()}_implementation",
                f"{task['repository']}_feature_pattern"
            ])
        elif task.get("type") == "ENHANCEMENT":
            components.extend([
                f"performance_optimization_pattern",
                f"{task['title'].replace(' ', '_').lower()}_approach"
            ])
        
        return components


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="AI Task Manager")
    parser.add_argument("command", choices=["generate", "review", "prioritize", "report", "analyze"],
                       help="Command to execute")
    parser.add_argument("--focus", default="auto",
                       help="Focus area for task generation")
    parser.add_argument("--max-tasks", type=int, default=5,
                       help="Maximum tasks to generate")
    
    args = parser.parse_args()
    
    # Initialize task manager
    manager = TaskManager()
    
    if args.command == "generate":
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Generating tasks with focus: {args.focus}")
        tasks = manager.generate_tasks(focus=args.focus, max_tasks=args.max_tasks)
        logger.info(f"Generated {len(tasks)} tasks")
        for task in tasks:
            logger.info(f"- {task['id']}: {task['title']}")
    
    elif args.command == "review":
        logger.info("Reviewing completed tasks...")
        results = manager.review_completed_tasks()
        logger.info(f"Reviewed {results.get('reviewed', 0)} tasks")
        logger.info(f"- Approved: {results.get('approved', 0)}")
        logger.info(f"- Needs revision: {results.get('needs_revision', 0)}")
    
    elif args.command == "prioritize":
        logger.info("Updating task priorities...")
        manager.prioritize_tasks()
        logger.info("Task priorities updated")
    
    elif args.command == "report":
        logger.info("Generating task report...")
        report = manager.generate_report()
        logger.info("Report generated: task_report.md")
    
    elif args.command == "analyze":
        logger.info("Analyzing existing tasks...")
        analysis = manager.analyze_existing_tasks()
        logger.info(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()