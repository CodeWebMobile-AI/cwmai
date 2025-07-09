"""
Intelligent Work Finder for Continuous 24/7 AI Operation - FIXED VERSION

Fixed issues:
1. No infinite research loops - uses cached problems immediately
2. Creates projects incrementally as workers become available
3. Doesn't wait for 10 problems before starting
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import uuid
import json
import os

from scripts.work_item_types import WorkItem, TaskPriority, WorkOpportunity
from scripts.ai_brain import IntelligentAIBrain
from scripts.repository_exclusion import RepositoryExclusion
from scripts.unified_task_state import UnifiedTaskState
from scripts.market_research_engine import MarketResearchEngine, ProjectOpportunity
from scripts.portfolio_intelligence import PortfolioIntelligence
from scripts.project_lifecycle_analyzer import ProjectLifecycleAnalyzer
from scripts.task_types import (
    SmartTaskType, TaskTypeSelector, get_task_type_for_string,
    ArchitectureType, TaskCategory
)


@dataclass
class CachedProblem:
    """A cached market problem ready for project creation."""
    id: str
    name: str
    problem: str
    validation_score: float
    features: List[str]
    monetization: str
    target_audience: str
    research_ref: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'problem': self.problem,
            'validation_score': self.validation_score,
            'features': self.features,
            'monetization': self.monetization,
            'target_audience': self.target_audience,
            'research_ref': self.research_ref,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedProblem':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ResearchCache:
    """Efficient cache for validated market problems."""
    
    def __init__(self, cache_file: str = "research_cache.json", system_state: Optional[Dict[str, Any]] = None):
        """Initialize the research cache.
        
        Args:
            cache_file: Path to cache file
            system_state: System state containing existing projects
        """
        self.cache_file = cache_file
        self.validated_problems: List[CachedProblem] = []
        self.last_research_time: Optional[datetime] = None
        self.research_interval = timedelta(hours=6)
        self.min_cache_size = 5
        self.max_cache_size = 20
        self.logger = logging.getLogger(__name__)
        self.system_state = system_state or {}
        self._research_in_progress = False  # Prevent multiple concurrent research
        
        # Load existing cache
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.validated_problems = [
                        CachedProblem.from_dict(p) for p in data.get('problems', [])
                    ]
                    if data.get('last_research_time'):
                        self.last_research_time = datetime.fromisoformat(data['last_research_time'])
                    self.logger.info(f"Loaded {len(self.validated_problems)} cached problems")
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            self.validated_problems = []
            
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {
                'problems': [p.to_dict() for p in self.validated_problems],
                'last_research_time': self.last_research_time.isoformat() if self.last_research_time else None
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache: {e}")
            
    def _needs_background_research(self) -> bool:
        """Check if we should run background research (not blocking)."""
        # Only run if cache is VERY low and not already researching
        if self._research_in_progress:
            return False
            
        # Only trigger if we have less than 3 problems
        if len(self.validated_problems) < 3:
            return True
            
        # Or if it's been a long time since last research
        if self.last_research_time:
            time_since_research = datetime.now(timezone.utc) - self.last_research_time
            if time_since_research > timedelta(hours=12):  # Increased from 6 hours
                return True
                
        return False
        
    def _generate_simple_name(self, problem: Any) -> str:
        """Generate a simple, deterministic project name."""
        # Extract domain from affected users
        domain = problem.affected_users.lower().replace(' ', '-')
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m')
        
        # Create simple name
        return f"{domain}-solution-{timestamp}"
        
    async def batch_research(self, count: int = 10) -> List[CachedProblem]:
        """Run batch research to fill the cache."""
        if self._research_in_progress:
            self.logger.info("Research already in progress, skipping")
            return []
            
        from scripts.market_discovery import MarketDiscovery
        
        self._research_in_progress = True
        try:
            self.logger.info(f"Running background research for {count} problems...")
            # Pass system state to market discovery
            market_discovery = MarketDiscovery(self.system_state)
            
            # Get multiple problems at once - efficient use of API
            problems = await market_discovery.discover_problems(count=count)
            
            cached_problems = []
            for i, problem in enumerate(problems):
                # Extract key features efficiently
                features = []
                if hasattr(problem, '_research_phases') and problem._research_phases:
                    phases = problem._research_phases
                    # Look in competition analysis
                    if 'phase_3_competition_analysis' in phases:
                        comp_data = phases['phase_3_competition_analysis']
                        if isinstance(comp_data, list) and comp_data:
                            features = comp_data[0].get('key_features', [])[:5]  # Top 5 features
                
                # Get monetization model
                monetization = 'subscription'  # Default
                if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    monetization = problem._full_opportunity.monetization_model or 'subscription'
                
                cached = CachedProblem(
                    id=f"prob_{datetime.now().strftime('%Y%m%d')}_{i:03d}",
                    name=self._generate_simple_name(problem),
                    problem=problem.problem_statement,
                    validation_score=problem.opportunity_score,
                    features=features,
                    monetization=monetization,
                    target_audience=problem.affected_users,  # Capture the actual affected users
                    research_ref=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                )
                cached_problems.append(cached)
                
            # Add to cache (keeping within max size)
            self.validated_problems.extend(cached_problems)
            if len(self.validated_problems) > self.max_cache_size:
                # Keep the newest problems
                self.validated_problems = self.validated_problems[-self.max_cache_size:]
                
            self.last_research_time = datetime.now(timezone.utc)
            self._save_cache()
            
            self.logger.info(f"Added {len(cached_problems)} problems to cache. Total cached: {len(self.validated_problems)}")
            return cached_problems
            
        finally:
            self._research_in_progress = False
            
    def get_cached_problems(self, limit: int = 1) -> List[CachedProblem]:
        """Get problems from cache WITHOUT triggering research.
        
        This is the key fix - just return what we have, don't research.
        """
        available = self.validated_problems[:limit]
        
        # Remove the returned problems from cache
        if available:
            self.validated_problems = self.validated_problems[len(available):]
            self._save_cache()
            
        return available
        
    def has_cached_problems(self) -> bool:
        """Check if we have any cached problems."""
        return len(self.validated_problems) > 0
        
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self.validated_problems)
        
    async def trigger_background_research(self):
        """Trigger background research if needed - non-blocking."""
        if self._needs_background_research():
            # Run research in background without blocking
            asyncio.create_task(self.batch_research(count=5))


class IntelligentWorkFinder:
    """Intelligent work discovery system for continuous operation."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, system_state: Dict[str, Any]):
        """Initialize the work finder.
        
        Args:
            ai_brain: AI brain for intelligent analysis
            system_state: Current system state
        """
        self.ai_brain = ai_brain
        self.system_state = system_state
        self.logger = logging.getLogger(__name__)
        
        # Work discovery history
        self.discovered_work: List[WorkOpportunity] = []
        self.work_sources_checked: Dict[str, datetime] = {}
        self.last_portfolio_analysis: Optional[datetime] = None
        
        # Unified task state for deduplication
        self.unified_task_state = UnifiedTaskState()
        
        # Initialize intelligent components
        self.market_research = MarketResearchEngine(ai_brain)
        self.portfolio_intelligence = PortfolioIntelligence(ai_brain)
        self.lifecycle_analyzer = ProjectLifecycleAnalyzer(ai_brain)
        
        # Configuration
        self.discovery_interval = 30  # Check for new work every 30 seconds
        self.portfolio_analysis_interval = 300  # Deep analysis every 5 minutes
        self.max_work_per_discovery = 5
        
        # Track completed tasks per repository
        self.completed_tasks_by_repo: Dict[str, Set[str]] = {}
        
        # Initialize research cache for efficient problem retrieval
        self.research_cache = ResearchCache(system_state=self.system_state)
        
    async def discover_work(self, max_items: int = 5, current_workload: int = 0) -> List[WorkItem]:
        """Discover new work opportunities.
        
        FIXED: Now creates work items incrementally from cache without triggering infinite research.
        
        Args:
            max_items: Maximum number of work items to return
            current_workload: Number of currently active workers
            
        Returns:
            List of discovered work items
        """
        self.logger.info(f"Discovering work (max: {max_items}, current load: {current_workload})")
        
        # Trigger background research if cache is low (non-blocking)
        await self.research_cache.trigger_background_research()
        
        # Check if there are any active projects in the system
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        from scripts.repository_exclusion import is_excluded_repo
        active_projects = {
            name: data for name, data in all_projects.items()
            if not is_excluded_repo(name)
        }
        
        # If no active projects exist, create NEW_PROJECT tasks from cache
        if not active_projects:
            self.logger.warning("⚠️ No active projects found. Creating projects from cache.")
            
            # Check cache size
            cache_size = self.research_cache.cache_size()
            self.logger.info(f"📦 Research cache has {cache_size} problems ready")
            
            # Get problems from cache (1-2 at a time for pipeline workflow)
            problems_to_use = min(2, max_items, cache_size)  # Max 2 projects at once
            cached_problems = self.research_cache.get_cached_problems(limit=problems_to_use)
            
            if not cached_problems:
                self.logger.warning("No cached problems available. Background research will run.")
                return []
            
            # Convert cached problems to work items
            work_items = []
            for problem in cached_problems:
                work_opportunity = WorkOpportunity(
                    source="research_cache",
                    type="NEW_PROJECT",
                    priority=TaskPriority.HIGH,
                    title=problem.name,
                    description=problem.problem,
                    repository=None,
                    estimated_cycles=5,
                    metadata={
                        'project_type': 'Market Solution',
                        'starter_kit': 'laravel/react-starter-kit',
                        'detailed_description': problem.problem,
                        'tech_stack': ['Laravel', 'React', 'TypeScript', 'MySQL', 'Redis'],
                        'selected_project': {
                            'project_name': problem.name,
                            'project_goal': problem.problem,
                            'problem_solved': problem.problem,
                            'target_audience': problem.target_audience,
                            'monetization_strategy': problem.monetization,
                            'key_features': problem.features
                        },
                        'architecture': {},  # Will be generated during creation
                        'validation_score': problem.validation_score,
                        'research_ref': problem.research_ref
                    }
                )
                
                work_item = work_opportunity.to_work_item()
                is_dup, _ = self.unified_task_state.is_duplicate_task(work_item)
                if not is_dup:
                    work_items.append(work_item)
            
            self.logger.info(f"📋 Created {len(work_items)} NEW_PROJECT tasks from cache")
            return work_items
        
        # If we have active projects, discover work from them
        opportunities = []
        
        try:
            # 1. Repository-based work discovery
            repo_work = await self._discover_repository_work()
            opportunities.extend(repo_work)
        except Exception as e:
            self.logger.error(f"Error in repository work discovery: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # ONLY add new projects if we have very few active projects
        if len(active_projects) < 3 and self.research_cache.has_cached_problems():
            # Add 1 new project opportunity from cache
            cached_problem = self.research_cache.get_cached_problems(limit=1)
            if cached_problem:
                problem = cached_problem[0]
                opportunities.append(WorkOpportunity(
                    source="research_cache",
                    type="NEW_PROJECT",
                    priority=TaskPriority.MEDIUM,  # Lower priority than repo work
                    title=problem.name,
                    description=problem.problem,
                    repository=None,
                    estimated_cycles=5,
                    metadata={
                        'project_type': 'Market Solution',
                        'starter_kit': 'laravel/react-starter-kit',
                        'detailed_description': problem.problem,
                        'tech_stack': ['Laravel', 'React', 'TypeScript', 'MySQL', 'Redis'],
                        'selected_project': {
                            'project_name': problem.name,
                            'project_goal': problem.problem,
                            'problem_solved': problem.problem,
                            'target_audience': problem.target_audience,
                            'monetization_strategy': problem.monetization,
                            'key_features': problem.features
                        },
                        'architecture': {},
                        'validation_score': problem.validation_score,
                        'research_ref': problem.research_ref
                    }
                ))
        
        # Filter and prioritize
        filtered_work = self._filter_and_prioritize(opportunities, max_items)
        
        # Convert to work items and filter duplicates
        work_items = []
        for opp in filtered_work:
            work_item = opp.to_work_item()
            
            # Check for duplicates using task persistence
            is_dup, _ = self.unified_task_state.is_duplicate_task(work_item)
            if not is_dup:
                work_items.append(work_item)
            else:
                self.logger.debug(f"🔄 Filtered duplicate work: {work_item.title}")
        
        # Log discovery results
        if work_items:
            self.logger.info(f"Discovered {len(work_items)} work items:")
            for item in work_items:
                self.logger.info(f"  - {item.task_type}: {item.title} (priority: {item.priority.name})")
        
        return work_items
    
    # ... (rest of the methods remain the same as original)
    
    def _filter_and_prioritize(self, opportunities: List[WorkOpportunity], max_items: int) -> List[WorkOpportunity]:
        """Filter and prioritize work opportunities."""
        # Remove duplicates
        unique_opportunities = []
        seen_keys = set()
        
        for opp in opportunities:
            key = f"{opp.type}:{opp.title}"
            if key not in seen_keys:
                unique_opportunities.append(opp)
                seen_keys.add(key)
        
        # Sort by priority (HIGH > MEDIUM > LOW > BACKGROUND)
        priority_order = {
            TaskPriority.HIGH: 0,
            TaskPriority.MEDIUM: 1,
            TaskPriority.LOW: 2,
            TaskPriority.BACKGROUND: 3
        }
        
        sorted_opportunities = sorted(
            unique_opportunities,
            key=lambda x: priority_order.get(x.priority, 99)
        )
        
        # Return top N items
        return sorted_opportunities[:max_items]