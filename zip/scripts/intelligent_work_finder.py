"""
Intelligent Work Finder for Continuous 24/7 AI Operation

Continuously discovers new work opportunities to ensure the AI system
never runs out of productive tasks to perform.
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
        
    # Removed _generate_simple_name - creative names must come from research phase
        
    def _extract_features_from_description(self, description: str) -> List[str]:
        """Extract features from opportunity description."""
        if not description:
            return []
            
        features = []
        import re
        
        # Pattern 1: Look for feature-indicating keywords
        feature_patterns = [
            r'(?:features?|capabilities?|includes?|provides?|offers?)[:\s]+([^.]+)',
            r'(?:•|\*|-|\d+\.)\s*([^•\*\-\n]+)',
            r'(?:AI-powered|automated|real-time|integrated|custom|advanced|smart)\s+(\w+(?:\s+\w+){0,3})',
            r'(?:with|using|through)\s+([\w\s]+(?:system|platform|tool|engine|module|interface))',
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                feature = match.strip()
                if 10 < len(feature) < 60 and not any(word in feature.lower() for word in ['the', 'and', 'or', 'but']):
                    features.append(feature.title())
        
        # Extract from specific technology mentions
        tech_patterns = [
            r'(machine learning|AI|blockchain|cloud|API|dashboard|analytics|automation)',
            r'(authentication|security|encryption|compliance)',
            r'(real-time|live|instant|automated|smart)',
            r'(integration|sync|connect|interface)'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                feature_phrase = f"{match.title()} {self._get_feature_suffix(match)}"
                if feature_phrase not in features:
                    features.append(feature_phrase)
        
        # Deduplicate and clean
        seen = set()
        clean_features = []
        for f in features:
            f_clean = ' '.join(f.split())  # Remove extra spaces
            f_lower = f_clean.lower()
            if f_lower not in seen and len(f_clean) > 10:
                seen.add(f_lower)
                clean_features.append(f_clean)
        
        # Return what we found - if empty, the async validation will handle it
        return clean_features[:5]
    
    def _get_feature_suffix(self, tech_term: str) -> str:
        """Get appropriate suffix for technology term."""
        suffixes = {
            'ai': 'Engine',
            'machine learning': 'Models',
            'blockchain': 'Integration',
            'cloud': 'Infrastructure',
            'api': 'Framework',
            'dashboard': 'Interface',
            'analytics': 'System',
            'automation': 'Workflow',
            'authentication': 'System',
            'security': 'Framework',
            'encryption': 'Protocol',
            'real-time': 'Processing',
            'integration': 'Platform'
        }
        return suffixes.get(tech_term.lower(), 'System')
    
    async def _generate_domain_features(self, context: str) -> List[str]:
        """Generate domain-specific features based on problem context using AI."""
        self.logger.info("Using AI to generate context-aware features...")
        
        # Use AI to generate features
        from scripts.http_ai_client import HTTPAIClient
        
        try:
            ai_client = HTTPAIClient()
            
            prompt = f"""You are a product architect. Analyze this context and generate 5 specific, implementable features:

{context}

Requirements:
1. Features must directly solve the stated problem
2. Each feature should be a concrete, buildable component
3. Features should be specific to this industry/domain
4. Each feature name should be 3-5 words
5. Features should represent core functionality, not generic items

Return ONLY a JSON array of exactly 5 feature names. No explanations.

Good example for NFT artists protecting IP:
["Blockchain IP Registry", "Smart Contract Royalties", "Plagiarism Detection Engine", "Automated DMCA System", "Decentralized Rights Management"]

Bad example (too generic):
["User Dashboard", "API Platform", "Database System", "Login Page", "Settings Menu"]
"""
            
            response = await ai_client.generate(prompt, temperature=0.2)  # Lower temp for consistency
            
            # Try multiple parsing strategies
            import json
            import re
            
            # Strategy 1: Direct JSON parsing
            try:
                # Remove markdown code blocks if present
                cleaned = re.sub(r'```json?\s*|\s*```', '', response.strip())
                features = json.loads(cleaned)
                if isinstance(features, list) and len(features) >= 5:
                    valid_features = [str(f).strip() for f in features if f and isinstance(f, str) and len(str(f).strip()) > 5]
                    if len(valid_features) >= 5:
                        self.logger.info(f"Successfully parsed {len(valid_features)} features from JSON")
                        return valid_features[:5]
            except:
                pass
            
            # Strategy 2: Extract from array-like format
            try:
                array_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
                if array_match:
                    items = re.findall(r'"([^"]+)"', array_match.group(1))
                    if len(items) >= 5:
                        self.logger.info(f"Extracted {len(items)} features from array format")
                        return items[:5]
            except:
                pass
            
            # Strategy 3: Line-by-line extraction
            lines = response.strip().split('\n')
            features = []
            for line in lines:
                # Clean common formats
                cleaned = re.sub(r'^[-*•\d.)\s]+', '', line.strip())
                cleaned = cleaned.strip('"\'[]')
                
                # Check if it looks like a feature
                if cleaned and 10 < len(cleaned) < 60 and not any(skip in cleaned.lower() for skip in ['example', 'json', 'array', 'return']):
                    features.append(cleaned)
            
            if len(features) >= 5:
                self.logger.info(f"Extracted {len(features)} features from text format")
                return features[:5]
            
            # Strategy 4: If we have some features but not enough, generate more
            if 0 < len(features) < 5:
                self.logger.warning(f"Only found {len(features)} features, generating additional...")
                # Create a follow-up prompt
                additional_prompt = f"Generate {5 - len(features)} more specific features for: {context[:200]}..."
                additional_response = await ai_client.generate(additional_prompt, temperature=0.3)
                
                # Try to extract more
                more_features = []
                for line in additional_response.strip().split('\n'):
                    cleaned = re.sub(r'^[-*•\d.)\s]+', '', line.strip()).strip('"\'[]')
                    if cleaned and len(cleaned) > 10:
                        more_features.append(cleaned)
                
                features.extend(more_features)
                return features[:5]
            
            self.logger.error(f"Could not extract features from AI response: {response[:300]}...")
            
        except Exception as e:
            self.logger.error(f"Error generating features with AI: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        # Return empty - validation will catch this and retry
        return []
        
    async def batch_research(self, count: int = 10) -> List[CachedProblem]:
        """Run batch research to fill the cache."""
        from scripts.market_discovery import MarketDiscovery
        
        if self._research_in_progress:
            self.logger.info("Research already in progress, skipping")
            return []
            
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
                
                # Option 1: Use key_features from opportunity (generated during research)
                if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    opp = problem._full_opportunity
                    if hasattr(opp, 'key_features') and opp.key_features:
                        features = [f for f in opp.key_features if f and f.strip()][:5]
                        if features:
                            self.logger.info(f"Using {len(features)} features from opportunity.key_features")
                
                # Option 3: Try research phases
                if not features and hasattr(problem, '_research_phases') and problem._research_phases:
                    phases = problem._research_phases
                    # Look in competition analysis
                    if 'phase_3_competition_analysis' in phases:
                        comp_data = phases['phase_3_competition_analysis']
                        if isinstance(comp_data, list) and comp_data:
                            features = comp_data[0].get('key_features', [])[:5]  # Top 5 features
                
                # Option 4: Try key_insights
                if not features and hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    opp = problem._full_opportunity
                    if hasattr(opp, 'key_insights') and opp.key_insights:
                        # Filter out empty strings and take top 5
                        features = [f for f in opp.key_insights if f and f.strip()][:5]
                        self.logger.info(f"Using {len(features)} features from key_insights")
                
                # Option 5: Last resort - extract from opportunity description
                if not features and hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    opp = problem._full_opportunity
                    if hasattr(opp, 'opportunity') and opp.opportunity:
                        features = self._extract_features_from_description(opp.opportunity)
                        self.logger.info(f"Extracted {len(features)} features from opportunity description")
                
                # Get monetization model
                monetization = 'subscription'  # Default
                if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    monetization = problem._full_opportunity.monetization_model or 'subscription'
                
                # Validation: Ensure features are never empty
                if not features:
                    self.logger.error(f"No features found for problem: {problem.problem_statement[:50]}...")
                    self.logger.error("This indicates the research phase failed to generate features!")
                    self.logger.error("The FeatureGenerator should have been called during MarketOpportunity creation.")
                    # Skip this problem - it's invalid without features
                    continue
                
                # Use creative name from opportunity if available
                project_name = ''
                if hasattr(problem, '_full_opportunity') and problem._full_opportunity:
                    if hasattr(problem._full_opportunity, 'project_name'):
                        project_name = problem._full_opportunity.project_name
                
                # NO FALLBACK - project_name must come from research
                if not project_name:
                    self.logger.error(f"No creative project name generated for problem: {problem.problem_statement[:50]}...")
                    self.logger.error("This indicates the research phase failed to generate a creative name!")
                    # Skip this problem - it's invalid without a creative name
                    continue
                
                cached = CachedProblem(
                    id=f"prob_{datetime.now().strftime('%Y%m%d')}_{i:03d}",
                    name=project_name,  # Use the creative name from research
                    problem=problem.problem_statement,
                    validation_score=problem.opportunity_score,
                    features=features,
                    monetization=monetization,
                    target_audience=problem.affected_users,  # Capture the actual affected users
                    research_ref=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                )
                
                # Final validation
                if not cached.features:
                    raise ValueError(f"Failed to generate features for problem: {problem.problem_statement[:50]}...")
                
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
        
    async def get_next_problem(self) -> Optional[CachedProblem]:
        """Get the next problem from cache, running research if needed."""
        # Check if we need more research
        if self._needs_background_research():
            await self.batch_research()
            
        # Return the next problem
        if self.validated_problems:
            problem = self.validated_problems.pop(0)
            self._save_cache()
            return problem
            
        return None
        
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
        
    def get_problems(self, count: int = 1) -> List[CachedProblem]:
        """Get multiple problems from cache without removing them."""
        return self.validated_problems[:count]


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
        
        Args:
            max_items: Maximum number of work items to return
            current_workload: Number of currently active workers
            
        Returns:
            List of discovered work items
        """
        self.logger.info(f"Discovering work (max: {max_items}, current load: {current_workload})")
        
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
        
        # If no active projects exist, only generate NEW_PROJECT tasks
        if not active_projects:
            self.logger.warning("⚠️ No active projects found. Only generating NEW_PROJECT tasks.")
            new_project_opportunities = []
            
            # Trigger background research if cache is low (non-blocking)
            await self.research_cache.trigger_background_research()
            
            # Check cache size
            cache_size = self.research_cache.cache_size()
            self.logger.info(f"📦 Research cache has {cache_size} problems ready")
            
            # Get problems from cache (1-2 at a time for pipeline workflow)
            problems_to_use = min(2, max_items, cache_size)  # Max 2 projects at once
            cached_problems = self.research_cache.get_cached_problems(limit=problems_to_use)
            
            if not cached_problems:
                self.logger.warning("No cached problems available")
                return []
            
            # Convert cached problems to work opportunities with simplified structure
            for problem in cached_problems:
                new_project_opportunities.append(WorkOpportunity(
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
                ))
            
            # Convert to work items
            work_items = []
            for opp in new_project_opportunities:
                work_item = opp.to_work_item()
                is_dup, _ = self.unified_task_state.is_duplicate_task(work_item)
                if not is_dup:
                    work_items.append(work_item)
            
            self.logger.info(f"📋 Generated {len(work_items)} NEW_PROJECT tasks (no existing projects)")
            return work_items
        
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
        
        try:
            # 2. Portfolio gap analysis
            gap_work = await self._discover_portfolio_gaps()
            opportunities.extend(gap_work)
        except Exception as e:
            self.logger.error(f"Error in portfolio gap discovery: {e}")
        
        # DISABLED: System improvement opportunities
        # try:
        #     # 3. System improvement opportunities
        #     system_work = await self._discover_system_improvements()
        #     opportunities.extend(system_work)
        # except Exception as e:
        #     self.logger.error(f"Error in system improvement discovery: {e}")
        
        try:
            # 4. Research opportunities
            research_work = await self._discover_research_opportunities()
            opportunities.extend(research_work)
        except Exception as e:
            self.logger.error(f"Error in research opportunity discovery: {e}")
        
        try:
            # 5. Maintenance tasks
            maintenance_work = await self._discover_maintenance_tasks()
            opportunities.extend(maintenance_work)
        except Exception as e:
            self.logger.error(f"Error in maintenance task discovery: {e}")
        
        try:
            # 6. Cross-repository integration opportunities
            integration_work = await self._discover_integration_opportunities()
            opportunities.extend(integration_work)
        except Exception as e:
            self.logger.error(f"Error in integration opportunity discovery: {e}")
        
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
        
        # If we have no new work due to duplicates, try to generate more diverse work
        if not work_items and len(filtered_work) > 0:
            self.logger.info("🔄 All discovered work was duplicates - generating more diverse opportunities")
            try:
                # Generate some alternative work types
                alternative_work = await self._generate_alternative_work()
                for opp in alternative_work:
                    work_item = opp.to_work_item()
                    is_dup, _ = self.unified_task_state.is_duplicate_task(work_item)
                    if not is_dup:
                        work_items.append(work_item)
                        if len(work_items) >= max_items:
                            break
            except Exception as e:
                self.logger.warning(f"Error generating alternative work: {e}")
        
        # Log discovery results
        if work_items:
            self.logger.info(f"Discovered {len(work_items)} work items:")
            for item in work_items:
                self.logger.info(f"  - {item.task_type}: {item.title} (priority: {item.priority.name})")
        
        # Track discovered work
        self.discovered_work.extend(filtered_work)
        
        # Log discovery summary
        if work_items:
            task_types = {}
            for item in work_items:
                task_types[item.task_type] = task_types.get(item.task_type, 0) + 1
            
            type_summary = ', '.join(f"{count} {task_type}" for task_type, count in task_types.items())
            self.logger.info(f"📋 Work discovery summary: {type_summary}")
        
        return work_items
    
    async def _discover_repository_work(self) -> List[WorkOpportunity]:
        """Discover work opportunities from repositories using lifecycle analysis."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        
        # Double-check that repositories still exist (defensive programming)
        # This prevents work generation for repos that might have been deleted
        # after system state was loaded but before work generation
        valid_projects = {}
        for repo_name, repo_data in filtered_projects.items():
            # Basic validation - ensure the repository has essential data
            if repo_data.get('name') or repo_data.get('full_name'):
                valid_projects[repo_name] = repo_data
            else:
                self.logger.warning(f"Skipping invalid repository entry: {repo_name}")
        
        self.logger.debug(f"Repository discovery: {len(all_projects)} total, {len(filtered_projects)} after exclusions, {len(valid_projects)} valid")
        
        for repo_name, repo_data in valid_projects.items():
            # Perform lifecycle analysis
            lifecycle_analysis = await self._analyze_repository_lifecycle(repo_name, repo_data)
            current_stage = lifecycle_analysis.get('current_stage', 'unknown')
            appropriate_task_types = lifecycle_analysis.get('appropriate_task_types', [])
            
            # Get architecture type
            architecture = self._determine_architecture_type(repo_data)
            
            # Get completed tasks for this repo
            completed_tasks = self.completed_tasks_by_repo.get(repo_name, set())
            
            # Get smart task types for this repository
            smart_tasks = TaskTypeSelector.get_appropriate_task_types(
                architecture=architecture,
                lifecycle_stage=current_stage,
                current_needs=repo_data.get('specific_needs', []),
                completed_tasks=completed_tasks
            )
            
            # Generate opportunities based on smart task types
            for task_type in smart_tasks[:3]:  # Limit to top 3 per repo
                opportunity = self._create_smart_opportunity(
                    task_type=task_type,
                    repo_name=repo_name,
                    repo_data=repo_data,
                    lifecycle_stage=current_stage
                )
                if opportunity:
                    opportunities.append(opportunity)
            
            # Fallback to basic analysis if no smart tasks generated
            if not smart_tasks:
                # Check repository health and activity
                health_score = repo_data.get('health_score', 100)
                recent_activity = repo_data.get('recent_activity', {})
            
                # Low health repositories need attention
                if health_score < 85:
                    opportunities.append(WorkOpportunity(
                        source=f"repository:{repo_name}",
                        type=SmartTaskType.REPOSITORY_HEALTH.value,
                        priority=TaskPriority.HIGH,
                        title=f"Improve {repo_name} health score",
                        description=f"Repository health is {health_score}%. Identify and fix issues.",
                        repository=repo_name,
                        estimated_cycles=3,
                        metadata={
                            'current_health': health_score,
                            'lifecycle_stage': current_stage,
                            'architecture': architecture.value if architecture else 'unknown'
                        }
                    ))
            
                # Recent commits suggest various improvement opportunities
                recent_commits = recent_activity.get('recent_commits', 0)
                if recent_commits > 0:
                    # Rotate through different types of improvements
                    improvement_types = [
                        {
                            'type': 'TESTING',
                            'title': f"Add comprehensive tests for {repo_name}",
                            'description': f"Recent {recent_commits} commits need test coverage to ensure reliability.",
                            'priority': TaskPriority.HIGH
                        },
                        {
                            'type': 'DOCUMENTATION',
                            'title': f"Update documentation for {repo_name} changes",
                            'description': f"Document recent features and changes in {repo_name} for better maintainability.",
                            'priority': TaskPriority.MEDIUM
                        },
                        {
                            'type': 'FEATURE',
                            'title': f"Enhance {repo_name} with new functionality",
                            'description': f"Build upon recent changes in {repo_name} to add valuable new features.",
                            'priority': TaskPriority.MEDIUM
                        },
                        {
                            'type': 'PERFORMANCE',
                            'title': f"Optimize {repo_name} performance and structure",
                            'description': f"Review and optimize {repo_name} codebase for better performance and maintainability.",
                            'priority': TaskPriority.LOW
                        }
                    ]
                
                    # Select improvement type based on repository characteristics
                    selected_improvement = improvement_types[0]  # Default to testing
                
                    if recent_commits > 10:  # Very active repo
                        selected_improvement = improvement_types[1]  # Documentation
                    elif 'typescript' in str(repo_data.get('topics', [])).lower():
                        selected_improvement = improvement_types[2]  # Feature
                    elif recent_commits > 5:
                        selected_improvement = improvement_types[3]  # Optimization
                
                    # Check if similar task already exists before creating opportunity
                    from scripts.task_manager import TaskManager, TaskType
                    # Check for duplicate without default repository
                    task_manager = TaskManager(repository=repo_name)
                
                    try:
                        # Check for duplicate in local task state
                        task_type = TaskType(selected_improvement['type'].lower())
                        existing_task = task_manager._find_duplicate_task(
                            task_type,
                            selected_improvement['title'],
                            selected_improvement['description'],
                            repo_name
                        )
                    
                        if not existing_task:
                            opportunities.append(WorkOpportunity(
                                source=f"repository:{repo_name}",
                                type=selected_improvement['type'],
                                priority=selected_improvement['priority'],
                                title=selected_improvement['title'],
                                description=selected_improvement['description'],
                                repository=repo_name,
                                estimated_cycles=2,
                                metadata={'recent_commits': recent_commits}
                            ))
                    except Exception as e:
                        self.logger.debug(f"Could not check for duplicate task: {e}")
        
        return opportunities


    async def _analyze_repository_lifecycle(self, repo_name: str, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository lifecycle stage."""
        try:
            # Prepare repository analysis data
            repo_analysis = {
                'repository': repo_name,
                'basic_info': repo_data,
                'health_metrics': {
                    'health_score': repo_data.get('health_score', 100),
                    'recent_commits': repo_data.get('recent_activity', {}).get('recent_commits', 0)
                },
                'code_analysis': repo_data.get('code_analysis', {}),
                'issues_analysis': repo_data.get('issues_analysis', {}),
                'recent_activity': repo_data.get('recent_activity', {}),
                'architecture': repo_data.get('architecture', {})
            }
            
            # Perform lifecycle analysis
            lifecycle_result = await self.lifecycle_analyzer.analyze_project_stage(repo_analysis)
            return lifecycle_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing lifecycle for {repo_name}: {e}")
            return {
                'current_stage': 'unknown',
                'appropriate_task_types': ['FEATURE', 'TESTING', 'DOCUMENTATION']
            }
    
    def _determine_architecture_type(self, repo_data: Dict[str, Any]) -> Optional[ArchitectureType]:
        """Determine repository architecture type."""
        # All our projects use Laravel React starter kit
        return ArchitectureType.LARAVEL_REACT
    
    def _create_smart_opportunity(self, task_type: SmartTaskType, repo_name: str,
                                repo_data: Dict[str, Any], lifecycle_stage: str) -> Optional[WorkOpportunity]:
        """Create a smart work opportunity based on task type."""
        metadata = TaskTypeSelector.get_task_metadata(task_type)
        
        # Generate appropriate title and description
        title = self._generate_task_title(task_type, repo_name, repo_data)
        description = self._generate_task_description(task_type, repo_name, repo_data, lifecycle_stage)
        
        # Calculate priority based on lifecycle stage
        priority = self._calculate_smart_priority(task_type, lifecycle_stage, metadata)
        
        # Estimate cycles based on repository complexity
        repo_complexity = repo_data.get('complexity_score', 1.0)
        estimated_cycles = TaskTypeSelector.estimate_task_duration(task_type, repo_complexity)
        
        return WorkOpportunity(
            source=f"repository:{repo_name}",
            type=task_type.value,
            priority=priority,
            title=title,
            description=description,
            repository=repo_name,
            estimated_cycles=estimated_cycles,
            metadata={
                'task_category': metadata.category.value,
                'complexity': metadata.complexity.value,
                'lifecycle_stage': lifecycle_stage,
                'required_skills': list(metadata.required_skills),
                'success_criteria': metadata.success_criteria
            }
        )
    
    def _generate_task_title(self, task_type: SmartTaskType, repo_name: str, 
                           repo_data: Dict[str, Any]) -> str:
        """Generate context-aware task title."""
        base_titles = {
            SmartTaskType.SETUP_LARAVEL_API: f"Set up Laravel API structure for {repo_name}",
            SmartTaskType.SETUP_SANCTUM_AUTH: f"Implement Sanctum authentication in {repo_name}",
            SmartTaskType.FEATURE_API_ENDPOINT: f"Add new API endpoint to {repo_name}",
            SmartTaskType.TESTING_UNIT_TESTS: f"Add unit tests for {repo_name}",
            SmartTaskType.DOCUMENTATION_API: f"Document API endpoints in {repo_name}",
            SmartTaskType.OPTIMIZATION_PERFORMANCE: f"Optimize {repo_name} performance",
        }
        
        return base_titles.get(task_type, f"Implement {task_type.value} for {repo_name}")
    
    def _generate_task_description(self, task_type: SmartTaskType, repo_name: str,
                                 repo_data: Dict[str, Any], lifecycle_stage: str) -> str:
        """Generate detailed task description based on context."""
        metadata = TaskTypeSelector.get_task_metadata(task_type)
        
        description = f"Repository {repo_name} is in {lifecycle_stage} stage. "
        
        if task_type == SmartTaskType.SETUP_LARAVEL_API:
            description += "Set up the Laravel API structure with proper routing, controllers, and middleware."
        elif task_type == SmartTaskType.FEATURE_API_ENDPOINT:
            description += "Implement a new API endpoint with proper validation, error handling, and documentation."
        else:
            description += f"Complete {task_type.value} to advance the project."
            
        # Add success criteria
        if metadata.success_criteria:
            criteria_text = ", ".join(f"{k}: {v}" for k, v in metadata.success_criteria.items())
            description += f" Success criteria: {criteria_text}."
            
        return description
    
    def _calculate_smart_priority(self, task_type: SmartTaskType, lifecycle_stage: str,
                                metadata: Any) -> TaskPriority:
        """Calculate priority based on task type and lifecycle stage."""
        # Stage-based priority adjustments
        stage_priority_boost = {
            'inception': {SmartTaskType.SETUP_PROJECT_STRUCTURE, SmartTaskType.SETUP_DATABASE_SCHEMA},
            'early_development': {SmartTaskType.FEATURE_CRUD_OPERATIONS, SmartTaskType.TESTING_UNIT_TESTS},
            'active_development': {SmartTaskType.FEATURE_API_ENDPOINT, SmartTaskType.TESTING_INTEGRATION_TESTS},
            'growth': {SmartTaskType.OPTIMIZATION_PERFORMANCE, SmartTaskType.INFRASTRUCTURE_SCALING},
            'mature': {SmartTaskType.MAINTENANCE_SECURITY_PATCH, SmartTaskType.OPTIMIZATION_CACHING}
        }
        
        # Check if this task type should be prioritized for current stage
        if lifecycle_stage in stage_priority_boost:
            if task_type in stage_priority_boost[lifecycle_stage]:
                return TaskPriority.HIGH
                
        # Use metadata priority modifier
        if metadata.priority_modifier > 1.2:
            return TaskPriority.HIGH
        elif metadata.priority_modifier > 1.0:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
            
    async def _discover_portfolio_gaps(self) -> List[WorkOpportunity]:
        """Discover gaps in the project portfolio using intelligent analysis."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        
        try:
            # Perform deep portfolio analysis
            portfolio_insights = await self.portfolio_intelligence.analyze_portfolio(
                filtered_projects,
                force_refresh=False
            )
            
            # Generate market-driven project opportunities (1 per research for focused execution)
            project_opportunities = await self.market_research.generate_project_opportunities(
                filtered_projects,
                max_opportunities=1  # Changed from 3 - intensive research = focused execution
            )
            
            # Convert ProjectOpportunity to WorkOpportunity
            for proj_opp in project_opportunities:
                # Create unique metadata for the opportunity
                metadata = {
                    'problem_statement': proj_opp.problem_statement,
                    'solution_approach': proj_opp.solution_approach,
                    'unique_value_proposition': proj_opp.unique_value_proposition,
                    'target_market': proj_opp.target_market,
                    'tech_stack': proj_opp.tech_stack,
                    'market_demand': proj_opp.market_demand,
                    'innovation_score': proj_opp.innovation_score,
                    'complexity': proj_opp.estimated_complexity
                }
                
                # Determine priority based on INTENSIVE market demand thresholds
                if proj_opp.market_demand > 0.9:  # Increased from 0.8
                    priority = TaskPriority.HIGH
                elif proj_opp.market_demand > 0.75:  # Increased from 0.6
                    priority = TaskPriority.MEDIUM
                else:
                    priority = TaskPriority.LOW
                
                # Estimate cycles based on complexity
                cycles_map = {'low': 3, 'medium': 5, 'high': 8}
                complexity = proj_opp.estimated_complexity if proj_opp.estimated_complexity else 'medium'
                estimated_cycles = cycles_map.get(complexity, 5)
                
                opportunities.append(WorkOpportunity(
                    source="market_research",
                    type="NEW_PROJECT",
                    priority=priority,
                    title=proj_opp.title,
                    description=proj_opp.description,
                    repository=None,
                    estimated_cycles=estimated_cycles,
                    metadata=metadata
                ))
            
            # Add strategic gap-based opportunities
            for gap in portfolio_insights.strategic_gaps[:2]:
                if gap.get('urgency') == 'high':
                    opportunities.append(WorkOpportunity(
                        source="portfolio_intelligence",
                        type="STRATEGIC_INITIATIVE",
                        priority=TaskPriority.HIGH,
                        title=f"Address {gap.get('gap_type', 'strategic')} gap",
                        description=gap.get('description', 'Strategic portfolio gap identified'),
                        repository=None,
                        estimated_cycles=4,
                        metadata={
                            'gap_details': gap,
                            'recommended_action': gap.get('recommended_action', '')
                        }
                    ))
            
            self.logger.info(f"📊 Portfolio analysis complete: {len(opportunities)} opportunities found")
            
        except Exception as e:
            self.logger.error(f"Error in intelligent portfolio analysis: {e}")
            # Fallback to basic analysis if intelligent analysis fails
            return await self._basic_portfolio_gap_analysis(filtered_projects)
        
        return opportunities
    
    async def _basic_portfolio_gap_analysis(self, filtered_projects: Dict[str, Any]) -> List[WorkOpportunity]:
        """Portfolio gap analysis using Gemini Deep Research."""
        opportunities = []
        
        # Use Gemini Deep Research for portfolio gap analysis
        project_ideas = await self._generate_dynamic_project_ideas()
        
        for i, project in enumerate(project_ideas[:2]):
            opportunities.append(WorkOpportunity(
                source="portfolio_gap",
                type="NEW_PROJECT",
                priority=TaskPriority.MEDIUM,
                title=project.get('title', f'Project {i+1}'),
                description=project.get('description', 'New project opportunity'),
                repository=None,
                estimated_cycles=5,
                metadata={
                    'project_type': project.get('type', 'general'),
                    'tech_stack': project.get('tech_stack', []),
                    'market_validation': project.get('market_validation', {})
                }
            ))
        
        return opportunities
    
    async def _discover_system_improvements(self) -> List[WorkOpportunity]:
        """Discover system improvement opportunities."""
        opportunities = []
        
        # Check if we're in development mode and skip if so
        import os
        is_development = os.getenv('NODE_ENV', 'production').lower() == 'development'
        if is_development:
            self.logger.info("⚠️ Skipping system improvement discovery in development mode")
            return opportunities
        
        # Check system performance metrics
        performance = self.system_state.get('system_performance', {})
        failed_actions = performance.get('failed_actions', 0)
        total_cycles = performance.get('total_cycles', 1)
        
        if failed_actions > 0:
            failure_rate = failed_actions / max(total_cycles, 1)
            if failure_rate > 0.1:  # More than 10% failure rate
                opportunities.append(WorkOpportunity(
                    source="system_monitoring",
                    type="SYSTEM_IMPROVEMENT",
                    priority=TaskPriority.CRITICAL,
                    title="Fix system reliability issues",
                    description=f"System has {failure_rate:.1%} failure rate. Investigate and fix.",
                    estimated_cycles=3,
                    metadata={'failure_rate': failure_rate, 'failed_actions': failed_actions}
                ))
        
        # Check for outdated dependencies or configurations - add variety
        import random
        maintenance_tasks = [
            {
                'title': "Update system dependencies and configurations",
                'description': "Regular maintenance to keep system components up to date.",
                'improvement_type': 'maintenance'
            },
            {
                'title': "Clean up unused code and dead imports",
                'description': "Remove technical debt by cleaning up unused code paths.",
                'improvement_type': 'cleanup'
            },
            {
                'title': "Optimize Docker container sizes",
                'description': "Reduce container bloat and improve deployment efficiency.",
                'improvement_type': 'containerization'
            },
            {
                'title': "Review and update security policies",
                'description': "Ensure security best practices are being followed.",
                'improvement_type': 'security'
            },
            {
                'title': "Implement automated backup strategy",
                'description': "Set up automated backups for critical system data.",
                'improvement_type': 'backup'
            }
        ]
        
        selected_maintenance = random.choice(maintenance_tasks)
        
        opportunities.append(WorkOpportunity(
            source="system_monitoring",
            type="SYSTEM_IMPROVEMENT", 
            priority=TaskPriority.LOW,
            title=selected_maintenance['title'],
            description=selected_maintenance['description'],
            estimated_cycles=2,
            metadata={'improvement_type': selected_maintenance['improvement_type']}
        ))
        
        # Performance optimization
        learning_metrics = performance.get('learning_metrics', {})
        efficiency = learning_metrics.get('resource_efficiency', 1.0)
        
        if efficiency < 0.8:  # Less than 80% efficiency
            # Add variety to system improvement tasks
            import random
            
            improvement_options = [
                {
                    'title': "Optimize system resource efficiency",
                    'description': f"System efficiency is {efficiency:.1%}. Analyze and optimize resource usage patterns.",
                    'improvement_type': 'resource_optimization'
                },
                {
                    'title': "Implement performance monitoring dashboard",
                    'description': f"Current efficiency: {efficiency:.1%}. Build real-time monitoring for better insights.",
                    'improvement_type': 'monitoring'
                },
                {
                    'title': "Refactor task scheduling algorithm",
                    'description': f"Efficiency at {efficiency:.1%}. Improve task distribution and scheduling.",
                    'improvement_type': 'scheduling'
                },
                {
                    'title': "Optimize database query performance",
                    'description': f"System running at {efficiency:.1%} efficiency. Tune database operations.",
                    'improvement_type': 'database'
                },
                {
                    'title': "Implement intelligent caching strategy",
                    'description': f"Boost efficiency from {efficiency:.1%} with smarter caching.",
                    'improvement_type': 'caching'
                }
            ]
            
            selected = random.choice(improvement_options)
            
            opportunities.append(WorkOpportunity(
                source="system_monitoring",
                type="SYSTEM_IMPROVEMENT",
                priority=TaskPriority.HIGH,
                title=selected['title'],
                description=selected['description'],
                estimated_cycles=4,
                metadata={
                    'efficiency': efficiency, 
                    'improvement_type': selected['improvement_type']
                }
            ))
        
        return opportunities
    
    async def _discover_research_opportunities(self) -> List[WorkOpportunity]:
        """Discover research opportunities."""
        opportunities = []
        
        # Technology trend research
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH",
            priority=TaskPriority.BACKGROUND,
            title="Research emerging AI development trends",
            description="Stay current with latest AI development tools and frameworks.",
            estimated_cycles=2,
            metadata={'research_area': 'ai_trends', 'frequency': 'weekly'}
        ))
        
        # Competitive analysis
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH", 
            priority=TaskPriority.LOW,
            title="Analyze competitor project structures",
            description="Research how similar AI development teams structure their projects.",
            estimated_cycles=3,
            metadata={'research_area': 'competitive_analysis', 'frequency': 'monthly'}
        ))
        
        # Best practices research
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH",
            priority=TaskPriority.MEDIUM,
            title="Research software architecture best practices",
            description="Study latest patterns for scalable AI-driven applications.",
            estimated_cycles=2,
            metadata={'research_area': 'architecture', 'frequency': 'bi-weekly'}
        ))
        
        return opportunities
    
    async def _discover_maintenance_tasks(self) -> List[WorkOpportunity]:
        """Discover maintenance tasks."""
        opportunities = []
        
        # Check when last backup was created
        last_backup_check = self.work_sources_checked.get('backup_system', datetime.min.replace(tzinfo=timezone.utc))
        if (datetime.now(timezone.utc) - last_backup_check).seconds > 3600:  # Check hourly
            opportunities.append(WorkOpportunity(
                source="maintenance_scheduler",
                type="MAINTENANCE",
                priority=TaskPriority.LOW,
                title="Verify system backup integrity",
                description="Check that system state backups are being created properly.",
                estimated_cycles=1,
                metadata={'maintenance_type': 'backup_verification'}
            ))
            self.work_sources_checked['backup_system'] = datetime.now(timezone.utc)
        
        # Log file cleanup
        opportunities.append(WorkOpportunity(
            source="maintenance_scheduler",
            type="MAINTENANCE",
            priority=TaskPriority.BACKGROUND,
            title="Clean up old log files",
            description="Archive or remove old log files to free up space.",
            estimated_cycles=1,
            metadata={'maintenance_type': 'log_cleanup'}
        ))
        
        # Performance monitoring
        opportunities.append(WorkOpportunity(
            source="maintenance_scheduler", 
            type="MONITORING",
            priority=TaskPriority.LOW,
            title="Generate performance metrics report",
            description="Analyze system performance and generate insights.",
            estimated_cycles=1,
            metadata={'maintenance_type': 'performance_report'}
        ))
        
        return opportunities
    
    async def _discover_integration_opportunities(self) -> List[WorkOpportunity]:
        """Discover cross-repository integration opportunities."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        project_list = list(filtered_projects.keys())
        
        if len(project_list) >= 2:
            # API integration opportunities
            cms_projects = [name for name, data in filtered_projects.items() 
                          if 'cms' in (data.get('description') or '').lower()]
            ai_projects = [name for name, data in filtered_projects.items()
                         if 'ai' in (data.get('description') or '').lower()]
            
            if cms_projects and ai_projects:
                opportunities.append(WorkOpportunity(
                    source="integration_analysis",
                    type="INTEGRATION",
                    priority=TaskPriority.MEDIUM,
                    title=f"Integrate {cms_projects[0]} with {ai_projects[0]}",
                    description="Create API connections between CMS and AI systems.",
                    estimated_cycles=5,
                    metadata={
                        'integration_type': 'api',
                        'source_repo': cms_projects[0],
                        'target_repo': ai_projects[0]
                    }
                ))
        
        # Shared component library
        if len(project_list) >= 3:
            opportunities.append(WorkOpportunity(
                source="integration_analysis",
                type="NEW_PROJECT",
                priority=TaskPriority.MEDIUM,
                title="Create shared component library",
                description="Build reusable components for all projects.",
                estimated_cycles=4,
                metadata={'integration_type': 'shared_library'}
            ))
        
        return opportunities
    
    async def _generate_alternative_work(self) -> List[WorkOpportunity]:
        """Generate alternative work when all regular work is duplicates."""
        opportunities = []
        
        # Generate time-based maintenance tasks
        current_hour = datetime.now(timezone.utc).hour
        
        if current_hour % 6 == 0:  # Every 6 hours
            opportunities.append(WorkOpportunity(
                source="alternative_scheduler",
                type="MAINTENANCE",
                priority=TaskPriority.LOW,
                title="Perform system health check",
                description="Comprehensive health check of the AI system components.",
                estimated_cycles=1,
                metadata={'maintenance_type': 'health_check', 'scheduled': True}
            ))
        
        if current_hour % 8 == 0:  # Every 8 hours  
            opportunities.append(WorkOpportunity(
                source="alternative_scheduler",
                type="RESEARCH",
                priority=TaskPriority.BACKGROUND,
                title="Research AI development best practices",
                description="Study latest trends in AI development and system optimization.",
                estimated_cycles=2,
                metadata={'research_area': 'best_practices', 'scheduled': True}
            ))
        
        # Generate code quality improvement tasks
        opportunities.append(WorkOpportunity(
            source="alternative_scheduler",
            type="SYSTEM_IMPROVEMENT",
            priority=TaskPriority.MEDIUM,
            title="Analyze code quality metrics",
            description="Review codebase for potential improvements and optimizations.",
            estimated_cycles=3,
            metadata={'improvement_type': 'code_quality', 'scheduled': True}
        ))
        
        # Generate random exploration task
        import random
        exploration_topics = [
            "container orchestration patterns",
            "microservices architecture",
            "database optimization techniques",
            "cloud deployment strategies",
            "API design patterns",
            "security vulnerability scanning",
            "performance monitoring tools",
            "automated testing strategies"
        ]
        
        topic = random.choice(exploration_topics)
        opportunities.append(WorkOpportunity(
            source="alternative_scheduler",
            type="RESEARCH",
            priority=TaskPriority.LOW,
            title=f"Explore {topic}",
            description=f"Research and document findings about {topic} for future implementation.",
            estimated_cycles=2,
            metadata={'research_area': 'exploration', 'topic': topic, 'scheduled': True}
        ))
        
        return opportunities
    
    def _filter_and_prioritize(self, opportunities: List[WorkOpportunity], max_items: int) -> List[WorkOpportunity]:
        """Filter and prioritize work opportunities."""
        # Remove duplicates based on title
        seen_titles = set()
        unique_opportunities = []
        
        for opp in opportunities:
            if opp.title not in seen_titles:
                unique_opportunities.append(opp)
                seen_titles.add(opp.title)
        
        # Sort by priority (lower value = higher priority)
        unique_opportunities.sort(key=lambda x: (x.priority.value, x.estimated_cycles))
        
        # Return top items
        return unique_opportunities[:max_items]
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get work discovery statistics."""
        total_discovered = len(self.discovered_work)
        by_source = {}
        by_type = {}
        by_priority = {}
        
        for work in self.discovered_work:
            # Count by source
            source = work.source.split(':')[0]  # Get base source name
            by_source[source] = by_source.get(source, 0) + 1
            
            # Count by type
            by_type[work.type] = by_type.get(work.type, 0) + 1
            
            # Count by priority
            priority_name = work.priority.name
            by_priority[priority_name] = by_priority.get(priority_name, 0) + 1
        
        return {
            'total_discovered': total_discovered,
            'by_source': by_source,
            'by_type': by_type, 
            'by_priority': by_priority,
            'last_discovery': datetime.now(timezone.utc).isoformat()
        }
    
    async def _generate_dynamic_project_ideas(self) -> List[Dict[str, Any]]:
        """Generate project ideas efficiently using cached research.
        
        Returns:
            List of project ideas from cache
        """
        self.logger.info("🔍 Retrieving validated project ideas from cache...")
        
        # Get problems from cache (this will trigger batch research if needed)
        problems = self.research_cache.get_problems(count=1)
        
        if not problems:
            # Cache is empty and couldn't be filled
            problem = await self.research_cache.get_next_problem()
            if not problem:
                raise ValueError("No cached problems available and research failed")
            problems = [problem]
        
        # Convert cached problems to project ideas with simplified structure
        project_ideas = []
        for cached_problem in problems:
            project_idea = {
                'type': 'Market Solution',
                'title': cached_problem.name,
                'description': cached_problem.problem,
                'tech_stack': ['Laravel', 'React', 'TypeScript', 'MySQL', 'Redis'],
                'market_validation': {
                    'validation_score': cached_problem.validation_score,
                    'research_ref': cached_problem.research_ref
                }
            }
            project_ideas.append(project_idea)
        
        self.logger.info(f"✅ Retrieved {len(project_ideas)} project ideas from cache")
        return project_ideas
    
    def _extract_key_features_from_research(self, research_phases: Dict[str, Any]) -> List[str]:
        """Extract key features from research phases data.
        
        Args:
            research_phases: Dictionary containing all research phase data
            
        Returns:
            List of key features extracted from competition analysis phase
        """
        key_features = []
        
        # Check Phase 3 - Competition Analysis for key_features
        if research_phases and 'phase_3_competition_analysis' in research_phases:
            competition_data = research_phases['phase_3_competition_analysis']
            
            # Competition data is a list of opportunities
            if isinstance(competition_data, list) and competition_data:
                # Get the first opportunity (should be the selected one)
                for opportunity in competition_data:
                    if isinstance(opportunity, dict) and 'key_features' in opportunity:
                        features = opportunity.get('key_features', [])
                        if isinstance(features, list):
                            key_features.extend(features)
                        break
        
        # If no features found in Phase 3, try Phase 5 - Final Selection
        if not key_features and research_phases and 'phase_5_final_selection' in research_phases:
            final_selection = research_phases['phase_5_final_selection']
            if isinstance(final_selection, list) and final_selection:
                for opportunity in final_selection:
                    if isinstance(opportunity, dict) and 'key_features' in opportunity:
                        features = opportunity.get('key_features', [])
                        if isinstance(features, list):
                            key_features.extend(features)
                        break
        
        # Deduplicate features
        unique_features = []
        seen = set()
        for feature in key_features:
            if feature and feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        self.logger.debug(f"Extracted {len(unique_features)} key features from research phases")
        return unique_features