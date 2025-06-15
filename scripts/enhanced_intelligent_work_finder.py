"""
Enhanced Intelligent Work Finder for Continuous 24/7 AI Operation

This is a truly intelligent work discovery system that uses AI to analyze,
learn, and adapt rather than relying on hard-coded templates.
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import uuid
import random

from scripts.work_item_types import WorkItem, TaskPriority, WorkOpportunity
from scripts.ai_brain import IntelligentAIBrain
from scripts.repository_exclusion import RepositoryExclusion
from scripts.task_persistence import TaskPersistence


class EnhancedIntelligentWorkFinder:
    """Truly intelligent work discovery system powered by AI analysis and learning."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, system_state: Dict[str, Any]):
        """Initialize the enhanced work finder.
        
        Args:
            ai_brain: AI brain for intelligent analysis
            system_state: Current system state
        """
        self.ai_brain = ai_brain
        self.system_state = system_state
        self.logger = logging.getLogger(__name__)
        
        # Work discovery history and learning
        self.discovered_work: List[WorkOpportunity] = []
        self.work_sources_checked: Dict[str, datetime] = {}
        self.last_portfolio_analysis: Optional[datetime] = None
        
        # Task persistence and learning
        self.task_persistence = TaskPersistence("enhanced_work_finder_tasks.json")
        
        # Learning system data
        self.task_outcomes: Dict[str, Dict[str, Any]] = {}  # task_id -> outcome data
        self.successful_patterns: List[Dict[str, Any]] = []
        self.repository_insights: Dict[str, Dict[str, Any]] = {}
        
        # Adaptive configuration (no hard-coded values)
        self.discovery_interval = 30  # Will adapt based on system load
        self.portfolio_analysis_interval = 300  # Will adapt based on activity
        self.max_work_per_discovery = 5  # Will adapt based on capacity
        
        # Market and trend data cache
        self.market_trends_cache: Dict[str, Any] = {}
        self.last_market_analysis: Optional[datetime] = None
        
        # Load historical data
        self._load_learning_data()
    
    async def discover_work(self, max_items: int = 5, current_workload: int = 0) -> List[WorkItem]:
        """Intelligently discover new work opportunities using AI analysis.
        
        Args:
            max_items: Maximum number of work items to return
            current_workload: Number of currently active workers
            
        Returns:
            List of discovered work items
        """
        self.logger.info(f"🧠 Intelligently discovering work (max: {max_items}, current load: {current_workload})")
        
        # Analyze system context
        context = await self._analyze_system_context(current_workload)
        
        # Check if we need to bootstrap with new projects
        if not context['has_active_projects']:
            return await self._bootstrap_new_projects(max_items)
        
        # Use AI to discover opportunities
        opportunities = await self._ai_discover_opportunities(context, max_items)
        
        # Predict value and prioritize
        prioritized_opportunities = await self._prioritize_by_predicted_value(opportunities)
        
        # Filter duplicates intelligently
        unique_opportunities = await self._intelligent_deduplication(prioritized_opportunities)
        
        # Convert to work items
        work_items = []
        for opp in unique_opportunities[:max_items]:
            work_item = opp.to_work_item()
            
            # Final check with enhanced duplicate detection
            if not await self._is_truly_duplicate(work_item):
                work_items.append(work_item)
        
        # If we couldn't find enough unique work, generate creative alternatives
        if len(work_items) < max_items:
            additional_work = await self._generate_creative_work(
                max_items - len(work_items), 
                context
            )
            work_items.extend(additional_work)
        
        # Log discovery with insights
        if work_items:
            await self._log_discovery_insights(work_items)
        
        # Update learning data
        await self._update_discovery_patterns(work_items, context)
        
        return work_items
    
    async def _analyze_system_context(self, current_workload: int) -> Dict[str, Any]:
        """Analyze current system context for intelligent decision making."""
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter excluded repositories
        active_projects = {
            name: data for name, data in all_projects.items()
            if not RepositoryExclusion.is_excluded_repo(name)
        }
        
        # Analyze recent task history
        recent_tasks = self.task_persistence.get_task_history(hours_back=24)
        task_distribution = {}
        for task in recent_tasks:
            task_distribution[task.task_type] = task_distribution.get(task.task_type, 0) + 1
        
        # Calculate system metrics
        performance = self.system_state.get('system_performance', {})
        
        context = {
            'has_active_projects': len(active_projects) > 0,
            'active_projects': active_projects,
            'current_workload': current_workload,
            'system_capacity': self._calculate_system_capacity(current_workload),
            'recent_task_distribution': task_distribution,
            'performance_metrics': performance,
            'time_context': {
                'hour': datetime.now(timezone.utc).hour,
                'day_of_week': datetime.now(timezone.utc).weekday(),
                'is_peak_hours': 9 <= datetime.now(timezone.utc).hour <= 17
            },
            'successful_patterns': self.successful_patterns[-10:],  # Recent successful patterns
            'repository_insights': self.repository_insights
        }
        
        return context
    
    async def _ai_discover_opportunities(self, context: Dict[str, Any], max_items: int) -> List[WorkOpportunity]:
        """Use AI to discover work opportunities based on context."""
        discovery_prompt = f"""
        Analyze this system context and discover valuable work opportunities:
        
        Context:
        - Active Projects: {len(context['active_projects'])}
        - Current Workload: {context['current_workload']}
        - System Capacity: {context['system_capacity']}
        - Recent Task Distribution: {json.dumps(context['recent_task_distribution'], indent=2)}
        - Time Context: {json.dumps(context['time_context'], indent=2)}
        
        Project Details:
        {json.dumps(context['active_projects'], indent=2)}
        
        Successful Patterns from History:
        {json.dumps(context['successful_patterns'], indent=2)}
        
        Instructions:
        1. Analyze each repository's actual state, recent activity, and needs
        2. Identify HIGH-VALUE opportunities that would genuinely improve the projects
        3. Consider task diversity - avoid suggesting similar tasks
        4. Factor in current system load and capacity
        5. Learn from successful patterns but don't just repeat them
        6. Look for integration opportunities between projects
        7. Consider market trends and real-world value
        
        For each opportunity provide:
        - source: Where this opportunity was identified (e.g., "repository:name", "integration_analysis")
        - type: Specific task type (FEATURE, TESTING, DOCUMENTATION, etc.)
        - priority: Based on actual value and urgency (CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND)
        - title: Specific, actionable title (not generic)
        - description: Detailed description of what needs to be done and why
        - repository: Which repository this applies to (if applicable)
        - estimated_cycles: Realistic estimate based on complexity
        - reasoning: Why this task is valuable right now
        - expected_value: What value this will create
        
        Generate up to {max_items * 2} opportunities (we'll filter and prioritize).
        Return as JSON array.
        """
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                discovery_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                # Parse JSON from response
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    opportunities_data = json.loads(json_match.group())
                    
                    # Convert to WorkOpportunity objects
                    opportunities = []
                    for opp_data in opportunities_data:
                        try:
                            # Map priority string to enum
                            priority_map = {
                                'CRITICAL': TaskPriority.CRITICAL,
                                'HIGH': TaskPriority.HIGH,
                                'MEDIUM': TaskPriority.MEDIUM,
                                'LOW': TaskPriority.LOW,
                                'BACKGROUND': TaskPriority.BACKGROUND
                            }
                            
                            priority = priority_map.get(
                                opp_data.get('priority', 'MEDIUM').upper(), 
                                TaskPriority.MEDIUM
                            )
                            
                            opportunity = WorkOpportunity(
                                source=opp_data['source'],
                                type=opp_data['type'],
                                priority=priority,
                                title=opp_data['title'],
                                description=opp_data['description'],
                                repository=opp_data.get('repository'),
                                estimated_cycles=opp_data.get('estimated_cycles', 3),
                                metadata={
                                    'reasoning': opp_data.get('reasoning', ''),
                                    'expected_value': opp_data.get('expected_value', ''),
                                    'ai_generated': True,
                                    'context_aware': True
                                }
                            )
                            opportunities.append(opportunity)
                            
                        except Exception as e:
                            self.logger.warning(f"Error parsing opportunity: {e}")
                            continue
                    
                    return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in AI opportunity discovery: {e}")
        
        # Fallback to basic discovery if AI fails
        return await self._fallback_discovery(context, max_items)
    
    async def _prioritize_by_predicted_value(self, opportunities: List[WorkOpportunity]) -> List[WorkOpportunity]:
        """Use AI to predict value and prioritize opportunities."""
        if not opportunities:
            return []
        
        # Prepare data for value prediction
        opportunities_data = []
        for opp in opportunities:
            opp_dict = {
                'title': opp.title,
                'type': opp.type,
                'description': opp.description,
                'repository': opp.repository,
                'priority': opp.priority.name,
                'metadata': opp.metadata
            }
            opportunities_data.append(opp_dict)
        
        value_prediction_prompt = f"""
        Predict the value and impact of these work opportunities:
        
        Opportunities:
        {json.dumps(opportunities_data, indent=2)}
        
        Historical Success Data:
        {json.dumps(self.successful_patterns[-5:], indent=2)}
        
        For each opportunity, provide:
        - predicted_value: Score from 0-100 based on expected impact
        - confidence: Your confidence in this prediction (0-100)
        - factors: Key factors influencing the value
        - risks: Potential risks or challenges
        
        Consider:
        - Business value and user impact
        - Technical debt reduction
        - System reliability improvement
        - Developer productivity gains
        - Market competitiveness
        
        Return as JSON array in the same order as input.
        """
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                value_prediction_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                # Parse predictions
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    predictions = json.loads(json_match.group())
                    
                    # Enhance opportunities with predictions
                    for i, (opp, pred) in enumerate(zip(opportunities, predictions)):
                        if isinstance(pred, dict):
                            opp.metadata['predicted_value'] = pred.get('predicted_value', 50)
                            opp.metadata['prediction_confidence'] = pred.get('confidence', 50)
                            opp.metadata['value_factors'] = pred.get('factors', [])
                            opp.metadata['risks'] = pred.get('risks', [])
        
        except Exception as e:
            self.logger.error(f"Error in value prediction: {e}")
            # Add default values if prediction fails
            for opp in opportunities:
                opp.metadata['predicted_value'] = 50
                opp.metadata['prediction_confidence'] = 30
        
        # Sort by predicted value and priority
        opportunities.sort(
            key=lambda x: (
                -x.metadata.get('predicted_value', 50),  # Higher value first
                x.priority.value,  # Higher priority first (lower enum value)
                x.estimated_cycles  # Shorter tasks first
            )
        )
        
        return opportunities
    
    async def _intelligent_deduplication(self, opportunities: List[WorkOpportunity]) -> List[WorkOpportunity]:
        """Intelligently deduplicate opportunities using semantic analysis."""
        if len(opportunities) <= 1:
            return opportunities
        
        dedup_prompt = f"""
        Analyze these work opportunities and identify true duplicates or overlapping work:
        
        Opportunities:
        {json.dumps([{
            'id': i,
            'title': opp.title,
            'type': opp.type,
            'description': opp.description,
            'repository': opp.repository
        } for i, opp in enumerate(opportunities)], indent=2)}
        
        Instructions:
        1. Identify opportunities that would result in the same or very similar work
        2. Consider semantic similarity, not just exact matches
        3. Look for tasks that would conflict or make each other redundant
        4. Preserve diversity - keep different types of valuable work
        
        Return a JSON object with:
        - keep: Array of opportunity IDs to keep (the best version of each unique task)
        - duplicates: Object mapping duplicate IDs to the ID they duplicate
        - reasoning: Brief explanation of deduplication decisions
        """
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                dedup_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                # Parse deduplication results
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    dedup_results = json.loads(json_match.group())
                    keep_ids = set(dedup_results.get('keep', []))
                    
                    # Filter opportunities
                    unique_opportunities = [
                        opp for i, opp in enumerate(opportunities)
                        if i in keep_ids
                    ]
                    
                    if dedup_results.get('reasoning'):
                        self.logger.debug(f"Deduplication reasoning: {dedup_results['reasoning']}")
                    
                    return unique_opportunities
        
        except Exception as e:
            self.logger.error(f"Error in intelligent deduplication: {e}")
        
        # Fallback to simple deduplication
        seen_titles = set()
        unique = []
        for opp in opportunities:
            if opp.title not in seen_titles:
                unique.append(opp)
                seen_titles.add(opp.title)
        
        return unique
    
    async def _is_truly_duplicate(self, work_item: WorkItem) -> bool:
        """Enhanced duplicate detection using AI and historical analysis."""
        # First check with existing persistence
        if self.task_persistence.is_duplicate_task(work_item):
            return True
        
        # Then do semantic similarity check with recent tasks
        recent_tasks = self.task_persistence.get_task_history(
            repository=work_item.repository,
            hours_back=48
        )
        
        if not recent_tasks:
            return False
        
        # Use AI for semantic similarity
        similarity_prompt = f"""
        Determine if this new task is essentially the same as any recent task:
        
        New Task:
        - Title: {work_item.title}
        - Type: {work_item.task_type}
        - Description: {work_item.description}
        - Repository: {work_item.repository}
        
        Recent Tasks:
        {json.dumps([{
            'title': task.title,
            'type': task.task_type,
            'description_hash': task.description_hash,
            'completed_at': task.completed_at.isoformat(),
            'repository': task.repository
        } for task in recent_tasks[:10]], indent=2)}
        
        Return JSON with:
        - is_duplicate: boolean
        - similar_to: task title if duplicate
        - similarity_score: 0-100
        - reasoning: explanation
        """
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                similarity_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    similarity_result = json.loads(json_match.group())
                    
                    if similarity_result.get('is_duplicate', False):
                        self.logger.debug(
                            f"AI detected duplicate: {work_item.title} similar to "
                            f"{similarity_result.get('similar_to')} "
                            f"(score: {similarity_result.get('similarity_score')})"
                        )
                        return True
        
        except Exception as e:
            self.logger.error(f"Error in AI duplicate detection: {e}")
        
        return False
    
    async def _generate_creative_work(self, needed_items: int, context: Dict[str, Any]) -> List[WorkItem]:
        """Generate creative, high-value work when standard discovery isn't enough."""
        creative_prompt = f"""
        Generate {needed_items} creative, high-value work items that haven't been done before.
        
        Context:
        - Current Projects: {list(context['active_projects'].keys())}
        - Recent Task Types: {list(context['recent_task_distribution'].keys())}
        - Time: {context['time_context']}
        
        Requirements:
        1. Be creative and innovative - suggest unexpected but valuable improvements
        2. Consider cross-project synergies
        3. Think about emerging tech trends and best practices
        4. Suggest proactive improvements before they become problems
        5. Each task must be unique and specific
        
        Ideas to consider:
        - Performance optimizations based on usage patterns
        - Security hardening and vulnerability prevention
        - Developer experience improvements
        - User experience enhancements
        - Automation opportunities
        - Technical debt that's not obvious
        - Future-proofing for scalability
        
        For each work item provide:
        - type: Task type
        - title: Creative, specific title
        - description: Detailed description with clear value proposition
        - repository: Target repository (if applicable)
        - innovation_type: What makes this creative/innovative
        - expected_impact: Specific impact this will have
        
        Return as JSON array.
        """
        
        work_items = []
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                creative_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    creative_ideas = json.loads(json_match.group())
                    
                    for idea in creative_ideas[:needed_items]:
                        work_item = WorkItem(
                            id=f"creative_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                            task_type=idea['type'],
                            title=idea['title'],
                            description=idea['description'],
                            priority=TaskPriority.MEDIUM,
                            repository=idea.get('repository'),
                            estimated_cycles=4,
                            metadata={
                                'creative': True,
                                'innovation_type': idea.get('innovation_type'),
                                'expected_impact': idea.get('expected_impact')
                            }
                        )
                        
                        if not await self._is_truly_duplicate(work_item):
                            work_items.append(work_item)
        
        except Exception as e:
            self.logger.error(f"Error generating creative work: {e}")
        
        return work_items
    
    async def _bootstrap_new_projects(self, max_items: int) -> List[WorkItem]:
        """Bootstrap system with new projects when none exist."""
        self.logger.info("🚀 Bootstrapping with new project ideas")
        
        # Update market trends
        await self._update_market_trends()
        
        bootstrap_prompt = f"""
        Generate {max_items} innovative project ideas based on current market needs and trends.
        
        Market Trends:
        {json.dumps(self.market_trends_cache, indent=2)}
        
        Requirements:
        1. Each project must solve a REAL, SPECIFIC problem
        2. Must have clear monetization potential
        3. Should use modern, appropriate technology stacks
        4. Project names should be memorable and brandable
        5. Avoid generic terms - be specific about what the project does
        
        For each project provide:
        - name: Brandable project name (like "Notion", "Stripe", "Discord")
        - problem: Specific problem it solves
        - solution: How it solves the problem
        - target_audience: Who will use this
        - tech_stack: Recommended technologies (be specific)
        - mvp_features: 3-5 core features for MVP
        - monetization: How it will make money
        - differentiator: What makes it unique
        
        Focus on:
        - Developer tools and productivity
        - AI-powered applications
        - Business automation
        - Creator economy tools
        - Data analytics platforms
        
        Return as JSON array.
        """
        
        work_items = []
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                bootstrap_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    project_ideas = json.loads(json_match.group())
                    
                    for idea in project_ideas[:max_items]:
                        work_item = WorkItem(
                            id=f"bootstrap_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                            task_type="NEW_PROJECT",
                            title=f"Create {idea['name']} - {idea['problem'][:50]}...",
                            description=(
                                f"Problem: {idea['problem']}\n\n"
                                f"Solution: {idea['solution']}\n\n"
                                f"Target Audience: {idea['target_audience']}\n\n"
                                f"Tech Stack: {idea['tech_stack']}\n\n"
                                f"MVP Features:\n" + 
                                '\n'.join(f"- {feature}" for feature in idea.get('mvp_features', [])) +
                                f"\n\nMonetization: {idea['monetization']}\n\n"
                                f"Differentiator: {idea['differentiator']}"
                            ),
                            priority=TaskPriority.HIGH,
                            repository=None,
                            estimated_cycles=8,
                            metadata={
                                'project_name': idea['name'],
                                'market_validated': True,
                                'tech_stack': idea['tech_stack'],
                                'monetization_strategy': idea['monetization']
                            }
                        )
                        work_items.append(work_item)
        
        except Exception as e:
            self.logger.error(f"Error bootstrapping projects: {e}")
            
            # Fallback to a single generic project
            work_items.append(WorkItem(
                id=f"fallback_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                task_type="NEW_PROJECT",
                title="Create AI-Powered Developer Tool",
                description="Create a tool that helps developers be more productive using AI",
                priority=TaskPriority.HIGH,
                repository=None,
                estimated_cycles=6
            ))
        
        return work_items
    
    async def _update_market_trends(self):
        """Update market trends cache for informed decision making."""
        # Only update if cache is old
        if self.last_market_analysis:
            time_since = datetime.now(timezone.utc) - self.last_market_analysis
            if time_since.total_seconds() < 3600:  # 1 hour cache
                return
        
        trends_prompt = """
        Analyze current software development market trends and opportunities.
        
        Research:
        1. Emerging technologies and frameworks
        2. Problems developers and businesses are facing
        3. Successful recent product launches
        4. Gaps in current tooling
        5. AI/ML application opportunities
        
        Provide:
        - hot_technologies: List of trending tech
        - problem_areas: Common pain points
        - opportunity_areas: Underserved markets
        - success_patterns: What's working well
        
        Return as JSON.
        """
        
        try:
            response = await self.ai_brain.http_ai_client.generate_enhanced_response(
                trends_prompt
            )
            
            if response and response.get('content'):
                result_text = response.get('content', '')
                
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    self.market_trends_cache = json.loads(json_match.group())
                    self.last_market_analysis = datetime.now(timezone.utc)
        
        except Exception as e:
            self.logger.error(f"Error updating market trends: {e}")
            # Use some default trends
            self.market_trends_cache = {
                'hot_technologies': ['AI/ML', 'TypeScript', 'Rust', 'WebAssembly'],
                'problem_areas': ['Developer productivity', 'Code quality', 'Deployment complexity'],
                'opportunity_areas': ['AI-powered tools', 'No-code platforms', 'Developer experience']
            }
    
    def _calculate_system_capacity(self, current_workload: int) -> float:
        """Calculate current system capacity for work."""
        # Base capacity
        max_workers = 10  # This should come from config
        
        # Calculate utilization
        utilization = current_workload / max_workers if max_workers > 0 else 0
        
        # Available capacity (0-1 scale)
        available_capacity = 1 - utilization
        
        # Adjust for time of day (reduce capacity during off-peak)
        hour = datetime.now(timezone.utc).hour
        if hour < 6 or hour > 22:  # Night time
            available_capacity *= 0.5
        
        # Adjust for system performance
        performance = self.system_state.get('system_performance', {})
        error_rate = performance.get('error_rate', 0)
        if error_rate > 0.1:  # High error rate
            available_capacity *= (1 - error_rate)
        
        return max(0, min(1, available_capacity))
    
    async def _log_discovery_insights(self, work_items: List[WorkItem]):
        """Log insights about discovered work."""
        insights = {
            'total_discovered': len(work_items),
            'by_type': {},
            'by_priority': {},
            'ai_generated': 0,
            'creative': 0,
            'predicted_total_value': 0
        }
        
        for item in work_items:
            # Count by type
            insights['by_type'][item.task_type] = insights['by_type'].get(item.task_type, 0) + 1
            
            # Count by priority
            priority_name = item.priority.name
            insights['by_priority'][priority_name] = insights['by_priority'].get(priority_name, 0) + 1
            
            # Count AI-generated
            if item.metadata.get('ai_generated'):
                insights['ai_generated'] += 1
            
            # Count creative
            if item.metadata.get('creative'):
                insights['creative'] += 1
            
            # Sum predicted value
            insights['predicted_total_value'] += item.metadata.get('predicted_value', 50)
        
        self.logger.info(f"📊 Discovery insights: {json.dumps(insights, indent=2)}")
    
    async def _update_discovery_patterns(self, work_items: List[WorkItem], context: Dict[str, Any]):
        """Update learning patterns based on discovered work."""
        pattern = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context_summary': {
                'workload': context['current_workload'],
                'capacity': context['system_capacity'],
                'time': context['time_context']
            },
            'discovered_items': len(work_items),
            'task_types': [item.task_type for item in work_items],
            'predicted_values': [item.metadata.get('predicted_value', 50) for item in work_items]
        }
        
        # This will be used for learning in future discoveries
        self.successful_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.successful_patterns) > 100:
            self.successful_patterns = self.successful_patterns[-100:]
    
    async def _fallback_discovery(self, context: Dict[str, Any], max_items: int) -> List[WorkOpportunity]:
        """Fallback discovery method if AI fails."""
        opportunities = []
        
        # At least try to analyze repositories
        for repo_name, repo_data in list(context['active_projects'].items())[:max_items]:
            opportunities.append(WorkOpportunity(
                source=f"repository:{repo_name}",
                type="ANALYSIS",
                priority=TaskPriority.MEDIUM,
                title=f"Analyze and improve {repo_name}",
                description=f"Comprehensive analysis of {repo_name} to identify improvements",
                repository=repo_name,
                estimated_cycles=3,
                metadata={'fallback': True}
            ))
        
        return opportunities
    
    def _load_learning_data(self):
        """Load historical learning data."""
        try:
            # This would load from a file in production
            self.logger.info("Loading historical learning data...")
            # For now, start with empty data
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")
    
    async def record_task_outcome(self, work_item: WorkItem, outcome: Dict[str, Any]):
        """Record the outcome of a task for learning purposes."""
        self.task_outcomes[work_item.id] = {
            'work_item': work_item.to_dict(),
            'outcome': outcome,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # If successful, add to successful patterns
        if outcome.get('success', False):
            pattern = {
                'task_type': work_item.task_type,
                'repository': work_item.repository,
                'predicted_value': work_item.metadata.get('predicted_value', 50),
                'actual_value': outcome.get('value_created', 0),
                'execution_time': outcome.get('execution_time', 0),
                'metadata': work_item.metadata
            }
            self.successful_patterns.append(pattern)
        
        # Update repository insights
        if work_item.repository:
            if work_item.repository not in self.repository_insights:
                self.repository_insights[work_item.repository] = {
                    'successful_tasks': 0,
                    'failed_tasks': 0,
                    'total_value': 0,
                    'preferred_task_types': {}
                }
            
            insights = self.repository_insights[work_item.repository]
            if outcome.get('success', False):
                insights['successful_tasks'] += 1
                insights['total_value'] += outcome.get('value_created', 0)
                task_type = work_item.task_type
                insights['preferred_task_types'][task_type] = insights['preferred_task_types'].get(task_type, 0) + 1
            else:
                insights['failed_tasks'] += 1
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics."""
        stats = {
            'total_discovered': len(self.discovered_work),
            'ai_powered_discoveries': sum(1 for w in self.discovered_work if w.metadata.get('ai_generated', False)),
            'creative_discoveries': sum(1 for w in self.discovered_work if w.metadata.get('creative', False)),
            'successful_patterns': len(self.successful_patterns),
            'repository_insights': self.repository_insights,
            'market_trends': self.market_trends_cache,
            'learning_metrics': {
                'total_outcomes_recorded': len(self.task_outcomes),
                'success_rate': self._calculate_success_rate(),
                'average_predicted_value': self._calculate_avg_predicted_value(),
                'value_prediction_accuracy': self._calculate_prediction_accuracy()
            }
        }
        
        return stats
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall task success rate."""
        if not self.task_outcomes:
            return 0.0
        
        successful = sum(1 for outcome in self.task_outcomes.values() 
                        if outcome['outcome'].get('success', False))
        
        return successful / len(self.task_outcomes)
    
    def _calculate_avg_predicted_value(self) -> float:
        """Calculate average predicted value of discovered work."""
        if not self.discovered_work:
            return 0.0
        
        total_value = sum(w.metadata.get('predicted_value', 50) for w in self.discovered_work)
        return total_value / len(self.discovered_work)
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate how accurate our value predictions are."""
        comparisons = []
        
        for task_id, outcome_data in self.task_outcomes.items():
            work_item = outcome_data['work_item']
            outcome = outcome_data['outcome']
            
            if 'predicted_value' in work_item.get('metadata', {}):
                predicted = work_item['metadata']['predicted_value']
                actual = outcome.get('value_created', 0) * 10  # Scale to 0-100
                
                # Calculate accuracy as 1 - (absolute difference / 100)
                accuracy = 1 - (abs(predicted - actual) / 100)
                comparisons.append(max(0, accuracy))
        
        if not comparisons:
            return 0.5  # Default 50% if no data
        
        return sum(comparisons) / len(comparisons)