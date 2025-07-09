"""
Enhanced Hybrid Continuous Orchestrator

Integrates all high-impact systems:
- Research Evolution Engine + Knowledge Store
- Pattern Learning Engine + Feedback Learning  
- Swarm Intelligence
- Multi-Objective Optimizer

This creates a truly intelligent, self-improving orchestration system.
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

# Import original Dynamic God Mode components
from scripts.dynamic_charter import DynamicCharter
from scripts.outcome_learning import OutcomeLearningSystem
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.dynamic_validator import DynamicTaskValidator
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.project_creator import ProjectCreator
from scripts.task_manager import TaskManager, TaskType
from scripts.multi_repo_coordinator import MultiRepoCoordinator
from scripts.storage_mode import StorageMode

# Import NEW high-impact systems
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.pattern_learning_engine import PatternLearningEngine
from scripts.feedback_learning_system import FeedbackLearningSystem, FeedbackCollector, PerformanceTracker
from scripts.swarm_intelligence import RealSwarmIntelligence
from scripts.multi_objective_optimizer import MultiObjectiveOptimizer

# Import utilities
from scripts.ai_brain_factory import AIBrainFactory
from scripts.state_manager import StateManager
from scripts.repository_exclusion import should_process_repo

# Import monitoring system
from scripts.monitoring_integration import MonitoringIntegration
from scripts.task_monitoring_system import get_monitoring_system


class EnhancedHybridOrchestrator(ContinuousOrchestrator):
    """
    Enhanced orchestrator with research evolution, pattern learning, 
    swarm intelligence, and multi-objective optimization.
    
    Features:
    - All original hybrid features
    - Research evolution with persistent knowledge
    - Pattern learning from system behavior
    - Feedback loops for continuous improvement
    - Swarm intelligence for better decision making
    - Multi-objective optimization for task prioritization
    """
    
    def __init__(self, max_workers: int = 5, enable_parallel: bool = True,
                 enable_research: bool = True, enable_round_robin: bool = False):
        """Initialize enhanced orchestrator with all systems."""
        # Initialize base orchestrator
        super().__init__(max_workers, enable_parallel, enable_round_robin)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize original Dynamic God Mode components
        self._initialize_dynamic_components()
        
        # Initialize NEW enhancement systems
        self._initialize_enhancement_systems()
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'research_cycles': 0,
            'patterns_learned': 0,
            'swarm_analyses': 0,
            'feedback_collected': 0,
            'optimizations_applied': 0,
            'knowledge_items_stored': 0
        }
        
        # Configuration
        self.config = {
            'enable_research_evolution': True,
            'enable_pattern_learning': True,
            'enable_swarm_intelligence': True,
            'enable_feedback_learning': True,
            'enable_multi_objective': True,
            'research_cycle_frequency': 100,  # Every N tasks
            'pattern_learning_frequency': 50,
            'swarm_analysis_threshold': 0.8,  # Use swarm for high-priority tasks
            'feedback_collection_rate': 1.0   # Collect feedback for all tasks
        }
        
        # Initialize monitoring system
        self.monitoring = MonitoringIntegration()
        self.monitor = get_monitoring_system()
        self.logger.info("✅ Task monitoring system initialized")
        
    def _initialize_dynamic_components(self):
        """Initialize original Dynamic God Mode components."""
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
        # Get storage mode from environment
        storage_mode = os.getenv('STORAGE_MODE', StorageMode.GITHUB)
        self.project_creator = ProjectCreator(github_token, self.ai_brain, storage_mode=storage_mode) if github_token or storage_mode == StorageMode.LOCAL else None
        
        # Multi-repo coordinator
        if github_token and self.system_state.get('projects'):
            active_repos = [p for p in self.system_state.get('projects', {}).values() 
                           if should_process_repo(p.get('full_name', ''))]
            max_concurrent = max(len(active_repos), 50)
            
            self.coordinator = MultiRepoCoordinator(github_token, max_concurrent=max_concurrent)
            for project_id, project_data in self.system_state.get('projects', {}).items():
                if project_data.get('clone_url') and should_process_repo(project_data.get('full_name', '')):
                    try:
                        self.coordinator.add_repository(project_data['clone_url'])
                    except Exception as e:
                        self.logger.warning(f"Failed to add repository {project_id}: {e}")
        else:
            self.coordinator = None
            
        # Load current charter
        self.current_charter = None
        asyncio.create_task(self._load_initial_charter())
        
        self.logger.info("Dynamic God Mode components initialized")
        
    def _initialize_enhancement_systems(self):
        """Initialize new enhancement systems."""
        self.logger.info("Initializing enhancement systems...")
        
        # Research Evolution System
        self.knowledge_store = ResearchKnowledgeStore()
        self.research_engine = ResearchEvolutionEngine(
            state_manager=self.state_manager,
            ai_brain=self.ai_brain,
            task_generator=self.task_generator,
            self_improver=None,  # Can be added later
            outcome_learning=self.learning_system
        )
        
        # Pattern Learning System
        self.pattern_learner = PatternLearningEngine()
        
        # Feedback Learning System
        self.feedback_system = FeedbackLearningSystem()
        self.feedback_collector = FeedbackCollector()
        self.performance_tracker = PerformanceTracker()
        
        # Swarm Intelligence
        self.swarm = RealSwarmIntelligence(
            ai_brain=self.ai_brain,
            num_agents=5  # Start with 5 specialized agents
        )
        
        # Multi-Objective Optimizer
        self.optimizer = MultiObjectiveOptimizer(self.state_manager)
        
        # Start background processes
        asyncio.create_task(self._start_research_evolution())
        asyncio.create_task(self._start_pattern_learning())
        
        self.logger.info("Enhancement systems initialized")
        
    async def _load_initial_charter(self):
        """Load or generate initial charter."""
        try:
            context = await self._gather_enhanced_context()
            self.current_charter = await self.charter_system.generate_charter(context)
            self.enhanced_metrics['charters_generated'] = 1
        except Exception as e:
            self.logger.error(f"Failed to load initial charter: {e}")
            
    async def _start_research_evolution(self):
        """Start background research evolution process."""
        if not self.config['enable_research_evolution']:
            return
            
        await asyncio.sleep(60)  # Initial delay
        
        while True:
            try:
                # Run research cycle
                await self.research_engine.execute_research_cycle()
                self.enhanced_metrics['research_cycles'] += 1
                
                # Wait before next cycle
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                self.logger.error(f"Research evolution error: {e}")
                await asyncio.sleep(300)  # 5 min on error
                
    async def _start_pattern_learning(self):
        """Start background pattern learning process."""
        if not self.config['enable_pattern_learning']:
            return
            
        await asyncio.sleep(120)  # Initial delay
        
        while True:
            try:
                # Learn patterns
                workflow_patterns = await self.pattern_learner.learn_workflow_patterns()
                error_patterns = await self.pattern_learner.learn_error_patterns()
                optimization_patterns = await self.pattern_learner.learn_optimization_patterns()
                
                self.enhanced_metrics['patterns_learned'] += (
                    len(workflow_patterns.get('patterns_discovered', [])) +
                    len(error_patterns.get('error_patterns', [])) +
                    len(optimization_patterns.get('inefficient_patterns', []))
                )
                
                # Apply learned patterns
                await self.pattern_learner.apply_learned_patterns()
                
                # Wait before next cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Pattern learning error: {e}")
                await asyncio.sleep(600)  # 10 min on error
                
    async def find_work(self) -> List[WorkItem]:
        """Enhanced work finding with research insights and swarm intelligence."""
        all_work = []
        
        try:
            # 1. First check GitHub issues (real user needs)
            github_work = await self._get_github_issues()
            all_work.extend(github_work)
            
            self.logger.info(f"Found {len(github_work)} GitHub issues")
            
            # 2. If we need more work, use enhanced generation
            if len(all_work) < self.max_workers:
                needed = self.max_workers - len(all_work)
                
                # Gather enhanced context with research insights
                context = await self._gather_enhanced_context_with_research()
                
                # Use swarm intelligence for complex decisions
                if self.config['enable_swarm_intelligence']:
                    swarm_suggestions = await self._get_swarm_work_suggestions(context, needed)
                    context['swarm_insights'] = swarm_suggestions
                
                # Generate tasks with research knowledge
                generated_tasks = await self._generate_intelligent_tasks_enhanced(context, needed)
                
                # Optimize task selection with multi-objective optimization
                if self.config['enable_multi_objective'] and generated_tasks:
                    generated_tasks = await self._optimize_task_selection(generated_tasks)
                
                # Convert to work items
                for task in generated_tasks:
                    work_item = self._task_to_work_item(task)
                    if work_item:
                        all_work.append(work_item)
                
                self.logger.info(f"Generated {len(generated_tasks)} intelligent tasks")
            
            # 3. Trigger research if needed
            if self.enhanced_metrics['swarm_analyses'] % self.config['research_cycle_frequency'] == 0:
                asyncio.create_task(self._trigger_targeted_research(all_work))
            
        except Exception as e:
            self.logger.error(f"Error finding work: {e}")
            traceback.print_exc()
        
        return all_work[:self.max_workers]
        
    async def _gather_enhanced_context_with_research(self) -> Dict[str, Any]:
        """Gather context enhanced with research insights and patterns."""
        # Get base context
        context = await self._gather_enhanced_context()
        
        # Add research insights
        if self.config['enable_research_evolution']:
            research_insights = self.knowledge_store.get_insights('continuous_improvement')
            context['research_insights'] = research_insights
            
            # Add research recommendations
            if hasattr(self.research_engine, 'need_analyzer'):
                immediate_needs = self.research_engine.need_analyzer.get_immediate_research_needs()
                context['research_priorities'] = immediate_needs
        
        # Add learned patterns
        if self.config['enable_pattern_learning']:
            context['learned_patterns'] = {
                'workflow_patterns': self.pattern_learner.learning_state.get('workflow_patterns', {}),
                'error_patterns': self.pattern_learner.learning_state.get('error_patterns', {}),
                'optimization_patterns': self.pattern_learner.learning_state.get('optimization_patterns', {})
            }
        
        # Add feedback insights
        if self.config['enable_feedback_learning']:
            pattern_analysis = await self.feedback_system.pattern_analyzer.analyze_patterns()
            context['user_preferences'] = pattern_analysis.get('user_preferences', {})
            context['improvement_opportunities'] = pattern_analysis.get('improvement_opportunities', [])
        
        return context
        
    async def _get_swarm_work_suggestions(self, context: Dict[str, Any], count: int) -> Dict[str, Any]:
        """Get work suggestions from swarm intelligence."""
        # Create a meta-task for the swarm to analyze
        meta_task = {
            'type': 'work_generation',
            'title': 'Generate High-Value Work Items',
            'description': f'Analyze system context and suggest {count} high-value tasks',
            'requirements': [
                'Consider current system state and performance',
                'Identify improvement opportunities', 
                'Suggest innovative tasks',
                'Balance different types of work'
            ],
            'context': context
        }
        
        # Get swarm analysis
        swarm_result = await self.swarm.process_task_swarm(meta_task)
        self.enhanced_metrics['swarm_analyses'] += 1
        
        return {
            'consensus': swarm_result.get('consensus', {}),
            'action_plan': swarm_result.get('action_plan', {}),
            'priority_insights': swarm_result.get('collective_review', {})
        }
        
    async def _generate_intelligent_tasks_enhanced(self, context: Dict[str, Any], 
                                                  count: int) -> List[Dict[str, Any]]:
        """Generate tasks using enhanced intelligence from all systems."""
        tasks = []
        
        # Get available repositories
        available_repos = self._get_available_repositories(context)
        
        # If no repositories, generate NEW_PROJECT tasks
        if not available_repos:
            self.logger.info("No repositories available - generating NEW_PROJECT tasks")
            for i in range(min(count, 2)):
                task = await self._generate_new_project_task(context)
                if task:
                    # Enhance with research insights
                    task = await self._enhance_task_with_research(task, context)
                    tasks.append(task)
            return tasks
        
        # Repository-first task generation with enhancements
        used_repositories = []
        
        for i in range(count):
            try:
                remaining_repos = [r for r in available_repos 
                                 if r.get('full_name') not in used_repositories]
                
                if not remaining_repos:
                    break
                
                # Use AI brain with swarm insights for repository selection
                repo_selection_context = {
                    **context,
                    'active_projects': remaining_repos,
                    'swarm_recommendations': context.get('swarm_insights', {}).get('consensus', {})
                }
                
                repo_selection = await self.ai_brain.select_repository_for_task(repo_selection_context)
                
                if not repo_selection:
                    continue
                
                # Analyze repository with pattern insights
                repo_name = repo_selection['repository']
                repo_analysis = await self.repository_analyzer.analyze_repository(repo_name)
                
                # Check for relevant patterns
                relevant_patterns = self._find_relevant_patterns(repo_name, context)
                if relevant_patterns:
                    repo_analysis['relevant_patterns'] = relevant_patterns
                
                # Generate task with all enhancements
                enhanced_context = {
                    **context,
                    'repository_analysis': repo_analysis,
                    'patterns': relevant_patterns
                }
                
                task = await self.task_generator.generate_task_for_repository(
                    repo_name,
                    repo_analysis,
                    enhanced_context
                )
                
                if task:
                    # Further enhance with research
                    task = await self._enhance_task_with_research(task, context)
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
                if validation['valid']:
                    validated_tasks.append(task)
                elif validation.get('corrected_task'):
                    validated_tasks.append(validation['corrected_task'])
            
            return validated_tasks
        
        return []
        
    async def _enhance_task_with_research(self, task: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a task with research insights."""
        if not self.config['enable_research_evolution']:
            return task
            
        # Find relevant research
        research_area = self._determine_research_area(task)
        relevant_research = self.knowledge_store.retrieve_research(
            research_type=research_area,
            min_quality=0.7,
            limit=3
        )
        
        if relevant_research:
            task['research_backed'] = True
            task['research_insights'] = [
                r.get('content', {}).get('key_insights', []) 
                for r in relevant_research
            ]
            
            # Store knowledge reference
            self.knowledge_store.record_value_delivered(
                relevant_research[0].get('id', ''),
                0.1  # Small value for being referenced
            )
            self.enhanced_metrics['knowledge_items_stored'] += 1
        
        return task
        
    def _find_relevant_patterns(self, repo_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find patterns relevant to a repository."""
        patterns = {}
        
        if not self.config['enable_pattern_learning']:
            return patterns
            
        learned = context.get('learned_patterns', {})
        
        # Check workflow patterns
        for date, data in learned.get('workflow_patterns', {}).items():
            for pattern in data.get('patterns', []):
                if repo_name in str(pattern):
                    patterns['workflow'] = pattern
                    break
                    
        # Check error patterns
        for date, data in learned.get('error_patterns', {}).items():
            for category, pattern in data.get('details', {}).items():
                if repo_name in str(pattern.get('sources', [])):
                    patterns['errors'] = {category: pattern}
                    
        return patterns
        
    def _determine_research_area(self, task: Dict[str, Any]) -> str:
        """Determine research area for a task."""
        task_type = task.get('type', '').lower()
        
        area_mapping = {
            'bug': 'outcome_patterns',
            'feature': 'innovation',
            'optimization': 'efficiency',
            'test': 'continuous_improvement',
            'security': 'critical_performance',
            'documentation': 'portfolio_management'
        }
        
        for key, area in area_mapping.items():
            if key in task_type:
                return area
                
        return 'general'
        
    async def _optimize_task_selection(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize task selection using multi-objective optimization."""
        if not tasks:
            return tasks
            
        # Prepare tasks for optimization
        for i, task in enumerate(tasks):
            task['id'] = task.get('id', f'task_{i}')
            task['name'] = task.get('title', f'Task {i}')
            
        # Run optimization
        optimized = await self.optimizer.optimize_for_objectives(tasks)
        
        # Return top tasks based on optimization
        return optimized[:self.max_workers]
        
    async def _trigger_targeted_research(self, work_items: List[WorkItem]):
        """Trigger targeted research based on current work."""
        if not self.config['enable_research_evolution']:
            return
            
        # Analyze work types
        work_types = [w.task_type for w in work_items]
        
        # Determine research focus
        if 'bug' in str(work_types).lower():
            research_area = 'error_prevention'
        elif 'feature' in str(work_types).lower():
            research_area = 'innovation'
        else:
            research_area = 'efficiency'
            
        # Execute targeted research
        await self.research_engine.execute_targeted_external_research(research_area)
        
    async def _execute_work_item(self, worker_id: str, work_item: WorkItem) -> Dict[str, Any]:
        """Execute work item with feedback collection and pattern tracking."""
        start_time = time.time()
        
        # Start monitoring for this task
        task_id = work_item.id
        task_type = work_item.task_type
        repository = work_item.repository or "unknown"
        
        # Get estimated duration from predictive analytics
        predicted_duration = self.monitoring.predict_task_duration(task_type, repository)
        
        # Start task monitoring
        metrics = self.monitor.start_task_monitoring(
            task_id=task_id,
            task_type=task_type,
            repository=repository,
            estimated_duration=predicted_duration
        )
        
        # Track lifecycle event
        self.monitoring.track_lifecycle_event(task_id, "started", {
            "worker_id": worker_id,
            "predicted_duration": predicted_duration
        })
        
        # Execute using parent method
        result = await super()._execute_work_item(worker_id, work_item)
        
        execution_time = time.time() - start_time
        success = result.get('status') == 'completed'
        
        # Complete monitoring
        self.monitor.complete_task_monitoring(
            task_id, 
            success=success, 
            error=result.get('error')
        )
        
        # Update worker performance
        self.monitoring.update_worker_performance(
            worker_id=worker_id,
            task_id=task_id,
            task_type=task_type,
            success=success,
            duration=execution_time
        )
        
        # Collect performance data
        if self.config['enable_feedback_learning']:
            await self.performance_tracker.track_execution(
                entity_type='task',
                entity_id=work_item.id,
                execution_time=execution_time,
                success=success,
                error=result.get('error')
            )
            self.enhanced_metrics['feedback_collected'] += 1
        
        # Record outcome for learning
        try:
            task = {
                'type': work_item.task_type,
                'title': work_item.title,
                'description': work_item.description,
                'repository': work_item.repository,
                'priority': work_item.priority.value if isinstance(work_item.priority, TaskPriority) else work_item.priority
            }
            
            # Original outcome learning
            await self.learning_system.record_outcome(task, result)
            
            # Research learning
            if self.config['enable_research_evolution']:
                await self._record_research_outcome(task, result)
                
        except Exception as e:
            self.logger.error(f"Error recording outcome: {e}")
        
        return result
        
    async def _record_research_outcome(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Record outcome for research learning."""
        # Determine if task was research-backed
        if task.get('research_backed'):
            # Update research quality based on outcome
            success = result.get('status') == 'completed'
            quality_boost = 0.1 if success else -0.05
            
            # This would update the research items that contributed
            # For now, just track that we used research
            self.enhanced_metrics['research_applied'] = self.enhanced_metrics.get('research_applied', 0) + 1
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with enhancement metrics."""
        # Get base status
        status = await super().get_system_status()
        
        # Add enhancement status
        status['enhancements'] = {
            'research_evolution': {
                'enabled': self.config['enable_research_evolution'],
                'cycles_completed': self.enhanced_metrics['research_cycles'],
                'knowledge_items': self.enhanced_metrics['knowledge_items_stored']
            },
            'pattern_learning': {
                'enabled': self.config['enable_pattern_learning'],
                'patterns_learned': self.enhanced_metrics['patterns_learned'],
                'last_learning': self.pattern_learner.learning_state.get('learned_at')
            },
            'swarm_intelligence': {
                'enabled': self.config['enable_swarm_intelligence'],
                'analyses_performed': self.enhanced_metrics['swarm_analyses'],
                'agent_count': len(self.swarm.agents) if hasattr(self, 'swarm') else 0
            },
            'feedback_learning': {
                'enabled': self.config['enable_feedback_learning'],
                'feedback_collected': self.enhanced_metrics['feedback_collected']
            },
            'multi_objective': {
                'enabled': self.config['enable_multi_objective'],
                'optimizations_applied': self.enhanced_metrics['optimizations_applied']
            }
        }
        
        # Add swarm status if available
        if hasattr(self, 'swarm'):
            status['swarm_status'] = self.swarm.get_swarm_status()
            
        return status
        
    async def run_learning_cycle(self):
        """Run a complete learning cycle across all systems."""
        self.logger.info("Running comprehensive learning cycle...")
        
        results = {}
        
        # Pattern learning
        if self.config['enable_pattern_learning']:
            try:
                pattern_results = await self.pattern_learner.apply_learned_patterns()
                results['patterns'] = pattern_results
            except Exception as e:
                self.logger.error(f"Pattern learning cycle error: {e}")
                
        # Feedback learning
        if self.config['enable_feedback_learning']:
            try:
                feedback_results = await self.feedback_system.run_learning_cycle()
                results['feedback'] = feedback_results
            except Exception as e:
                self.logger.error(f"Feedback learning cycle error: {e}")
                
        # Research evolution (manual trigger)
        if self.config['enable_research_evolution']:
            try:
                research_results = await self.research_engine.execute_research_cycle()
                results['research'] = research_results
            except Exception as e:
                self.logger.error(f"Research cycle error: {e}")
                
        self.logger.info(f"Learning cycle completed: {results}")
        return results
        
    async def shutdown(self):
        """Graceful shutdown with learning persistence."""
        self.logger.info("Shutting down Enhanced Hybrid Orchestrator...")
        
        # Save all learning states
        try:
            # Pattern learner state
            if hasattr(self, 'pattern_learner'):
                self.pattern_learner.save_learning_state()
                
            # Research knowledge
            if hasattr(self, 'knowledge_store'):
                self.knowledge_store.backup_research()
                
            # Final metrics
            final_state = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'enhanced_metrics': self.enhanced_metrics,
                'final_charter': self.current_charter
            }
            
            self.state_manager.update_state({'enhanced_final_state': final_state})
            
        except Exception as e:
            self.logger.error(f"Error saving final state: {e}")
        
        # Call parent shutdown
        await super().shutdown()


# Example usage
if __name__ == "__main__":
    async def run_enhanced():
        orchestrator = EnhancedHybridOrchestrator(
            max_workers=5,
            enable_parallel=True,
            enable_research=True
        )
        
        # Show initial status
        status = await orchestrator.get_system_status()
        print(f"Enhanced Orchestrator Status: {json.dumps(status, indent=2)}")
        
        # Run the orchestrator
        await orchestrator.run()
    
    asyncio.run(run_enhanced())