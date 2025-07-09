"""
Feedback Learning System
========================

This module implements feedback loops that:
1. Collect user feedback on tool and workflow effectiveness
2. Track performance metrics and usage patterns
3. Learn from successes and failures
4. Automatically improve tool generation and workflow creation
5. Adapt to user preferences over time
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics
from collections import defaultdict, deque

from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient


class FeedbackCollector:
    """Collects and processes user feedback."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.feedback_queue = deque(maxlen=1000)  # Keep last 1000 feedback items
        
    async def collect_feedback(
        self,
        entity_type: str,  # 'tool', 'workflow', 'response'
        entity_id: str,
        rating: Optional[int] = None,  # 1-5 scale
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Collect feedback on an entity."""
        
        feedback = {
            'id': f"feedback_{datetime.now().timestamp()}",
            'entity_type': entity_type,
            'entity_id': entity_id,
            'rating': rating,
            'comment': comment,
            'metrics': metrics or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add to queue
        self.feedback_queue.append(feedback)
        
        # Persist to state
        self._save_feedback(feedback)
        
        # Process immediately for critical feedback
        if rating and rating <= 2:
            await self._process_negative_feedback(feedback)
            
        return {
            'success': True,
            'feedback_id': feedback['id'],
            'message': 'Feedback collected successfully'
        }
        
    def _save_feedback(self, feedback: Dict[str, Any]):
        """Save feedback to persistent state."""
        state = self.state_manager.get_state()
        
        if 'feedback_history' not in state:
            state['feedback_history'] = []
            
        state['feedback_history'].append(feedback)
        
        # Keep only recent feedback
        state['feedback_history'] = state['feedback_history'][-1000:]
        
        self.state_manager.state = state
        self.state_manager.save_state()
        
    async def _process_negative_feedback(self, feedback: Dict[str, Any]):
        """Process negative feedback immediately."""
        self.logger.warning(
            f"Negative feedback for {feedback['entity_type']} {feedback['entity_id']}: "
            f"Rating {feedback['rating']}, Comment: {feedback.get('comment', 'N/A')}"
        )
        
        # Mark entity for improvement
        state = self.state_manager.get_state()
        
        if 'entities_needing_improvement' not in state:
            state['entities_needing_improvement'] = []
            
        state['entities_needing_improvement'].append({
            'entity_type': feedback['entity_type'],
            'entity_id': feedback['entity_id'],
            'feedback': feedback,
            'marked_at': datetime.now(timezone.utc).isoformat()
        })
        
        self.state_manager.state = state
        self.state_manager.save_state()
        
    def get_entity_feedback(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific entity."""
        return [
            fb for fb in self.feedback_queue
            if fb['entity_type'] == entity_type and fb['entity_id'] == entity_id
        ]
        
    def get_aggregated_feedback(self, entity_type: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated feedback statistics."""
        feedback_list = list(self.feedback_queue)
        
        if entity_type:
            feedback_list = [fb for fb in feedback_list if fb['entity_type'] == entity_type]
            
        if not feedback_list:
            return {'count': 0, 'average_rating': None}
            
        ratings = [fb['rating'] for fb in feedback_list if fb['rating']]
        
        return {
            'count': len(feedback_list),
            'average_rating': statistics.mean(ratings) if ratings else None,
            'rating_distribution': self._calculate_distribution(ratings),
            'recent_comments': [fb['comment'] for fb in feedback_list[-5:] if fb.get('comment')]
        }
        
    def _calculate_distribution(self, ratings: List[int]) -> Dict[int, int]:
        """Calculate rating distribution."""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            if rating in distribution:
                distribution[rating] += 1
        return distribution


class PerformanceTracker:
    """Tracks performance metrics for tools and workflows."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.metrics_buffer = defaultdict(list)
        
    async def track_execution(
        self,
        entity_type: str,
        entity_id: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        resource_usage: Optional[Dict[str, float]] = None
    ):
        """Track execution metrics."""
        
        metric = {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'execution_time': execution_time,
            'success': success,
            'error': error,
            'resource_usage': resource_usage or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add to buffer
        key = f"{entity_type}:{entity_id}"
        self.metrics_buffer[key].append(metric)
        
        # Persist periodically (every 10 executions)
        if len(self.metrics_buffer[key]) >= 10:
            await self._persist_metrics(key)
            
    async def _persist_metrics(self, key: str):
        """Persist metrics to state."""
        state = self.state_manager.get_state()
        
        if 'performance_metrics' not in state:
            state['performance_metrics'] = {}
            
        if key not in state['performance_metrics']:
            state['performance_metrics'][key] = []
            
        # Add buffered metrics
        state['performance_metrics'][key].extend(self.metrics_buffer[key])
        
        # Keep only recent metrics (last 100)
        state['performance_metrics'][key] = state['performance_metrics'][key][-100:]
        
        # Clear buffer
        self.metrics_buffer[key] = []
        
        self.state_manager.state = state
        self.state_manager.save_state()
        
    def get_performance_summary(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get performance summary for an entity."""
        key = f"{entity_type}:{entity_id}"
        
        # Combine buffered and persisted metrics
        metrics = self.metrics_buffer[key].copy()
        
        state = self.state_manager.get_state()
        if 'performance_metrics' in state and key in state['performance_metrics']:
            metrics.extend(state['performance_metrics'][key])
            
        if not metrics:
            return {'executions': 0}
            
        execution_times = [m['execution_time'] for m in metrics]
        success_count = sum(1 for m in metrics if m['success'])
        
        return {
            'executions': len(metrics),
            'success_rate': success_count / len(metrics),
            'avg_execution_time': statistics.mean(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'recent_errors': [m['error'] for m in metrics[-5:] if m.get('error')],
            'performance_trend': self._calculate_trend(metrics)
        }
        
    def _calculate_trend(self, metrics: List[Dict[str, Any]]) -> str:
        """Calculate performance trend."""
        if len(metrics) < 10:
            return 'insufficient_data'
            
        # Compare recent vs older performance
        mid_point = len(metrics) // 2
        old_times = [m['execution_time'] for m in metrics[:mid_point]]
        new_times = [m['execution_time'] for m in metrics[mid_point:]]
        
        old_avg = statistics.mean(old_times)
        new_avg = statistics.mean(new_times)
        
        improvement = (old_avg - new_avg) / old_avg * 100
        
        if improvement > 10:
            return 'improving'
        elif improvement < -10:
            return 'degrading'
        else:
            return 'stable'


class UsagePatternAnalyzer:
    """Analyzes usage patterns to identify improvement opportunities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.ai_client = HTTPAIClient()
        
    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns across all tools and workflows."""
        state = self.state_manager.get_state()
        
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tool_patterns': await self._analyze_tool_patterns(state),
            'workflow_patterns': await self._analyze_workflow_patterns(state),
            'user_preferences': await self._analyze_user_preferences(state),
            'improvement_opportunities': []
        }
        
        # Identify improvement opportunities
        analysis['improvement_opportunities'] = await self._identify_improvements(analysis)
        
        return analysis
        
    async def _analyze_tool_patterns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tool usage patterns."""
        performance_metrics = state.get('performance_metrics', {})
        
        tool_stats = {}
        
        for key, metrics in performance_metrics.items():
            if key.startswith('tool:'):
                tool_name = key.split(':', 1)[1]
                
                if metrics:
                    execution_times = [m['execution_time'] for m in metrics]
                    success_count = sum(1 for m in metrics if m['success'])
                    
                    tool_stats[tool_name] = {
                        'usage_count': len(metrics),
                        'success_rate': success_count / len(metrics),
                        'avg_time': statistics.mean(execution_times),
                        'common_errors': self._get_common_errors(metrics)
                    }
                    
        return {
            'total_tools': len(tool_stats),
            'most_used': sorted(tool_stats.items(), key=lambda x: x[1]['usage_count'], reverse=True)[:5],
            'least_successful': sorted(tool_stats.items(), key=lambda x: x[1]['success_rate'])[:5],
            'slowest': sorted(tool_stats.items(), key=lambda x: x[1]['avg_time'], reverse=True)[:5]
        }
        
    async def _analyze_workflow_patterns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow usage patterns."""
        workflows = state.get('workflows', [])
        
        workflow_stats = {}
        
        # Handle both list and dict formats
        if isinstance(workflows, list):
            for wf_data in workflows:
                if isinstance(wf_data, dict) and wf_data.get('status') == 'completed':
                    wf_name = wf_data.get('name', 'unknown')
                    workflow_stats[wf_name] = {
                        'executions': 1,
                        'avg_duration': wf_data.get('duration_ms', 0),
                        'step_count': len(wf_data.get('steps', []))
                    }
        elif isinstance(workflows, dict):
            for wf_id, wf_data in workflows.items():
                if isinstance(wf_data, dict) and wf_data.get('status') == 'completed':
                    workflow_stats[wf_data.get('name', wf_id)] = {
                        'executions': 1,  # Would need to track multiple executions
                        'avg_duration': wf_data.get('duration_ms', 0),
                        'step_count': len(wf_data.get('steps', []))
                    }
                
        return {
            'total_workflows': len(workflow_stats),
            'common_patterns': self._identify_common_patterns(workflows),
            'optimization_candidates': self._find_optimization_candidates(workflows)
        }
        
    async def _analyze_user_preferences(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user preferences from feedback and usage."""
        feedback_history = state.get('feedback_history', [])
        
        preferences = {
            'preferred_tools': [],
            'avoided_tools': [],
            'feedback_themes': []
        }
        
        if feedback_history:
            # Analyze ratings
            tool_ratings = defaultdict(list)
            
            for fb in feedback_history:
                if fb['entity_type'] == 'tool' and fb.get('rating'):
                    tool_ratings[fb['entity_id']].append(fb['rating'])
                    
            # Find preferred and avoided tools
            for tool, ratings in tool_ratings.items():
                avg_rating = statistics.mean(ratings)
                if avg_rating >= 4:
                    preferences['preferred_tools'].append(tool)
                elif avg_rating <= 2:
                    preferences['avoided_tools'].append(tool)
                    
            # Extract themes from comments
            comments = [fb['comment'] for fb in feedback_history if fb.get('comment')]
            if comments:
                preferences['feedback_themes'] = await self._extract_themes(comments)
                
        return preferences
        
    async def _extract_themes(self, comments: List[str]) -> List[str]:
        """Extract common themes from feedback comments."""
        if not comments:
            return []
            
        prompt = f"""
        Extract the top 5 common themes from these user feedback comments:
        
        {json.dumps(comments[:20], indent=2)}
        
        Return a JSON list of theme strings.
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            return json.loads(response.get('content', '[]'))
        except:
            return []
            
    def _get_common_errors(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Get common errors from metrics."""
        errors = [m['error'] for m in metrics if m.get('error')]
        
        # Count occurrences
        error_counts = defaultdict(int)
        for error in errors:
            # Simplify error message
            error_key = error.split(':')[0] if ':' in error else error
            error_counts[error_key] += 1
            
        # Return top 3
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
    def _identify_common_patterns(self, workflows: Any) -> List[Dict[str, Any]]:
        """Identify common workflow patterns."""
        step_sequences = defaultdict(int)
        
        # Handle both list and dict formats
        if isinstance(workflows, list):
            workflow_items = workflows
        elif isinstance(workflows, dict):
            workflow_items = workflows.values()
        else:
            return []
            
        for wf_data in workflow_items:
            if isinstance(wf_data, dict) and 'steps' in wf_data:
                # Extract step sequence
                sequence = tuple(step.get('type', 'unknown') for step in wf_data['steps'])
                step_sequences[sequence] += 1
                
        # Return common sequences
        return [
            {'sequence': list(seq), 'count': count}
            for seq, count in sorted(step_sequences.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
    def _find_optimization_candidates(self, workflows: Any) -> List[str]:
        """Find workflows that could be optimized."""
        candidates = []
        
        # Handle both list and dict formats
        if isinstance(workflows, list):
            workflow_items = [(f"workflow_{i}", wf) for i, wf in enumerate(workflows)]
        elif isinstance(workflows, dict):
            workflow_items = workflows.items()
        else:
            return []
            
        for wf_id, wf_data in workflow_items:
            if isinstance(wf_data, dict):
                # Check for sequential steps that could be parallelized
                steps = wf_data.get('steps', [])
                if len(steps) > 3:
                    # Simple heuristic: workflows with many sequential steps
                    sequential_count = sum(1 for s in steps if s.get('type') == 'tool')
                    if sequential_count > 3:
                        candidates.append(wf_data.get('name', wf_id))
                        
        return candidates[:5]
        
    async def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities."""
        improvements = []
        
        # Slow tools
        for tool, stats in analysis['tool_patterns'].get('slowest', []):
            if stats['avg_time'] > 5.0:
                improvements.append({
                    'type': 'optimize_tool',
                    'target': tool,
                    'reason': f'Average execution time {stats["avg_time"]:.1f}s',
                    'suggestion': 'Add caching or optimize algorithm'
                })
                
        # Unreliable tools
        for tool, stats in analysis['tool_patterns'].get('least_successful', []):
            if stats['success_rate'] < 0.8:
                improvements.append({
                    'type': 'fix_tool',
                    'target': tool,
                    'reason': f'Success rate only {stats["success_rate"]:.1%}',
                    'suggestion': 'Review error handling and fix common errors'
                })
                
        # Workflow optimizations
        for workflow in analysis['workflow_patterns'].get('optimization_candidates', []):
            improvements.append({
                'type': 'optimize_workflow',
                'target': workflow,
                'reason': 'Many sequential steps could be parallelized',
                'suggestion': 'Convert sequential steps to parallel execution'
            })
            
        return improvements


class AdaptiveLearner:
    """Learns and adapts based on feedback and usage patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.feedback_collector = FeedbackCollector()
        self.performance_tracker = PerformanceTracker()
        self.pattern_analyzer = UsagePatternAnalyzer()
        self.ai_client = HTTPAIClient()
        
    async def learn_and_adapt(self) -> Dict[str, Any]:
        """Main learning cycle - analyze and adapt."""
        
        # Analyze current patterns
        analysis = await self.pattern_analyzer.analyze_patterns()
        
        # Generate adaptations
        adaptations = await self._generate_adaptations(analysis)
        
        # Apply adaptations
        results = await self._apply_adaptations(adaptations)
        
        # Store learning results
        self._store_learning_results(analysis, adaptations, results)
        
        return {
            'analysis': analysis,
            'adaptations': adaptations,
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    async def _generate_adaptations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptations based on analysis."""
        adaptations = []
        
        # Process improvement opportunities
        for improvement in analysis.get('improvement_opportunities', []):
            if improvement['type'] == 'optimize_tool':
                adaptation = await self._generate_tool_optimization(improvement)
                if adaptation:
                    adaptations.append(adaptation)
                    
            elif improvement['type'] == 'fix_tool':
                adaptation = await self._generate_tool_fix(improvement)
                if adaptation:
                    adaptations.append(adaptation)
                    
            elif improvement['type'] == 'optimize_workflow':
                adaptation = await self._generate_workflow_optimization(improvement)
                if adaptation:
                    adaptations.append(adaptation)
                    
        # Add preference-based adaptations
        preferences = analysis.get('user_preferences', {})
        if preferences.get('avoided_tools'):
            for tool in preferences['avoided_tools'][:3]:
                adaptations.append({
                    'type': 'improve_avoided_tool',
                    'target': tool,
                    'action': 'redesign',
                    'reason': 'Low user ratings'
                })
                
        return adaptations
        
    async def _generate_tool_optimization(self, improvement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate optimization for a slow tool."""
        tool_name = improvement['target']
        
        # Get tool code
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        if not tool_path.exists():
            return None
            
        current_code = tool_path.read_text()
        
        # Generate optimization suggestions
        prompt = f"""
        This tool is slow (avg {improvement['reason']}). Suggest specific code optimizations:
        
        Current code:
        {current_code[:1000]}...
        
        Provide specific optimization techniques like:
        - Add caching with @lru_cache
        - Use asyncio.gather for parallel operations
        - Optimize data structures
        - Add early returns
        
        Return JSON with optimization steps.
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            optimizations = json.loads(response.get('content', '{}'))
            return {
                'type': 'tool_optimization',
                'target': tool_name,
                'optimizations': optimizations,
                'current_performance': improvement['reason']
            }
        except:
            return None
            
    async def _generate_tool_fix(self, improvement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate fixes for an unreliable tool."""
        tool_name = improvement['target']
        
        # Get recent errors
        performance = self.performance_tracker.get_performance_summary('tool', tool_name)
        recent_errors = performance.get('recent_errors', [])
        
        if not recent_errors:
            return None
            
        return {
            'type': 'tool_fix',
            'target': tool_name,
            'errors_to_fix': recent_errors,
            'success_rate': improvement['reason']
        }
        
    async def _generate_workflow_optimization(self, improvement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate workflow optimization."""
        return {
            'type': 'workflow_optimization',
            'target': improvement['target'],
            'optimization': 'parallelize_steps',
            'reason': improvement['reason']
        }
        
    async def _apply_adaptations(self, adaptations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply the generated adaptations."""
        results = []
        
        for adaptation in adaptations:
            try:
                if adaptation['type'] == 'tool_optimization':
                    result = await self._apply_tool_optimization(adaptation)
                elif adaptation['type'] == 'tool_fix':
                    result = await self._apply_tool_fix(adaptation)
                elif adaptation['type'] == 'workflow_optimization':
                    result = await self._apply_workflow_optimization(adaptation)
                else:
                    result = {'success': False, 'reason': 'Unknown adaptation type'}
                    
                results.append({
                    'adaptation': adaptation,
                    'result': result,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Failed to apply adaptation: {e}")
                results.append({
                    'adaptation': adaptation,
                    'result': {'success': False, 'error': str(e)},
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
        return results
        
    async def _apply_tool_optimization(self, adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization to a tool."""
        # This would integrate with the tool regeneration system
        return {
            'success': True,
            'message': f"Tool {adaptation['target']} marked for optimization",
            'optimizations': adaptation.get('optimizations', [])
        }
        
    async def _apply_tool_fix(self, adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fixes to a tool."""
        # This would integrate with the tool error fixing system
        return {
            'success': True,
            'message': f"Tool {adaptation['target']} marked for error fixes",
            'errors_to_fix': adaptation.get('errors_to_fix', [])
        }
        
    async def _apply_workflow_optimization(self, adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization to a workflow."""
        # This would integrate with the workflow optimization system
        return {
            'success': True,
            'message': f"Workflow {adaptation['target']} marked for optimization",
            'optimization_type': adaptation.get('optimization', 'parallelize')
        }
        
    def _store_learning_results(
        self,
        analysis: Dict[str, Any],
        adaptations: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ):
        """Store learning results for future reference."""
        state = self.state_manager.get_state()
        
        if 'learning_history' not in state:
            state['learning_history'] = []
            
        state['learning_history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_summary': {
                'tool_count': analysis['tool_patterns']['total_tools'],
                'workflow_count': analysis['workflow_patterns']['total_workflows'],
                'improvement_count': len(analysis['improvement_opportunities'])
            },
            'adaptations_applied': len([r for r in results if r['result'].get('success')]),
            'adaptations_failed': len([r for r in results if not r['result'].get('success')])
        })
        
        # Keep only recent history
        state['learning_history'] = state['learning_history'][-50:]
        
        self.state_manager.state = state
        self.state_manager.save_state()


# Main feedback learning system
class FeedbackLearningSystem:
    """Main system that coordinates all feedback and learning components."""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.performance_tracker = PerformanceTracker()
        self.pattern_analyzer = UsagePatternAnalyzer()
        self.adaptive_learner = AdaptiveLearner()
        
    async def process_execution_feedback(
        self,
        entity_type: str,
        entity_id: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        user_rating: Optional[int] = None,
        user_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process feedback from an execution."""
        
        # Track performance
        await self.performance_tracker.track_execution(
            entity_type, entity_id, execution_time, success, error
        )
        
        # Collect user feedback if provided
        if user_rating is not None or user_comment:
            await self.feedback_collector.collect_feedback(
                entity_type, entity_id, user_rating, user_comment,
                {'execution_time': execution_time, 'success': success}
            )
            
        return {
            'success': True,
            'message': 'Feedback processed',
            'performance_tracked': True,
            'user_feedback_collected': user_rating is not None or user_comment is not None
        }
        
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """Run a complete learning cycle."""
        return await self.adaptive_learner.learn_and_adapt()
        
    async def get_entity_insights(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get insights about a specific entity."""
        
        # Get performance data
        performance = self.performance_tracker.get_performance_summary(entity_type, entity_id)
        
        # Get feedback data
        feedback = self.feedback_collector.get_entity_feedback(entity_type, entity_id)
        feedback_summary = self.feedback_collector.get_aggregated_feedback(entity_type)
        
        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'performance': performance,
            'user_feedback': {
                'count': len(feedback),
                'average_rating': statistics.mean([f['rating'] for f in feedback if f.get('rating')]) if feedback else None,
                'recent_comments': [f['comment'] for f in feedback[-3:] if f.get('comment')]
            },
            'insights': self._generate_insights(performance, feedback)
        }
        
    def _generate_insights(self, performance: Dict[str, Any], feedback: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable insights."""
        insights = []
        
        # Performance insights
        if performance.get('success_rate', 1) < 0.8:
            insights.append(f"Low success rate ({performance['success_rate']:.1%}) - needs reliability improvements")
            
        if performance.get('avg_execution_time', 0) > 5:
            insights.append(f"Slow execution ({performance['avg_execution_time']:.1f}s) - consider optimization")
            
        if performance.get('performance_trend') == 'degrading':
            insights.append("Performance is degrading over time - investigate recent changes")
            
        # Feedback insights
        if feedback:
            ratings = [f['rating'] for f in feedback if f.get('rating')]
            if ratings and statistics.mean(ratings) < 3:
                insights.append("Low user satisfaction - review user comments for improvement areas")
                
        return insights


# Convenience functions
async def track_execution(
    entity_type: str,
    entity_id: str,
    execution_time: float,
    success: bool,
    **kwargs
) -> Dict[str, Any]:
    """Track execution and collect feedback."""
    system = FeedbackLearningSystem()
    return await system.process_execution_feedback(
        entity_type, entity_id, execution_time, success, **kwargs
    )
    
async def learn_and_improve() -> Dict[str, Any]:
    """Run learning cycle to improve system."""
    system = FeedbackLearningSystem()
    return await system.run_learning_cycle()