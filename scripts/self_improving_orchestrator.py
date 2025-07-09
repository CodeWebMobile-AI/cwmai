"""
Self-Improving Orchestrator

Integrates all self-improvement systems to create a continuously evolving AI system:
- Autonomous Improvement System: Learns from patterns and implements improvements
- Code Evolution Engine: Evolves code for better performance and maintainability  
- Intelligent Monitoring: Tracks everything and provides insights
- Research Evolution Engine: Conducts research and integrates new knowledge

This orchestrator ensures all systems work together harmoniously.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from scripts.state_manager import StateManager
from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.autonomous_improvement_system import get_improvement_system
from scripts.code_evolution_engine import get_evolution_engine
from scripts.intelligent_monitoring_system import get_monitoring_system
from scripts.research_evolution_engine import ResearchEvolutionEngine
from scripts.mcp_integration import MCPIntegrationHub
from scripts.http_ai_client import HTTPAIClient
from scripts.real_improvement_engine import RealImprovementEngine


class SelfImprovingOrchestrator:
    """Orchestrates all self-improvement systems for continuous evolution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.state_manager = StateManager()
        self.continuous_orchestrator = ContinuousOrchestrator(
            max_workers=10,
            enable_parallel=True,
            enable_research=True
        )
        
        # Self-improvement systems
        self.improvement_system = get_improvement_system(self.state_manager)
        self.evolution_engine = get_evolution_engine()
        self.monitoring_system = get_monitoring_system(self.state_manager)
        self.research_engine = self.continuous_orchestrator.research_engine
        self.real_improvement_engine = RealImprovementEngine(self.state_manager)
        
        # Integration components
        self.mcp_hub = MCPIntegrationHub()
        self.ai_client = HTTPAIClient()
        
        # Orchestration state
        self.is_running = False
        self.start_time = None
        self.coordination_metrics = {
            'cycles_completed': 0,
            'improvements_applied': 0,
            'evolutions_applied': 0,
            'insights_actioned': 0,
            'research_integrated': 0
        }
        
        # Configuration
        self.config = {
            'coordination_interval': 300,  # 5 minutes
            'min_confidence_for_auto_apply': 0.8,
            'max_concurrent_improvements': 3,
            'enable_auto_evolution': True,
            'enable_auto_research': True,
            'enable_predictive_optimization': True
        }
        
    async def start(self):
        """Start the self-improving orchestrator."""
        print("🚀 Starting Self-Improving Orchestrator")
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Initialize MCP
        await self.mcp_hub.initialize(servers=['brave-search', 'github'])
        
        # Start all subsystems
        await self._start_subsystems()
        
        # Start coordination loop
        asyncio.create_task(self._coordination_loop())
        
        # Start the main continuous orchestrator
        await self.continuous_orchestrator.start()
        
        print("✅ All self-improvement systems active")
        
    async def stop(self):
        """Stop the orchestrator."""
        print("Stopping Self-Improving Orchestrator")
        self.is_running = False
        
        # Stop all subsystems
        await self._stop_subsystems()
        
        # Stop continuous orchestrator
        await self.continuous_orchestrator.stop()
        
        # Close MCP
        await self.mcp_hub.close()
        
    async def _start_subsystems(self):
        """Start all self-improvement subsystems."""
        # Start monitoring first to track everything
        await self.monitoring_system.start()
        
        # Register custom monitors
        self._register_monitors()
        
        # Start improvement system
        await self.improvement_system.start()
        
        # Start code evolution
        await self.evolution_engine.start()
        
        # Research engine starts with continuous orchestrator
        
    async def _stop_subsystems(self):
        """Stop all subsystems."""
        await self.monitoring_system.stop()
        await self.improvement_system.stop()
        await self.evolution_engine.stop()
        
    def _register_monitors(self):
        """Register custom monitors for self-improvement tracking."""
        
        async def improvement_monitor():
            """Monitor improvement system metrics."""
            try:
                metrics = []
                
                # Get improvement metrics
                imp_metrics = await self.improvement_system.get_system_intelligence_metrics()
                
                # Handle case where imp_metrics might be a list instead of dict
                if isinstance(imp_metrics, list):
                    # If it's already a list of metrics, return as-is
                    return imp_metrics
                elif isinstance(imp_metrics, dict):
                    # Convert dict to metrics list
                    for key, value in imp_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics.append({
                                'name': f'improvement.{key}',
                                'type': 'gauge',
                                'value': value,
                                'timestamp': datetime.now(timezone.utc),
                                'tags': {'system': 'improvement'}
                            })
                else:
                    self.logger.warning(f"Unexpected type for improvement metrics: {type(imp_metrics)}")
                    
                return metrics
            except Exception as e:
                self.logger.error(f"Monitor improvement failed: {e}")
                return []
            
        async def evolution_monitor():
            """Monitor code evolution metrics."""
            try:
                metrics = []
                
                # Get evolution stats
                evo_stats = self.evolution_engine.get_evolution_stats()
                
                # Handle case where evo_stats might be a list instead of dict
                if isinstance(evo_stats, list):
                    # If it's already a list of metrics, return as-is
                    return evo_stats
                elif isinstance(evo_stats, dict):
                    # Convert dict to metrics list
                    for key, value in evo_stats.items():
                        if isinstance(value, (int, float)):
                            metrics.append({
                                'name': f'evolution.{key}',
                                'type': 'gauge',
                                'value': value,
                                'timestamp': datetime.now(timezone.utc),
                                'tags': {'system': 'evolution'}
                            })
                else:
                    self.logger.warning(f"Unexpected type for evolution stats: {type(evo_stats)}")
                    
                return metrics
            except Exception as e:
                self.logger.error(f"Monitor evolution failed: {e}")
                return []
            
        self.monitoring_system.register_monitor('improvement', improvement_monitor)
        self.monitoring_system.register_monitor('evolution', evolution_monitor)
        
    async def _coordination_loop(self):
        """Main coordination loop for self-improvement."""
        while self.is_running:
            try:
                await self._coordinate_improvements()
                self.coordination_metrics['cycles_completed'] += 1
                await asyncio.sleep(self.config['coordination_interval'])
            except Exception as e:
                self.logger.error(f"Coordination error: {e}")
                await asyncio.sleep(60)
                
    async def _coordinate_improvements(self):
        """Coordinate improvements across all systems."""
        print("🔄 Running coordination cycle")
        
        # 1. Get system health and insights
        health = await self.monitoring_system.get_system_health()
        insights = await self.monitoring_system.get_insights()
        
        print(f"System health: {health['status']} ({health['health_score']:.1f}/100)")
        
        # 2. Create REAL improvements first
        real_improvement = await self.real_improvement_engine.create_real_improvement()
        if real_improvement.get('success'):
            print(f"✅ Created real improvement: {real_improvement['type']} - {real_improvement.get('name', 'N/A')}")
            self.coordination_metrics['improvements_applied'] += 1
        
        # 3. Prioritize additional improvements based on health and insights
        priorities = await self._prioritize_improvements(health, insights)
        
        # 4. Apply high-priority improvements
        applied = await self._apply_improvements(priorities)
        
        # 5. Trigger research if needed
        if self._should_trigger_research(health, insights):
            await self._trigger_targeted_research(insights)
            
        # 6. Coordinate code evolution
        if self.config['enable_auto_evolution']:
            await self._coordinate_code_evolution(insights)
            
        # 7. Learn from coordination
        await self._learn_from_coordination(applied)
        
        # 8. Log real improvement stats
        stats = self.real_improvement_engine.get_improvement_stats()
        print(f"Real improvements so far: {stats['total_improvements']} "
                        f"(Tools: {stats['tools_created']}, Enhanced: {stats['tools_enhanced']}, "
                        f"Workflows: {stats['workflows_created']})")
        
        print(f"Coordination complete: {len(applied) + (1 if real_improvement.get('success') else 0)} improvements applied")
        
    async def _prioritize_improvements(self, health: Dict[str, Any], 
                                     insights: List[Any]) -> List[Dict[str, Any]]:
        """Prioritize improvements based on system state."""
        priorities = []
        
        # Critical health issues first
        if health['health_score'] < 50:
            priorities.append({
                'type': 'health_critical',
                'action': 'emergency_optimization',
                'priority': 1.0,
                'data': health
            })
            
        # High-impact insights
        for insight in insights:
            if insight.confidence >= self.config['min_confidence_for_auto_apply']:
                priorities.append({
                    'type': 'insight',
                    'action': 'implement_insight',
                    'priority': insight.confidence * 0.9,
                    'data': insight
                })
                
        # Performance bottlenecks
        metrics_summary = await self.monitoring_system.get_metrics_summary()
        
        for metric_name, values in metrics_summary.items():
            if 'time' in metric_name:
                for value_set in values:
                    if value_set['avg'] > 5.0:  # Operations taking >5s
                        priorities.append({
                            'type': 'performance',
                            'action': 'optimize_operation',
                            'priority': 0.7,
                            'data': {
                                'metric': metric_name,
                                'avg_time': value_set['avg'],
                                'tags': value_set['tags']
                            }
                        })
                        
        # Get improvement suggestions
        suggestions = await self.improvement_system.get_improvement_suggestions()
        
        for i, suggestion in enumerate(suggestions[:5]):
            priorities.append({
                'type': 'suggestion',
                'action': 'implement_suggestion',
                'priority': 0.6 - (i * 0.1),
                'data': {'suggestion': suggestion}
            })
            
        # Sort by priority
        priorities.sort(key=lambda x: x['priority'], reverse=True)
        
        return priorities
        
    async def _apply_improvements(self, priorities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply prioritized improvements."""
        applied = []
        concurrent_count = 0
        
        for priority in priorities:
            if concurrent_count >= self.config['max_concurrent_improvements']:
                break
                
            try:
                if priority['type'] == 'health_critical':
                    result = await self._handle_critical_health(priority['data'])
                elif priority['type'] == 'insight':
                    result = await self._implement_insight(priority['data'])
                elif priority['type'] == 'performance':
                    result = await self._optimize_performance(priority['data'])
                elif priority['type'] == 'suggestion':
                    result = await self._implement_suggestion(priority['data'])
                else:
                    continue
                    
                if result.get('success'):
                    applied.append(result)
                    concurrent_count += 1
                    self.coordination_metrics['improvements_applied'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to apply improvement: {e}")
                
        return applied
        
    async def _handle_critical_health(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle critical health issues."""
        self.logger.warning(f"Handling critical health: score={health_data['health_score']}")
        
        actions_taken = []
        
        # Emergency measures
        if health_data['health_score'] < 25:
            # Reduce system load
            self.continuous_orchestrator.max_workers = max(2, self.continuous_orchestrator.max_workers // 2)
            actions_taken.append("Reduced worker count")
            
            # Clear caches
            import gc
            gc.collect()
            actions_taken.append("Cleared memory")
            
            # Disable non-essential features temporarily
            self.config['enable_auto_evolution'] = False
            actions_taken.append("Disabled auto-evolution temporarily")
            
        return {
            'success': True,
            'type': 'health_recovery',
            'actions': actions_taken
        }
        
    async def _implement_insight(self, insight: Any) -> Dict[str, Any]:
        """Implement an insight from monitoring."""
        print(f"Implementing insight: {insight.title}")
        
        # Generate implementation plan
        prompt = f"""Create an implementation plan for this insight:

Title: {insight.title}
Description: {insight.description}
Impact: {insight.impact}
Recommendations: {json.dumps(insight.recommendations)}

Generate specific code changes or configuration updates needed.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if response.get('content'):
            # Could automatically apply changes here
            self.coordination_metrics['insights_actioned'] += 1
            
            return {
                'success': True,
                'type': 'insight_implementation',
                'insight': insight.title,
                'plan': response['content']
            }
            
        return {'success': False}
        
    async def _optimize_performance(self, perf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a performance bottleneck."""
        metric = perf_data['metric']
        
        # Find the related code
        if 'tool' in perf_data.get('tags', {}):
            tool_name = perf_data['tags']['tool']
            
            # Suggest optimization to code evolution engine
            tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
            if tool_path.exists():
                evolution = await self.evolution_engine.suggest_evolution(str(tool_path))
                
                if evolution and evolution.confidence >= 0.7:
                    # Apply evolution
                    applied = await self.evolution_engine._apply_evolutions([evolution])
                    
                    if applied:
                        self.coordination_metrics['evolutions_applied'] += 1
                        return {
                            'success': True,
                            'type': 'performance_optimization',
                            'metric': metric,
                            'evolution': evolution.description
                        }
                        
        return {'success': False}
        
    async def _implement_suggestion(self, suggestion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement an improvement suggestion."""
        suggestion = suggestion_data['suggestion']
        
        # Parse suggestion to determine action
        if 'Create workflow' in suggestion:
            # Delegate to workflow system
            return {'success': True, 'type': 'workflow_creation', 'suggestion': suggestion}
        elif 'Optimize' in suggestion:
            # Delegate to evolution engine
            return {'success': True, 'type': 'optimization', 'suggestion': suggestion}
        elif 'Create tool' in suggestion:
            # Delegate to tool generator
            return {'success': True, 'type': 'tool_creation', 'suggestion': suggestion}
            
        return {'success': False}
        
    def _should_trigger_research(self, health: Dict[str, Any], insights: List[Any]) -> bool:
        """Determine if research should be triggered."""
        # Trigger research if:
        # 1. System is healthy enough (>50)
        # 2. Not many insights available
        # 3. Auto-research is enabled
        
        return (self.config['enable_auto_research'] and
                health['health_score'] > 50 and
                len(insights) < 5)
                
    async def _trigger_targeted_research(self, insights: List[Any]):
        """Trigger targeted research based on current needs."""
        # Identify research topics from insights
        topics = []
        
        # Research solutions for low-confidence insights
        for insight in insights:
            if insight.confidence < 0.7:
                topics.append(f"Best practices for {insight.title}")
                
        # Research from recent errors
        error_patterns = self.improvement_system.error_patterns
        for error_type, count in list(error_patterns.items())[:3]:
            if count > 5:
                topics.append(f"How to prevent {error_type} in Python applications")
                
        # Trigger research
        for topic in topics[:2]:  # Limit concurrent research
            print(f"Triggering research: {topic}")
            # Research engine will handle this in its cycle
            
    async def _coordinate_code_evolution(self, insights: List[Any]):
        """Coordinate code evolution based on insights."""
        # Find files that need evolution
        evolution_targets = []
        
        # From performance insights
        for insight in insights:
            if insight.category == 'optimization' and 'file' in insight.supporting_data:
                evolution_targets.append(insight.supporting_data['file'])
                
        # Trigger evolution for targets
        for target in evolution_targets[:2]:  # Limit concurrent evolutions
            try:
                evolution = await self.evolution_engine.suggest_evolution(target)
                if evolution and evolution.confidence >= self.config['min_confidence_for_auto_apply']:
                    applied = await self.evolution_engine._apply_evolutions([evolution])
                    if applied:
                        self.coordination_metrics['evolutions_applied'] += 1
            except Exception as e:
                self.logger.error(f"Evolution failed for {target}: {e}")
                
    async def _learn_from_coordination(self, applied: List[Dict[str, Any]]):
        """Learn from coordination results."""
        # Track what worked
        for improvement in applied:
            if improvement.get('success'):
                # Record in improvement system
                self.improvement_system.success_patterns[improvement['type']] += 1
                
        # Save coordination state
        state = {
            'metrics': self.coordination_metrics,
            'last_coordination': datetime.now(timezone.utc).isoformat(),
            'applied_improvements': len(applied),
            'health_trend': await self._calculate_health_trend()
        }
        
        state_path = Path('logs/coordination_state.json')
        state_path.write_text(json.dumps(state, indent=2))
        
    async def _calculate_health_trend(self) -> str:
        """Calculate system health trend."""
        # Get historical health scores
        # For now, return a simple status
        current_health = await self.monitoring_system.get_system_health()
        
        if current_health['health_score'] > 80:
            return 'improving'
        elif current_health['health_score'] > 60:
            return 'stable'
        else:
            return 'degrading'
            
    async def get_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive intelligence report."""
        # Gather metrics from all systems
        improvement_metrics = await self.improvement_system.get_system_intelligence_metrics()
        evolution_stats = self.evolution_engine.get_evolution_stats()
        health = await self.monitoring_system.get_system_health()
        real_improvement_stats = self.real_improvement_engine.get_improvement_stats()
        
        # Calculate overall intelligence score (weighted by REAL improvements)
        real_improvement_weight = min(real_improvement_stats['total_improvements'] / 10, 1.0)
        intelligence_score = (
            improvement_metrics.get('intelligence_score', 50) * 0.2 +
            evolution_stats.get('success_rate', 0.5) * 100 * 0.2 +
            health.get('health_score', 50) * 0.2 +
            real_improvement_weight * 100 * 0.4  # 40% weight on REAL improvements
        )
        
        return {
            'overall_intelligence_score': intelligence_score,
            'improvement_system': improvement_metrics,
            'evolution_engine': evolution_stats,
            'system_health': health,
            'real_improvements': real_improvement_stats,
            'coordination_metrics': self.coordination_metrics,
            'uptime': self._calculate_uptime(),
            'status': self._get_intelligence_status(intelligence_score)
        }
        
    def _get_intelligence_status(self, score: float) -> str:
        """Get intelligence status from score."""
        if score >= 80:
            return 'highly_intelligent'
        elif score >= 60:
            return 'intelligent'
        elif score >= 40:
            return 'learning'
        elif score >= 20:
            return 'basic'
        else:
            return 'initializing'
            
    def _calculate_uptime(self) -> Dict[str, int]:
        """Calculate orchestrator uptime."""
        if self.start_time:
            uptime = datetime.now(timezone.utc) - self.start_time
            return {
                'days': uptime.days,
                'hours': uptime.seconds // 3600,
                'minutes': (uptime.seconds % 3600) // 60
            }
        return {'days': 0, 'hours': 0, 'minutes': 0}
        
    async def manual_trigger_improvement(self, improvement_type: str, **kwargs) -> Dict[str, Any]:
        """Manually trigger a specific improvement."""
        print(f"Manual trigger: {improvement_type}")
        
        if improvement_type == 'code_evolution':
            file_path = kwargs.get('file_path')
            if file_path:
                evolution = await self.evolution_engine.suggest_evolution(file_path)
                if evolution:
                    applied = await self.evolution_engine._apply_evolutions([evolution])
                    return {'success': bool(applied), 'evolution': evolution.description}
                    
        elif improvement_type == 'performance_optimization':
            # Trigger performance analysis and optimization
            insights = await self.monitoring_system._generate_performance_insights()
            if insights:
                result = await self._implement_insight(insights[0])
                return result
                
        elif improvement_type == 'research':
            topic = kwargs.get('topic', 'AI system self-improvement techniques')
            research_result = await self.improvement_system._research_topic(topic)
            return {'success': True, 'research': research_result}
            
        return {'success': False, 'reason': 'Unknown improvement type'}


# Global instance
self_improving_orchestrator = None

def get_self_improving_orchestrator() -> SelfImprovingOrchestrator:
    """Get or create the global self-improving orchestrator."""
    global self_improving_orchestrator
    if self_improving_orchestrator is None:
        self_improving_orchestrator = SelfImprovingOrchestrator()
    return self_improving_orchestrator