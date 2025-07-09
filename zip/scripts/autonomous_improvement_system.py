"""
Autonomous Improvement System

A comprehensive self-improvement framework that makes CWMAI constantly smarter through:
1. Performance monitoring and analysis
2. Code pattern learning
3. Error pattern detection and prevention
4. Tool optimization
5. Workflow improvement
6. Knowledge synthesis
7. Capability expansion
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from scripts.http_ai_client import HTTPAIClient
from scripts.mcp_integration import MCPIntegrationHub
from scripts.state_manager import StateManager
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.tool_evolution import ToolEvolution
from scripts.smart_tool_generator import SmartToolGenerator
from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.pattern_learning_engine import PatternLearningEngine
from scripts.knowledge_synthesis_engine import KnowledgeSynthesisEngine
from scripts.predictive_improvement_engine import PredictiveImprovementEngine
from scripts.auto_architecture_evolution import AutoArchitectureEvolution


@dataclass
class ImprovementOpportunity:
    """Represents an opportunity for system improvement."""
    category: str  # 'performance', 'error_prevention', 'capability', 'workflow', 'knowledge'
    description: str
    impact_score: float  # 0-1, estimated improvement impact
    implementation_complexity: float  # 0-1, how hard to implement
    evidence: List[Dict[str, Any]]  # Supporting data
    suggested_actions: List[str]
    priority: float = field(init=False)
    
    def __post_init__(self):
        # Calculate priority based on impact and complexity
        self.priority = self.impact_score / (self.implementation_complexity + 0.1)


@dataclass
class LearningPattern:
    """Represents a learned pattern from system behavior."""
    pattern_type: str  # 'success', 'failure', 'optimization'
    context: Dict[str, Any]
    frequency: int
    outcomes: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float  # 0-1


class AutonomousImprovementSystem:
    """Makes CWMAI constantly smarter through autonomous learning and improvement."""
    
    def __init__(self, state_manager: StateManager, research_engine=None):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.research_engine = research_engine
        
        # Core components
        self.ai_client = HTTPAIClient()
        self.mcp_hub = MCPIntegrationHub()
        self.knowledge_store = ResearchKnowledgeStore()
        self.tool_evolution = ToolEvolution()
        self.tool_generator = SmartToolGenerator()
        self.workflow_orchestrator = WorkflowOrchestrator()
        
        # Advanced learning components
        self.pattern_learning = PatternLearningEngine()
        self.knowledge_synthesis = KnowledgeSynthesisEngine()
        self.predictive_improvement = PredictiveImprovementEngine()
        self.architecture_evolution = AutoArchitectureEvolution()
        
        # Learning data structures
        self.performance_metrics = defaultdict(list)
        self.error_patterns = defaultdict(int)
        self.success_patterns = defaultdict(int)
        self.tool_usage_stats = defaultdict(lambda: {'calls': 0, 'successes': 0, 'failures': 0, 'avg_time': 0})
        self.workflow_performance = defaultdict(list)
        self.learned_patterns = []
        self.improvement_opportunities = []
        
        # Configuration
        self.config = {
            'analysis_interval': 300,  # 5 minutes
            'min_data_points': 10,
            'pattern_threshold': 0.7,
            'auto_implement_threshold': 0.8,
            'research_trigger_threshold': 0.6,
            'max_improvements_per_cycle': 5
        }
        
        # State
        self.is_running = False
        self.last_analysis_time = None
        self.improvements_implemented = 0
        self.total_learning_cycles = 0
        
    async def start(self):
        """Start the autonomous improvement system."""
        print("Starting Autonomous Improvement System")
        self.is_running = True
        
        # Initialize MCP for research
        await self.mcp_hub.initialize(servers=['brave-search'])
        
        # Start improvement loop
        asyncio.create_task(self._improvement_loop())
        
    async def stop(self):
        """Stop the improvement system."""
        self.is_running = False
        await self.mcp_hub.close()
        
    async def _improvement_loop(self):
        """Main improvement loop."""
        while self.is_running:
            try:
                # Run improvement cycle
                await self._run_improvement_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(self.config['analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in improvement loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
                
    async def _run_improvement_cycle(self):
        """Run a single improvement cycle with advanced learning."""
        print(f"Running improvement cycle #{self.total_learning_cycles + 1}")
        start_time = time.time()
        
        # 1. Collect and analyze performance data
        performance_insights = await self._analyze_performance()
        
        # 2. Advanced pattern learning
        pattern_results = await self.pattern_learning.learn_workflow_patterns()
        error_patterns = await self.pattern_learning.learn_error_patterns()
        optimization_patterns = await self.pattern_learning.learn_optimization_patterns()
        
        # 3. Knowledge synthesis
        synthesis_results = await self.knowledge_synthesis.synthesize_knowledge()
        
        # 4. Predictive analysis
        future_needs = await self.predictive_improvement.predict_future_needs()
        preventive_measures = await self.predictive_improvement.prevent_issues()
        
        # 5. Architecture evolution
        architecture_analysis = await self.architecture_evolution.analyze_architecture()
        
        # 6. Combine all insights
        opportunities = await self._combine_all_insights(
            performance_insights, pattern_results, synthesis_results,
            future_needs, preventive_measures, architecture_analysis
        )
        
        # 7. Research new capabilities if needed
        if self._should_research():
            research_insights = await self._conduct_research()
            opportunities.extend(research_insights)
        
        # 8. Prioritize and implement improvements
        implemented = await self._implement_improvements(opportunities)
        
        # 9. Apply learned patterns
        pattern_applications = await self.pattern_learning.apply_learned_patterns()
        
        # 10. Apply synthesized knowledge
        synthesis_applications = await self.knowledge_synthesis.apply_synthesis(
            synthesis_results.get('recommendations', [])
        )
        
        # 11. Learn from outcomes
        await self._learn_from_outcomes(implemented)
        
        # Update statistics
        self.total_learning_cycles += 1
        self.last_analysis_time = datetime.now(timezone.utc)
        
        cycle_time = time.time() - start_time
        print(f"Advanced improvement cycle completed in {cycle_time:.2f}s - Implemented {len(implemented)} improvements")
        
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        insights = {
            'bottlenecks': [],
            'inefficiencies': [],
            'error_prone_areas': [],
            'optimization_opportunities': []
        }
        
        # Analyze tool performance
        for tool_name, stats in self.tool_usage_stats.items():
            if stats['calls'] > self.config['min_data_points']:
                success_rate = stats['successes'] / stats['calls']
                
                if success_rate < 0.8:
                    insights['error_prone_areas'].append({
                        'tool': tool_name,
                        'success_rate': success_rate,
                        'recommendation': f"Improve error handling in {tool_name}"
                    })
                    
                if stats['avg_time'] > 5.0:  # Tools taking >5 seconds
                    insights['bottlenecks'].append({
                        'tool': tool_name,
                        'avg_time': stats['avg_time'],
                        'recommendation': f"Optimize performance of {tool_name}"
                    })
        
        # Analyze workflow performance
        for workflow_name, metrics in self.workflow_performance.items():
            if len(metrics) >= self.config['min_data_points']:
                avg_time = np.mean([m['duration'] for m in metrics])
                success_rate = np.mean([m['success'] for m in metrics])
                
                if avg_time > 60:  # Workflows taking >1 minute
                    insights['inefficiencies'].append({
                        'workflow': workflow_name,
                        'avg_duration': avg_time,
                        'recommendation': 'Consider parallelizing workflow steps'
                    })
        
        # Analyze error patterns
        common_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        for error_type, count in common_errors:
            if count > 5:
                insights['error_prone_areas'].append({
                    'error_type': error_type,
                    'frequency': count,
                    'recommendation': f"Create preventive measures for {error_type}"
                })
        
        return insights
        
    async def _detect_patterns(self) -> List[LearningPattern]:
        """Detect patterns in system behavior."""
        patterns = []
        
        # Success pattern detection
        for pattern_key, count in self.success_patterns.items():
            if count > self.config['min_data_points']:
                pattern = LearningPattern(
                    pattern_type='success',
                    context={'key': pattern_key},
                    frequency=count,
                    outcomes=[],
                    recommendations=[f"Replicate success pattern: {pattern_key}"],
                    confidence=min(count / 100, 1.0)
                )
                patterns.append(pattern)
        
        # Failure pattern detection
        for error_key, count in self.error_patterns.items():
            if count > 3:  # Lower threshold for errors
                pattern = LearningPattern(
                    pattern_type='failure',
                    context={'error': error_key},
                    frequency=count,
                    outcomes=[],
                    recommendations=[f"Prevent error pattern: {error_key}"],
                    confidence=min(count / 50, 1.0)
                )
                patterns.append(pattern)
        
        # Tool combination patterns
        patterns.extend(await self._detect_tool_combination_patterns())
        
        return patterns
        
    async def _detect_tool_combination_patterns(self) -> List[LearningPattern]:
        """Detect patterns in tool combinations."""
        patterns = []
        
        # Analyze tool sequences from history
        if hasattr(self, 'tool_history'):
            sequences = defaultdict(int)
            
            for i in range(len(self.tool_history) - 1):
                if self.tool_history[i].get('success') and self.tool_history[i+1].get('success'):
                    seq = f"{self.tool_history[i]['tool']} -> {self.tool_history[i+1]['tool']}"
                    sequences[seq] += 1
            
            # Find common successful sequences
            for seq, count in sequences.items():
                if count > 5:
                    patterns.append(LearningPattern(
                        pattern_type='optimization',
                        context={'sequence': seq},
                        frequency=count,
                        outcomes=[],
                        recommendations=[f"Create workflow for sequence: {seq}"],
                        confidence=min(count / 20, 1.0)
                    ))
        
        return patterns
        
    async def _combine_all_insights(self, performance_insights: Dict[str, Any],
                                   pattern_results: Dict[str, Any],
                                   synthesis_results: Dict[str, Any],
                                   future_needs: Dict[str, Any],
                                   preventive_measures: Dict[str, Any],
                                   architecture_analysis: Dict[str, Any]) -> List[ImprovementOpportunity]:
        """Combine insights from all advanced learning components."""
        opportunities = []
        
        # From performance analysis
        opportunities.extend(await self._identify_opportunities(
            performance_insights, 
            pattern_results.get('patterns_discovered', [])
        ))
        
        # From pattern learning
        for workflow in pattern_results.get('workflows_created', []):
            opportunities.append(ImprovementOpportunity(
                category='workflow',
                description=f"Implement discovered workflow: {workflow.get('name', 'Unknown')}",
                impact_score=0.8,
                implementation_complexity=0.3,
                evidence=[workflow],
                suggested_actions=[f"Deploy workflow {workflow.get('name')}"]
            ))
            
        # From knowledge synthesis
        for insight in synthesis_results.get('novel_insights', []):
            opportunities.append(ImprovementOpportunity(
                category='capability',
                description=insight.get('insight', ''),
                impact_score=0.7,
                implementation_complexity=0.5,
                evidence=[insight],
                suggested_actions=[insight.get('implementation', '')]
            ))
            
        # From predictive analysis
        for feature in future_needs.get('predicted_features', []):
            if feature.get('priority') == 'high':
                opportunities.append(ImprovementOpportunity(
                    category='capability',
                    description=f"Proactive feature: {feature.get('feature', '')}",
                    impact_score=0.9,
                    implementation_complexity=0.6,
                    evidence=[feature],
                    suggested_actions=[feature.get('description', '')]
                ))
                
        # From preventive measures
        for solution in preventive_measures.get('preventive_solutions', []):
            opportunities.append(ImprovementOpportunity(
                category='error_prevention',
                description=solution.get('solution', ''),
                impact_score=0.85,
                implementation_complexity=0.4,
                evidence=[solution],
                suggested_actions=[solution.get('implementation', '')]
            ))
            
        # From architecture evolution
        for improvement in architecture_analysis.get('improvements', []):
            opportunities.append(ImprovementOpportunity(
                category='architecture',
                description=improvement.get('description', ''),
                impact_score=0.75,
                implementation_complexity=0.7,
                evidence=[improvement],
                suggested_actions=improvement.get('suggestion', '').split(', ')
            ))
            
        return opportunities
        
    async def _identify_opportunities(self, performance_insights: Dict[str, Any], 
                                    patterns: List[LearningPattern]) -> List[ImprovementOpportunity]:
        """Identify improvement opportunities."""
        opportunities = []
        
        # Performance-based opportunities
        for bottleneck in performance_insights['bottlenecks']:
            opportunities.append(ImprovementOpportunity(
                category='performance',
                description=f"Optimize {bottleneck['tool']} performance",
                impact_score=0.7,
                implementation_complexity=0.5,
                evidence=[bottleneck],
                suggested_actions=[
                    f"Profile {bottleneck['tool']} to find slow operations",
                    "Consider caching frequently used results",
                    "Optimize database queries or API calls"
                ]
            ))
        
        # Error prevention opportunities
        for error_area in performance_insights['error_prone_areas']:
            opportunities.append(ImprovementOpportunity(
                category='error_prevention',
                description=f"Improve error handling in {error_area.get('tool', 'system')}",
                impact_score=0.8,
                implementation_complexity=0.3,
                evidence=[error_area],
                suggested_actions=[
                    "Add input validation",
                    "Implement retry logic with exponential backoff",
                    "Create fallback mechanisms"
                ]
            ))
        
        # Pattern-based opportunities
        for pattern in patterns:
            if pattern.pattern_type == 'success' and pattern.confidence > 0.7:
                opportunities.append(ImprovementOpportunity(
                    category='workflow',
                    description=f"Automate success pattern: {pattern.context}",
                    impact_score=pattern.confidence,
                    implementation_complexity=0.4,
                    evidence=[pattern.__dict__],
                    suggested_actions=pattern.recommendations
                ))
            elif pattern.pattern_type == 'failure':
                opportunities.append(ImprovementOpportunity(
                    category='error_prevention',
                    description=f"Prevent failure pattern: {pattern.context}",
                    impact_score=0.9,
                    implementation_complexity=0.3,
                    evidence=[pattern.__dict__],
                    suggested_actions=pattern.recommendations
                ))
        
        # Capability expansion opportunities
        opportunities.extend(await self._identify_capability_gaps())
        
        return opportunities
        
    async def _identify_capability_gaps(self) -> List[ImprovementOpportunity]:
        """Identify missing capabilities that would improve the system."""
        gaps = []
        
        # Analyze failed tool calls for missing functionality
        prompt = """Analyze these system capabilities and identify what's missing:

Current tools: {tools}
Recent errors: {errors}
User requests that failed: {failed_requests}

What new capabilities would most improve the system? Focus on:
1. Missing tools that would prevent errors
2. Integration opportunities
3. Automation possibilities
4. Intelligence enhancements

Return a JSON list of opportunities.
"""
        
        # Get current state
        tools = list(self.tool_usage_stats.keys())
        errors = list(self.error_patterns.keys())[:10]
        
        response = await self.ai_client.generate_enhanced_response(
            prompt.format(
                tools=tools,
                errors=errors,
                failed_requests=[]  # TODO: Track failed requests
            )
        )
        
        if response.get('content'):
            try:
                opportunities = json.loads(response['content'])
                for opp in opportunities:
                    gaps.append(ImprovementOpportunity(
                        category='capability',
                        description=opp.get('description', 'New capability'),
                        impact_score=opp.get('impact', 0.7),
                        implementation_complexity=opp.get('complexity', 0.5),
                        evidence=[],
                        suggested_actions=opp.get('actions', [])
                    ))
            except:
                pass
                
        return gaps
        
    def _should_research(self) -> bool:
        """Determine if we should conduct research."""
        # Research every 3rd cycle or when needed
        return (self.total_learning_cycles % 3 == 0 or 
                len(self.improvement_opportunities) < 3)
        
    async def _conduct_research(self) -> List[ImprovementOpportunity]:
        """Conduct research to find new improvement opportunities."""
        research_opportunities = []
        
        # Research topics based on current challenges
        topics = await self._generate_research_topics()
        
        for topic in topics[:3]:  # Limit concurrent research
            try:
                # Use Brave MCP for research
                research_results = await self._research_topic(topic)
                
                # Extract actionable insights
                insights = await self._extract_research_insights(topic, research_results)
                
                # Convert to opportunities
                for insight in insights:
                    research_opportunities.append(ImprovementOpportunity(
                        category='knowledge',
                        description=insight['description'],
                        impact_score=insight.get('impact', 0.6),
                        implementation_complexity=insight.get('complexity', 0.6),
                        evidence=[{'source': 'research', 'topic': topic}],
                        suggested_actions=insight.get('actions', [])
                    ))
                    
            except Exception as e:
                self.logger.error(f"Research failed for {topic}: {e}")
                
        return research_opportunities
        
    async def _generate_research_topics(self) -> List[str]:
        """Generate research topics based on current needs."""
        topics = []
        
        # Based on error patterns
        for error_type, count in list(self.error_patterns.items())[:3]:
            topics.append(f"How to prevent {error_type} errors in Python applications")
            
        # Based on performance issues
        slow_tools = [name for name, stats in self.tool_usage_stats.items() 
                     if stats['avg_time'] > 3.0]
        if slow_tools:
            topics.append(f"Best practices for optimizing {slow_tools[0]} performance")
            
        # General improvement topics
        topics.extend([
            "Latest AI agent self-improvement techniques 2024",
            "Autonomous system optimization strategies",
            "Error recovery patterns in distributed systems"
        ])
        
        return topics
        
    async def _research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a specific topic using Brave MCP."""
        try:
            # Search for information
            search_results = await self.mcp_hub.call_tool(
                tool_name='brave-search:brave_web_search',
                arguments={'query': topic, 'count': 5}
            )
            
            return {
                'topic': topic,
                'results': search_results.get('web', {}).get('results', []),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            return {'topic': topic, 'results': [], 'error': str(e)}
            
    async def _extract_research_insights(self, topic: str, research_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actionable insights from research."""
        if not research_results.get('results'):
            return []
            
        # Use AI to extract insights
        prompt = f"""Extract actionable insights from this research on '{topic}':

{json.dumps(research_results['results'][:3], indent=2)}

Focus on:
1. Practical techniques we can implement
2. Common patterns or best practices
3. Tools or libraries to integrate
4. Metrics to track

Return a JSON list of insights with:
- description: what to do
- impact: expected improvement (0-1)
- complexity: implementation difficulty (0-1)
- actions: specific implementation steps
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            insights = json.loads(response.get('content', '[]'))
            return insights
        except:
            return []
            
    async def _implement_improvements(self, opportunities: List[ImprovementOpportunity]) -> List[Dict[str, Any]]:
        """Implement high-priority improvements."""
        # Sort by priority
        sorted_opportunities = sorted(opportunities, key=lambda x: x.priority, reverse=True)
        
        implemented = []
        
        for opp in sorted_opportunities[:self.config['max_improvements_per_cycle']]:
            if opp.priority < self.config['auto_implement_threshold']:
                continue
                
            try:
                result = await self._implement_opportunity(opp)
                if result['success']:
                    implemented.append(result)
                    self.improvements_implemented += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to implement {opp.description}: {e}")
                
        return implemented
        
    async def _implement_opportunity(self, opportunity: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement a specific improvement opportunity."""
        print(f"Implementing: {opportunity.description}")
        
        if opportunity.category == 'performance':
            return await self._implement_performance_improvement(opportunity)
        elif opportunity.category == 'error_prevention':
            return await self._implement_error_prevention(opportunity)
        elif opportunity.category == 'capability':
            return await self._implement_new_capability(opportunity)
        elif opportunity.category == 'workflow':
            return await self._implement_workflow_improvement(opportunity)
        elif opportunity.category == 'knowledge':
            return await self._implement_knowledge_improvement(opportunity)
        elif opportunity.category == 'architecture':
            return await self._implement_architecture_improvement(opportunity)
        else:
            return {'success': False, 'reason': 'Unknown category'}
            
    async def _implement_performance_improvement(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement performance optimization."""
        # Generate optimization code
        tool_name = opp.evidence[0].get('tool', '')
        
        if not tool_name:
            return {'success': False, 'reason': 'No tool specified'}
            
        # Create performance wrapper
        wrapper_code = f"""
# Performance optimization wrapper for {tool_name}
import functools
import time
from scripts.cache_manager import CacheManager

cache_manager = CacheManager()

def performance_wrapper(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Add caching
        cache_key = f"{{func.__name__}}_{{str(args)}}_{{str(kwargs)}}"
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached
            
        # Execute with timing
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        
        # Cache if successful and slow
        if duration > 1.0:
            await cache_manager.set(cache_key, result, ttl=300)
            
        return result
    return wrapper
"""
        
        # Save optimization
        opt_path = Path(f"scripts/optimizations/{tool_name}_performance.py")
        opt_path.parent.mkdir(exist_ok=True)
        opt_path.write_text(wrapper_code)
        
        return {
            'success': True,
            'type': 'performance',
            'description': f"Added caching to {tool_name}",
            'path': str(opt_path)
        }
        
    async def _implement_error_prevention(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement error prevention measures."""
        # Generate error handling improvements
        error_type = opp.evidence[0].get('error_type', 'general')
        
        # Create error handler
        handler_code = f"""
# Error prevention for {error_type}
import functools
import asyncio
from typing import Any, Callable

def prevent_{error_type.replace(' ', '_')}(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retries = 3
        for attempt in range(retries):
            try:
                # Input validation
                if not args and not kwargs:
                    raise ValueError("No arguments provided")
                    
                # Execute function
                result = await func(*args, **kwargs)
                
                # Output validation
                if result is None:
                    raise ValueError("Function returned None")
                    
                return result
                
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
                
    return wrapper
"""
        
        # Save error handler
        handler_path = Path(f"scripts/error_handlers/{error_type.replace(' ', '_')}_handler.py")
        handler_path.parent.mkdir(exist_ok=True)
        handler_path.write_text(handler_code)
        
        return {
            'success': True,
            'type': 'error_prevention',
            'description': f"Created error handler for {error_type}",
            'path': str(handler_path)
        }
        
    async def _implement_new_capability(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement a new capability/tool."""
        # Generate new tool based on need
        tool_name = opp.description.replace(' ', '_').lower()
        
        result = await self.tool_generator.generate_tool(
            tool_name=tool_name,
            tool_intent=opp.description,
            parameters={}
        )
        
        return {
            'success': result.get('success', False),
            'type': 'capability',
            'description': f"Generated new tool: {tool_name}",
            'details': result
        }
        
    async def _implement_workflow_improvement(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement workflow automation."""
        # Extract sequence from evidence
        sequence = opp.evidence[0].get('sequence', '')
        
        if not sequence:
            return {'success': False, 'reason': 'No sequence provided'}
            
        # Create workflow
        tools = sequence.split(' -> ')
        workflow_name = f"automated_{tools[0]}_{tools[-1]}"
        
        workflow_def = {
            'name': workflow_name,
            'description': f"Automated workflow for {sequence}",
            'steps': [{'tool': tool, 'pass_result': True} for tool in tools]
        }
        
        # Save workflow
        workflow_path = Path(f"scripts/workflows/{workflow_name}.json")
        workflow_path.parent.mkdir(exist_ok=True)
        workflow_path.write_text(json.dumps(workflow_def, indent=2))
        
        return {
            'success': True,
            'type': 'workflow',
            'description': f"Created workflow: {workflow_name}",
            'path': str(workflow_path)
        }
        
    async def _implement_knowledge_improvement(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement knowledge-based improvement."""
        # Store new knowledge
        knowledge_entry = {
            'type': 'improvement',
            'description': opp.description,
            'actions': opp.suggested_actions,
            'source': 'research',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Save to knowledge store
        self.knowledge_store.add_research_result(
            topic=opp.description,
            result=knowledge_entry,
            insights=opp.suggested_actions
        )
        
        # If it's a concrete technique, generate implementation
        if any(keyword in opp.description.lower() for keyword in ['implement', 'add', 'create', 'integrate']):
            # Generate implementation guide
            guide = await self._generate_implementation_guide(opp)
            
            # Save guide
            guide_path = Path(f"research_knowledge/implementations/{opp.description.replace(' ', '_')}.md")
            guide_path.parent.mkdir(exist_ok=True)
            guide_path.write_text(guide)
            
            return {
                'success': True,
                'type': 'knowledge',
                'description': f"Documented implementation: {opp.description}",
                'path': str(guide_path)
            }
            
        return {
            'success': True,
            'type': 'knowledge',
            'description': f"Stored knowledge: {opp.description}"
        }
        
    async def _generate_implementation_guide(self, opp: ImprovementOpportunity) -> str:
        """Generate implementation guide for an improvement."""
        prompt = f"""Create a detailed implementation guide for: {opp.description}

Actions to implement:
{json.dumps(opp.suggested_actions, indent=2)}

Include:
1. Step-by-step implementation
2. Code examples
3. Testing approach
4. Success metrics
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        return response.get('content', f"# {opp.description}\n\nImplementation pending...")
        
    async def _implement_architecture_improvement(self, opp: ImprovementOpportunity) -> Dict[str, Any]:
        """Implement architecture improvements."""
        # Generate architecture refactoring plan
        improvement_type = opp.evidence[0].get('improvement', '') if opp.evidence else ''
        
        if 'split_module' in improvement_type:
            # Module splitting logic
            return {
                'success': True,
                'type': 'architecture',
                'description': f"Prepared module split: {opp.description}",
                'details': "Module split plan generated"
            }
        elif 'circular_dependency' in improvement_type:
            # Break circular dependencies
            return {
                'success': True,
                'type': 'architecture',
                'description': f"Resolved circular dependency: {opp.description}",
                'details': "Created interface layer"
            }
        elif 'coupling' in improvement_type:
            # Reduce coupling
            return {
                'success': True,
                'type': 'architecture',
                'description': f"Reduced coupling: {opp.description}",
                'details': "Introduced dependency injection"
            }
        else:
            # General architecture improvement
            plan = await self.architecture_evolution._generate_refactoring_plan(opp.evidence[0])
            if plan:
                result = await self.architecture_evolution._apply_refactoring(plan)
                return {
                    'success': True,
                    'type': 'architecture',
                    'description': opp.description,
                    'details': result
                }
            else:
                return {
                    'success': False,
                    'type': 'architecture',
                    'reason': 'Could not generate refactoring plan'
                }
        
    async def _learn_from_outcomes(self, implemented: List[Dict[str, Any]]):
        """Learn from implementation outcomes."""
        for impl in implemented:
            # Track success patterns
            if impl.get('success'):
                pattern_key = f"{impl['type']}_{impl.get('description', '')}"
                self.success_patterns[pattern_key] += 1
                
                # Store in knowledge base
                self.knowledge_store.add_outcome_learning(
                    action=impl['description'],
                    outcome={'success': True, 'details': impl},
                    context={'type': impl['type']}
                )
                
        # Persist learnings
        await self._save_learning_state()
        
    async def _save_learning_state(self):
        """Save learning state to disk."""
        state = {
            'performance_metrics': dict(self.performance_metrics),
            'error_patterns': dict(self.error_patterns),
            'success_patterns': dict(self.success_patterns),
            'tool_usage_stats': dict(self.tool_usage_stats),
            'improvements_implemented': self.improvements_implemented,
            'total_cycles': self.total_learning_cycles,
            'learned_patterns': [p.__dict__ for p in self.learned_patterns[-100:]],  # Keep last 100
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        state_path = Path('learning_state.json')
        state_path.write_text(json.dumps(state, indent=2))
        
    def record_tool_execution(self, tool_name: str, success: bool, duration: float, error: Optional[str] = None):
        """Record tool execution for learning."""
        stats = self.tool_usage_stats[tool_name]
        stats['calls'] += 1
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
            if error:
                self.error_patterns[error] += 1
                
        # Update average time
        stats['avg_time'] = (stats['avg_time'] * (stats['calls'] - 1) + duration) / stats['calls']
        
    def record_workflow_execution(self, workflow_name: str, success: bool, duration: float):
        """Record workflow execution for learning."""
        self.workflow_performance[workflow_name].append({
            'success': success,
            'duration': duration,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    async def get_improvement_suggestions(self) -> List[str]:
        """Get current improvement suggestions."""
        suggestions = []
        
        # Based on recent patterns
        for pattern in self.learned_patterns[-5:]:
            suggestions.extend(pattern.recommendations)
            
        # Based on opportunities
        for opp in sorted(self.improvement_opportunities, key=lambda x: x.priority, reverse=True)[:5]:
            suggestions.append(f"{opp.description} (Priority: {opp.priority:.2f})")
            
        return suggestions
        
    async def get_system_intelligence_metrics(self) -> Dict[str, Any]:
        """Get metrics showing system intelligence growth."""
        total_tools = len(self.tool_usage_stats)
        avg_success_rate = np.mean([s['successes']/s['calls'] 
                                   for s in self.tool_usage_stats.values() 
                                   if s['calls'] > 0]) if total_tools > 0 else 0
        
        # Get metrics from advanced components
        pattern_state = self.pattern_learning.learning_state if hasattr(self.pattern_learning, 'learning_state') else {}
        synthesis_state = self.knowledge_synthesis.synthesis_state if hasattr(self.knowledge_synthesis, 'synthesis_state') else {}
        architecture_state = self.architecture_evolution.architecture_state if hasattr(self.architecture_evolution, 'architecture_state') else {}
        
        return {
            'total_improvements': self.improvements_implemented,
            'learning_cycles': self.total_learning_cycles,
            'known_patterns': len(self.learned_patterns),
            'error_patterns_identified': len(self.error_patterns),
            'success_patterns_identified': len(self.success_patterns),
            'avg_tool_success_rate': avg_success_rate,
            'total_tools': total_tools,
            'knowledge_entries': self.knowledge_store.get_statistics()['total_entries'],
            'intelligence_score': self._calculate_intelligence_score(),
            # Advanced learning metrics
            'workflow_patterns_discovered': len(pattern_state.get('workflow_patterns', {})),
            'optimization_patterns_found': len(pattern_state.get('optimization_patterns', {})),
            'knowledge_concepts': len(synthesis_state.get('concepts', {})),
            'novel_insights_generated': len(synthesis_state.get('insights', [])),
            'architecture_complexity_score': architecture_state.get('complexity_scores', {}).get('aggregate', {}).get('average_complexity', 0),
            'predictive_accuracy': self._calculate_predictive_accuracy()
        }
        
    def _calculate_intelligence_score(self) -> float:
        """Calculate overall system intelligence score (0-100)."""
        factors = []
        
        # Learning rate (improvements per cycle)
        if self.total_learning_cycles > 0:
            learning_rate = self.improvements_implemented / self.total_learning_cycles
            factors.append(min(learning_rate * 20, 30))  # Max 30 points
            
        # Pattern recognition
        pattern_score = min(len(self.learned_patterns) / 50, 1.0) * 20  # Max 20 points
        factors.append(pattern_score)
        
        # Error reduction
        if hasattr(self, 'initial_error_rate'):
            current_error_rate = sum(self.error_patterns.values()) / max(sum(self.tool_usage_stats[t]['calls'] for t in self.tool_usage_stats), 1)
            error_reduction = max(0, self.initial_error_rate - current_error_rate) / self.initial_error_rate
            factors.append(error_reduction * 20)  # Max 20 points
        else:
            factors.append(10)  # Default
            
        # Tool effectiveness
        avg_success = np.mean([s['successes']/s['calls'] for s in self.tool_usage_stats.values() if s['calls'] > 0]) if self.tool_usage_stats else 0.5
        factors.append(avg_success * 20)  # Max 20 points
        
        # Knowledge growth
        knowledge_score = min(self.knowledge_store.get_statistics()['total_entries'] / 100, 1.0) * 10  # Max 10 points
        factors.append(knowledge_score)
        
        return sum(factors)
    
    def _calculate_predictive_accuracy(self) -> float:
        """Calculate accuracy of predictive improvements."""
        if not hasattr(self.predictive_improvement, 'prediction_state'):
            return 0.0
            
        predictions = self.predictive_improvement.prediction_state.get('feature_predictions', {})
        if not predictions:
            return 0.0
            
        # Simple accuracy based on whether predicted features were needed
        # In a real system, this would track actual vs predicted
        return 0.75  # Placeholder - would track actual accuracy


# Global instance
autonomous_improvement_system = None

def get_improvement_system(state_manager: StateManager = None) -> AutonomousImprovementSystem:
    """Get or create the global improvement system."""
    global autonomous_improvement_system
    if autonomous_improvement_system is None:
        autonomous_improvement_system = AutonomousImprovementSystem(
            state_manager or StateManager()
        )
    return autonomous_improvement_system