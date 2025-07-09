"""
Self-Improvement Orchestrator

Autonomous learning and improvement system that continuously enhances the AI assistant's
capabilities based on user interactions and system performance.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.pattern_learning_engine import PatternLearningEngine
from scripts.tool_calling_system import ToolCallingSystem


@dataclass
class LearningInsight:
    """Represents an insight learned from user interactions."""
    timestamp: datetime
    interaction_id: str
    insight_type: str  # pattern, preference, failure, success
    content: Dict[str, Any]
    confidence: float
    applied: bool = False
    
    
@dataclass
class ImprovementCandidate:
    """Represents a potential improvement to the system."""
    id: str
    improvement_type: str  # tool_creation, tool_enhancement, response_optimization, workflow
    description: str
    target: str  # what to improve
    implementation: Dict[str, Any]
    expected_benefit: float
    risk_level: float
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, testing, applied, rejected
    

class SelfImprovementOrchestrator:
    """Orchestrates autonomous learning and improvement of the AI system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.pattern_engine = PatternLearningEngine()
        self.tool_system = ToolCallingSystem()
        
        # Learning storage
        self.insights: List[LearningInsight] = []
        self.improvements: List[ImprovementCandidate] = []
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = {
            "successful_interactions": 0,
            "failed_interactions": 0,
            "tool_creation_success": 0,
            "tool_creation_failure": 0,
            "response_satisfaction": defaultdict(int),
            "common_intents": defaultdict(int),
            "error_patterns": defaultdict(int)
        }
        
        # Learning configuration
        self.config = {
            "min_confidence_for_improvement": 0.7,
            "interaction_batch_size": 10,
            "improvement_test_duration": 300,  # 5 minutes
            "max_concurrent_improvements": 3
        }
        
        # State file
        self.state_file = Path.home() / ".cwmai" / "self_improvement_state.json"
        self._load_state()
        
    def _load_state(self):
        """Load previous learning state."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Convert loaded metrics back to defaultdicts where needed
                    loaded_metrics = state.get('metrics', {})
                    self.metrics = {
                        "successful_interactions": loaded_metrics.get("successful_interactions", 0),
                        "failed_interactions": loaded_metrics.get("failed_interactions", 0),
                        "tool_creation_success": loaded_metrics.get("tool_creation_success", 0),
                        "tool_creation_failure": loaded_metrics.get("tool_creation_failure", 0),
                        "response_satisfaction": defaultdict(int, loaded_metrics.get("response_satisfaction", {})),
                        "common_intents": defaultdict(int, loaded_metrics.get("common_intents", {})),
                        "error_patterns": defaultdict(int, loaded_metrics.get("error_patterns", {}))
                    }
                    # Load insights and improvements
                    for insight_data in state.get('insights', [])[-100:]:  # Keep last 100
                        self.insights.append(LearningInsight(
                            timestamp=datetime.fromisoformat(insight_data['timestamp']),
                            interaction_id=insight_data['interaction_id'],
                            insight_type=insight_data['insight_type'],
                            content=insight_data['content'],
                            confidence=insight_data['confidence'],
                            applied=insight_data['applied']
                        ))
        except Exception as e:
            self.logger.debug(f"Could not load state: {e}")
            
    def _save_state(self):
        """Save current learning state."""
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            
            # Prepare insights for serialization
            insights_data = []
            for insight in self.insights[-100:]:  # Keep last 100
                insights_data.append({
                    'timestamp': insight.timestamp.isoformat(),
                    'interaction_id': insight.interaction_id,
                    'insight_type': insight.insight_type,
                    'content': insight.content,
                    'confidence': insight.confidence,
                    'applied': insight.applied
                })
                
            with open(self.state_file, 'w') as f:
                json.dump({
                    'metrics': self.metrics,
                    'insights': insights_data,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state: {e}")
            
    async def analyze_interaction(self, user_input: str, assistant_response: str, 
                                context: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze a single interaction for learning opportunities."""
        insights = []
        interaction_id = hashlib.md5(f"{user_input}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Analyze intent patterns
        intent_analysis = await self._analyze_intent(user_input, context)
        if intent_analysis:
            intent = intent_analysis.get('intent', 'unknown')
            if "common_intents" not in self.metrics:
                self.metrics["common_intents"] = defaultdict(int)
            self.metrics["common_intents"][intent] += 1
            insights.append(LearningInsight(
                timestamp=datetime.now(timezone.utc),
                interaction_id=interaction_id,
                insight_type="pattern",
                content=intent_analysis,
                confidence=intent_analysis.get('confidence', 0.8)
            ))
            
        # Analyze tool usage
        if 'tool_calls' in context:
            tool_insights = await self._analyze_tool_usage(context['tool_calls'])
            insights.extend(tool_insights)
            
        # Analyze errors or failures
        if 'error' in context or 'failed' in assistant_response.lower():
            error_insight = await self._analyze_error(user_input, assistant_response, context)
            if error_insight:
                insights.append(error_insight)
                self.metrics["failed_interactions"] += 1
        else:
            self.metrics["successful_interactions"] += 1
            
        # Store interaction for batch analysis
        self.interaction_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_input': user_input,
            'assistant_response': assistant_response,
            'context': context,
            'insights': [i.content for i in insights]
        })
        
        # Trigger batch analysis if enough interactions
        if len(self.interaction_history) >= self.config["interaction_batch_size"]:
            asyncio.create_task(self._batch_analyze_interactions())
            
        self.insights.extend(insights)
        self._save_state()
        
        return insights
        
    async def _analyze_intent(self, user_input: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze user intent from input."""
        prompt = f"""Analyze the user's intent from this input:
User input: {user_input}
Context: {json.dumps(context, indent=2)}

Provide a JSON response with:
{{
    "intent": "primary intent category",
    "sub_intent": "specific action requested",
    "entities": ["list of key entities mentioned"],
    "confidence": 0.0-1.0,
    "suggested_tools": ["tools that could help with this intent"]
}}
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(
                prompt,
                model="gemini"
            )
            
            if response and response.get('content'):
                # Extract JSON from response
                content = response['content']
                json_match = content.find('{')
                if json_match != -1:
                    json_str = content[json_match:]
                    json_end = json_str.rfind('}')
                    if json_end != -1:
                        json_str = json_str[:json_end + 1]
                        return json.loads(json_str)
        except Exception as e:
            self.logger.debug(f"Intent analysis failed: {e}")
            
        return None
        
    async def _analyze_tool_usage(self, tool_calls: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze tool usage patterns."""
        insights = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('tool_name')
            success = tool_call.get('success', False)
            
            if success:
                self.metrics["tool_creation_success"] += 1
            else:
                self.metrics["tool_creation_failure"] += 1
                
                # Analyze why the tool failed
                error_msg = tool_call.get('error', '')
                if 'not found' in error_msg:
                    # Tool doesn't exist - candidate for creation
                    insights.append(LearningInsight(
                        timestamp=datetime.now(timezone.utc),
                        interaction_id=tool_call.get('id', ''),
                        insight_type="failure",
                        content={
                            "type": "missing_tool",
                            "tool_name": tool_name,
                            "parameters": tool_call.get('parameters', {})
                        },
                        confidence=0.9
                    ))
                    
        return insights
        
    async def _analyze_error(self, user_input: str, response: str, context: Dict[str, Any]) -> Optional[LearningInsight]:
        """Analyze errors to learn from failures."""
        error_type = "unknown"
        
        # Categorize error
        if "tool" in response.lower() and "not found" in response.lower():
            error_type = "missing_tool"
        elif "error executing" in response.lower():
            error_type = "tool_execution_error"
        elif "understand" in response.lower() or "clarify" in response.lower():
            error_type = "understanding_error"
        else:
            error_type = "unknown_error"
            
        # Initialize error type in metrics if not present
        if error_type not in self.metrics["error_patterns"]:
            self.metrics["error_patterns"][error_type] = 0
            
        self.metrics["error_patterns"][error_type] += 1
        
        return LearningInsight(
            timestamp=datetime.now(timezone.utc),
            interaction_id=context.get('interaction_id', ''),
            insight_type="failure",
            content={
                "error_type": error_type,
                "user_input": user_input,
                "response": response,
                "context": context
            },
            confidence=0.8
        )
        
    async def _batch_analyze_interactions(self):
        """Analyze a batch of interactions for patterns."""
        if len(self.interaction_history) < self.config["interaction_batch_size"]:
            return
            
        batch = self.interaction_history[-self.config["interaction_batch_size"]:]
        
        # Find common patterns
        patterns = await self.pattern_engine.find_patterns(batch)
        
        # Generate improvement candidates based on patterns
        for pattern in patterns:
            if pattern['confidence'] >= self.config["min_confidence_for_improvement"]:
                improvement = await self._generate_improvement(pattern)
                if improvement:
                    self.improvements.append(improvement)
                    
        # Clear old history
        self.interaction_history = self.interaction_history[-50:]
        
    async def _generate_improvement(self, pattern: Dict[str, Any]) -> Optional[ImprovementCandidate]:
        """Generate an improvement candidate from a pattern."""
        improvement_id = hashlib.md5(f"{pattern}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Determine improvement type based on pattern
        if pattern['type'] == 'missing_tool':
            return ImprovementCandidate(
                id=improvement_id,
                improvement_type="tool_creation",
                description=f"Create tool: {pattern['tool_name']}",
                target=pattern['tool_name'],
                implementation={
                    "action": "create_tool",
                    "tool_name": pattern['tool_name'],
                    "requirements": pattern.get('requirements', '')
                },
                expected_benefit=0.8,
                risk_level=0.2
            )
        elif pattern['type'] == 'repeated_error':
            return ImprovementCandidate(
                id=improvement_id,
                improvement_type="error_fix",
                description=f"Fix recurring error: {pattern['error_type']}",
                target=pattern['error_type'],
                implementation={
                    "action": "enhance_error_handling",
                    "error_type": pattern['error_type'],
                    "solution": pattern.get('suggested_solution', '')
                },
                expected_benefit=0.7,
                risk_level=0.3
            )
            
        return None
        
    async def apply_improvements(self, max_improvements: int = 3) -> List[Dict[str, Any]]:
        """Apply pending improvements to the system."""
        applied = []
        
        # Sort improvements by expected benefit and risk
        pending = [imp for imp in self.improvements if imp.status == "pending"]
        pending.sort(key=lambda x: x.expected_benefit - x.risk_level, reverse=True)
        
        for improvement in pending[:max_improvements]:
            try:
                result = await self._apply_improvement(improvement)
                if result['success']:
                    improvement.status = "applied"
                    applied.append({
                        "improvement_id": improvement.id,
                        "type": improvement.improvement_type,
                        "description": improvement.description,
                        "result": result
                    })
                    
                    # Mark related insights as applied
                    for insight in self.insights:
                        if not insight.applied and insight.content.get('type') == improvement.target:
                            insight.applied = True
                else:
                    improvement.status = "rejected"
                    
            except Exception as e:
                self.logger.error(f"Failed to apply improvement {improvement.id}: {e}")
                improvement.status = "rejected"
                
        self._save_state()
        return applied
        
    async def _apply_improvement(self, improvement: ImprovementCandidate) -> Dict[str, Any]:
        """Apply a specific improvement."""
        impl = improvement.implementation
        
        if impl['action'] == 'create_tool':
            # Create a new tool
            result = await self.tool_system._create_new_tool(
                name=impl['tool_name'],
                description=improvement.description,
                requirements=impl.get('requirements', '')
            )
            return result
            
        elif impl['action'] == 'enhance_error_handling':
            # Enhance error handling (placeholder for now)
            return {"success": True, "message": "Error handling enhanced"}
            
        return {"success": False, "error": "Unknown improvement action"}
        
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of what the system has learned."""
        return {
            "total_insights": len(self.insights),
            "applied_insights": sum(1 for i in self.insights if i.applied),
            "pending_improvements": sum(1 for i in self.improvements if i.status == "pending"),
            "applied_improvements": sum(1 for i in self.improvements if i.status == "applied"),
            "metrics": dict(self.metrics),
            "top_intents": sorted(
                [(k, v) for k, v in self.metrics["common_intents"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_errors": sorted(
                [(k, v) for k, v in self.metrics["error_patterns"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
    async def start_background_learning(self):
        """Start the background learning process."""
        while True:
            try:
                # Apply improvements every 5 minutes
                await asyncio.sleep(300)
                
                # Apply pending improvements
                if self.improvements:
                    applied = await self.apply_improvements()
                    if applied:
                        self.logger.info(f"Applied {len(applied)} improvements")
                        
                # Clean up old data
                self.interaction_history = self.interaction_history[-100:]
                self.insights = self.insights[-200:]
                
                self._save_state()
                
            except Exception as e:
                self.logger.error(f"Background learning error: {e}")
                await asyncio.sleep(60)  # Wait before retrying