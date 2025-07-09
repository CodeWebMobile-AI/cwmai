"""
Intelligent Workflow Generator
==============================

This module provides AI-driven workflow generation capabilities that:
1. Automatically decompose complex tasks into workflow steps
2. Generate workflows from natural language descriptions
3. Optimize workflow structures based on patterns
4. Learn from successful workflow executions
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import re

from scripts.http_ai_client import HTTPAIClient
from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.state_manager import StateManager
from scripts.tool_calling_system import ToolCallingSystem


class IntelligentWorkflowGenerator:
    """Generates workflows intelligently from task descriptions and patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
        self.orchestrator = WorkflowOrchestrator()
        self.state_manager = StateManager()
        self.tool_system = ToolCallingSystem()
        
        # Load workflow patterns and templates
        self.workflow_patterns = self._load_workflow_patterns()
        self.successful_workflows = self._load_successful_workflows()
        
    def _load_workflow_patterns(self) -> Dict[str, Any]:
        """Load common workflow patterns."""
        return {
            'data_processing': {
                'pattern': ['fetch', 'validate', 'transform', 'store'],
                'tools': ['fetch_data', 'validate_data', 'transform_data', 'save_results']
            },
            'analysis_pipeline': {
                'pattern': ['collect', 'analyze', 'report'],
                'tools': ['gather_data', 'run_analysis', 'generate_report']
            },
            'maintenance': {
                'pattern': ['check', 'identify_issues', 'fix', 'verify'],
                'tools': ['health_check', 'diagnose_problems', 'apply_fixes', 'verify_health']
            },
            'development': {
                'pattern': ['plan', 'implement', 'test', 'deploy'],
                'tools': ['create_plan', 'write_code', 'run_tests', 'deploy_changes']
            }
        }
        
    def _load_successful_workflows(self) -> List[Dict[str, Any]]:
        """Load previously successful workflows for learning."""
        state = self.state_manager.get_state()
        workflows = state.get('successful_workflows', [])
        return workflows[-100:]  # Keep last 100 successful workflows
        
    async def generate_workflow_from_description(
        self, 
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a complete workflow from natural language description."""
        
        # Extract key entities and intents
        entities = await self._extract_entities(description)
        intent = await self._identify_intent(description)
        
        # Find similar successful workflows
        similar_workflows = self._find_similar_workflows(description, intent)
        
        # Generate workflow structure
        workflow_structure = await self._generate_workflow_structure(
            description, entities, intent, similar_workflows
        )
        
        # Optimize the workflow
        optimized_workflow = await self._optimize_workflow(workflow_structure)
        
        # Add metadata
        workflow_def = {
            'name': self._generate_workflow_name(description),
            'description': description,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'context': context or {},
            'metadata': {
                'entities': entities,
                'intent': intent,
                'pattern_match': workflow_structure.get('pattern_used'),
                'confidence': workflow_structure.get('confidence', 0.8)
            },
            'steps': optimized_workflow['steps']
        }
        
        return workflow_def
        
    async def _extract_entities(self, description: str) -> Dict[str, List[str]]:
        """Extract entities from description."""
        prompt = f"""
        Extract key entities from this workflow description:
        "{description}"
        
        Return a JSON object with these entity types:
        - actions: List of actions to perform
        - resources: Resources mentioned (repos, files, etc.)
        - conditions: Any conditions or criteria
        - outputs: Expected outputs or results
        
        Example:
        {{
            "actions": ["analyze", "create", "deploy"],
            "resources": ["repository-name", "config-file"],
            "conditions": ["if errors exist", "when complete"],
            "outputs": ["report", "fixed issues"]
        }}
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            entities = json.loads(response.get('content', '{}'))
            return entities
        except:
            # Fallback to regex extraction
            return self._extract_entities_regex(description)
            
    def _extract_entities_regex(self, description: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns."""
        entities = {
            'actions': [],
            'resources': [],
            'conditions': [],
            'outputs': []
        }
        
        # Action verbs
        action_pattern = r'\b(analyze|create|generate|check|fix|deploy|test|validate|fetch|update|delete)\b'
        entities['actions'] = list(set(re.findall(action_pattern, description.lower())))
        
        # Resources (repos, files, etc.)
        resource_pattern = r'(?:repository|repo|file|database|api|service|tool)\s+([a-zA-Z0-9_-]+)'
        entities['resources'] = re.findall(resource_pattern, description, re.IGNORECASE)
        
        # Conditions
        condition_pattern = r'(?:if|when|unless|after|before)\s+([^,\.]+)'
        entities['conditions'] = re.findall(condition_pattern, description, re.IGNORECASE)
        
        return entities
        
    async def _identify_intent(self, description: str) -> str:
        """Identify the primary intent of the workflow."""
        prompt = f"""
        Identify the primary intent of this workflow description:
        "{description}"
        
        Choose one of these categories:
        - data_processing: Processing, transforming, or analyzing data
        - maintenance: System maintenance, cleanup, or fixes
        - development: Code generation, testing, or deployment
        - analysis: Research, investigation, or reporting
        - automation: Automating repetitive tasks
        - integration: Connecting systems or tools
        
        Return only the category name.
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        intent = response.get('content', '').strip().lower()
        
        # Validate intent
        valid_intents = ['data_processing', 'maintenance', 'development', 'analysis', 'automation', 'integration']
        if intent not in valid_intents:
            # Fallback to keyword matching
            intent = self._identify_intent_keywords(description)
            
        return intent
        
    def _identify_intent_keywords(self, description: str) -> str:
        """Identify intent using keyword matching."""
        desc_lower = description.lower()
        
        intent_keywords = {
            'data_processing': ['process', 'transform', 'convert', 'parse', 'extract'],
            'maintenance': ['clean', 'fix', 'repair', 'maintain', 'optimize'],
            'development': ['create', 'build', 'develop', 'implement', 'code'],
            'analysis': ['analyze', 'investigate', 'research', 'examine', 'study'],
            'automation': ['automate', 'schedule', 'repeat', 'routine'],
            'integration': ['connect', 'integrate', 'sync', 'link', 'combine']
        }
        
        scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in desc_lower)
            scores[intent] = score
            
        return max(scores, key=scores.get) if scores else 'automation'
        
    def _find_similar_workflows(self, description: str, intent: str) -> List[Dict[str, Any]]:
        """Find similar successful workflows."""
        similar = []
        
        for workflow in self.successful_workflows:
            # Check intent match
            if workflow.get('metadata', {}).get('intent') == intent:
                similar.append(workflow)
                continue
                
            # Check description similarity
            if self._calculate_similarity(description, workflow.get('description', '')) > 0.7:
                similar.append(workflow)
                
        return similar[:5]  # Return top 5 similar workflows
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    async def _generate_workflow_structure(
        self,
        description: str,
        entities: Dict[str, List[str]],
        intent: str,
        similar_workflows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate the workflow structure."""
        
        # Check if we can use a pattern
        if intent in self.workflow_patterns:
            pattern = self.workflow_patterns[intent]
            return await self._generate_from_pattern(description, entities, pattern)
            
        # Learn from similar workflows
        if similar_workflows:
            return await self._generate_from_similar(description, entities, similar_workflows[0])
            
        # Generate from scratch using AI
        return await self._generate_from_scratch(description, entities, intent)
        
    async def _generate_from_pattern(
        self,
        description: str,
        entities: Dict[str, List[str]],
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workflow from a known pattern."""
        steps = []
        
        for i, (step_type, tool) in enumerate(zip(pattern['pattern'], pattern['tools'])):
            # Check if tool exists
            if self._tool_exists(tool):
                steps.append({
                    'id': f'step_{i+1}',
                    'name': step_type.replace('_', ' ').title(),
                    'type': 'tool',
                    'tool': tool,
                    'params': self._generate_tool_params(tool, entities)
                })
            else:
                # Generate a placeholder for missing tool
                steps.append({
                    'id': f'step_{i+1}',
                    'name': step_type.replace('_', ' ').title(),
                    'type': 'log',
                    'message': f'Tool {tool} not found - would perform: {step_type}'
                })
                
        return {
            'steps': steps,
            'pattern_used': pattern,
            'confidence': 0.9
        }
        
    async def _generate_from_similar(
        self,
        description: str,
        entities: Dict[str, List[str]],
        similar_workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workflow based on similar successful workflow."""
        # Clone the similar workflow structure
        steps = []
        
        for step in similar_workflow.get('steps', []):
            new_step = step.copy()
            
            # Update parameters based on new entities
            if 'params' in new_step:
                new_step['params'] = self._adapt_params(step['params'], entities)
                
            steps.append(new_step)
            
        return {
            'steps': steps,
            'based_on': similar_workflow.get('name', 'Unknown'),
            'confidence': 0.8
        }
        
    async def _generate_from_scratch(
        self,
        description: str,
        entities: Dict[str, List[str]],
        intent: str
    ) -> Dict[str, Any]:
        """Generate workflow from scratch using AI."""
        
        # Get available tools
        available_tools = self._get_available_tools()
        
        prompt = f"""
        Generate a workflow structure for this task:
        "{description}"
        
        Intent: {intent}
        Entities: {json.dumps(entities, indent=2)}
        
        Available tools:
        {json.dumps(available_tools, indent=2)}
        
        Create a JSON workflow with steps. Each step should have:
        - id: Unique identifier
        - name: Human-readable name
        - type: "tool", "parallel", "conditional", "loop", "transform", or "log"
        - tool: Tool name (if type is "tool")
        - params: Parameters for the tool
        
        Example:
        {{
            "steps": [
                {{
                    "id": "step_1",
                    "name": "Fetch Data",
                    "type": "tool",
                    "tool": "fetch_data",
                    "params": {{"source": "api"}}
                }}
            ]
        }}
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            workflow_data = json.loads(response.get('content', '{}'))
            return {
                'steps': workflow_data.get('steps', []),
                'ai_generated': True,
                'confidence': 0.7
            }
        except:
            # Fallback to simple sequential workflow
            return self._generate_simple_workflow(entities)
            
    def _generate_simple_workflow(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate a simple sequential workflow."""
        steps = []
        
        for i, action in enumerate(entities.get('actions', [])):
            steps.append({
                'id': f'step_{i+1}',
                'name': action.title(),
                'type': 'log',
                'message': f'Would perform: {action}'
            })
            
        return {
            'steps': steps,
            'fallback': True,
            'confidence': 0.5
        }
        
    async def _optimize_workflow(self, workflow_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the workflow structure."""
        steps = workflow_structure.get('steps', [])
        
        # Identify parallelizable steps
        optimized_steps = self._identify_parallel_steps(steps)
        
        # Add error handling
        optimized_steps = self._add_error_handling(optimized_steps)
        
        # Add logging and monitoring
        optimized_steps = self._add_monitoring(optimized_steps)
        
        return {
            'steps': optimized_steps,
            'optimizations': ['parallelization', 'error_handling', 'monitoring']
        }
        
    def _identify_parallel_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify steps that can run in parallel."""
        optimized = []
        i = 0
        
        while i < len(steps):
            # Look for independent consecutive steps
            parallel_group = [steps[i]]
            j = i + 1
            
            while j < len(steps) and self._can_parallelize(steps[j], parallel_group):
                parallel_group.append(steps[j])
                j += 1
                
            if len(parallel_group) > 1:
                # Create parallel step
                optimized.append({
                    'id': f'parallel_{i}',
                    'name': f'Parallel Execution ({len(parallel_group)} tasks)',
                    'type': 'parallel',
                    'steps': parallel_group
                })
                i = j
            else:
                optimized.append(steps[i])
                i += 1
                
        return optimized
        
    def _can_parallelize(self, step: Dict[str, Any], group: List[Dict[str, Any]]) -> bool:
        """Check if a step can be parallelized with a group."""
        # Simple heuristic: different tools with no dependencies
        if step.get('type') != 'tool':
            return False
            
        step_tool = step.get('tool', '')
        for other in group:
            if other.get('tool') == step_tool:
                return False  # Same tool might have conflicts
                
        return True
        
    def _add_error_handling(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add error handling to critical steps."""
        enhanced = []
        
        for step in steps:
            if step.get('type') == 'tool':
                # Wrap in error handling
                enhanced.append({
                    'id': f"{step['id']}_with_retry",
                    'name': step['name'],
                    'type': 'conditional',
                    'max_retries': 3,
                    'condition': {'check': 'success'},
                    'then': step,
                    'else': {
                        'type': 'log',
                        'message': f"Failed to execute {step['name']} after retries"
                    }
                })
            else:
                enhanced.append(step)
                
        return enhanced
        
    def _add_monitoring(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add monitoring and logging steps."""
        monitored = []
        
        # Add initial log
        monitored.append({
            'id': 'log_start',
            'name': 'Log Workflow Start',
            'type': 'log',
            'message': 'Starting workflow execution'
        })
        
        # Add existing steps
        monitored.extend(steps)
        
        # Add final summary
        monitored.append({
            'id': 'generate_summary',
            'name': 'Generate Execution Summary',
            'type': 'transform',
            'transform': 'summarize',
            'input': '{{workflow_context}}'
        })
        
        return monitored
        
    def _tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists."""
        try:
            # Check in custom tools
            tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
            if tool_path.exists():
                return True
                
            # Check in registered tools
            # This would need actual implementation
            return False
        except:
            return False
            
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        tools = []
        
        # Get custom tools
        custom_tools_dir = Path("scripts/custom_tools")
        if custom_tools_dir.exists():
            for tool_file in custom_tools_dir.glob("*.py"):
                if not tool_file.name.startswith("_"):
                    tools.append(tool_file.stem)
                    
        return tools
        
    def _generate_tool_params(self, tool_name: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate appropriate parameters for a tool based on entities."""
        params = {}
        
        # Map common parameter patterns
        if entities.get('resources'):
            params['target'] = entities['resources'][0]
            
        if entities.get('conditions'):
            params['conditions'] = entities['conditions']
            
        return params
        
    def _adapt_params(self, original_params: Dict[str, Any], entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Adapt parameters from similar workflow to new entities."""
        adapted = original_params.copy()
        
        # Update resource references
        if entities.get('resources') and 'repo' in adapted:
            adapted['repo'] = entities['resources'][0]
            
        return adapted
        
    def _generate_workflow_name(self, description: str) -> str:
        """Generate a meaningful workflow name."""
        # Extract key words
        words = description.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [w for w in words if w not in stop_words][:4]
        
        return '_'.join(keywords) + '_workflow'
        
    async def learn_from_execution(self, workflow_id: str, execution_result: Dict[str, Any]):
        """Learn from workflow execution results."""
        if execution_result.get('success'):
            # Store successful workflow
            state = self.state_manager.get_state()
            
            if 'successful_workflows' not in state:
                state['successful_workflows'] = []
                
            # Get workflow details
            workflow = self.orchestrator.workflows.get(workflow_id)
            if workflow:
                state['successful_workflows'].append({
                    'id': workflow_id,
                    'name': workflow.name,
                    'description': workflow.description,
                    'steps': [s.to_dict() for s in workflow.steps],
                    'metadata': workflow.context.get('metadata', {}),
                    'execution_time': execution_result.get('duration_ms', 0),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Keep only recent workflows
                state['successful_workflows'] = state['successful_workflows'][-100:]
                
                self.state_manager.state = state
                self.state_manager.save_state()
                
                print(f"Learned from successful workflow: {workflow_id}")


# Convenience function
async def generate_intelligent_workflow(description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate an intelligent workflow from description."""
    generator = IntelligentWorkflowGenerator()
    return await generator.generate_workflow_from_description(description, context)