#!/usr/bin/env python3
"""
Multi-Tool Orchestrator - Complex workflow management for multiple tools
Handles query decomposition, tool chaining, and result aggregation
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

from scripts.ai_brain import AIBrain
from scripts.task_coordinator import TaskCoordinator


class ExecutionStrategy(Enum):
    """Different strategies for executing tools"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    PIPELINE = "pipeline"


@dataclass
class ToolNode:
    """Represents a tool in the orchestration graph"""
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None
    retry_count: int = 3
    timeout: int = 300  # seconds
    
    
@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    success: bool
    results: Dict[str, Any]
    errors: List[str]
    execution_time: float
    execution_path: List[str]


class MultiToolOrchestrator:
    """Orchestrates complex workflows involving multiple tools"""
    
    def __init__(self, tool_system=None):
        self.tool_system = tool_system
        self.ai_brain = AIBrain()
        self.task_coordinator = TaskCoordinator(self.ai_brain)
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose a complex query into sub-tasks"""
        decomposition_prompt = f"""
        Analyze this complex query and break it down into individual sub-tasks:
        Query: {query}
        
        For each sub-task, identify:
        1. The specific action to perform
        2. What tool would be best suited
        3. Required parameters
        4. Dependencies on other sub-tasks
        5. Expected output
        
        Return as JSON array of tasks with structure:
        [{{
            "id": "task_1",
            "description": "...",
            "tool_hint": "tool_name or description",
            "parameters": {{}},
            "dependencies": ["task_id"],
            "expected_output": "..."
        }}]
        """
        
        response = await self.ai_brain.generate_enhanced_response(decomposition_prompt)
        response_content = response.get('content', '') if isinstance(response, dict) else str(response)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group())
                return tasks
            else:
                # Fallback to simple decomposition
                return [{
                    "id": "task_1",
                    "description": query,
                    "tool_hint": "auto",
                    "parameters": {},
                    "dependencies": [],
                    "expected_output": "result"
                }]
        except Exception as e:
            self.logger.error(f"Failed to decompose query: {e}")
            return [{
                "id": "task_1",
                "description": query,
                "tool_hint": "auto",
                "parameters": {},
                "dependencies": [],
                "expected_output": "result"
            }]
            
    async def build_tool_chain(self, tasks: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build a directed graph representing the tool execution chain"""
        graph = nx.DiGraph()
        
        for task in tasks:
            # Find or create appropriate tool
            tool_name = await self._resolve_tool(task)
            
            node = ToolNode(
                id=task['id'],
                tool_name=tool_name,
                parameters=task.get('parameters', {}),
                dependencies=task.get('dependencies', [])
            )
            
            graph.add_node(task['id'], data=node)
            
            # Add edges for dependencies
            for dep in task.get('dependencies', []):
                graph.add_edge(dep, task['id'])
                
        return graph
        
    async def _resolve_tool(self, task: Dict[str, Any]) -> str:
        """Resolve the appropriate tool for a task"""
        tool_hint = task.get('tool_hint', 'auto')
        
        if tool_hint != 'auto' and self.tool_system.get_tool(tool_hint):
            return tool_hint
            
        # Use AI to find or create appropriate tool
        tool_finder_prompt = f"""
        Find or suggest a tool for this task:
        Description: {task['description']}
        Expected output: {task.get('expected_output', 'result')}
        
        Available tools: {list(self.tool_system.list_tools().keys())}
        
        Return the exact tool name if one exists, or 'CREATE_NEW' if a new tool is needed.
        """
        
        response = await self.ai_brain.generate_enhanced_response(tool_finder_prompt)
        response_content = response.get('content', '') if isinstance(response, dict) else str(response)
        
        if 'CREATE_NEW' in response_content:
            # Create new tool
            tool_spec = f"Tool to {task['description']}"
            # Create tool using the create_new_tool method
            create_result = await self.tool_system.call_tool(
                "create_new_tool",
                name=f"tool_{task['id']}",
                description=task['description'],
                requirements=tool_spec
            )
            new_tool_name = f"tool_{task['id']}" if create_result.get('success') else None
            return new_tool_name
        else:
            # Extract tool name from response
            for tool_name in self.tool_system.list_tools():
                if tool_name in response_content:
                    return tool_name
                    
        # Fallback
        return list(self.tool_system.list_tools().keys())[0]
        
    async def execute_workflow(
        self,
        graph: nx.DiGraph,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Execute the workflow according to the specified strategy"""
        context = context or {}
        results = {}
        errors = []
        execution_path = []
        start_time = time.time()
        
        try:
            if strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_sequential(graph, context)
            elif strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(graph, context)
            elif strategy == ExecutionStrategy.PIPELINE:
                results = await self._execute_pipeline(graph, context)
            else:
                results = await self._execute_sequential(graph, context)
                
            success = len(errors) == 0
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            errors.append(str(e))
            success = False
            
        execution_time = time.time() - start_time
        
        return WorkflowResult(
            success=success,
            results=results,
            errors=errors,
            execution_time=execution_time,
            execution_path=execution_path
        )
        
    async def _execute_sequential(
        self,
        graph: nx.DiGraph,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools sequentially in topological order"""
        results = {}
        
        # Get topological order
        try:
            execution_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            raise ValueError("Workflow contains cycles")
            
        for node_id in execution_order:
            node_data = graph.nodes[node_id]['data']
            
            # Prepare parameters with results from dependencies
            params = node_data.parameters.copy()
            for dep in node_data.dependencies:
                if dep in results:
                    params[f"{dep}_result"] = results[dep]
                    
            # Execute tool
            try:
                result = await self._execute_tool(node_data, params)
                results[node_id] = result
                context[node_id] = result  # Update context
            except Exception as e:
                self.logger.error(f"Failed to execute {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                
        return results
        
    async def _execute_parallel(
        self,
        graph: nx.DiGraph,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute independent tools in parallel"""
        results = {}
        
        # Find groups of independent nodes
        levels = self._get_execution_levels(graph)
        
        for level in levels:
            # Execute all nodes in this level in parallel
            tasks = []
            for node_id in level:
                node_data = graph.nodes[node_id]['data']
                
                # Prepare parameters
                params = node_data.parameters.copy()
                for dep in node_data.dependencies:
                    if dep in results:
                        params[f"{dep}_result"] = results[dep]
                        
                task = self._execute_tool(node_data, params)
                tasks.append((node_id, task))
                
            # Wait for all tasks in this level
            level_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Store results
            for (node_id, _), result in zip(tasks, level_results):
                if isinstance(result, Exception):
                    results[node_id] = {"error": str(result)}
                else:
                    results[node_id] = result
                    context[node_id] = result
                    
        return results
        
    async def _execute_pipeline(
        self,
        graph: nx.DiGraph,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tools in a pipeline, passing output to next tool"""
        results = {}
        
        # Get linear path (assumes single path)
        path = list(nx.topological_sort(graph))
        
        previous_result = None
        for node_id in path:
            node_data = graph.nodes[node_id]['data']
            
            # Use previous result as input
            params = node_data.parameters.copy()
            if previous_result is not None:
                params['input'] = previous_result
                
            # Execute tool
            try:
                result = await self._execute_tool(node_data, params)
                results[node_id] = result
                previous_result = result
            except Exception as e:
                self.logger.error(f"Pipeline failed at {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                break
                
        return results
        
    def _get_execution_levels(self, graph: nx.DiGraph) -> List[List[str]]:
        """Group nodes into levels for parallel execution"""
        levels = []
        remaining = set(graph.nodes())
        completed = set()
        
        while remaining:
            # Find nodes with all dependencies completed
            current_level = []
            for node in remaining:
                deps = set(graph.predecessors(node))
                if deps.issubset(completed):
                    current_level.append(node)
                    
            if not current_level:
                raise ValueError("Cannot determine execution order - possible cycle")
                
            levels.append(current_level)
            completed.update(current_level)
            remaining.difference_update(current_level)
            
        return levels
        
    async def _execute_tool(
        self,
        node: ToolNode,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute a single tool with retry logic"""
        for attempt in range(node.retry_count):
            try:
                # Set timeout
                result = await asyncio.wait_for(
                    self.tool_system.call_tool(node.tool_name, **parameters),
                    timeout=node.timeout
                )
                return result
            except asyncio.TimeoutError:
                self.logger.warning(f"Tool {node.tool_name} timed out (attempt {attempt + 1})")
                if attempt == node.retry_count - 1:
                    raise
            except Exception as e:
                self.logger.warning(f"Tool {node.tool_name} failed (attempt {attempt + 1}): {e}")
                if attempt == node.retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
    async def aggregate_results(
        self,
        results: Dict[str, Any],
        aggregation_strategy: str = "combine"
    ) -> Any:
        """Aggregate results from multiple tools"""
        if aggregation_strategy == "combine":
            # Simply combine all results
            return results
        elif aggregation_strategy == "summarize":
            # Use AI to summarize results
            summary_prompt = f"""
            Summarize these results from multiple tools:
            {json.dumps(results, indent=2)}
            
            Provide a coherent summary that combines the key findings.
            """
            summary_response = await self.ai_brain.generate_enhanced_response(summary_prompt)
            summary = summary_response.get('content', '') if isinstance(summary_response, dict) else str(summary_response)
            return {"summary": summary, "detailed_results": results}
        elif aggregation_strategy == "merge":
            # Merge dictionaries
            merged = {}
            for result in results.values():
                if isinstance(result, dict):
                    merged.update(result)
                else:
                    merged[str(uuid.uuid4())] = result
            return merged
        else:
            return results
            
    async def handle_complex_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for handling complex queries"""
        self.logger.info(f"Handling complex query: {query}")
        
        # Decompose query into tasks
        tasks = await self.decompose_query(query)
        self.logger.info(f"Decomposed into {len(tasks)} tasks")
        
        # Build execution graph
        graph = await self.build_tool_chain(tasks)
        
        # Determine execution strategy
        strategy = self._determine_strategy(graph)
        
        # Execute workflow
        workflow_result = await self.execute_workflow(graph, strategy)
        
        # Aggregate results
        if workflow_result.success:
            final_result = await self.aggregate_results(
                workflow_result.results,
                "summarize"
            )
        else:
            final_result = {
                "error": "Workflow execution failed",
                "errors": workflow_result.errors,
                "partial_results": workflow_result.results
            }
            
        return {
            "query": query,
            "result": final_result,
            "execution_time": workflow_result.execution_time,
            "tasks_executed": len(tasks),
            "strategy": strategy.value
        }
        
    def _determine_strategy(self, graph: nx.DiGraph) -> ExecutionStrategy:
        """Determine the best execution strategy based on graph structure"""
        # Check if graph is linear (pipeline)
        if nx.is_directed_acyclic_graph(graph):
            nodes = list(graph.nodes())
            if all(graph.out_degree(n) <= 1 for n in nodes):
                return ExecutionStrategy.PIPELINE
                
        # Check for parallelizable structure
        levels = self._get_execution_levels(graph)
        if any(len(level) > 1 for level in levels):
            return ExecutionStrategy.PARALLEL
            
        # Default to sequential
        return ExecutionStrategy.SEQUENTIAL


# Example workflow definitions
class WorkflowTemplates:
    """Pre-defined workflow templates for common scenarios"""
    
    @staticmethod
    def security_audit_workflow() -> List[Dict[str, Any]]:
        """Workflow for comprehensive security audit"""
        return [
            {
                "id": "scan_repos",
                "description": "Scan all repositories for security files",
                "tool_hint": "analyze_repository",
                "parameters": {"focus": "security"},
                "dependencies": [],
                "expected_output": "repository_list"
            },
            {
                "id": "find_vulnerabilities",
                "description": "Identify security vulnerabilities",
                "tool_hint": "security_scanner",
                "parameters": {},
                "dependencies": ["scan_repos"],
                "expected_output": "vulnerability_list"
            },
            {
                "id": "create_issues",
                "description": "Create GitHub issues for vulnerabilities",
                "tool_hint": "create_issue",
                "parameters": {"priority": "high"},
                "dependencies": ["find_vulnerabilities"],
                "expected_output": "issue_urls"
            },
            {
                "id": "generate_report",
                "description": "Generate security audit report",
                "tool_hint": "report_generator",
                "parameters": {"format": "markdown"},
                "dependencies": ["find_vulnerabilities", "create_issues"],
                "expected_output": "report_content"
            }
        ]
        
    @staticmethod
    def code_improvement_workflow() -> List[Dict[str, Any]]:
        """Workflow for code improvement across repositories"""
        return [
            {
                "id": "analyze_code",
                "description": "Analyze code quality metrics",
                "tool_hint": "code_analyzer",
                "parameters": {},
                "dependencies": [],
                "expected_output": "quality_metrics"
            },
            {
                "id": "identify_improvements",
                "description": "Identify improvement opportunities",
                "tool_hint": "improvement_finder",
                "parameters": {},
                "dependencies": ["analyze_code"],
                "expected_output": "improvement_list"
            },
            {
                "id": "generate_patches",
                "description": "Generate code patches",
                "tool_hint": "patch_generator",
                "parameters": {},
                "dependencies": ["identify_improvements"],
                "expected_output": "patches"
            },
            {
                "id": "test_patches",
                "description": "Test generated patches",
                "tool_hint": "test_runner",
                "parameters": {},
                "dependencies": ["generate_patches"],
                "expected_output": "test_results"
            }
        ]


if __name__ == "__main__":
    # Example usage
    async def demo():
        orchestrator = MultiToolOrchestrator()
        
        # Example complex query
        query = "Analyze all repositories, identify security vulnerabilities, and create GitHub issues for each"
        
        result = await orchestrator.handle_complex_query(query)
        print(json.dumps(result, indent=2))
        
    asyncio.run(demo())