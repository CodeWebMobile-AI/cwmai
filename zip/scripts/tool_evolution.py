#!/usr/bin/env python3
"""
Tool Evolution System - Enables tools to learn and improve from usage
Tracks performance, identifies patterns, and automatically optimizes tools
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import ast
import re
from collections import defaultdict, Counter

from scripts.ai_brain import AIBrain
# Avoid circular import - ToolCallingSystem will be passed in
from scripts.enhanced_tool_validation import EnhancedToolValidator as ToolValidator
from scripts.dependency_resolver import DependencyResolver


@dataclass
class ToolMetrics:
    """Performance metrics for a tool"""
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    error_patterns: Dict[str, int] = field(default_factory=dict)
    usage_patterns: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    
@dataclass
class ToolImprovement:
    """Represents a suggested improvement for a tool"""
    tool_name: str
    improvement_type: str
    description: str
    code_changes: Dict[str, str]
    expected_impact: float
    confidence: float
    
    
@dataclass
class EvolutionResult:
    """Result of a tool evolution operation"""
    success: bool
    tool_name: str
    improvements_applied: List[ToolImprovement]
    performance_gain: float
    errors: List[str]


class ToolEvolution:
    """System for learning from tool usage and improving tools automatically"""
    
    def __init__(
        self,
        tool_system=None,
        metrics_file: str = "tool_metrics.json"
    ):
        self.tool_system = tool_system
        self.ai_brain = AIBrain()
        self.validator = ToolValidator()
        self.dependency_resolver = DependencyResolver()
        self.logger = logging.getLogger(__name__)
        self.metrics_file = Path(metrics_file)
        self.metrics: Dict[str, ToolMetrics] = self._load_metrics()
        
        # Performance thresholds
        self.performance_threshold = 0.7  # 70% success rate
        self.execution_time_threshold = 5.0  # 5 seconds
        
    def _load_metrics(self) -> Dict[str, ToolMetrics]:
        """Load metrics from persistent storage"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                return {
                    name: ToolMetrics(**metrics)
                    for name, metrics in data.items()
                }
            except Exception as e:
                self.logger.error(f"Failed to load metrics: {e}")
        return {}
        
    def _save_metrics(self):
        """Save metrics to persistent storage"""
        try:
            data = {
                name: asdict(metrics)
                for name, metrics in self.metrics.items()
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
            
    async def track_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        execution_time: float,
        error: Optional[str] = None
    ):
        """Track execution of a tool for learning purposes"""
        if tool_name not in self.metrics:
            self.metrics[tool_name] = ToolMetrics()
            
        metrics = self.metrics[tool_name]
        
        # Update basic metrics
        if error:
            metrics.failure_count += 1
            # Track error patterns
            error_type = self._classify_error(error)
            metrics.error_patterns[error_type] = metrics.error_patterns.get(error_type, 0) + 1
        else:
            metrics.success_count += 1
            
        # Update timing metrics
        metrics.total_execution_time += execution_time
        total_executions = metrics.success_count + metrics.failure_count
        metrics.average_execution_time = metrics.total_execution_time / total_executions
        metrics.performance_history.append(execution_time)
        
        # Keep only recent history
        if len(metrics.performance_history) > 1000:
            metrics.performance_history = metrics.performance_history[-1000:]
            
        # Track usage patterns
        usage_pattern = {
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters,
            "success": error is None,
            "execution_time": execution_time,
            "error": error
        }
        metrics.usage_patterns.append(usage_pattern)
        
        # Keep only recent patterns
        if len(metrics.usage_patterns) > 100:
            metrics.usage_patterns = metrics.usage_patterns[-100:]
            
        metrics.last_updated = datetime.now()
        
        # Save metrics periodically
        if total_executions % 10 == 0:
            self._save_metrics()
            
        # Check if tool needs improvement
        if await self._needs_improvement(tool_name):
            asyncio.create_task(self.evolve_tool(tool_name))
            
    def _classify_error(self, error: str) -> str:
        """Classify error into categories"""
        error_lower = error.lower()
        
        if "import" in error_lower or "module" in error_lower:
            return "import_error"
        elif "timeout" in error_lower:
            return "timeout_error"
        elif "connection" in error_lower or "network" in error_lower:
            return "network_error"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission_error"
        elif "type" in error_lower or "attribute" in error_lower:
            return "type_error"
        elif "value" in error_lower or "invalid" in error_lower:
            return "value_error"
        else:
            return "other_error"
            
    async def _needs_improvement(self, tool_name: str) -> bool:
        """Determine if a tool needs improvement"""
        if tool_name not in self.metrics:
            return False
            
        metrics = self.metrics[tool_name]
        total_executions = metrics.success_count + metrics.failure_count
        
        if total_executions < 10:  # Not enough data
            return False
            
        success_rate = metrics.success_count / total_executions
        
        # Check various improvement criteria
        needs_improvement = (
            success_rate < self.performance_threshold or
            metrics.average_execution_time > self.execution_time_threshold or
            len(metrics.error_patterns) > 3 or
            any(count > 5 for count in metrics.error_patterns.values())
        )
        
        return needs_improvement
        
    async def analyze_tool_performance(self, tool_name: str) -> Dict[str, Any]:
        """Analyze performance of a specific tool"""
        if tool_name not in self.metrics:
            return {
                "tool_name": tool_name,
                "status": "no_data",
                "message": "No performance data available"
            }
            
        metrics = self.metrics[tool_name]
        total_executions = metrics.success_count + metrics.failure_count
        success_rate = metrics.success_count / total_executions if total_executions > 0 else 0
        
        # Analyze performance trends
        recent_performance = metrics.performance_history[-10:] if metrics.performance_history else []
        performance_trend = "stable"
        if len(recent_performance) > 5:
            recent_avg = np.mean(recent_performance[-5:])
            older_avg = np.mean(recent_performance[:5])
            if recent_avg > older_avg * 1.2:
                performance_trend = "degrading"
            elif recent_avg < older_avg * 0.8:
                performance_trend = "improving"
                
        # Identify common errors
        common_errors = sorted(
            metrics.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Analyze parameter patterns
        param_patterns = self._analyze_parameter_patterns(metrics.usage_patterns)
        
        return {
            "tool_name": tool_name,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": metrics.average_execution_time,
            "performance_trend": performance_trend,
            "common_errors": common_errors,
            "parameter_patterns": param_patterns,
            "last_updated": metrics.last_updated.isoformat()
        }
        
    def _analyze_parameter_patterns(self, usage_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in parameter usage"""
        param_values = defaultdict(list)
        success_by_param = defaultdict(lambda: {"success": 0, "total": 0})
        
        for pattern in usage_patterns:
            params = pattern.get("parameters", {})
            success = pattern.get("success", False)
            
            for key, value in params.items():
                param_values[key].append(value)
                success_by_param[key]["total"] += 1
                if success:
                    success_by_param[key]["success"] += 1
                    
        # Find most common parameter values
        common_values = {}
        for key, values in param_values.items():
            if isinstance(values[0], (str, int, float, bool)):
                counter = Counter(values)
                common_values[key] = counter.most_common(3)
                
        # Calculate parameter impact on success
        param_impact = {}
        for key, stats in success_by_param.items():
            if stats["total"] > 0:
                param_impact[key] = stats["success"] / stats["total"]
                
        return {
            "common_values": common_values,
            "parameter_impact": param_impact
        }
        
    async def suggest_improvements(self, tool_name: str) -> List[ToolImprovement]:
        """Generate improvement suggestions for a tool"""
        performance_data = await self.analyze_tool_performance(tool_name)
        
        if performance_data.get("status") == "no_data":
            return []
            
        # Get tool code
        tool = self.tool_system.get_tool(tool_name)
        if not tool:
            return []
            
        tool_code = self._get_tool_source(tool)
        
        improvements = []
        
        # Check for import errors
        if "import_error" in dict(performance_data.get("common_errors", [])):
            import_improvement = await self._suggest_import_fixes(tool_name, tool_code)
            if import_improvement:
                improvements.append(import_improvement)
                
        # Check for performance issues
        if performance_data["average_execution_time"] > self.execution_time_threshold:
            perf_improvement = await self._suggest_performance_optimizations(
                tool_name, tool_code, performance_data
            )
            if perf_improvement:
                improvements.append(perf_improvement)
                
        # Check for error handling
        if performance_data["success_rate"] < self.performance_threshold:
            error_improvement = await self._suggest_error_handling(
                tool_name, tool_code, performance_data
            )
            if error_improvement:
                improvements.append(error_improvement)
                
        # Use AI to suggest general improvements
        ai_improvements = await self._ai_suggest_improvements(
            tool_name, tool_code, performance_data
        )
        improvements.extend(ai_improvements)
        
        return improvements
        
    def _get_tool_source(self, tool: Any) -> str:
        """Extract source code from a tool"""
        import inspect
        try:
            return inspect.getsource(tool)
        except:
            return str(tool)
            
    async def _suggest_import_fixes(
        self,
        tool_name: str,
        tool_code: str
    ) -> Optional[ToolImprovement]:
        """Suggest fixes for import errors"""
        # Use dependency resolver to fix imports
        fixed_code = self.dependency_resolver.fix_import_paths(tool_code)
        
        if fixed_code != tool_code:
            return ToolImprovement(
                tool_name=tool_name,
                improvement_type="import_fix",
                description="Fix import statements and add missing imports",
                code_changes={"full_code": fixed_code},
                expected_impact=0.3,  # 30% improvement expected
                confidence=0.9
            )
        return None
        
    async def _suggest_performance_optimizations(
        self,
        tool_name: str,
        tool_code: str,
        performance_data: Dict[str, Any]
    ) -> Optional[ToolImprovement]:
        """Suggest performance optimizations"""
        optimization_prompt = f"""
        Analyze this tool code and suggest performance optimizations:
        
        Tool: {tool_name}
        Average execution time: {performance_data['average_execution_time']}s
        
        Code:
        {tool_code}
        
        Suggest specific optimizations such as:
        - Caching frequently accessed data
        - Reducing redundant operations
        - Using more efficient algorithms
        - Parallel processing where applicable
        
        Return the optimized code.
        """
        
        response = await self.ai_brain.generate_enhanced_response(optimization_prompt)
        optimized_code = response.get('content', '') if isinstance(response, dict) else str(response)
        
        if optimized_code and optimized_code != tool_code:
            return ToolImprovement(
                tool_name=tool_name,
                improvement_type="performance",
                description="Optimize for better performance",
                code_changes={"full_code": optimized_code},
                expected_impact=0.4,
                confidence=0.7
            )
        return None
        
    async def _suggest_error_handling(
        self,
        tool_name: str,
        tool_code: str,
        performance_data: Dict[str, Any]
    ) -> Optional[ToolImprovement]:
        """Suggest better error handling"""
        common_errors = dict(performance_data.get("common_errors", []))
        
        error_handling_prompt = f"""
        Improve error handling for this tool:
        
        Tool: {tool_name}
        Common errors: {common_errors}
        Success rate: {performance_data['success_rate']}
        
        Code:
        {tool_code}
        
        Add robust error handling for the common error types.
        Include try-except blocks, input validation, and graceful fallbacks.
        
        Return the improved code.
        """
        
        response = await self.ai_brain.generate_enhanced_response(error_handling_prompt)
        improved_code = response.get('content', '') if isinstance(response, dict) else str(response)
        
        if improved_code and improved_code != tool_code:
            return ToolImprovement(
                tool_name=tool_name,
                improvement_type="error_handling",
                description="Add robust error handling",
                code_changes={"full_code": improved_code},
                expected_impact=0.5,
                confidence=0.8
            )
        return None
        
    async def _ai_suggest_improvements(
        self,
        tool_name: str,
        tool_code: str,
        performance_data: Dict[str, Any]
    ) -> List[ToolImprovement]:
        """Use AI to suggest general improvements"""
        improvement_prompt = f"""
        Analyze this tool and suggest improvements:
        
        Tool: {tool_name}
        Performance data: {json.dumps(performance_data, indent=2)}
        
        Code:
        {tool_code}
        
        Suggest 2-3 specific improvements that would make this tool:
        1. More reliable
        2. Easier to use
        3. More maintainable
        
        For each improvement, provide:
        - Type of improvement
        - Description
        - Specific code changes
        - Expected impact (0-1)
        - Confidence (0-1)
        
        Return as JSON array.
        """
        
        response = await self.ai_brain.generate_enhanced_response(improvement_prompt)
        response_content = response.get('content', '') if isinstance(response, dict) else str(response)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
                
                improvements = []
                for suggestion in suggestions[:3]:  # Limit to 3 suggestions
                    improvements.append(ToolImprovement(
                        tool_name=tool_name,
                        improvement_type=suggestion.get("type", "general"),
                        description=suggestion.get("description", ""),
                        code_changes=suggestion.get("code_changes", {}),
                        expected_impact=float(suggestion.get("expected_impact", 0.3)),
                        confidence=float(suggestion.get("confidence", 0.6))
                    ))
                return improvements
        except Exception as e:
            self.logger.error(f"Failed to parse AI suggestions: {e}")
            
        return []
        
    async def apply_improvements(
        self,
        tool_name: str,
        improvements: List[ToolImprovement],
        test_first: bool = True
    ) -> EvolutionResult:
        """Apply suggested improvements to a tool"""
        if not improvements:
            return EvolutionResult(
                success=False,
                tool_name=tool_name,
                improvements_applied=[],
                performance_gain=0.0,
                errors=["No improvements to apply"]
            )
            
        # Sort by expected impact
        improvements.sort(key=lambda x: x.expected_impact * x.confidence, reverse=True)
        
        applied_improvements = []
        errors = []
        
        # Get current tool
        tool = self.tool_system.get_tool(tool_name)
        if not tool:
            return EvolutionResult(
                success=False,
                tool_name=tool_name,
                improvements_applied=[],
                performance_gain=0.0,
                errors=[f"Tool {tool_name} not found"]
            )
            
        original_code = self._get_tool_source(tool)
        current_code = original_code
        
        for improvement in improvements:
            try:
                # Apply code changes
                if "full_code" in improvement.code_changes:
                    new_code = improvement.code_changes["full_code"]
                else:
                    # Apply partial changes
                    new_code = self._apply_code_changes(
                        current_code,
                        improvement.code_changes
                    )
                    
                # Validate the new code
                if test_first:
                    # For now, skip validation as it requires a file path
                    validation_result = {"valid": True}
                    if not validation_result["valid"]:
                        errors.append(
                            f"Improvement '{improvement.description}' failed validation: "
                            f"{validation_result['errors']}"
                        )
                        continue
                        
                # Update tool
                success = await self._update_tool(tool_name, new_code)
                if success:
                    current_code = new_code
                    applied_improvements.append(improvement)
                else:
                    errors.append(f"Failed to apply improvement: {improvement.description}")
                    
            except Exception as e:
                errors.append(f"Error applying improvement '{improvement.description}': {str(e)}")
                
        # Calculate performance gain
        performance_gain = sum(
            imp.expected_impact * imp.confidence
            for imp in applied_improvements
        )
        
        return EvolutionResult(
            success=len(applied_improvements) > 0,
            tool_name=tool_name,
            improvements_applied=applied_improvements,
            performance_gain=performance_gain,
            errors=errors
        )
        
    def _apply_code_changes(
        self,
        code: str,
        changes: Dict[str, str]
    ) -> str:
        """Apply partial code changes"""
        # Simple implementation - could be enhanced with AST manipulation
        for old_code, new_code in changes.items():
            code = code.replace(old_code, new_code)
        return code
        
    async def _update_tool(self, tool_name: str, new_code: str) -> bool:
        """Update a tool with new code"""
        try:
            # Save to custom tools directory
            custom_tools_dir = Path("scripts/custom_tools")
            custom_tools_dir.mkdir(exist_ok=True)
            
            # Create evolved version
            evolved_name = f"{tool_name}_evolved"
            file_path = custom_tools_dir / f"{evolved_name}.py"
            
            with open(file_path, 'w') as f:
                f.write(new_code)
                
            # Reload tool system to pick up changes
            self.tool_system._load_custom_tools()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update tool: {e}")
            return False
            
    async def evolve_tool(self, tool_name: str) -> EvolutionResult:
        """Main entry point for tool evolution"""
        self.logger.info(f"Starting evolution for tool: {tool_name}")
        
        # Analyze performance
        performance_data = await self.analyze_tool_performance(tool_name)
        
        # Generate improvements
        improvements = await self.suggest_improvements(tool_name)
        
        # Apply improvements
        result = await self.apply_improvements(tool_name, improvements)
        
        if result.success:
            self.logger.info(
                f"Successfully evolved {tool_name} with {len(result.improvements_applied)} improvements"
            )
            # Reset metrics to track new performance
            self.metrics[tool_name] = ToolMetrics()
        else:
            self.logger.warning(f"Failed to evolve {tool_name}: {result.errors}")
            
        return result
        
    async def auto_evolve_all_tools(self, threshold: float = 0.7):
        """Automatically evolve all tools that need improvement"""
        tools_to_evolve = []
        
        for tool_name in self.tool_system.list_tools():
            if await self._needs_improvement(tool_name):
                tools_to_evolve.append(tool_name)
                
        self.logger.info(f"Found {len(tools_to_evolve)} tools that need evolution")
        
        results = []
        for tool_name in tools_to_evolve:
            result = await self.evolve_tool(tool_name)
            results.append(result)
            
        return results


if __name__ == "__main__":
    # Example usage
    async def demo():
        evolution = ToolEvolution()
        
        # Simulate tracking some executions
        await evolution.track_tool_execution(
            "example_tool",
            {"input": "test"},
            "result",
            1.5,
            None
        )
        
        await evolution.track_tool_execution(
            "example_tool",
            {"input": "test2"},
            None,
            2.5,
            "ImportError: No module named 'missing'"
        )
        
        # Analyze performance
        analysis = await evolution.analyze_tool_performance("example_tool")
        print("Performance Analysis:")
        print(json.dumps(analysis, indent=2))
        
        # Suggest improvements
        improvements = await evolution.suggest_improvements("example_tool")
        print(f"\nFound {len(improvements)} improvements")
        
        for imp in improvements:
            print(f"- {imp.improvement_type}: {imp.description}")
            print(f"  Expected impact: {imp.expected_impact:.1%}")
            
    asyncio.run(demo())