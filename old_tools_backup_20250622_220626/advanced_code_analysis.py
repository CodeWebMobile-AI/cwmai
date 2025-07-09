#!/usr/bin/env python3
"""
Advanced Code Analysis Tool - Integrates all new advanced features
"""

import asyncio
import json
import os
import hashlib
from typing import Dict, List, Any, Optional

from scripts.deep_code_analyzer import DeepCodeAnalyzer
from scripts.benchmarking_system import BenchmarkingSystem
from scripts.self_debugging_engine import SelfDebuggingEngine
from scripts.specialized_agents.multi_agent_collaboration import AgentCoordinator
import logging

logger = logging.getLogger(__name__)

# Configuration for caching
CACHE_ENABLED = True
CACHE_DIR = ".code_analysis_cache"
if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class AdvancedCodeAnalysisTool:
    """Tool that integrates all advanced code analysis features"""
    
    def __init__(self):
        self.deep_analyzer = DeepCodeAnalyzer()
        self.benchmarking = BenchmarkingSystem()
        self.debugging_engine = SelfDebuggingEngine()
        self.agent_coordinator = AgentCoordinator()
        
    def _generate_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a cache key based on function name and arguments."""
        arg_string = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        combined = f"{func_name}_{arg_string}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _cached_call(self, func, func_name: str, *args, **kwargs):
        """
        Wrapper to cache results of async functions.
        """
        if not CACHE_ENABLED:
            return await func(*args, **kwargs)
        
        cache_key = self._generate_cache_key(func_name, *args, **kwargs)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    logger.debug(f"Cache hit for {func_name} with key {cache_key}")
                    return json.load(f)
            
            result = await func(*args, **kwargs)
            
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=4)
                logger.debug(f"Cache miss for {func_name}, saving to cache with key {cache_key}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during caching for {func_name}: {e}")
            return await func(*args, **kwargs)
    
    def _validate_code(self, code: str) -> None:
        """
        Validate input code to prevent common issues.
        
        Args:
            code: The code string to validate.
        
        Raises:
            ValueError: If code is empty or exceeds length limits.
        """
        if not code:
            raise ValueError("Code cannot be empty.")
        if len(code) > 10000:  # Example limit, adjust as needed
            raise ValueError("Code length exceeds maximum allowed (10000 characters).")

    def _format_results(self, results: Dict[str, Any], indent: int = 4) -> str:
        """
        Formats the results dictionary into a pretty JSON string.
        """
        try:
            return json.dumps(results, indent=indent)
        except (TypeError, ValueError) as e:
            logger.error(f"Error formatting results to JSON: {e}")
            return str(results)
    
    async def analyze_code_advanced(self, code: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform advanced code analysis using all available systems
        
        Args:
            code: The code to analyze
            analysis_type: Type of analysis (comprehensive, security, performance, quality)
            
        Returns:
            Comprehensive analysis results
        """
        # Input validation
        try:
            self._validate_code(code)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return {"error": str(e), "analysis_type": analysis_type}
        
        async def _analyze_code(code: str, analysis_type: str) -> Dict[str, Any]:
            logger.info(f"Starting advanced {analysis_type} analysis")
            
            results = {
                "analysis_type": analysis_type,
                "code_length": len(code.split('\n'))
            }
            
            try:
                if analysis_type in ["comprehensive", "quality"]:
                    # Deep code analysis
                    intent = self.deep_analyzer.analyze_code_intent(code)
                    refactorings = self.deep_analyzer.suggest_refactoring(code, intent)
                    
                    results["code_intent"] = {
                        "purpose": intent.purpose,
                        "quality_score": intent.quality_score,
                        "patterns_used": intent.patterns_used,
                        "improvement_areas": intent.improvement_areas
                    }
                    
                    results["refactoring_suggestions"] = [
                        {
                            "description": r.description,
                            "rationale": r.rationale,
                            "impact": r.impact,
                            "risk_level": r.risk_level
                        } for r in refactorings[:5]
                    ]
                
                if analysis_type in ["comprehensive", "security"]:
                    # Bug detection
                    bugs = await self.debugging_engine.detect_bugs()
                    
                    # Generate fixes for critical bugs
                    critical_bugs = [b for b in bugs if b.severity in ["critical", "high"]][:3]
                    fixes = {}
                    
                    for bug in critical_bugs:
                        bug_fixes = await self.debugging_engine.generate_fixes(bug)
                        fixes[bug.bug_id] = [
                            {
                                "fix_type": f.fix_type,
                                "explanation": f.explanation,
                                "confidence": f.confidence,
                                "validation_status": f.validation_status
                            } for f in bug_fixes
                        ]
                    
                    results["bug_analysis"] = {
                        "total_bugs": len(bugs),
                        "critical_bugs": len([b for b in bugs if b.severity == "critical"]),
                        "high_severity_bugs": len([b for b in bugs if b.severity == "high"]),
                        "fixes_generated": len(fixes)
                    }
                    
                    results["bug_fixes"] = fixes
                
                if analysis_type in ["comprehensive", "performance"]:
                    # Performance benchmarking
                    tool_results = await self.benchmarking.benchmark_tools(iterations=3)
                    
                    results["performance_analysis"] = {
                        "benchmarked_tools": len(tool_results),
                        "average_execution_times": {
                            name: report.average_time 
                            for name, report in tool_results.items()
                        },
                        "regressions_detected": [
                            name for name, report in tool_results.items() 
                            if report.regression_detected
                        ]
                    }
                
                # Multi-agent analysis for comprehensive analysis
                if analysis_type == "comprehensive":
                    agent_task = {
                        "type": "comprehensive",
                        "code": code,
                        "context": {"analysis_type": analysis_type}
                    }
                    
                    agent_results = await self.agent_coordinator.coordinate_task(agent_task)
                    
                    results["multi_agent_analysis"] = {
                        "summary": agent_results["summary"],
                        "final_recommendations": agent_results["final_recommendations"],
                        "agent_insights": {
                            agent: result.get("summary", "")
                            for agent, result in agent_results["agent_results"].items()
                            if "error" not in result
                        }
                    }
                
                # Generate overall summary
                results["summary"] = self._generate_summary(results)
                
            except Exception as e:
                logger.info(f"Error in advanced analysis: {str(e)}")
                results["error"] = str(e)
            
            return results
        
        return await self._cached_call(_analyze_code, "analyze_code_advanced", code, analysis_type)
    
    async def optimize_code(self, code: str, target: str = "balanced") -> Dict[str, Any]:
        """
        Optimize code using multi-agent collaboration
        
        Args:
            code: Code to optimize
            target: Optimization target (performance, security, readability, balanced)
            
        Returns:
            Optimized code and recommendations
        """
        # Input validation
        try:
            self._validate_code(code)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return {"error": str(e), "target": target}
        
        async def _optimize_code(code: str, target: str) -> Dict[str, Any]:
            logger.info(f"Optimizing code for {target}")
            
            # Use agent coordinator for optimization
            agent_task = {
                "type": f"{target}_focused" if target != "balanced" else "comprehensive",
                "code": code,
                "target": target
            }
            
            results = await self.agent_coordinator.coordinate_task(agent_task)
            
            # Extract optimized code from results
            optimized_code = code  # Default to original
            
            if "code_optimizer" in results["agent_results"]:
                opt_result = results["agent_results"]["code_optimizer"]
                if "optimized_code" in opt_result:
                    optimized_code = opt_result["optimized_code"]
            
            return {
                "original_code": code,
                "optimized_code": optimized_code,
                "optimization_target": target,
                "improvements": results.get("final_recommendations", []),
                "metrics": results.get("summary", ""),
                "agent_details": results.get("agent_results", {})
            }
        
        return await self._cached_call(_optimize_code, "optimize_code", code, target)
    
    async def debug_and_fix(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect and fix bugs in code
        
        Args:
            file_path: Optional specific file to debug
            
        Returns:
            Bug report and fixes
        """
        async def _debug_and_fix(file_path: Optional[str]) -> Dict[str, Any]:
            logger.info("Running debug and fix analysis")
            
            # Detect bugs
            file_paths = [file_path] if file_path else None
            bugs = await self.debugging_engine.detect_bugs(file_paths)
            
            # Generate fixes for high-priority bugs
            fixes = {}
            fixed_count = 0
            
            for bug in bugs[:10]:  # Fix up to 10 bugs
                if bug.severity in ["critical", "high"]:
                    bug_fixes = await self.debugging_engine.generate_fixes(bug)
                    
                    # Apply first validated fix
                    for fix in bug_fixes:
                        if fix.validation_status == "passed":
                            success = self.debugging_engine.apply_fix(fix, backup=True)
                            if success:
                                fixed_count += 1
                                fixes[bug.bug_id] = {
                                    "applied": True,
                                    "fix_type": fix.fix_type,
                                    "explanation": fix.explanation
                                }
                            break
            
            # Generate report
            report = self.debugging_engine.generate_bug_report(bugs, fixes)
            
            return {
                "bugs_found": len(bugs),
                "bugs_fixed": fixed_count,
                "bug_summary": {
                    "critical": len([b for b in bugs if b.severity == "critical"]),
                    "high": len([b for b in bugs if b.severity == "high"]),
                    "medium": len([b for b in bugs if b.severity == "medium"]),
                    "low": len([b for b in bugs if b.severity == "low"])
                },
                "fixes_applied": fixes,
                "report": report
            }
        
        return await self._cached_call(_debug_and_fix, "debug_and_fix", file_path)
    
    async def benchmark_performance(self, tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Benchmark tool performance
        
        Args:
            tools: Optional list of specific tools to benchmark
            
        Returns:
            Performance benchmarking results
        """
        async def _benchmark_performance(tools: Optional[List[str]]) -> Dict[str, Any]:
            logger.info("Running performance benchmarks")
            
            # Benchmark tools
            tool_results = await self.benchmarking.benchmark_tools(tools, iterations=5)
            
            # Benchmark against external standards
            external_results = await self.benchmarking.benchmark_against_external()
            
            # Generate comprehensive report
            all_results = {
                "tool_benchmarks": tool_results,
                "external_benchmarks": external_results
            }
            
            report = self.benchmarking.generate_benchmark_report(all_results)
            
            return {
                "tools_benchmarked": len(tool_results),
                "performance_summary": {
                    name: {
                        "avg_time": report.average_time,
                        "success_rate": report.success_rate,
                        "regression": report.regression_detected
                    }
                    for name, report in tool_results.items()
                },
                "external_comparison": {
                    op: result.get("comparison", {})
                    for op, result in external_results.items()
                    if "error" not in result
                },
                "full_report": report
            }
        
        return await self._cached_call(_benchmark_performance, "benchmark_performance", tools)
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of analysis results"""
        summary_parts = []
        
        if "code_intent" in results:
            summary_parts.append(f"Code quality: {results['code_intent']['quality_score']:.1f}/1.0")
        
        if "bug_analysis" in results:
            bugs = results["bug_analysis"]
            summary_parts.append(f"Bugs found: {bugs['total_bugs']} ({bugs['critical_bugs']} critical)")
        
        if "performance_analysis" in results:
            perf = results["performance_analysis"]
            summary_parts.append(f"Tools benchmarked: {perf['benchmarked_tools']}")
        
        if "multi_agent_analysis" in results:
            summary_parts.append(results["multi_agent_analysis"]["summary"])
        
        return " | ".join(summary_parts)


# Tool interface for integration
async def analyze_code_advanced(code: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Analyze code using advanced features"""
    tool = AdvancedCodeAnalysisTool()
    return await tool.analyze_code_advanced(code, analysis_type)


async def optimize_code(code: str, target: str = "balanced") -> Dict[str, Any]:
    """Optimize code for specific targets"""
    tool = AdvancedCodeAnalysisTool()
    return await tool.optimize_code(code, target)


async def debug_and_fix(file_path: Optional[str] = None) -> Dict[str, Any]:
    """Debug and fix issues in code"""
    tool = AdvancedCodeAnalysisTool()
    return await tool.debug_and_fix(file_path)


async def benchmark_performance(tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """Benchmark system performance"""
    tool = AdvancedCodeAnalysisTool()
    return await tool.benchmark_performance(tools)


# Example usage
if __name__ == "__main__":
    async def main():
        # Test code
        test_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Potential ZeroDivisionError

def process_data(data):
    result = []
    for item in data:
        if item > 10:
            result.append(item * 2)
    return result
"""
        
        # Run comprehensive analysis
        print("Running comprehensive analysis...")
        analysis_result = await analyze_code_advanced(test_code, "comprehensive")
        
        print(f"\nSummary: {analysis_result.get('summary', 'No summary available')}")

        if "error" in analysis_result:
            print(f"Error: {analysis_result['error']}")
        else:
            if "code_intent" in analysis_result:
                print(f"Quality Score: {analysis_result['code_intent']['quality_score']}")
                print(f"Purpose: {analysis_result['code_intent']['purpose']}")
        
            if "refactoring_suggestions" in analysis_result:
                print("\nRefactoring Suggestions:")
                for i, suggestion in enumerate(analysis_result["refactoring_suggestions"], 1):
                    print(f"{i}. {suggestion['description']}")
        
        # Optimize code
        print("\n\nOptimizing code...")
        optimization_result = await optimize_code(test_code, "performance")
        
        if "error" in optimization_result:
            print(f"Error: {optimization_result['error']}")
        else:
            if optimization_result["optimized_code"] != test_code:
                print("Code was optimized!")
                print(f"Improvements: {', '.join(optimization_result['improvements'][:3])}")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())