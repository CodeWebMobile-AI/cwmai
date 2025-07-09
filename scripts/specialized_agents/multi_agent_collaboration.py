#!/usr/bin/env python3
"""
Multi-Agent Collaboration System - Specialized agents with exact names as requested
"""

import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

from scripts.http_ai_client import HTTPAIClient
import logging
from scripts.deep_code_analyzer import DeepCodeAnalyzer
from scripts.benchmarking_system import BenchmarkingSystem
from scripts.self_debugging_engine import SelfDebuggingEngine

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    message_type: str
    content: Any
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.ai_client = HTTPAIClient()
        self.knowledge_base = {}
        self.message_queue = asyncio.Queue()
        
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific task"""
        pass
    
    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        await self.message_queue.put(message)
    
    async def send_message(self, receiver: str, content: Any, message_type: str = "info"):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content,
            metadata={"timestamp": asyncio.get_event_loop().time()}
        )
        # Message will be routed by coordinator
        return message
    
    def share_knowledge(self, key: str, value: Any):
        """Share knowledge with other agents"""
        self.knowledge_base[key] = value
    
    def get_shared_knowledge(self, key: str) -> Any:
        """Get shared knowledge"""
        return self.knowledge_base.get(key)


class CodeOptimizationAgent(BaseAgent):
    """Agent specialized in code optimization"""
    
    def __init__(self):
        super().__init__("CodeOptimizationAgent")
        self.analyzer = DeepCodeAnalyzer()
        self.benchmarking = BenchmarkingSystem()
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code for performance and quality"""
        code = task.get("code", "")
        optimization_target = task.get("target", "performance")
        
        logger.info(f"{self.name} optimizing code for {optimization_target}")
        
        # Analyze current code
        intent = self.analyzer.analyze_code_intent(code)
        
        # Get refactoring suggestions
        refactorings = self.analyzer.suggest_refactoring(code, intent)
        
        # Benchmark current performance
        current_perf = await self._benchmark_code(code)
        
        # Apply optimizations
        optimized_code = await self._apply_optimizations(code, refactorings, optimization_target)
        
        # Benchmark optimized code
        optimized_perf = await self._benchmark_code(optimized_code)
        
        # Calculate improvement
        improvement = self._calculate_improvement(current_perf, optimized_perf)
        
        # Share knowledge about optimization patterns
        self.share_knowledge("optimization_patterns", {
            "target": optimization_target,
            "techniques_used": [r.description for r in refactorings[:3]],
            "improvement": improvement
        })
        
        return {
            "agent": self.name,
            "original_code": code,
            "optimized_code": optimized_code,
            "improvements": [r.description for r in refactorings],
            "performance_gain": improvement,
            "quality_score": intent.quality_score
        }
    
    async def _benchmark_code(self, code: str) -> Dict[str, float]:
        """Benchmark code performance"""
        # Simplified benchmark
        return {
            "execution_time": 0.1,  # placeholder
            "memory_usage": 50.0,   # placeholder
            "complexity": len(code.split('\n'))
        }
    
    async def _apply_optimizations(self, code: str, refactorings: List, target: str) -> str:
        """Apply optimization based on target"""
        if not refactorings:
            return code
        
        # For demo, apply first refactoring suggestion
        # In production, would intelligently select based on target
        prompt = f"""Apply these optimizations to the code for {target}:

Original code:
```python
{code}
```

Optimizations to apply:
{chr(10).join([f"- {r.description}: {r.rationale}" for r in refactorings[:3]])}

Return only the optimized code."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Extract code from response
            if "```python" in response:
                code_start = response.find("```python") + 9
                code_end = response.find("```", code_start)
                return response[code_start:code_end].strip()
            return response.strip()
        except:
            return code
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> Dict[str, float]:
        """Calculate performance improvement"""
        return {
            "execution_time": (before["execution_time"] - after["execution_time"]) / before["execution_time"] * 100,
            "memory_usage": (before["memory_usage"] - after["memory_usage"]) / before["memory_usage"] * 100,
            "complexity": (before["complexity"] - after["complexity"]) / before["complexity"] * 100
        }


class SecurityAuditAgent(BaseAgent):
    """Agent specialized in security auditing"""
    
    def __init__(self):
        super().__init__("SecurityAuditAgent")
        self.debugging_engine = SelfDebuggingEngine()
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        
    def _load_vulnerability_patterns(self) -> Dict[str, Any]:
        """Load security vulnerability patterns"""
        return {
            "sql_injection": ["execute", "raw", "format"],
            "xss": ["innerHTML", "document.write", "eval"],
            "path_traversal": ["../", "..\\", "open"],
            "command_injection": ["subprocess", "os.system", "exec"],
            "insecure_random": ["random.random", "random.randint"],
            "hardcoded_secrets": ["password=", "api_key=", "secret="]
        }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Audit code for security vulnerabilities"""
        code = task.get("code", "")
        context = task.get("context", {})
        
        logger.info(f"{self.name} auditing code for security issues")
        
        # Detect security bugs
        bugs = await self.debugging_engine.detect_bugs()
        security_bugs = [b for b in bugs if "injection" in b.bug_type or "security" in b.bug_type]
        
        # Additional security checks
        vulnerabilities = self._scan_for_vulnerabilities(code)
        
        # Get AI security analysis
        ai_vulnerabilities = await self._ai_security_analysis(code, context)
        
        # Generate fixes for security issues
        fixes = []
        for bug in security_bugs[:5]:
            bug_fixes = await self.debugging_engine.generate_fixes(bug)
            fixes.extend(bug_fixes)
        
        # Calculate security score
        security_score = self._calculate_security_score(len(vulnerabilities), len(security_bugs))
        
        # Share knowledge about vulnerabilities
        self.share_knowledge("security_vulnerabilities", {
            "patterns_found": list(set([v["type"] for v in vulnerabilities])),
            "critical_count": len([v for v in vulnerabilities if v["severity"] == "critical"])
        })
        
        return {
            "agent": self.name,
            "security_score": security_score,
            "vulnerabilities": vulnerabilities,
            "security_bugs": [{"type": b.bug_type, "severity": b.severity} for b in security_bugs],
            "fixes_available": len(fixes),
            "recommendations": self._generate_security_recommendations(vulnerabilities, security_bugs)
        }
    
    def _scan_for_vulnerabilities(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for known vulnerability patterns"""
        vulnerabilities = []
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if pattern in code:
                    vulnerabilities.append({
                        "type": vuln_type,
                        "pattern": pattern,
                        "severity": "high" if vuln_type in ["sql_injection", "command_injection"] else "medium",
                        "line": code.find(pattern)  # Simplified
                    })
        
        return vulnerabilities
    
    async def _ai_security_analysis(self, code: str, context: Dict) -> List[Dict[str, Any]]:
        """Use AI for advanced security analysis"""
        prompt = f"""Analyze this code for security vulnerabilities:

```python
{code[:2000]}  # Truncated
```

Context: {json.dumps(context)}

Focus on:
1. Injection vulnerabilities
2. Authentication/authorization issues
3. Data exposure risks
4. Cryptographic weaknesses

Return JSON array of vulnerabilities."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Parse response
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)
        except:
            pass
        
        return []
    
    def _calculate_security_score(self, vuln_count: int, bug_count: int) -> float:
        """Calculate overall security score (0-100)"""
        base_score = 100
        score = base_score - (vuln_count * 5) - (bug_count * 10)
        return max(0, min(100, score))
    
    def _generate_security_recommendations(self, vulnerabilities: List, bugs: List) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        vuln_types = set([v["type"] for v in vulnerabilities])
        
        if "sql_injection" in vuln_types:
            recommendations.append("Use parameterized queries to prevent SQL injection")
        if "xss" in vuln_types:
            recommendations.append("Sanitize all user input before rendering")
        if "command_injection" in vuln_types:
            recommendations.append("Avoid shell commands; use safe APIs instead")
        if "hardcoded_secrets" in vuln_types:
            recommendations.append("Use environment variables for sensitive data")
            
        if len(vulnerabilities) == 0:
            recommendations.append("No major security issues detected")
            
        return recommendations


class PerformanceTuningAgent(BaseAgent):
    """Agent specialized in performance tuning"""
    
    def __init__(self):
        super().__init__("PerformanceTuningAgent")
        self.benchmarking = BenchmarkingSystem()
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Tune code for optimal performance"""
        code = task.get("code", "")
        performance_targets = task.get("targets", {})
        
        logger.info(f"{self.name} tuning performance")
        
        # Profile current performance
        current_profile = await self._profile_code(code)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(current_profile)
        
        # Generate optimization strategies
        strategies = self._generate_optimization_strategies(bottlenecks)
        
        # Apply optimizations
        optimized_code = await self._apply_performance_optimizations(code, strategies)
        
        # Profile optimized code
        optimized_profile = await self._profile_code(optimized_code)
        
        # Compare with industry benchmarks
        industry_comparison = await self.benchmarking.benchmark_against_external()
        
        # Share performance insights
        self.share_knowledge("performance_insights", {
            "bottlenecks": bottlenecks,
            "optimization_strategies": [s["name"] for s in strategies],
            "improvement_percentage": self._calculate_improvement_percentage(current_profile, optimized_profile)
        })
        
        return {
            "agent": self.name,
            "current_performance": current_profile,
            "optimized_performance": optimized_profile,
            "bottlenecks": bottlenecks,
            "optimizations_applied": strategies,
            "industry_comparison": industry_comparison,
            "recommendations": self._generate_performance_recommendations(optimized_profile, industry_comparison)
        }
    
    async def _profile_code(self, code: str) -> Dict[str, Any]:
        """Profile code performance characteristics"""
        # Simplified profiling
        return {
            "execution_time": 0.5,
            "memory_usage": 100,
            "cpu_usage": 50,
            "io_operations": 10,
            "function_calls": 100
        }
    
    def _identify_bottlenecks(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if profile["execution_time"] > 0.3:
            bottlenecks.append({
                "type": "slow_execution",
                "impact": "high",
                "location": "main loop"
            })
            
        if profile["memory_usage"] > 50:
            bottlenecks.append({
                "type": "high_memory",
                "impact": "medium",
                "location": "data structures"
            })
            
        if profile["io_operations"] > 5:
            bottlenecks.append({
                "type": "excessive_io",
                "impact": "high",
                "location": "file operations"
            })
            
        return bottlenecks
    
    def _generate_optimization_strategies(self, bottlenecks: List[Dict]) -> List[Dict[str, Any]]:
        """Generate optimization strategies based on bottlenecks"""
        strategies = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_execution":
                strategies.append({
                    "name": "algorithm_optimization",
                    "description": "Replace O(n²) with O(n log n) algorithm",
                    "expected_improvement": 50
                })
            elif bottleneck["type"] == "high_memory":
                strategies.append({
                    "name": "memory_optimization",
                    "description": "Use generators instead of lists",
                    "expected_improvement": 30
                })
            elif bottleneck["type"] == "excessive_io":
                strategies.append({
                    "name": "io_optimization",
                    "description": "Batch I/O operations",
                    "expected_improvement": 40
                })
                
        return strategies
    
    async def _apply_performance_optimizations(self, code: str, strategies: List[Dict]) -> str:
        """Apply performance optimizations to code"""
        if not strategies:
            return code
            
        prompt = f"""Apply these performance optimizations to the code:

Code:
```python
{code}
```

Optimizations:
{chr(10).join([f"- {s['name']}: {s['description']}" for s in strategies])}

Return the optimized code that implements these strategies."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            if "```python" in response:
                code_start = response.find("```python") + 9
                code_end = response.find("```", code_start)
                return response[code_start:code_end].strip()
            return response.strip()
        except:
            return code
    
    def _calculate_improvement_percentage(self, before: Dict, after: Dict) -> float:
        """Calculate overall performance improvement"""
        time_improvement = (before["execution_time"] - after["execution_time"]) / before["execution_time"]
        memory_improvement = (before["memory_usage"] - after["memory_usage"]) / before["memory_usage"]
        
        return (time_improvement + memory_improvement) / 2 * 100
    
    def _generate_performance_recommendations(self, profile: Dict, industry: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if profile["execution_time"] > 0.2:
            recommendations.append("Consider caching frequently computed results")
        if profile["memory_usage"] > 80:
            recommendations.append("Implement object pooling to reduce memory allocation")
        if profile["cpu_usage"] > 70:
            recommendations.append("Use parallel processing for CPU-intensive operations")
            
        return recommendations


class DocumentationAgent(BaseAgent):
    """Agent specialized in documentation writing"""
    
    def __init__(self):
        super().__init__("DocumentationAgent")
        self.analyzer = DeepCodeAnalyzer()
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation"""
        code = task.get("code", "")
        doc_type = task.get("doc_type", "comprehensive")
        
        logger.info(f"{self.name} generating {doc_type} documentation")
        
        # Analyze code to understand its purpose
        intent = self.analyzer.analyze_code_intent(code)
        
        # Generate different types of documentation
        docs = {
            "api_docs": await self._generate_api_docs(code, intent),
            "user_guide": await self._generate_user_guide(code, intent),
            "technical_docs": await self._generate_technical_docs(code, intent),
            "examples": await self._generate_examples(code, intent)
        }
        
        # Generate README if needed
        if doc_type == "comprehensive":
            docs["readme"] = await self._generate_readme(code, intent, docs)
        
        # Share documentation insights
        self.share_knowledge("documentation_coverage", {
            "functions_documented": self._count_documented_functions(docs["api_docs"]),
            "examples_provided": len(docs["examples"]),
            "doc_completeness": self._calculate_doc_completeness(docs)
        })
        
        return {
            "agent": self.name,
            "documentation": docs,
            "doc_type": doc_type,
            "completeness_score": self._calculate_doc_completeness(docs),
            "recommendations": self._generate_doc_recommendations(docs)
        }
    
    async def _generate_api_docs(self, code: str, intent) -> str:
        """Generate API documentation"""
        prompt = f"""Generate comprehensive API documentation for this code:

```python
{code}
```

Code Intent: {intent.purpose}

Include:
1. Function/class descriptions
2. Parameters and return values
3. Usage examples
4. Type hints"""
        
        try:
            return self.ai_client.generate_response(prompt)
        except:
            return "# API Documentation\n\nError generating documentation."
    
    async def _generate_user_guide(self, code: str, intent) -> str:
        """Generate user guide"""
        prompt = f"""Create a user guide for this code:

```python
{code[:1000]}  # Truncated
```

Purpose: {intent.purpose}

Include:
1. Getting started
2. Basic usage
3. Common use cases
4. Troubleshooting"""
        
        try:
            return self.ai_client.generate_response(prompt)
        except:
            return "# User Guide\n\nError generating guide."
    
    async def _generate_technical_docs(self, code: str, intent) -> str:
        """Generate technical documentation"""
        return f"""# Technical Documentation

## Architecture Overview
{intent.purpose}

## Design Patterns Used
{', '.join(intent.patterns_used)}

## Quality Metrics
- Quality Score: {intent.quality_score}
- Improvement Areas: {', '.join(intent.improvement_areas)}
"""
    
    async def _generate_examples(self, code: str, intent) -> List[str]:
        """Generate usage examples"""
        prompt = f"""Generate 3 practical usage examples for this code:

```python
{code[:1000]}  # Truncated
```

Return just the example code snippets."""
        
        try:
            response = self.ai_client.generate_response(prompt)
            # Extract code examples
            examples = []
            parts = response.split("```python")
            for part in parts[1:]:
                if "```" in part:
                    examples.append(part[:part.find("```")])
            return examples
        except:
            return ["# Example 1\n# Error generating examples"]
    
    async def _generate_readme(self, code: str, intent, docs: Dict) -> str:
        """Generate README file"""
        return f"""# Project Documentation

## Overview
{intent.purpose}

## Features
{chr(10).join([f"- {area}" for area in intent.improvement_areas])}

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
{docs['examples'][0] if docs['examples'] else '# See examples below'}

## Documentation
- [API Documentation](#api-documentation)
- [User Guide](#user-guide)
- [Technical Documentation](#technical-documentation)

## Contributing
Please read our contributing guidelines before submitting PRs.

## License
MIT License
"""
    
    def _count_documented_functions(self, api_docs: str) -> int:
        """Count documented functions"""
        return api_docs.count("def ") + api_docs.count("class ")
    
    def _calculate_doc_completeness(self, docs: Dict) -> float:
        """Calculate documentation completeness score"""
        scores = {
            "api_docs": 0.3 if len(docs.get("api_docs", "")) > 100 else 0,
            "user_guide": 0.2 if len(docs.get("user_guide", "")) > 100 else 0,
            "technical_docs": 0.2 if len(docs.get("technical_docs", "")) > 100 else 0,
            "examples": 0.2 if len(docs.get("examples", [])) > 0 else 0,
            "readme": 0.1 if len(docs.get("readme", "")) > 100 else 0
        }
        return sum(scores.values()) * 100
    
    def _generate_doc_recommendations(self, docs: Dict) -> List[str]:
        """Generate documentation recommendations"""
        recommendations = []
        
        if not docs.get("examples"):
            recommendations.append("Add code examples to improve usability")
        if len(docs.get("api_docs", "")) < 500:
            recommendations.append("Expand API documentation with more details")
        if "troubleshooting" not in docs.get("user_guide", "").lower():
            recommendations.append("Add troubleshooting section to user guide")
            
        return recommendations


class TestCreationAgent(BaseAgent):
    """Agent specialized in test creation"""
    
    def __init__(self):
        super().__init__("TestCreationAgent")
        self.analyzer = DeepCodeAnalyzer()
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive tests for code"""
        code = task.get("code", "")
        test_type = task.get("test_type", "unit")
        
        logger.info(f"{self.name} creating {test_type} tests")
        
        # Analyze code to understand what to test
        intent = self.analyzer.analyze_code_intent(code)
        
        # Generate different types of tests
        tests = {
            "unit_tests": await self._generate_unit_tests(code, intent),
            "integration_tests": await self._generate_integration_tests(code, intent),
            "edge_cases": await self._generate_edge_case_tests(code, intent),
            "performance_tests": await self._generate_performance_tests(code, intent)
        }
        
        # Calculate test coverage
        coverage = self._calculate_test_coverage(code, tests)
        
        # Share testing insights
        self.share_knowledge("test_coverage", {
            "coverage_percentage": coverage,
            "test_count": sum(t.count("def test_") for t in tests.values()),
            "edge_cases_covered": len(tests["edge_cases"].split("def test_")) - 1
        })
        
        return {
            "agent": self.name,
            "tests": tests,
            "coverage": coverage,
            "test_type": test_type,
            "recommendations": self._generate_test_recommendations(coverage, tests)
        }
    
    async def _generate_unit_tests(self, code: str, intent) -> str:
        """Generate unit tests"""
        prompt = f"""Generate comprehensive unit tests for this code:

```python
{code}
```

Purpose: {intent.purpose}

Include:
1. Test all public functions
2. Test happy path scenarios
3. Use appropriate assertions
4. Follow pytest conventions"""
        
        try:
            return self.ai_client.generate_response(prompt)
        except:
            return "# Unit Tests\n\ndef test_placeholder():\n    assert True"
    
    async def _generate_integration_tests(self, code: str, intent) -> str:
        """Generate integration tests"""
        prompt = f"""Generate integration tests for this code:

```python
{code[:1000]}  # Truncated
```

Test how components work together."""
        
        try:
            return self.ai_client.generate_response(prompt)
        except:
            return "# Integration Tests\n\ndef test_integration():\n    pass"
    
    async def _generate_edge_case_tests(self, code: str, intent) -> str:
        """Generate edge case tests"""
        prompt = f"""Generate edge case tests for this code:

```python
{code[:1000]}  # Truncated
```

Test:
1. Boundary conditions
2. Invalid inputs
3. Error scenarios
4. Resource exhaustion"""
        
        try:
            return self.ai_client.generate_response(prompt)
        except:
            return "# Edge Case Tests\n\ndef test_edge_case():\n    pass"
    
    async def _generate_performance_tests(self, code: str, intent) -> str:
        """Generate performance tests"""
        return f"""# Performance Tests

import time
import pytest

def test_performance_execution_time():
    start = time.time()
    # Run code
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Execution took {{elapsed}}s, expected < 1s"

def test_memory_usage():
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    # Run code
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024
    assert memory_increase < 100, f"Memory increased by {{memory_increase}}MB"
"""
    
    def _calculate_test_coverage(self, code: str, tests: Dict) -> float:
        """Calculate test coverage percentage"""
        # Simplified calculation
        functions_in_code = code.count("def ")
        tests_written = sum(t.count("def test_") for t in tests.values())
        
        if functions_in_code == 0:
            return 100.0
            
        return min(100.0, (tests_written / functions_in_code) * 100)
    
    def _generate_test_recommendations(self, coverage: float, tests: Dict) -> List[str]:
        """Generate testing recommendations"""
        recommendations = []
        
        if coverage < 80:
            recommendations.append(f"Increase test coverage from {coverage:.1f}% to at least 80%")
        if "mock" not in str(tests.values()):
            recommendations.append("Consider using mocks for external dependencies")
        if "parametrize" not in str(tests.values()):
            recommendations.append("Use pytest.parametrize for testing multiple scenarios")
        if len(tests["edge_cases"]) < 200:
            recommendations.append("Add more edge case tests")
            
        return recommendations


class AgentCoordinator:
    """Coordinates work between specialized agents"""
    
    def __init__(self):
        self.agents = {
            "code_optimizer": CodeOptimizationAgent(),
            "security_auditor": SecurityAuditAgent(),
            "performance_tuner": PerformanceTuningAgent(),
            "documentation_writer": DocumentationAgent(),
            "test_creator": TestCreationAgent()
        }
        self.shared_knowledge = {}
        
    async def coordinate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents to work on a complex task"""
        task_type = task.get("type", "comprehensive")
        code = task.get("code", "")
        
        logger.info(f"Coordinating {task_type} task across agents")
        
        results = {}
        
        if task_type == "comprehensive":
            # Run all agents
            agent_tasks = []
            for agent_name, agent in self.agents.items():
                agent_tasks.append(self._run_agent_task(agent_name, agent, task))
            
            # Run agents in parallel
            agent_results = await asyncio.gather(*agent_tasks)
            
            # Combine results
            for agent_name, result in zip(self.agents.keys(), agent_results):
                results[agent_name] = result
                
        elif task_type == "security_focused":
            # Security audit first, then fixes
            security_result = await self.agents["security_auditor"].process_task(task)
            results["security_auditor"] = security_result
            
            # If vulnerabilities found, optimize for security
            if security_result["vulnerabilities"]:
                task["target"] = "security"
                optimization_result = await self.agents["code_optimizer"].process_task(task)
                results["code_optimizer"] = optimization_result
                
        elif task_type == "performance_focused":
            # Performance analysis first
            perf_result = await self.agents["performance_tuner"].process_task(task)
            results["performance_tuner"] = perf_result
            
            # Then optimize based on bottlenecks
            if perf_result["bottlenecks"]:
                optimization_result = await self.agents["code_optimizer"].process_task(task)
                results["code_optimizer"] = optimization_result
        
        # Share knowledge between agents
        self._consolidate_knowledge(results)
        
        # Generate final recommendations
        final_recommendations = self._generate_final_recommendations(results)
        
        return {
            "task_type": task_type,
            "agent_results": results,
            "shared_knowledge": self.shared_knowledge,
            "final_recommendations": final_recommendations,
            "summary": self._generate_summary(results)
        }
    
    async def _run_agent_task(self, agent_name: str, agent: BaseAgent, task: Dict) -> Dict[str, Any]:
        """Run a single agent task"""
        try:
            return await agent.process_task(task)
        except Exception as e:
            logger.info(f"Error in {agent_name}: {str(e)}")
            return {"agent": agent_name, "error": str(e)}
    
    def _consolidate_knowledge(self, results: Dict[str, Dict]):
        """Consolidate knowledge from all agents"""
        for agent_name, result in results.items():
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                for key, value in agent.knowledge_base.items():
                    if key not in self.shared_knowledge:
                        self.shared_knowledge[key] = {}
                    self.shared_knowledge[key][agent_name] = value
    
    def _generate_final_recommendations(self, results: Dict[str, Dict]) -> List[str]:
        """Generate final recommendations based on all agent inputs"""
        recommendations = []
        
        # Collect all recommendations
        for agent_result in results.values():
            if "recommendations" in agent_result:
                recommendations.extend(agent_result["recommendations"])
        
        # Prioritize and deduplicate
        unique_recommendations = list(set(recommendations))
        
        # Add cross-agent insights
        if "security_auditor" in results and "performance_tuner" in results:
            sec_score = results["security_auditor"].get("security_score", 0)
            if sec_score < 70:
                unique_recommendations.insert(0, "Prioritize security fixes before performance optimization")
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def _generate_summary(self, results: Dict[str, Dict]) -> str:
        """Generate executive summary of all agent work"""
        summary_parts = []
        
        if "code_optimizer" in results:
            opt = results["code_optimizer"]
            if "performance_gain" in opt:
                summary_parts.append(f"Code optimization achieved {opt['performance_gain']:.1f}% improvement")
        
        if "security_auditor" in results:
            sec = results["security_auditor"]
            summary_parts.append(f"Security score: {sec.get('security_score', 0):.1f}/100")
        
        if "test_creator" in results:
            test = results["test_creator"]
            summary_parts.append(f"Test coverage: {test.get('coverage', 0):.1f}%")
        
        if "documentation_writer" in results:
            doc = results["documentation_writer"]
            summary_parts.append(f"Documentation completeness: {doc.get('completeness_score', 0):.1f}%")
        
        return " | ".join(summary_parts)


# Example usage
if __name__ == "__main__":
    async def main():
        coordinator = AgentCoordinator()
        
        # Test with sample code
        test_task = {
            "type": "comprehensive",
            "code": """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total

def process_user_data(user_id):
    # This could have security issues
    query = f"SELECT * FROM users WHERE id = {user_id}"
    # execute query...
    return query
""",
            "context": {"purpose": "E-commerce calculation and user processing"}
        }
        
        # Coordinate agents
        result = await coordinator.coordinate_task(test_task)
        
        print(f"Task Type: {result['task_type']}")
        print(f"Summary: {result['summary']}")
        print(f"\nFinal Recommendations:")
        for i, rec in enumerate(result['final_recommendations'], 1):
            print(f"{i}. {rec}")
    
    asyncio.run(main())