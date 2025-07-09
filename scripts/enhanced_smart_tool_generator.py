"""
Enhanced Smart Tool Generator
=============================

Advanced tool generation with:
1. Context-aware generation based on user intent and system state
2. Tool composition - combining multiple tools into higher-level tools
3. Performance optimization through caching and parallel execution
4. Domain-specific tool generation (DevOps, Data Science, etc.)
5. Learning from tool usage patterns
"""

import asyncio
import json
import logging
import ast
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from pathlib import Path
import hashlib

from scripts.http_ai_client import HTTPAIClient
from scripts.smart_tool_generator import SmartToolGenerator
from scripts.state_manager import StateManager
from scripts.module_discovery import get_module_discovery


class ToolComposer:
    """Composes multiple tools into higher-level tools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def compose_tools(
        self,
        base_tools: List[str],
        composition_intent: str,
        output_tool_name: str
    ) -> str:
        """Compose multiple tools into a single higher-level tool."""
        
        # Generate the composition code
        code = f'''"""
{output_tool_name.replace('_', ' ').title()} - Composed Tool
Generated from: {', '.join(base_tools)}
Intent: {composition_intent}
"""

import asyncio
from typing import Dict, List, Any, Optional
from scripts.tool_calling_system import ToolCallingSystem

async def {output_tool_name}(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    {composition_intent}
    
    This tool composes functionality from: {', '.join(base_tools)}
    """
    tool_system = ToolCallingSystem()
    results = {{}}
    errors = []
    
    try:
'''
        
        # Add tool calls based on composition pattern
        if 'sequential' in composition_intent.lower():
            code += self._generate_sequential_composition(base_tools)
        elif 'parallel' in composition_intent.lower():
            code += self._generate_parallel_composition(base_tools)
        elif 'conditional' in composition_intent.lower():
            code += self._generate_conditional_composition(base_tools)
        else:
            code += self._generate_pipeline_composition(base_tools)
            
        # Complete the function
        code += '''
        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "composed_from": base_tools
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": results
        }

# Tool metadata
__description__ = f"{composition_intent}"
__parameters__ = {
    "kwargs": "Parameters to pass to composed tools"
}
__examples__ = [
    {
        "description": "Execute composed tool",
        "params": {}
    }
]
'''
        
        return code
        
    def _generate_sequential_composition(self, tools: List[str]) -> str:
        """Generate sequential tool execution."""
        code = "        # Sequential execution\n"
        
        for i, tool in enumerate(tools):
            code += f'''        # Step {i+1}: Execute {tool}
        result_{i} = await tool_system.call_tool("{tool}", **kwargs)
        results["{tool}"] = result_{i}
        
        if result_{i}.get("error"):
            errors.append({{
                "tool": "{tool}",
                "error": result_{i}.get("error")
            }})
            # Continue despite errors - may want to change this
            
        # Pass output to next tool
        if result_{i}.get("data"):
            kwargs.update(result_{i}["data"])
            
'''
        return code
        
    def _generate_parallel_composition(self, tools: List[str]) -> str:
        """Generate parallel tool execution."""
        code = '''        # Parallel execution
        tasks = []
        
'''
        for tool in tools:
            code += f'        tasks.append(tool_system.call_tool("{tool}", **kwargs))\n'
            
        code += '''        
        # Execute all tools in parallel
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for tool, result in zip(base_tools, parallel_results):
            if isinstance(result, Exception):
                errors.append({
                    "tool": tool,
                    "error": str(result)
                })
            else:
                results[tool] = result
                if result.get("error"):
                    errors.append({
                        "tool": tool,
                        "error": result.get("error")
                    })
                    
'''
        return code
        
    def _generate_conditional_composition(self, tools: List[str]) -> str:
        """Generate conditional tool execution."""
        code = '''        # Conditional execution based on context
        condition = kwargs.get("condition", "default")
        
'''
        for i, tool in enumerate(tools):
            if_clause = "if" if i == 0 else "elif"
            code += f'''        {if_clause} condition == "{tool}" or (condition == "auto" and "{tool}" in kwargs.get("context", "")):
            result = await tool_system.call_tool("{tool}", **kwargs)
            results["{tool}"] = result
            if result.get("error"):
                errors.append({{
                    "tool": "{tool}",
                    "error": result.get("error")
                }})
'''
        
        code += '''        else:
            # Default: try first tool
            result = await tool_system.call_tool(base_tools[0], **kwargs)
            results[base_tools[0]] = result
            
'''
        return code
        
    def _generate_pipeline_composition(self, tools: List[str]) -> str:
        """Generate pipeline composition where output flows through tools."""
        code = '''        # Pipeline execution - output flows through tools
        pipeline_data = kwargs.copy()
        
'''
        for tool in tools:
            code += f'''        # Process through {tool}
        result = await tool_system.call_tool("{tool}", **pipeline_data)
        results["{tool}"] = result
        
        if result.get("error"):
            errors.append({{
                "tool": "{tool}",
                "error": result.get("error")
            }})
            break  # Stop pipeline on error
            
        # Update pipeline data with output
        if result.get("data"):
            pipeline_data.update(result["data"])
        elif result.get("output"):
            pipeline_data["input"] = result["output"]
            
'''
        return code


class DomainSpecificGenerator:
    """Generates domain-specific tools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.domains = {
            'devops': self._generate_devops_tool,
            'data_science': self._generate_data_science_tool,
            'security': self._generate_security_tool,
            'monitoring': self._generate_monitoring_tool,
            'automation': self._generate_automation_tool
        }
        
    async def generate_domain_tool(
        self,
        domain: str,
        tool_name: str,
        specific_purpose: str
    ) -> str:
        """Generate a domain-specific tool."""
        
        if domain not in self.domains:
            domain = 'automation'  # Default domain
            
        generator = self.domains[domain]
        return await generator(tool_name, specific_purpose)
        
    async def _generate_devops_tool(self, tool_name: str, purpose: str) -> str:
        """Generate DevOps-specific tool."""
        return f'''"""
{tool_name.replace('_', ' ').title()} - DevOps Tool
Purpose: {purpose}
"""

import asyncio
import subprocess
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

async def {tool_name}(
    environment: str = "development",
    service: Optional[str] = None,
    action: str = "status",
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    {purpose}
    
    Args:
        environment: Target environment (development, staging, production)
        service: Service name to operate on
        action: Action to perform (status, deploy, rollback, scale)
    """
    results = {{
        "environment": environment,
        "service": service,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }}
    
    try:
        if action == "status":
            # Check service status
            cmd = ["kubectl", "get", "pods", "-l", f"app={{service}}"] if service else ["kubectl", "get", "pods"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            results["status"] = result.stdout
            results["success"] = result.returncode == 0
            
        elif action == "deploy":
            # Deploy service
            results["message"] = f"Would deploy {{service}} to {{environment}}"
            results["success"] = True
            
        elif action == "rollback":
            # Rollback deployment
            results["message"] = f"Would rollback {{service}} in {{environment}}"
            results["success"] = True
            
        elif action == "scale":
            # Scale service
            replicas = kwargs.get("replicas", 3)
            results["message"] = f"Would scale {{service}} to {{replicas}} replicas"
            results["success"] = True
            
        else:
            results["error"] = f"Unknown action: {{action}}"
            results["success"] = False
            
        return results
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "environment": environment,
            "service": service
        }}

# Tool metadata
__description__ = "{purpose}"
__parameters__ = {{
    "environment": "Target environment",
    "service": "Service name",
    "action": "Action to perform"
}}
__examples__ = [
    {{
        "description": "Check service status",
        "params": {{"service": "api-server", "action": "status"}}
    }}
]
'''
        
    async def _generate_data_science_tool(self, tool_name: str, purpose: str) -> str:
        """Generate Data Science tool."""
        return f'''"""
{tool_name.replace('_', ' ').title()} - Data Science Tool
Purpose: {purpose}
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
from pathlib import Path

async def {tool_name}(
    data_source: Optional[str] = None,
    operation: str = "analyze",
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    {purpose}
    
    Args:
        data_source: Path to data file or data dict
        operation: Operation to perform (analyze, transform, visualize)
        parameters: Additional parameters for the operation
    """
    results = {{
        "operation": operation,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }}
    
    try:
        # Load data
        if isinstance(data_source, str) and Path(data_source).exists():
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                df = pd.read_json(data_source)
            else:
                df = pd.DataFrame(kwargs.get("data", []))
        else:
            df = pd.DataFrame(kwargs.get("data", []))
            
        results["data_shape"] = list(df.shape)
        results["columns"] = list(df.columns)
        
        if operation == "analyze":
            # Basic statistical analysis
            results["summary"] = df.describe().to_dict()
            results["missing_values"] = df.isnull().sum().to_dict()
            results["data_types"] = df.dtypes.astype(str).to_dict()
            
        elif operation == "transform":
            # Data transformation
            transforms = parameters or {{}}
            
            if "normalize" in transforms:
                df = (df - df.mean()) / df.std()
                
            if "fillna" in transforms:
                df = df.fillna(transforms["fillna"])
                
            results["transformed_shape"] = list(df.shape)
            results["sample"] = df.head(5).to_dict()
            
        elif operation == "visualize":
            # Generate visualization data
            results["visualization_data"] = {{
                "x": list(df.index),
                "y": df.iloc[:, 0].tolist() if len(df.columns) > 0 else [],
                "type": "line"
            }}
            
        results["success"] = True
        return results
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "operation": operation
        }}

# Tool metadata
__description__ = "{purpose}"
__parameters__ = {{
    "data_source": "Path to data or data dict",
    "operation": "Operation to perform",
    "parameters": "Operation-specific parameters"
}}
'''
        
    async def _generate_security_tool(self, tool_name: str, purpose: str) -> str:
        """Generate Security tool."""
        return f'''"""
{tool_name.replace('_', ' ').title()} - Security Tool
Purpose: {purpose}
"""

import hashlib
import secrets
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

async def {tool_name}(
    action: str = "scan",
    target: Optional[str] = None,
    security_level: str = "medium",
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    {purpose}
    
    Args:
        action: Security action (scan, audit, generate, validate)
        target: Target to operate on
        security_level: Security level (low, medium, high)
    """
    results = {{
        "action": action,
        "target": target,
        "security_level": security_level,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }}
    
    try:
        if action == "scan":
            # Security scan
            vulnerabilities = []
            
            # Check for common patterns
            if target and Path(target).exists():
                content = Path(target).read_text()
                
                # Check for hardcoded secrets
                if re.search(r'(password|secret|key)\\s*=\\s*["\'][^"\']+["\']', content, re.I):
                    vulnerabilities.append({{
                        "type": "hardcoded_secret",
                        "severity": "high",
                        "description": "Possible hardcoded credentials found"
                    }})
                    
            results["vulnerabilities"] = vulnerabilities
            results["risk_score"] = len(vulnerabilities) * 10
            
        elif action == "generate":
            # Generate secure values
            results["generated"] = {{
                "token": secrets.token_urlsafe(32),
                "password": secrets.token_urlsafe(16),
                "api_key": f"sk_{{secrets.token_hex(16)}}"
            }}
            
        elif action == "validate":
            # Validate security measures
            validations = []
            
            # Check password strength
            password = kwargs.get("password", "")
            if len(password) < 12:
                validations.append("Password too short")
            if not re.search(r'[A-Z]', password):
                validations.append("Missing uppercase letters")
                
            results["valid"] = len(validations) == 0
            results["issues"] = validations
            
        results["success"] = True
        return results
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "action": action
        }}

# Tool metadata
__description__ = "{purpose}"
__parameters__ = {{
    "action": "Security action to perform",
    "target": "Target for the action",
    "security_level": "Security level"
}}
'''
        
    async def _generate_monitoring_tool(self, tool_name: str, purpose: str) -> str:
        """Generate Monitoring tool."""
        return f'''"""
{tool_name.replace('_', ' ').title()} - Monitoring Tool
Purpose: {purpose}
"""

import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import deque

async def {tool_name}(
    metric_type: str = "system",
    duration: int = 60,
    interval: int = 5,
    threshold: Optional[Dict[str, float]] = None,
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    {purpose}
    
    Args:
        metric_type: Type of metrics to monitor (system, process, custom)
        duration: Monitoring duration in seconds
        interval: Sampling interval in seconds
        threshold: Alert thresholds
    """
    results = {{
        "metric_type": metric_type,
        "duration": duration,
        "interval": interval,
        "start_time": datetime.now(timezone.utc).isoformat()
    }}
    
    metrics = []
    alerts = []
    threshold = threshold or {{"cpu": 80, "memory": 85, "disk": 90}}
    
    try:
        samples = min(duration // interval, 20)  # Limit samples
        
        for i in range(samples):
            sample = {{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_number": i + 1
            }}
            
            if metric_type == "system":
                # System metrics
                sample["cpu_percent"] = psutil.cpu_percent(interval=1)
                sample["memory_percent"] = psutil.virtual_memory().percent
                sample["disk_percent"] = psutil.disk_usage('/').percent
                
                # Check thresholds
                if sample["cpu_percent"] > threshold.get("cpu", 100):
                    alerts.append({{
                        "metric": "cpu",
                        "value": sample["cpu_percent"],
                        "threshold": threshold["cpu"],
                        "timestamp": sample["timestamp"]
                    }})
                    
            elif metric_type == "process":
                # Process metrics
                process_name = kwargs.get("process_name", "python")
                processes = []
                
                for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
                    if process_name.lower() in proc.info['name'].lower():
                        processes.append({{
                            "name": proc.info['name'],
                            "cpu": proc.info['cpu_percent'],
                            "memory": proc.info['memory_percent']
                        }})
                        
                sample["processes"] = processes
                
            elif metric_type == "custom":
                # Custom metrics from kwargs
                sample["custom_metrics"] = kwargs.get("custom_metrics", {{}})
                
            metrics.append(sample)
            
            if i < samples - 1:
                await asyncio.sleep(interval)
                
        # Calculate summary statistics
        if metrics and metric_type == "system":
            cpu_values = [m["cpu_percent"] for m in metrics]
            memory_values = [m["memory_percent"] for m in metrics]
            
            results["summary"] = {{
                "cpu": {{
                    "avg": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                }},
                "memory": {{
                    "avg": sum(memory_values) / len(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values)
                }}
            }}
            
        results["metrics"] = metrics
        results["alerts"] = alerts
        results["success"] = True
        
        return results
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "metric_type": metric_type
        }}

# Tool metadata
__description__ = "{purpose}"
__parameters__ = {{
    "metric_type": "Type of metrics to monitor",
    "duration": "Monitoring duration",
    "interval": "Sampling interval",
    "threshold": "Alert thresholds"
}}
'''
        
    async def _generate_automation_tool(self, tool_name: str, purpose: str) -> str:
        """Generate general automation tool."""
        return f'''"""
{tool_name.replace('_', ' ').title()} - Automation Tool
Purpose: {purpose}
"""

import asyncio
import subprocess
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone

async def {tool_name}(
    task: str = "execute",
    target: Optional[Union[str, List[str]]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    {purpose}
    
    Args:
        task: Automation task to perform
        target: Target(s) for automation
        config: Configuration for the task
    """
    results = {{
        "task": task,
        "target": target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config or {{}}
    }}
    
    try:
        if task == "execute":
            # Execute automation
            commands = config.get("commands", [])
            outputs = []
            
            for cmd in commands:
                if isinstance(cmd, str):
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    outputs.append({{
                        "command": cmd,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    }})
                    
            results["outputs"] = outputs
            results["success"] = all(o["returncode"] == 0 for o in outputs)
            
        elif task == "schedule":
            # Schedule automation
            schedule_config = config.get("schedule", {{}})
            results["scheduled"] = {{
                "frequency": schedule_config.get("frequency", "daily"),
                "time": schedule_config.get("time", "00:00"),
                "next_run": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            }}
            results["success"] = True
            
        elif task == "batch":
            # Batch processing
            targets = target if isinstance(target, list) else [target]
            batch_results = []
            
            for t in targets:
                # Process each target
                batch_results.append({{
                    "target": t,
                    "processed": True,
                    "result": f"Processed {{t}}"
                }})
                
            results["batch_results"] = batch_results
            results["success"] = True
            
        else:
            results["error"] = f"Unknown task: {{task}}"
            results["success"] = False
            
        return results
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "task": task
        }}

# Tool metadata
__description__ = "{purpose}"
__parameters__ = {{
    "task": "Automation task",
    "target": "Target(s) for automation",
    "config": "Task configuration"
}}
'''


class EnhancedSmartToolGenerator(SmartToolGenerator):
    """Enhanced tool generator with advanced capabilities."""
    
    def __init__(self):
        super().__init__()
        self.composer = ToolComposer()
        self.domain_generator = DomainSpecificGenerator()
        self.state_manager = StateManager()
        self.usage_tracker = self._load_usage_patterns()
        self.performance_cache = {}
        
    def _load_usage_patterns(self) -> Dict[str, Any]:
        """Load tool usage patterns for learning."""
        state = self.state_manager.get_state()
        return state.get('tool_usage_patterns', {})
        
    async def generate_context_aware_tool(
        self,
        tool_name: str,
        user_context: str,
        system_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate tool based on user context and system state."""
        
        # Analyze context
        context_analysis = await self._analyze_context(user_context)
        
        # Check if we can compose from existing tools
        if context_analysis.get('can_compose'):
            return await self._generate_composed_tool(
                tool_name,
                context_analysis['base_tools'],
                user_context
            )
            
        # Check for domain-specific generation
        if context_analysis.get('domain'):
            return await self._generate_domain_tool(
                tool_name,
                context_analysis['domain'],
                user_context
            )
            
        # Generate optimized tool
        return await self._generate_optimized_tool(
            tool_name,
            user_context,
            context_analysis
        )
        
    async def _analyze_context(self, context: str) -> Dict[str, Any]:
        """Analyze user context to determine generation strategy."""
        prompt = f"""
        Analyze this tool request context:
        "{context}"
        
        Determine:
        1. Can this be composed from existing tools? If yes, which ones?
        2. What domain does this belong to? (devops, data_science, security, monitoring, automation, general)
        3. What are the key operations needed?
        4. What performance optimizations would help?
        
        Return JSON:
        {{
            "can_compose": boolean,
            "base_tools": ["tool1", "tool2"],
            "domain": "domain_name",
            "operations": ["op1", "op2"],
            "optimizations": ["caching", "parallel", "async"]
        }}
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        try:
            return json.loads(response.get('content', '{}'))
        except:
            # Fallback analysis
            return self._analyze_context_fallback(context)
            
    def _analyze_context_fallback(self, context: str) -> Dict[str, Any]:
        """Fallback context analysis using patterns."""
        context_lower = context.lower()
        
        analysis = {
            'can_compose': False,
            'base_tools': [],
            'domain': 'general',
            'operations': [],
            'optimizations': []
        }
        
        # Domain detection
        domains = {
            'devops': ['deploy', 'kubernetes', 'docker', 'ci/cd', 'infrastructure'],
            'data_science': ['analyze', 'data', 'model', 'predict', 'statistics'],
            'security': ['scan', 'vulnerability', 'encrypt', 'authenticate', 'secure'],
            'monitoring': ['monitor', 'track', 'alert', 'metrics', 'performance'],
            'automation': ['automate', 'schedule', 'batch', 'workflow', 'process']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in context_lower for keyword in keywords):
                analysis['domain'] = domain
                break
                
        # Check for composition opportunities
        if 'and then' in context_lower or 'followed by' in context_lower:
            analysis['can_compose'] = True
            
        # Optimization detection
        if 'fast' in context_lower or 'performance' in context_lower:
            analysis['optimizations'].append('caching')
        if 'parallel' in context_lower or 'concurrent' in context_lower:
            analysis['optimizations'].append('parallel')
            
        return analysis
        
    async def _generate_composed_tool(
        self,
        tool_name: str,
        base_tools: List[str],
        context: str
    ) -> Dict[str, Any]:
        """Generate a composed tool from base tools."""
        
        # Filter to existing tools
        existing_tools = []
        for tool in base_tools:
            tool_path = Path(f"scripts/custom_tools/{tool}.py")
            if tool_path.exists():
                existing_tools.append(tool)
                
        if len(existing_tools) < 2:
            # Not enough tools to compose, fall back to regular generation
            return await self.generate_tool(tool_name, context)
            
        # Generate composition
        code = await self.composer.compose_tools(
            existing_tools,
            context,
            tool_name
        )
        
        # Save the composed tool
        tool_path = self.custom_tools_dir / f"{tool_name}.py"
        tool_path.write_text(code)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "tool_path": str(tool_path),
            "composed_from": existing_tools,
            "message": f"Composed tool {tool_name} from {len(existing_tools)} base tools"
        }
        
    async def _generate_domain_tool(
        self,
        tool_name: str,
        domain: str,
        context: str
    ) -> Dict[str, Any]:
        """Generate domain-specific tool."""
        
        code = await self.domain_generator.generate_domain_tool(
            domain,
            tool_name,
            context
        )
        
        # Save the tool
        tool_path = self.custom_tools_dir / f"{tool_name}.py"
        tool_path.write_text(code)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "tool_path": str(tool_path),
            "domain": domain,
            "message": f"Generated {domain} tool: {tool_name}"
        }
        
    async def _generate_optimized_tool(
        self,
        tool_name: str,
        context: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate tool with performance optimizations."""
        
        # Check usage patterns for similar tools
        similar_patterns = self._find_similar_usage_patterns(context)
        
        # Generate with optimizations
        optimizations = analysis.get('optimizations', [])
        
        # Add optimization hints to context
        optimization_context = context
        if 'caching' in optimizations:
            optimization_context += "\n\nIMPORTANT: Add caching using @lru_cache or similar for expensive operations."
        if 'parallel' in optimizations:
            optimization_context += "\n\nIMPORTANT: Use asyncio.gather() for parallel operations where possible."
        if 'async' in optimizations:
            optimization_context += "\n\nIMPORTANT: Make all I/O operations async."
            
        # Add performance hints from usage patterns
        if similar_patterns:
            optimization_context += f"\n\nBased on similar tools, consider: {similar_patterns}"
            
        # Generate the tool
        result = await self.generate_tool(tool_name, optimization_context)
        
        if result['success']:
            # Track this generation for future learning
            self._track_tool_generation(tool_name, context, analysis)
            
        return result
        
    def _find_similar_usage_patterns(self, context: str) -> str:
        """Find optimization hints from similar tool usage."""
        hints = []
        
        context_words = set(context.lower().split())
        
        for tool_name, patterns in self.usage_tracker.items():
            if patterns.get('performance_issues'):
                # Check similarity
                tool_words = set(tool_name.replace('_', ' ').split())
                if len(context_words.intersection(tool_words)) > 2:
                    for issue in patterns['performance_issues']:
                        hints.append(f"Avoid {issue['pattern']} (caused {issue['impact']})")
                        
        return '; '.join(hints[:3])  # Limit hints
        
    def _track_tool_generation(self, tool_name: str, context: str, analysis: Dict[str, Any]):
        """Track tool generation for learning."""
        state = self.state_manager.get_state()
        
        if 'tool_generations' not in state:
            state['tool_generations'] = []
            
        state['tool_generations'].append({
            'tool_name': tool_name,
            'context': context,
            'analysis': analysis,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only recent generations
        state['tool_generations'] = state['tool_generations'][-100:]
        
        self.state_manager.save_state()
        
    async def learn_from_tool_execution(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Learn from tool execution results."""
        if tool_name not in self.usage_tracker:
            self.usage_tracker[tool_name] = {
                'executions': 0,
                'successes': 0,
                'avg_time': 0,
                'performance_issues': []
            }
            
        tracker = self.usage_tracker[tool_name]
        tracker['executions'] += 1
        
        if success:
            tracker['successes'] += 1
            
        # Update average time
        tracker['avg_time'] = (
            (tracker['avg_time'] * (tracker['executions'] - 1) + execution_time) /
            tracker['executions']
        )
        
        # Track performance issues
        if execution_time > 5.0:  # Slow execution
            tracker['performance_issues'].append({
                'pattern': 'slow_execution',
                'impact': f'{execution_time:.1f}s execution time',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        # Save patterns
        state = self.state_manager.get_state()
        state['tool_usage_patterns'] = self.usage_tracker
        self.state_manager.save_state()


# Convenience functions
async def generate_smart_tool(
    tool_name: str,
    context: str,
    system_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate a smart, context-aware tool."""
    generator = EnhancedSmartToolGenerator()
    return await generator.generate_context_aware_tool(tool_name, context, system_state)
    
async def compose_tools(base_tools: List[str], intent: str, output_name: str) -> Dict[str, Any]:
    """Compose multiple tools into one."""
    generator = EnhancedSmartToolGenerator()
    return await generator._generate_composed_tool(output_name, base_tools, intent)