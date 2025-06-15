"""
Smart Tool Generator with Dependency Management

Intelligently generates tools that:
1. Check for existing dependencies before importing
2. Create stub implementations when needed
3. Fall back to available modules
"""

import os
import ast
import json
import asyncio
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from scripts.http_ai_client import HTTPAIClient


class DependencyResolver:
    """Resolves and manages tool dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scripts_path = Path(__file__).parent
        self.available_modules = self._scan_available_modules()
        
    def _scan_available_modules(self) -> Set[str]:
        """Scan for all available modules in scripts directory."""
        modules = set()
        
        for py_file in self.scripts_path.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue
            module_name = f"scripts.{py_file.stem}"
            modules.add(module_name)
            
        # Also scan subdirectories
        for subdir in self.scripts_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                init_file = subdir / "__init__.py"
                if init_file.exists():
                    modules.add(f"scripts.{subdir.name}")
                    # Add sub-modules
                    for py_file in subdir.glob("*.py"):
                        if not py_file.name.startswith("_"):
                            modules.add(f"scripts.{subdir.name}.{py_file.stem}")
                            
        return modules
    
    def check_import_availability(self, import_name: str) -> bool:
        """Check if an import is available."""
        # Check standard library
        try:
            __import__(import_name.split('.')[0])
            return True
        except ImportError:
            pass
            
        # Check our modules
        return import_name in self.available_modules
    
    def suggest_alternatives(self, import_name: str) -> List[str]:
        """Suggest alternative imports for unavailable modules."""
        alternatives = []
        
        # Common mappings
        mappings = {
            "scripts.alert_system": ["scripts.logger", "scripts.error_handler"],
            "scripts.alert_manager": ["scripts.logger", "scripts.error_handler"],
            "scripts.metrics_tracker": ["scripts.state_manager", "scripts.cache_manager"],
            "scripts.metric_tracker": ["scripts.state_manager", "scripts.cache_manager"],
            "scripts.logging_utils": ["scripts.logger"],
            "scripts.worker_utils": ["scripts.worker_status_monitor", "scripts.simple_worker_monitor"],
        }
        
        if import_name in mappings:
            alternatives.extend(mappings[import_name])
            
        # Also suggest similar names
        import_parts = import_name.split('.')
        if len(import_parts) > 1:
            base_name = import_parts[-1]
            for module in self.available_modules:
                if base_name.lower() in module.lower() and module != import_name:
                    alternatives.append(module)
                    
        return list(set(alternatives))
    
    def generate_stub_module(self, module_name: str, required_functions: List[str]) -> str:
        """Generate a stub module implementation."""
        stub_code = f'''"""
Auto-generated stub for {module_name}
Created by Smart Tool Generator
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

'''
        
        # Generate stub implementations based on module type
        if "alert" in module_name.lower():
            stub_code += '''
class AlertSystem:
    """Stub alert system implementation."""
    
    def __init__(self):
        self.alerts = []
        
    def check_alerts(self, state: Dict[str, Any], metrics: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Check for alerts based on state and metrics."""
        alerts = []
        
        # Basic alert logic
        if state.get("error_count", 0) > 0:
            alerts.append({
                "type": "error",
                "message": f"Found {state['error_count']} errors",
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            })
            
        return alerts
    
    def send_alert(self, message: str, severity: str = "info"):
        """Log an alert."""
        logger.warning(f"ALERT [{severity}]: {message}")
        self.alerts.append({
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        })

# Convenience functions
def check_alerts(*args, **kwargs):
    """Check for alerts."""
    alert_system = AlertSystem()
    return alert_system.check_alerts(*args, **kwargs)
'''
        
        elif "metric" in module_name.lower():
            stub_code += '''
class MetricTracker:
    """Stub metric tracker implementation."""
    
    def __init__(self):
        self.metrics = {}
        
    def track_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Track metrics from state."""
        metrics = {
            "total_items": len(state.get("items", [])),
            "active_count": sum(1 for item in state.get("items", []) if item.get("status") == "active"),
            "error_count": sum(1 for item in state.get("items", []) if item.get("status") == "error"),
            "timestamp": datetime.now().isoformat()
        }
        self.metrics.update(metrics)
        return metrics
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

# Convenience functions
def track_metrics(*args, **kwargs):
    """Track metrics."""
    tracker = MetricTracker()
    return tracker.track_metrics(*args, **kwargs)
'''
        
        elif "logging" in module_name.lower():
            stub_code += '''
def log_activity(message: str, category: str = "info"):
    """Log an activity."""
    logger.info(f"[{category.upper()}] {message}")

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger."""
    return logging.getLogger(name)
'''
        
        elif "worker" in module_name.lower() and "utils" in module_name.lower():
            stub_code += '''
async def get_worker_metrics(workers: Dict[str, Any]) -> Dict[str, Any]:
    """Get metrics for workers."""
    active_workers = [w for w in workers.values() if w.get("status") == "active"]
    
    return {
        "total_workers": len(workers),
        "active_workers": len(active_workers),
        "cpu_usage_avg": sum(w.get("cpu_usage", 0) for w in active_workers) / max(len(active_workers), 1),
        "memory_usage_avg": sum(w.get("memory_usage", 0) for w in active_workers) / max(len(active_workers), 1),
        "tasks_processed": sum(w.get("tasks_processed", 0) for w in workers.values()),
    }

async def analyze_worker_performance(workers: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    """Analyze worker performance and return alerts."""
    alerts = []
    
    # Check for idle workers
    idle_count = sum(1 for w in workers.values() if w.get("status") == "idle")
    if idle_count > len(workers) * 0.5:
        alerts.append(f"High idle rate: {idle_count}/{len(workers)} workers are idle")
    
    # Check for errors
    error_count = sum(1 for w in workers.values() if w.get("status") == "error")
    if error_count > 0:
        alerts.append(f"Worker errors detected: {error_count} workers in error state")
    
    # Check resource usage
    if metrics.get("cpu_usage_avg", 0) > 80:
        alerts.append(f"High CPU usage: {metrics['cpu_usage_avg']:.1f}%")
        
    return alerts
'''
        
        # Add imports if needed
        if "datetime" in stub_code and "from datetime import datetime" not in stub_code:
            stub_code = "from datetime import datetime, timezone\n" + stub_code
            
        return stub_code


class SmartToolGenerator:
    """Generates tools with intelligent dependency handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient(enable_round_robin=True)
        self.dependency_resolver = DependencyResolver()
        self.custom_tools_dir = Path(__file__).parent / "custom_tools"
        self.custom_tools_dir.mkdir(exist_ok=True)
        
        # Stub modules directory
        self.stubs_dir = Path(__file__).parent / "custom_stubs"
        self.stubs_dir.mkdir(exist_ok=True)
        
    async def generate_tool(self, tool_name: str, tool_intent: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a tool with smart dependency handling."""
        
        # First, generate the tool with existing modules list
        prompt = f"""Create a Python tool with these specifications:

Tool Name: {tool_name}
Intent: {tool_intent}
Parameters: {parameters or {}}

CRITICAL: You MUST only use imports from this list of AVAILABLE modules:
{chr(10).join(sorted(self.dependency_resolver.available_modules))}

Standard library imports are also allowed (os, json, asyncio, logging, datetime, etc.)

DO NOT import any modules not in the above list. If you need functionality that would normally come from a missing module, implement it inline or use the closest available alternative.

For example:
- Instead of scripts.alert_system, use scripts.logger
- Instead of scripts.metrics_tracker, use scripts.state_manager
- Instead of non-existent worker utils, implement the logic directly

The tool should:
1. Be an async function named {tool_name}
2. Have proper error handling
3. Return a dictionary
4. Include __description__, __parameters__, and __examples__ metadata
"""

        try:
            response = await self.ai_client.generate(prompt, preferred_provider="anthropic")
            
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            # Analyze and fix dependencies
            fixed_code = await self._fix_dependencies(code, tool_name)
            
            # Save the tool
            tool_path = self.custom_tools_dir / f"{tool_name}.py"
            tool_path.write_text(fixed_code)
            
            self.logger.info(f"Successfully generated tool: {tool_name}")
            
            return {
                "success": True,
                "tool_name": tool_name,
                "tool_path": str(tool_path),
                "dependencies_fixed": code != fixed_code,
                "message": f"Tool {tool_name} created successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tool {tool_name}: {e}")
            return {
                "success": False,
                "tool_name": tool_name,
                "error": str(e),
                "message": f"Failed to generate tool: {e}"
            }
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from AI response."""
        # Try to find code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            # Assume entire response is code
            return response.strip()
    
    async def _fix_dependencies(self, code: str, tool_name: str) -> str:
        """Fix dependencies in generated code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.logger.error(f"Syntax error in generated code: {e}")
            return code
            
        # Collect all imports
        imports_to_fix = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module
                if module_name and module_name.startswith("scripts."):
                    if not self.dependency_resolver.check_import_availability(module_name):
                        imports_to_fix.append((node, module_name))
                        
        # Fix unavailable imports
        lines = code.split('\n')
        for node, module_name in imports_to_fix:
            # Find alternatives
            alternatives = self.dependency_resolver.suggest_alternatives(module_name)
            
            if alternatives:
                # Replace with first alternative
                old_line = lines[node.lineno - 1]
                new_module = alternatives[0]
                new_line = old_line.replace(module_name, new_module)
                lines[node.lineno - 1] = new_line
                self.logger.info(f"Replaced import {module_name} with {new_module}")
            else:
                # Generate stub module
                self.logger.info(f"Generating stub for {module_name}")
                stub_code = self.dependency_resolver.generate_stub_module(module_name, [])
                
                # Save stub
                stub_path = self.stubs_dir / f"{module_name.replace('scripts.', '')}.py"
                stub_path.parent.mkdir(parents=True, exist_ok=True)
                stub_path.write_text(stub_code)
                
                # Update sys.path to include stubs
                lines.insert(0, f"import sys; sys.path.insert(0, '{self.stubs_dir}')")
                
        return '\n'.join(lines)
    
    async def validate_and_improve_tool(self, tool_path: Path) -> Dict[str, Any]:
        """Validate a tool and improve it if needed."""
        code = tool_path.read_text()
        
        # Try to import and test
        try:
            spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Test execution
            tool_func = getattr(module, tool_path.stem)
            result = await tool_func()
            
            return {
                "valid": True,
                "executable": True,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Tool validation failed: {e}")
            
            # Try to fix and retry
            fixed_code = await self._fix_dependencies(code, tool_path.stem)
            if fixed_code != code:
                tool_path.write_text(fixed_code)
                return await self.validate_and_improve_tool(tool_path)
                
            return {
                "valid": False,
                "executable": False,
                "error": str(e)
            }


# Convenience function for tool calling system integration
async def create_smart_tool(tool_name: str, tool_intent: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a tool using the smart generator."""
    generator = SmartToolGenerator()
    return await generator.generate_tool(tool_name, tool_intent, parameters)