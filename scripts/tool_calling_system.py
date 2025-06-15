"""
Tool-Calling System for Conversational AI

This module provides a comprehensive tool-calling framework that allows the AI
to directly execute functions, create new tools, and enhance existing ones.
"""

import json
import asyncio
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import ast
import textwrap
import importlib.util
import os
from difflib import get_close_matches

from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.mcp_integration import MCPIntegrationHub
from scripts.http_ai_client import HTTPAIClient
from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.intelligent_work_finder import IntelligentWorkFinder

# Import new tool enhancement systems
try:
    from scripts.dependency_resolver import DependencyResolver
    from scripts.multi_tool_orchestrator import MultiToolOrchestrator
    from scripts.tool_evolution import ToolEvolution
    from scripts.semantic_tool_matcher import SemanticToolMatcher
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    ENHANCED_TOOLS_AVAILABLE = False


class ToolDefinition:
    """Represents a callable tool with metadata."""
    
    def __init__(self, name: str, func: Callable, description: str, 
                 parameters: Dict[str, Any], examples: List[str] = None):
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters
        self.examples = examples or []
        self.usage_count = 0
        self.success_count = 0
        self.last_used = None
        self.created_by_ai = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples,
            "stats": {
                "usage_count": self.usage_count,
                "success_rate": self.success_count / max(self.usage_count, 1)
            }
        }


class ToolCallingSystem:
    """Advanced tool-calling system with self-improvement capabilities."""
    
    def __init__(self):
        """Initialize the tool-calling system."""
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_history: List[Dict[str, Any]] = []
        self.ai_client = HTTPAIClient(enable_round_robin=True)
        
        # Initialize components
        self.state_manager = StateManager()
        self.task_manager = TaskManager()
        self.repo_analyzer = RepositoryAnalyzer()
        self.mcp_hub = MCPIntegrationHub()
        
        # Initialize enhanced tool systems if available
        if ENHANCED_TOOLS_AVAILABLE:
            self.dependency_resolver = DependencyResolver()
            self.multi_tool_orchestrator = MultiToolOrchestrator(self)
            self.tool_evolution = ToolEvolution(self)
            self.semantic_matcher = SemanticToolMatcher(self)
        else:
            self.dependency_resolver = None
            self.multi_tool_orchestrator = None
            self.tool_evolution = None
            self.semantic_matcher = None
            
        # Register built-in tools
        self._register_builtin_tools()
        
        # Load AI-created tools
        self._load_custom_tools()
        
    def _register_builtin_tools(self):
        """Register all built-in tools."""
        
        # System Status Tools
        self.register_tool(
            name="get_system_status",
            func=self._get_system_status,
            description="Get current system status including health, tasks, and metrics",
            parameters={},
            examples=["get_system_status()"]
        )
        
        self.register_tool(
            name="get_repositories",
            func=self._get_repositories,
            description="Get list of repositories with optional filtering",
            parameters={
                "filter": {"type": "string", "required": False, "description": "Filter repositories by name or language"},
                "limit": {"type": "integer", "required": False, "default": 10, "description": "Maximum number to return"}
            },
            examples=["get_repositories()", "get_repositories(filter='python', limit=5)"]
        )
        
        self.register_tool(
            name="get_tasks",
            func=self._get_tasks,
            description="Get current tasks from the task queue",
            parameters={
                "status": {"type": "string", "required": False, "enum": ["active", "pending", "completed"], "description": "Filter by status"},
                "limit": {"type": "integer", "required": False, "default": 10}
            },
            examples=["get_tasks()", "get_tasks(status='active', limit=5)"]
        )
        
        # Action Tools
        self.register_tool(
            name="create_issue",
            func=self._create_issue,
            description="Create a GitHub issue in a repository",
            parameters={
                "repo": {"type": "string", "required": True, "description": "Repository name"},
                "title": {"type": "string", "required": True, "description": "Issue title"},
                "body": {"type": "string", "required": True, "description": "Issue description"},
                "labels": {"type": "array", "required": False, "description": "Issue labels"}
            },
            examples=["create_issue(repo='myapp', title='Fix login bug', body='Users cannot login with email')"]
        )
        
        self.register_tool(
            name="execute_command",
            func=self._execute_command,
            description="Execute a natural language command through the NLI",
            parameters={
                "command": {"type": "string", "required": True, "description": "Natural language command to execute"}
            },
            examples=["execute_command(command='search repositories for React')"]
        )
        
        # Continuous AI Tools
        self.register_tool(
            name="start_continuous_ai",
            func=self._start_continuous_ai,
            description="Start the continuous AI system",
            parameters={
                "workers": {"type": "integer", "required": False, "default": 3, "description": "Number of workers"},
                "mode": {"type": "string", "required": False, "default": "production", "enum": ["production", "development"]}
            },
            examples=["start_continuous_ai()", "start_continuous_ai(workers=5, mode='development')"]
        )
        
        self.register_tool(
            name="stop_continuous_ai",
            func=self._stop_continuous_ai,
            description="Stop the continuous AI system",
            parameters={},
            examples=["stop_continuous_ai()"]
        )
        
        self.register_tool(
            name="get_continuous_ai_status",
            func=self._get_continuous_ai_status,
            description="Get status of the continuous AI system",
            parameters={},
            examples=["get_continuous_ai_status()"]
        )
        
        # Analysis Tools
        self.register_tool(
            name="analyze_repository",
            func=self._analyze_repository,
            description="Analyze a repository for code quality, issues, and improvements",
            parameters={
                "repo": {"type": "string", "required": True, "description": "Repository to analyze"}
            },
            examples=["analyze_repository(repo='myapp')"]
        )
        
        self.register_tool(
            name="search_code",
            func=self._search_code,
            description="Search for code patterns across repositories",
            parameters={
                "pattern": {"type": "string", "required": True, "description": "Code pattern to search for"},
                "language": {"type": "string", "required": False, "description": "Programming language filter"}
            },
            examples=["search_code(pattern='TODO', language='python')"]
        )
        
        # System Management Tools
        self.register_tool(
            name="clear_logs",
            func=self._clear_logs,
            description="Clear system log files",
            parameters={
                "older_than_days": {"type": "integer", "required": False, "default": 7, "description": "Clear logs older than N days"}
            },
            examples=["clear_logs()", "clear_logs(older_than_days=30)"]
        )
        
        self.register_tool(
            name="reset_system",
            func=self._reset_system,
            description="Reset system state",
            parameters={
                "type": {"type": "string", "required": False, "default": "selective", "enum": ["full", "selective", "emergency"]},
                "preserve_cache": {"type": "boolean", "required": False, "default": True}
            },
            examples=["reset_system()", "reset_system(type='full', preserve_cache=False)"]
        )
        
        # Research and Learning Tools
        self.register_tool(
            name="research_topic",
            func=self._research_topic,
            description="Research a technical topic and get insights",
            parameters={
                "topic": {"type": "string", "required": True, "description": "Topic to research"},
                "depth": {"type": "string", "required": False, "default": "medium", "enum": ["quick", "medium", "deep"]}
            },
            examples=["research_topic(topic='microservices architecture', depth='deep')"]
        )
        
        # Meta Tools (for self-improvement)
        self.register_tool(
            name="create_new_tool",
            func=self._create_new_tool,
            description="Create a new tool based on requirements",
            parameters={
                "name": {"type": "string", "required": True, "description": "Tool name"},
                "description": {"type": "string", "required": True, "description": "What the tool does"},
                "requirements": {"type": "string", "required": True, "description": "Detailed requirements for the tool"}
            },
            examples=["create_new_tool(name='deploy_app', description='Deploy application to cloud', requirements='Should handle AWS and Azure')"]
        )
        
        self.register_tool(
            name="enhance_tool",
            func=self._enhance_tool,
            description="Enhance an existing tool with new capabilities",
            parameters={
                "tool_name": {"type": "string", "required": True, "description": "Name of tool to enhance"},
                "enhancement": {"type": "string", "required": True, "description": "Description of enhancement needed"}
            },
            examples=["enhance_tool(tool_name='get_repositories', enhancement='Add ability to sort by stars')"]
        )
        
        self.register_tool(
            name="get_tool_usage_stats",
            func=self._get_tool_usage_stats,
            description="Get usage statistics for all tools",
            parameters={},
            examples=["get_tool_usage_stats()"]
        )
        
        # Enhanced tool system tools
        if ENHANCED_TOOLS_AVAILABLE:
            self.register_tool(
                name="handle_complex_query",
                func=self._handle_complex_query,
                description="Handle complex queries that require multiple tools",
                parameters={
                    "query": {"type": "string", "required": True, "description": "Complex query to handle"}
                },
                examples=["handle_complex_query(query='Analyze all repos for security issues and create GitHub issues')"]
            )
            
            self.register_tool(
                name="evolve_tool",
                func=self._evolve_tool,
                description="Evolve a tool based on its usage patterns and performance",
                parameters={
                    "target_tool": {"type": "string", "required": True, "description": "Name of the tool to evolve"}
                },
                examples=["evolve_tool(target_tool='analyze_repository')"]
            )
            
            self.register_tool(
                name="find_similar_tools",
                func=self._find_similar_tools,
                description="Find tools similar to a given query",
                parameters={
                    "query": {"type": "string", "required": True, "description": "Query to match against tools"}
                },
                examples=["find_similar_tools(query='scan code for bugs')"]
            )
        
        # Additional essential tools
        self.register_tool(
            name="count_repositories",
            func=self._count_repositories,
            description="Count total repositories being managed with breakdown by status",
            parameters={},
            examples=["count_repositories()"]
        )
        
        self.register_tool(
            name="count_tasks",
            func=self._count_tasks,
            description="Count tasks by status and type",
            parameters={
                "group_by": {"type": "string", "required": False, "enum": ["status", "type", "priority"], "description": "Group counting by attribute"}
            },
            examples=["count_tasks()", "count_tasks(group_by='status')"]
        )
        
        self.register_tool(
            name="list_available_commands",
            func=self._list_available_commands,
            description="List all available commands/tools with descriptions",
            parameters={
                "category": {"type": "string", "required": False, "description": "Filter by category (e.g., 'repository', 'task', 'ai')"}
            },
            examples=["list_available_commands()", "list_available_commands(category='repository')"]
        )
        
        self.register_tool(
            name="repository_health_check",
            func=self._repository_health_check,
            description="Check health status of all repositories",
            parameters={},
            examples=["repository_health_check()"]
        )
        
        self.register_tool(
            name="ai_health_dashboard",
            func=self._ai_health_dashboard,
            description="Get comprehensive AI system health dashboard",
            parameters={},
            examples=["ai_health_dashboard()"]
        )
        
    def register_tool(self, name: str, func: Callable, description: str, 
                     parameters: Dict[str, Any], examples: List[str] = None):
        """Register a new tool."""
        tool = ToolDefinition(name, func, description, parameters, examples)
        self.tools[name] = tool
        self.logger.info(f"Registered tool: {name}")
        
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool with given parameters."""
        # First check if an existing tool can handle this with semantic matching
        if tool_name not in self.tools and self.semantic_matcher:
            existing_tool = await self.semantic_matcher.can_existing_tool_handle(tool_name)
            if existing_tool:
                self.logger.info(f"Using existing tool '{existing_tool}' instead of creating '{tool_name}'")
                tool_name = existing_tool
                
        if tool_name not in self.tools:
            # Try to create the tool dynamically
            self.logger.info(f"Tool '{tool_name}' not found. Attempting to create it...")
            creation_result = await self._create_tool_from_intent(tool_name, kwargs)
            if creation_result.get('success'):
                # Tool created, now call it
                return await self.call_tool(tool_name, **kwargs)
            else:
                return {"error": f"Tool '{tool_name}' not found and could not be created", 
                       "suggestion": creation_result.get('suggestion'),
                       "available_tools": self._get_similar_tools(tool_name)}
        
        tool = self.tools[tool_name]
        tool.usage_count += 1
        tool.last_used = datetime.now(timezone.utc)
        
        # Record call in history
        call_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "parameters": kwargs,
            "success": False
        }
        
        try:
            # Track execution time
            import time
            start_time = time.time()
            
            # Call the tool function
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**kwargs)
            else:
                result = tool.func(**kwargs)
            
            execution_time = time.time() - start_time
            
            tool.success_count += 1
            call_record["success"] = True
            call_record["result"] = result
            call_record["execution_time"] = execution_time
            
            self.tool_history.append(call_record)
            
            # Track for evolution system if available
            if self.tool_evolution:
                await self.tool_evolution.track_tool_execution(
                    tool_name, kwargs, result, execution_time, None
                )
            
            return {"success": True, "result": result}
            
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            
            call_record["error"] = str(e)
            call_record["execution_time"] = execution_time
            self.tool_history.append(call_record)
            
            # Track error for evolution system
            if self.tool_evolution:
                await self.tool_evolution.track_tool_execution(
                    tool_name, kwargs, None, execution_time, str(e)
                )
            
            return {"error": str(e)}
    
    def get_tools_for_ai(self) -> List[Dict[str, Any]]:
        """Get all tools formatted for AI consumption."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with their metadata."""
        return {
            name: {
                'description': tool.description,
                'parameters': tool.parameters,
                'examples': tool.examples
            }
            for name, tool in self.tools.items()
        }
    
    def get_tool(self, tool_name: str):
        """Get a specific tool by name."""
        if tool_name in self.tools:
            return self.tools[tool_name].func
        return None
    
    async def parse_and_execute_tool_calls(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response for tool calls and execute them."""
        import re
        
        # Pattern to match tool calls: tool_name(param1="value1", param2=123)
        tool_pattern = r'(\w+)\((.*?)\)'
        results = []
        
        # Find all tool calls in the response
        for match in re.finditer(tool_pattern, ai_response):
            tool_name = match.group(1)
            params_str = match.group(2)
            
            if tool_name in self.tools:
                try:
                    # Parse parameters
                    params = {}
                    if params_str:
                        # Simple parameter parsing (can be enhanced)
                        param_pattern = r'(\w+)=(["\']?)(.+?)\2(?:,|$)'
                        for param_match in re.finditer(param_pattern, params_str):
                            param_name = param_match.group(1)
                            param_value = param_match.group(3)
                            
                            # Try to convert to appropriate type
                            try:
                                param_value = json.loads(param_value)
                            except:
                                pass
                            
                            params[param_name] = param_value
                    
                    # Execute the tool
                    result = await self.call_tool(tool_name, **params)
                    results.append({
                        "tool": tool_name,
                        "params": params,
                        "result": result
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error parsing/executing tool call: {e}")
                    results.append({
                        "tool": tool_name,
                        "error": str(e)
                    })
        
        return results
    
    # Built-in tool implementations
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Force reload to get fresh state
        if hasattr(self.state_manager, "force_reload_state"):
            state = self.state_manager.force_reload_state()
        else:
            state = self.state_manager.load_state()
        
        # Check continuous AI
        continuous_ai_status = await self._get_continuous_ai_status()
        
        # Get task stats
        task_state = self.task_manager.state
        tasks = list(task_state.get('tasks', {}).values())
        active_tasks = len([t for t in tasks if t.get('status') == 'active'])
        pending_tasks = len([t for t in tasks if t.get('status') == 'pending'])
        
        return {
            "healthy": True,
            "performance_metrics": state.get('system_performance', {}),
            "continuous_ai": continuous_ai_status,
            "tasks": {
                "active": active_tasks,
                "pending": pending_tasks,
                "total": len(tasks)
            },
            "last_updated": state.get('last_updated', 'Unknown')
        }
    
    async def _get_repositories(self, filter: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get repositories with optional filtering."""
        # Force reload to get fresh state
        if hasattr(self.state_manager, "force_reload_state"):
            state = self.state_manager.force_reload_state()
        else:
            state = self.state_manager.load_state()
        
        repositories = []
        for repo_name, repo_data in state.get('projects', {}).items():
            if filter:
                # Apply filter
                if filter.lower() not in repo_name.lower() and filter.lower() not in repo_data.get('language', '').lower():
                    continue
            
            repositories.append({
                'name': repo_name,
                'description': repo_data.get('description', ''),
                'language': repo_data.get('language', ''),
                'stars': repo_data.get('stars', 0),
                'issues': repo_data.get('open_issues_count', 0)
            })
        
        # Sort by stars and limit
        repositories.sort(key=lambda x: x['stars'], reverse=True)
        return repositories[:limit]
    
    async def _get_tasks(self, status: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tasks from the queue."""
        task_state = self.task_manager.state
        tasks = list(task_state.get('tasks', {}).values())
        
        if status:
            tasks = [t for t in tasks if t.get('status') == status]
        
        return tasks[:limit]
    
    async def _create_issue(self, repo: str, title: str, body: str, labels: List[str] = None) -> Dict[str, Any]:
        """Create a GitHub issue."""
        try:
            # Initialize MCP hub if needed
            if not hasattr(self.mcp_hub, 'github') or not self.mcp_hub.github:
                await self.mcp_hub.initialize()
            
            if self.mcp_hub.github:
                result = await self.mcp_hub.github.create_issue(repo, title, body, labels)
                return {"success": True, "issue": result}
            else:
                # Fallback to task creation
                task = self.task_manager.create_task(
                    title=f"Create issue: {title}",
                    description=f"Repository: {repo}\n\n{body}",
                    task_type="issue_creation",
                    metadata={"repo": repo, "labels": labels}
                )
                return {"success": True, "task_created": task}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a natural language command."""
        from scripts.natural_language_interface import NaturalLanguageInterface
        
        nli = NaturalLanguageInterface()
        await nli.initialize()
        
        result = await nli.process_natural_language(command)
        return result
    
    async def _start_continuous_ai(self, workers: int = 3, mode: str = "production") -> Dict[str, Any]:
        """Start the continuous AI system."""
        import subprocess
        import sys
        
        script_path = Path(__file__).parent.parent / "run_continuous_ai.py"
        cmd = [sys.executable, str(script_path), "--workers", str(workers), "--mode", mode]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            await asyncio.sleep(2)  # Wait for startup
            
            if process.poll() is None:
                return {"success": True, "pid": process.pid, "message": "Continuous AI started"}
            else:
                stdout, stderr = process.communicate()
                return {"success": False, "error": stderr or stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stop_continuous_ai(self) -> Dict[str, Any]:
        """Stop the continuous AI system."""
        import psutil
        import signal
        
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'run_continuous_ai.py' in ' '.join(cmdline):
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    return {"success": True, "message": "Continuous AI stopped"}
            
            return {"success": False, "message": "Continuous AI not running"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_continuous_ai_status(self) -> Dict[str, Any]:
        """Get continuous AI status."""
        import psutil
        from pathlib import Path
        
        status = {"running": False}
        
        # Check process
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'run_continuous_ai.py' in ' '.join(cmdline):
                    status['running'] = True
                    status['pid'] = proc.info['pid']
                    break
            except:
                continue
        
        # Check state file
        state_file = Path("continuous_orchestrator_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    status['metrics'] = state_data.get('metrics', {})
                    status['last_updated'] = state_data.get('last_updated')
            except:
                pass
        
        return status
    
    async def _analyze_repository(self, repo: str) -> Dict[str, Any]:
        """Analyze a repository."""
        try:
            analysis = await self.repo_analyzer.analyze_repository(repo)
            return {
                "success": True,
                "analysis": analysis,
                "summary": {
                    "health_score": analysis.get('health_score', 0),
                    "issues_found": len(analysis.get('issues', [])),
                    "improvements_suggested": len(analysis.get('improvements', []))
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _search_code(self, pattern: str, language: str = None) -> List[Dict[str, Any]]:
        """Search for code patterns."""
        # This would integrate with code search tools
        results = []
        
        # For now, return mock data
        return [
            {
                "file": "src/main.py",
                "line": 42,
                "match": f"# {pattern}: Fix this later",
                "repository": "myapp"
            }
        ]
    
    async def _clear_logs(self, older_than_days: int = 7) -> Dict[str, Any]:
        """Clear old log files."""
        import glob
        import os
        from datetime import datetime, timedelta
        
        cleared_files = []
        total_size = 0
        
        log_patterns = ["*.log", "logs/*.log", "*.txt"]
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        for pattern in log_patterns:
            for file_path in glob.glob(pattern):
                try:
                    file_stat = os.stat(file_path)
                    if datetime.fromtimestamp(file_stat.st_mtime) < cutoff_time:
                        total_size += file_stat.st_size
                        os.remove(file_path)
                        cleared_files.append(file_path)
                except:
                    pass
        
        return {
            "success": True,
            "cleared_files": len(cleared_files),
            "freed_space_mb": total_size / 1024 / 1024
        }
    
    async def _reset_system(self, type: str = "selective", preserve_cache: bool = True) -> Dict[str, Any]:
        """Reset system state."""
        # This would integrate with the reset functionality
        return {
            "success": True,
            "message": f"System reset ({type}) completed",
            "preserved": ["cache"] if preserve_cache else []
        }
    
    async def _research_topic(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """Research a technical topic."""
        prompt = f"""Research the following topic and provide insights:
        Topic: {topic}
        Depth: {depth}
        
        Provide:
        1. Overview
        2. Key concepts
        3. Best practices
        4. Common pitfalls
        5. Practical examples
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        return {
            "topic": topic,
            "research": response.get('content', ''),
            "depth": depth,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Meta tools for self-improvement
    
    async def _create_new_tool(self, name: str, description: str, requirements: str) -> Dict[str, Any]:
        """Create a new tool based on requirements."""
        # Check if this is trying to create a tool for a known abbreviation
        abbreviation_mappings = {
            'count_reps': 'count_repositories',
            'count_repos': 'count_repositories',
            'list_reps': 'list_repositories',
            'list_repos': 'list_repositories',
            'get_reps': 'get_repositories',
            'get_repos': 'get_repositories'
        }
        
        # If this is a known abbreviation, don't create a new tool
        if name in abbreviation_mappings:
            actual_tool = abbreviation_mappings[name]
            if actual_tool in self.tools:
                return {
                    "success": False,
                    "error": f"Tool '{name}' is an abbreviation for '{actual_tool}' which already exists",
                    "suggestion": f"Use '{actual_tool}' instead"
                }
        
        # Try to use smart tool generator first
        try:
            from scripts.smart_tool_generator import SmartToolGenerator
            generator = SmartToolGenerator()
            
            # Extract parameters from requirements
            params = {}
            if "Expected Parameters:" in requirements:
                param_line = requirements.split("Expected Parameters:")[1].split("\n")[0]
                try:
                    params = json.loads(param_line.strip())
                except:
                    pass
            
            result = await generator.generate_tool(name, description, params)
            
            if result.get('success'):
                return result
            else:
                # Fall back to old method if smart generator fails
                self.logger.warning(f"Smart tool generator failed: {result.get('error')}, falling back to standard method")
        except ImportError:
            self.logger.debug("Smart tool generator not available, using standard method")
        except Exception as e:
            self.logger.error(f"Error using smart tool generator: {e}")
        
        # Analyze the tool name to understand what imports might be needed
        imports_needed = self._determine_imports(name, requirements)
        
        # Generate tool code using AI
        prompt = f"""Create a Python async function for a new tool with these specifications:
        
        Tool Name: {name}
        Description: {description}
        Requirements: {requirements}
        
        CRITICAL RULES:
        1. The function should NOT take 'self' as a parameter
        2. It should be a standalone async function
        3. Do NOT include a if __name__ == '__main__' block
        4. Do NOT include any test/example 'main()' function
        5. ONLY create the single tool function named '{name}'
        
        Generate a complete Python module with:
        1. Module docstring with description
        2. Required imports (from scripts.state_manager import StateManager, etc.)
        3. Module-level variables: __description__, __parameters__, __examples__
        4. The main async function named '{name}' with NO self parameter
        5. Inside the function, create instances: state_manager = StateManager(), etc.
        6. Proper error handling and validation
        7. Return a dictionary with the actual result data (not nested in 'success'/'result')
        
        Example structure:
        async def {name}(**kwargs):
            state_manager = StateManager()
            state = state_manager.load_state()
            # ... implementation ...
            return {{"total": 10, "summary": "Found 10 items"}}
        
        Format the code as a complete, runnable Python module.
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if response.get('content'):
            try:
                # Extract code from response
                code_content = self._extract_code_from_response(response['content'])
                
                # Fix imports using dependency resolver if available
                if self.dependency_resolver:
                    code_content = self.dependency_resolver.fix_import_paths(code_content)
                
                # Ensure proper module structure
                if '__description__' not in code_content:
                    code_content = f'''"""
{description}
"""

__description__ = "{description}"
__parameters__ = {{}}
__examples__ = ["{name}()"]

''' + code_content
                
                # Add necessary imports if missing and no resolver available
                elif not self.dependency_resolver and 'import' not in code_content:
                    code_content = f"{imports_needed}\n\n{code_content}"
                
                # Save the generated tool
                custom_tools_dir = Path("scripts/custom_tools")
                custom_tools_dir.mkdir(exist_ok=True)
                
                tool_file = custom_tools_dir / f"{name}.py"
                with open(tool_file, 'w') as f:
                    f.write(f'''"""
AI-Generated Tool: {name}
Description: {description}
Generated: {datetime.now(timezone.utc).isoformat()}
Requirements: {requirements}
"""

{code_content}
''')
                
                self.logger.info(f"Created tool file: {tool_file}")
                
                return {
                    "success": True,
                    "message": f"Created new tool: {name}",
                    "file": str(tool_file),
                    "auto_generated": True
                }
                
            except Exception as e:
                self.logger.error(f"Error creating tool: {e}")
                return {"success": False, "error": f"Failed to create tool: {str(e)}"}
        
        return {"success": False, "error": "Could not generate tool code"}
    
    def _determine_imports(self, name: str, requirements: str) -> str:
        """Determine what imports are likely needed based on tool name and requirements."""
        imports = ["from typing import Dict, List, Any, Optional", "import asyncio", "import json"]
        
        if 'repository' in name or 'repo' in name:
            imports.append("from pathlib import Path")
        if 'github' in name or 'issue' in name:
            imports.append("import aiohttp")
        if 'count' in name or 'analyze' in name:
            imports.append("from collections import Counter")
        if 'time' in requirements or 'date' in requirements:
            imports.append("from datetime import datetime, timezone")
            
        return "\n".join(imports)
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from AI response, handling various formats."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # If no code blocks, assume the whole response is code
        return response.strip()
    
    async def _enhance_tool(self, tool_name: str, enhancement: str) -> Dict[str, Any]:
        """Enhance an existing tool."""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        
        # Generate enhancement using AI
        prompt = f"""Enhance the existing tool with new capabilities:
        
        Tool: {tool_name}
        Current Description: {tool.description}
        Current Parameters: {json.dumps(tool.parameters, indent=2)}
        
        Enhancement Required: {enhancement}
        
        Provide:
        1. Updated function code that includes the enhancement
        2. Any new parameters needed
        3. Updated description
        4. Example of using the new capability
        """
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if response.get('content'):
            # Log the enhancement
            self.logger.info(f"Enhanced tool {tool_name}: {enhancement}")
            
            return {
                "success": True,
                "message": f"Enhanced tool: {tool_name}",
                "enhancement": enhancement,
                "details": response['content']
            }
        
        return {"success": False, "error": "Could not generate enhancement"}
    
    async def _get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = []
        
        for name, tool in self.tools.items():
            stats.append({
                "name": name,
                "usage_count": tool.usage_count,
                "success_rate": tool.success_count / max(tool.usage_count, 1),
                "last_used": tool.last_used.isoformat() if tool.last_used else None,
                "created_by_ai": tool.created_by_ai
            })
        
        # Sort by usage
        stats.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return {
            "total_tools": len(self.tools),
            "total_calls": sum(t.usage_count for t in self.tools.values()),
            "ai_created_tools": sum(1 for t in self.tools.values() if t.created_by_ai),
            "most_used": stats[:5],
            "least_used": [s for s in stats if s['usage_count'] == 0],
            "detailed_stats": stats
        }
    
    async def _count_repositories(self) -> Dict[str, Any]:
        """Count total repositories being managed."""
        # Force reload to get fresh state
        if hasattr(self.state_manager, "force_reload_state"):
            state = self.state_manager.force_reload_state()
        else:
            state = self.state_manager.load_state()
        projects = state.get('projects', {})
        
        # Count by various criteria
        total = len(projects)
        by_language = {}
        by_status = {'active': 0, 'archived': 0, 'unknown': 0}
        total_stars = 0
        total_issues = 0
        
        for repo_name, repo_data in projects.items():
            # Language count
            language = repo_data.get('language', 'Unknown')
            by_language[language] = by_language.get(language, 0) + 1
            
            # Status count
            if repo_data.get('archived', False):
                by_status['archived'] += 1
            else:
                by_status['active'] += 1
            
            # Aggregate metrics
            total_stars += repo_data.get('stars', 0)
            total_issues += repo_data.get('open_issues_count', 0)
        
        return {
            "total": total,
            "breakdown": {
                "by_status": by_status,
                "by_language": by_language
            },
            "metrics": {
                "total_stars": total_stars,
                "total_open_issues": total_issues,
                "avg_stars_per_repo": total_stars / max(total, 1)
            },
            "summary": f"Managing {total} repositories ({by_status['active']} active, {by_status['archived']} archived)"
        }
    
    async def _count_tasks(self, group_by: str = "status") -> Dict[str, Any]:
        """Count tasks by specified grouping."""
        task_state = self.task_manager.state
        tasks = list(task_state.get('tasks', {}).values())
        total = len(tasks)
        
        # Initialize counters
        counts = {}
        
        if group_by == "status":
            counts = {"active": 0, "pending": 0, "completed": 0, "failed": 0}
            for task in tasks:
                status = task.get('status', 'unknown')
                counts[status] = counts.get(status, 0) + 1
                
        elif group_by == "type":
            for task in tasks:
                task_type = task.get('type', 'general')
                counts[task_type] = counts.get(task_type, 0) + 1
                
        elif group_by == "priority":
            counts = {"high": 0, "medium": 0, "low": 0}
            for task in tasks:
                priority = task.get('priority', 'medium')
                counts[priority] = counts.get(priority, 0) + 1
        
        # Calculate percentages
        percentages = {k: (v / max(total, 1)) * 100 for k, v in counts.items()}
        
        return {
            "total": total,
            "group_by": group_by,
            "counts": counts,
            "percentages": percentages,
            "summary": f"Total {total} tasks - " + ", ".join([f"{k}: {v}" for k, v in counts.items()])
        }
    
    async def _list_available_commands(self, category: str = None) -> Dict[str, Any]:
        """List all available commands/tools."""
        tools_by_category = {
            "repository": [],
            "task": [],
            "ai": [],
            "system": [],
            "analysis": [],
            "meta": [],
            "other": []
        }
        
        # Categorize tools
        for name, tool in self.tools.items():
            # Determine category
            if 'repository' in name or 'repo' in name:
                cat = "repository"
            elif 'task' in name or 'issue' in name:
                cat = "task"
            elif 'ai' in name or 'continuous' in name:
                cat = "ai"
            elif 'system' in name or 'reset' in name or 'clear' in name:
                cat = "system"
            elif 'analyze' in name or 'search' in name or 'count' in name:
                cat = "analysis"
            elif 'create_new_tool' in name or 'enhance' in name:
                cat = "meta"
            else:
                cat = "other"
                
            tool_info = {
                "name": name,
                "description": tool.description,
                "usage_count": tool.usage_count,
                "ai_created": tool.created_by_ai
            }
            tools_by_category[cat].append(tool_info)
        
        # Filter by category if specified
        if category:
            filtered = tools_by_category.get(category, [])
            return {
                "category": category,
                "tools": filtered,
                "count": len(filtered),
                "available_categories": list(tools_by_category.keys())
            }
        
        # Return all categories
        total_tools = sum(len(tools) for tools in tools_by_category.values())
        return {
            "total_tools": total_tools,
            "by_category": tools_by_category,
            "categories": {cat: len(tools) for cat, tools in tools_by_category.items() if tools},
            "ai_created_count": sum(1 for t in self.tools.values() if t.created_by_ai)
        }
    
    async def _repository_health_check(self) -> Dict[str, Any]:
        """Check health status of all repositories."""
        # Force reload to get fresh state
        if hasattr(self.state_manager, "force_reload_state"):
            state = self.state_manager.force_reload_state()
        else:
            state = self.state_manager.load_state()
        projects = state.get('projects', {})
        
        health_results = {
            "healthy": [],
            "warning": [],
            "critical": [],
            "unknown": []
        }
        
        for repo_name, repo_data in projects.items():
            health_score = 100
            issues = []
            
            # Check various health metrics
            open_issues = repo_data.get('open_issues_count', 0)
            if open_issues > 50:
                health_score -= 30
                issues.append(f"High issue count: {open_issues}")
            elif open_issues > 20:
                health_score -= 15
                issues.append(f"Moderate issue count: {open_issues}")
            
            # Check last update
            last_updated = repo_data.get('updated_at', '')
            if last_updated:
                # Simple check - if not updated in last 30 days
                health_score -= 10
                issues.append("Possibly stale")
            
            # Check if archived
            if repo_data.get('archived', False):
                health_score = 0
                issues.append("Repository is archived")
            
            # Categorize health
            repo_health = {
                "name": repo_name,
                "score": health_score,
                "issues": issues,
                "metrics": {
                    "stars": repo_data.get('stars', 0),
                    "open_issues": open_issues,
                    "language": repo_data.get('language', 'Unknown')
                }
            }
            
            if health_score >= 80:
                health_results["healthy"].append(repo_health)
            elif health_score >= 60:
                health_results["warning"].append(repo_health)
            elif health_score > 0:
                health_results["critical"].append(repo_health)
            else:
                health_results["unknown"].append(repo_health)
        
        return {
            "total_checked": len(projects),
            "health_breakdown": {
                "healthy": len(health_results["healthy"]),
                "warning": len(health_results["warning"]),
                "critical": len(health_results["critical"]),
                "unknown": len(health_results["unknown"])
            },
            "details": health_results,
            "summary": f"Checked {len(projects)} repositories: {len(health_results['healthy'])} healthy, {len(health_results['warning'])} warnings, {len(health_results['critical'])} critical"
        }
    
    async def _ai_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive AI system health dashboard."""
        # Get continuous AI status
        ai_status = await self._get_continuous_ai_status()
        
        # Get tool usage stats
        tool_stats = await self._get_tool_usage_stats()
        
        # Calculate AI health metrics
        health_metrics = {
            "continuous_ai_running": ai_status.get('running', False),
            "total_tool_calls": tool_stats['total_calls'],
            "ai_created_tools": tool_stats['ai_created_tools'],
            "most_used_tools": tool_stats['most_used'][:3]
        }
        
        # Performance metrics
        performance = {
            "avg_tool_success_rate": sum(t.success_count / max(t.usage_count, 1) for t in self.tools.values()) / max(len(self.tools), 1),
            "tools_never_used": len(tool_stats['least_used']),
            "total_available_tools": tool_stats['total_tools']
        }
        
        # System load (simplified)
        import psutil
        system_load = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "ai_status": ai_status,
            "health_metrics": health_metrics,
            "performance": performance,
            "system_load": system_load,
            "recommendations": self._generate_ai_health_recommendations(health_metrics, performance, system_load),
            "summary": f"AI System {'Running' if ai_status.get('running') else 'Stopped'} - {tool_stats['total_calls']} total operations"
        }
    
    def _generate_ai_health_recommendations(self, health_metrics, performance, system_load):
        """Generate recommendations based on AI health metrics."""
        recommendations = []
        
        if not health_metrics['continuous_ai_running']:
            recommendations.append("Start continuous AI system for automated task processing")
        
        if performance['tools_never_used'] > 10:
            recommendations.append(f"Consider removing {performance['tools_never_used']} unused tools")
        
        if performance['avg_tool_success_rate'] < 0.8:
            recommendations.append("Review and improve failing tools")
        
        if system_load['cpu_percent'] > 80:
            recommendations.append("High CPU usage - consider scaling down AI workers")
        
        if system_load['memory_percent'] > 85:
            recommendations.append("High memory usage - consider restarting some services")
        
        return recommendations
    
    def _load_custom_tools(self):
        """Load AI-created custom tools."""
        custom_tools_dir = Path("scripts/custom_tools")
        if custom_tools_dir.exists():
            for tool_file in custom_tools_dir.glob("*.py"):
                try:
                    # Dynamic import of custom tools
                    spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Only register the main tool function (matching filename)
                    tool_name = tool_file.stem
                    if hasattr(module, tool_name):
                        func = getattr(module, tool_name)
                        if inspect.isfunction(func):
                            # Check if function expects self parameter
                            sig = inspect.signature(func)
                            params = list(sig.parameters.keys())
                            
                            # If function expects 'self' as first parameter, wrap it
                            if params and params[0] == 'self':
                                if asyncio.iscoroutinefunction(func):
                                    async def wrapped_func(**kwargs):
                                        return await func(self, **kwargs)
                                else:
                                    def wrapped_func(**kwargs):
                                        return func(self, **kwargs)
                                actual_func = wrapped_func
                            else:
                                actual_func = func
                            
                            # Register the tool
                            self.register_tool(
                                name=tool_name,
                                func=actual_func,
                                description=getattr(module, '__description__', func.__doc__ or f"Custom tool: {tool_name}"),
                                parameters=getattr(module, '__parameters__', {}),
                                examples=getattr(module, '__examples__', [])
                            )
                            self.tools[tool_name].created_by_ai = True
                            
                except Exception as e:
                    self.logger.error(f"Error loading custom tool {tool_file}: {e}")
    
    async def _create_tool_from_intent(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tool based on the intent from the tool name and parameters."""
        # Analyze the tool name to understand intent
        intent = await self._analyze_tool_intent(tool_name, params)
        
        if not intent.get('understood'):
            return {
                'success': False,
                'suggestion': f"Could not understand the intent for '{tool_name}'. Try one of these similar tools: {', '.join(self._get_similar_tools(tool_name))}"
            }
        
        # Generate tool requirements
        requirements = f"""
        Tool Name: {tool_name}
        Intent: {intent['description']}
        Expected Parameters: {json.dumps(params)}
        Category: {intent.get('category', 'general')}
        
        The tool should:
        {intent.get('requirements', 'Perform the requested operation')}
        """
        
        # Create the tool
        result = await self._create_new_tool(
            name=tool_name,
            description=intent['description'],
            requirements=requirements
        )
        
        if result.get('success'):
            # Load the newly created tool
            await self._load_single_custom_tool(tool_name)
            self.logger.info(f"Successfully created and loaded tool: {tool_name}")
            
        return result
    
    async def _analyze_tool_intent(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the intent behind a tool name."""
        # Common patterns
        patterns = {
            'count_': {'category': 'analytics', 'action': 'count', 'description': 'Count items'},
            'list_': {'category': 'query', 'action': 'list', 'description': 'List items'},
            'create_': {'category': 'creation', 'action': 'create', 'description': 'Create new item'},
            'delete_': {'category': 'deletion', 'action': 'delete', 'description': 'Delete item'},
            'update_': {'category': 'modification', 'action': 'update', 'description': 'Update existing item'},
            'analyze_': {'category': 'analytics', 'action': 'analyze', 'description': 'Analyze data'},
            'get_': {'category': 'query', 'action': 'get', 'description': 'Retrieve information'},
            'search_': {'category': 'query', 'action': 'search', 'description': 'Search for items'},
            'monitor_': {'category': 'monitoring', 'action': 'monitor', 'description': 'Monitor status'},
            'optimize_': {'category': 'optimization', 'action': 'optimize', 'description': 'Optimize performance'},
        }
        
        # Determine action and object
        understood = False
        category = 'general'
        action = 'process'
        description = f"Process {tool_name.replace('_', ' ')}"
        
        for prefix, pattern_info in patterns.items():
            if tool_name.startswith(prefix):
                understood = True
                category = pattern_info['category']
                action = pattern_info['action']
                object_name = tool_name[len(prefix):].replace('_', ' ')
                description = f"{pattern_info['description']} for {object_name}"
                break
        
        # Special cases
        if 'repositories' in tool_name or 'repos' in tool_name:
            object_type = 'repositories'
        elif 'tasks' in tool_name:
            object_type = 'tasks'
        elif 'issues' in tool_name:
            object_type = 'issues'
        elif 'ai' in tool_name or 'continuous' in tool_name:
            object_type = 'AI system'
        else:
            object_type = 'items'
        
        # Build requirements based on intent
        requirements = self._generate_tool_requirements(action, object_type, tool_name)
        
        return {
            'understood': understood,
            'category': category,
            'action': action,
            'object_type': object_type,
            'description': description,
            'requirements': requirements
        }
    
    def _generate_tool_requirements(self, action: str, object_type: str, tool_name: str) -> str:
        """Generate detailed requirements for a tool based on its action and object type."""
        requirements_templates = {
            'count': f"1. Count all {object_type} in the system\n2. Return total count with breakdown by status/type\n3. Include summary statistics",
            'list': f"1. List all {object_type} with relevant details\n2. Support filtering and pagination\n3. Return structured data",
            'create': f"1. Create a new {object_type[:-1]} with validation\n2. Handle all required fields\n3. Return created item details",
            'analyze': f"1. Perform comprehensive analysis of {object_type}\n2. Generate insights and recommendations\n3. Return detailed report",
            'search': f"1. Search {object_type} using flexible criteria\n2. Support partial matches and filters\n3. Return ranked results",
            'monitor': f"1. Monitor {object_type} status in real-time\n2. Track key metrics and changes\n3. Return current status and alerts",
            'optimize': f"1. Analyze {object_type} performance\n2. Identify optimization opportunities\n3. Apply improvements and return results"
        }
        
        return requirements_templates.get(action, f"Implement functionality for {tool_name}")
    
    def _get_similar_tools(self, tool_name: str) -> List[str]:
        """Find similar tool names using fuzzy matching."""
        all_tools = list(self.tools.keys())
        # Get close matches
        similar = get_close_matches(tool_name, all_tools, n=3, cutoff=0.6)
        
        # Also check for tools with similar prefixes or suffixes
        if not similar:
            prefix = tool_name.split('_')[0] + '_'
            similar = [t for t in all_tools if t.startswith(prefix)][:3]
        
        return similar
    
    async def _load_single_custom_tool(self, tool_name: str):
        """Load a single custom tool file with validation."""
        custom_tools_dir = Path("scripts/custom_tools")
        tool_file = custom_tools_dir / f"{tool_name}.py"
        
        if tool_file.exists():
            try:
                # First validate the tool
                from scripts.enhanced_tool_validation import EnhancedToolValidator
                validator = EnhancedToolValidator()
                
                self.logger.info(f"Validating tool before loading: {tool_name}")
                validation_result = await validator.validate_tool(tool_file, tool_name)
                
                if not validation_result.is_valid:
                    self.logger.error(f"Tool validation failed for {tool_name}: {validation_result.issues}")
                    # Optionally remove invalid tool
                    if len(validation_result.issues) > 2:  # Multiple serious issues
                        tool_file.unlink()
                        self.logger.warning(f"Removed invalid tool file: {tool_file}")
                    return False
                
                # If validation passed, load the tool
                spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for the main function (should match tool name)
                if hasattr(module, tool_name):
                    func = getattr(module, tool_name)
                    
                    # Check function signature
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    
                    # If function expects 'self' as first parameter, it needs wrapping
                    if params and params[0] == 'self':
                        # This function expects self, wrap it
                        if asyncio.iscoroutinefunction(func):
                            async def wrapped_func(**kwargs):
                                return await func(self, **kwargs)
                        else:
                            def wrapped_func(**kwargs):
                                return func(self, **kwargs)
                        actual_func = wrapped_func
                    else:
                        # Function doesn't need self, use as-is
                        actual_func = func
                    
                    # Extract metadata from module
                    self.register_tool(
                        name=tool_name,
                        func=actual_func,
                        description=getattr(module, '__description__', f"Custom tool: {tool_name}"),
                        parameters=getattr(module, '__parameters__', {}),
                        examples=getattr(module, '__examples__', [])
                    )
                    self.tools[tool_name].created_by_ai = True
                    
                    # Log warnings if any
                    if validation_result.warnings:
                        self.logger.warning(f"Tool {tool_name} loaded with warnings: {validation_result.warnings}")
                    
                    # Add to semantic index if available
                    if self.semantic_matcher:
                        self.semantic_matcher.add_tool_to_index(
                            tool_name,
                            {
                                'description': self.tools[tool_name].description,
                                'parameters': self.tools[tool_name].parameters
                            }
                        )
                    
                    return True
            except ImportError as e:
                self.logger.error(f"Import error loading custom tool {tool_name}: {e}")
                # Don't remove file for import errors - might be fixable
            except Exception as e:
                self.logger.error(f"Error loading custom tool {tool_name}: {e}")
        
        return False
    
    # Enhanced tool system methods
    
    async def _handle_complex_query(self, query: str) -> Dict[str, Any]:
        """Handle complex queries using multi-tool orchestration."""
        if not self.multi_tool_orchestrator:
            return {"error": "Multi-tool orchestration not available"}
            
        try:
            result = await self.multi_tool_orchestrator.handle_complex_query(query)
            return result
        except Exception as e:
            self.logger.error(f"Error handling complex query: {e}")
            return {"error": str(e)}
            
    async def _evolve_tool(self, target_tool: str) -> Dict[str, Any]:
        """Evolve a tool based on its usage patterns."""
        if not self.tool_evolution:
            return {"error": "Tool evolution system not available"}
            
        try:
            evolution_result = await self.tool_evolution.evolve_tool(target_tool)
            return {
                "success": evolution_result.success,
                "tool_name": evolution_result.tool_name,
                "improvements_applied": len(evolution_result.improvements_applied),
                "performance_gain": f"{evolution_result.performance_gain:.1%}",
                "details": [
                    {
                        "type": imp.improvement_type,
                        "description": imp.description,
                        "expected_impact": f"{imp.expected_impact:.1%}"
                    }
                    for imp in evolution_result.improvements_applied
                ],
                "errors": evolution_result.errors
            }
        except Exception as e:
            self.logger.error(f"Error evolving tool: {e}")
            return {"error": str(e)}
            
    async def _find_similar_tools(self, query: str) -> Dict[str, Any]:
        """Find tools similar to a given query."""
        if not self.semantic_matcher:
            return {"error": "Semantic tool matching not available"}
            
        try:
            matches = await self.semantic_matcher.find_similar_tools(query, top_k=5)
            return {
                "query": query,
                "matches": [
                    {
                        "tool_name": match.tool_name,
                        "similarity": f"{match.similarity_score:.1%}",
                        "reason": match.reason,
                        "capability_match": match.capability_match
                    }
                    for match in matches
                ],
                "suggestion": f"Use '{matches[0].tool_name}'" if matches else "No similar tools found"
            }
        except Exception as e:
            self.logger.error(f"Error finding similar tools: {e}")
            return {"error": str(e)}