"""
Background Intelligence Daemon

Runs continuously in the background to monitor system performance, identify improvement
opportunities, and apply optimizations without user intervention.
"""

import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
import traceback

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.tool_calling_system import ToolCallingSystem
from scripts.self_improvement_orchestrator import SelfImprovementOrchestrator
from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine


class SystemHealthMonitor:
    """Monitors system health and performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "response_times": deque(maxlen=100),
            "error_rates": deque(maxlen=100),
            "tool_success_rates": defaultdict(lambda: {"success": 0, "total": 0})
        }
        self.alerts = []
        
    def record_metric(self, metric_type: str, value: float):
        """Record a metric value."""
        if metric_type in self.metrics:
            self.metrics[metric_type].append({
                "timestamp": datetime.now(timezone.utc),
                "value": value
            })
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
        
    def check_health(self) -> List[Dict[str, Any]]:
        """Check system health and return any issues."""
        issues = []
        stats = self.get_system_stats()
        
        # Check CPU usage
        if stats["cpu_percent"] > 80:
            issues.append({
                "type": "high_cpu",
                "severity": "warning",
                "value": stats["cpu_percent"],
                "threshold": 80
            })
            
        # Check memory usage
        if stats["memory_percent"] > 85:
            issues.append({
                "type": "high_memory",
                "severity": "warning",
                "value": stats["memory_percent"],
                "threshold": 85
            })
            
        # Check error rates
        if self.metrics["error_rates"]:
            recent_errors = [m["value"] for m in list(self.metrics["error_rates"])[-10:]]
            avg_error_rate = sum(recent_errors) / len(recent_errors)
            if avg_error_rate > 0.2:  # More than 20% errors
                issues.append({
                    "type": "high_error_rate",
                    "severity": "critical",
                    "value": avg_error_rate,
                    "threshold": 0.2
                })
                
        return issues


class BackgroundIntelligenceDaemon:
    """Main daemon that orchestrates background intelligence tasks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Initialize components
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.tool_system = ToolCallingSystem()
        self.improvement_orchestrator = SelfImprovementOrchestrator()
        self.evolutionary_engine = EvolutionaryImprovementEngine(self.state_manager)
        self.health_monitor = SystemHealthMonitor()
        
        # Task scheduling
        self.tasks = {
            "health_check": {"interval": 60, "last_run": None},  # Every minute
            "apply_improvements": {"interval": 300, "last_run": None},  # Every 5 minutes
            "optimize_tools": {"interval": 600, "last_run": None},  # Every 10 minutes
            "evolutionary_improvement": {"interval": 900, "last_run": None},  # Every 15 minutes
            "cleanup": {"interval": 3600, "last_run": None},  # Every hour
            "deep_analysis": {"interval": 7200, "last_run": None}  # Every 2 hours
        }
        
        # Performance tracking
        self.performance_history = []
        self.optimization_history = []
        
        # State persistence
        self.state_file = Path.home() / ".cwmai" / "background_daemon_state.json"
        self._load_state()
        
    def _load_state(self):
        """Load previous daemon state."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Restore task last run times
                    for task_name, task_info in state.get('tasks', {}).items():
                        if task_name in self.tasks:
                            self.tasks[task_name]['last_run'] = datetime.fromisoformat(task_info['last_run']) if task_info['last_run'] else None
        except Exception as e:
            self.logger.debug(f"Could not load daemon state: {e}")
            
    def _save_state(self):
        """Save current daemon state."""
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            
            # Prepare tasks for serialization
            tasks_data = {}
            for task_name, task_info in self.tasks.items():
                tasks_data[task_name] = {
                    'interval': task_info['interval'],
                    'last_run': task_info['last_run'].isoformat() if task_info['last_run'] else None
                }
                
            with open(self.state_file, 'w') as f:
                json.dump({
                    'tasks': tasks_data,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save daemon state: {e}")
            
    async def start(self):
        """Start the background intelligence daemon."""
        self.running = True
        self.logger.info("Background Intelligence Daemon started")
        
        # Start all background tasks
        tasks = [
            asyncio.create_task(self._run_scheduled_tasks()),
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._process_improvements()),
            asyncio.create_task(self._optimize_continuously())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Daemon error: {e}")
            self.running = False
            
    async def stop(self):
        """Stop the daemon gracefully."""
        self.logger.info("Stopping Background Intelligence Daemon...")
        self.running = False
        self._save_state()
        
    async def _run_scheduled_tasks(self):
        """Run scheduled maintenance tasks."""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for task_name, task_info in self.tasks.items():
                    if task_info['last_run'] is None or \
                       (current_time - task_info['last_run']).total_seconds() >= task_info['interval']:
                        
                        self.logger.debug(f"Running scheduled task: {task_name}")
                        
                        try:
                            if task_name == "health_check":
                                await self._perform_health_check()
                            elif task_name == "apply_improvements":
                                await self._apply_pending_improvements()
                            elif task_name == "optimize_tools":
                                await self._optimize_tool_system()
                            elif task_name == "evolutionary_improvement":
                                await self._run_evolutionary_improvement()
                            elif task_name == "cleanup":
                                await self._perform_cleanup()
                            elif task_name == "deep_analysis":
                                await self._perform_deep_analysis()
                                
                            task_info['last_run'] = current_time
                            self._save_state()
                            
                        except Exception as e:
                            self.logger.error(f"Error in task {task_name}: {e}")
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Scheduled tasks error: {e}")
                await asyncio.sleep(60)
                
    async def _monitor_system_health(self):
        """Continuously monitor system health."""
        while self.running:
            try:
                # Get system stats
                stats = self.health_monitor.get_system_stats()
                
                # Record metrics
                self.health_monitor.record_metric("cpu_usage", stats["cpu_percent"])
                self.health_monitor.record_metric("memory_usage", stats["memory_percent"])
                
                # Check for issues
                issues = self.health_monitor.check_health()
                if issues:
                    await self._handle_health_issues(issues)
                    
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _process_improvements(self):
        """Process and apply system improvements."""
        while self.running:
            try:
                # Get learning summary
                summary = await self.improvement_orchestrator.get_learning_summary()
                
                # Log progress
                if summary["pending_improvements"] > 0:
                    self.logger.info(f"Processing {summary['pending_improvements']} pending improvements")
                    
                # Let the orchestrator handle improvements
                # It has its own timing logic
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Improvement processing error: {e}")
                await asyncio.sleep(120)
                
    async def _optimize_continuously(self):
        """Continuously optimize system performance."""
        while self.running:
            try:
                # Measure current performance
                start_time = time.time()
                
                # Test a sample operation
                test_result = await self._test_system_performance()
                
                response_time = time.time() - start_time
                self.health_monitor.record_metric("response_times", response_time)
                
                # If performance is degrading, trigger optimization
                if response_time > 2.0:  # More than 2 seconds
                    await self._trigger_performance_optimization()
                    
                await asyncio.sleep(300)  # Test every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(600)
                
    async def _perform_health_check(self):
        """Perform a comprehensive health check."""
        issues = self.health_monitor.check_health()
        
        if issues:
            # Prioritize critical issues
            critical_issues = [i for i in issues if i["severity"] == "critical"]
            if critical_issues:
                await self._handle_critical_issues(critical_issues)
                
        # Log health status
        self.logger.debug(f"Health check completed. Issues found: {len(issues)}")
        
    async def _apply_pending_improvements(self):
        """Apply pending system improvements."""
        applied = await self.improvement_orchestrator.apply_improvements(max_improvements=2)
        
        if applied:
            self.logger.info(f"Applied {len(applied)} improvements")
            for improvement in applied:
                self.optimization_history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": improvement["type"],
                    "description": improvement["description"]
                })
                
    async def _optimize_tool_system(self):
        """Optimize the tool system by removing unused tools and enhancing popular ones."""
        # Get tool usage statistics
        tool_stats = {}
        for tool_name, tool_def in self.tool_system.tools.items():
            tool_stats[tool_name] = {
                "usage_count": tool_def.usage_count,
                "success_rate": tool_def.success_count / max(tool_def.usage_count, 1),
                "last_used": tool_def.last_used
            }
            
        # Remove tools not used in the last 30 days
        current_time = datetime.now(timezone.utc)
        tools_to_remove = []
        
        for tool_name, stats in tool_stats.items():
            if stats["last_used"] and (current_time - stats["last_used"]).days > 30:
                if stats["usage_count"] < 5:  # Rarely used
                    tools_to_remove.append(tool_name)
                    
        # Log optimization
        if tools_to_remove:
            self.logger.info(f"Removing {len(tools_to_remove)} unused tools")
            
    async def _run_evolutionary_improvement(self):
        """Run evolutionary improvement cycle."""
        try:
            # Initialize population if needed
            if not self.evolutionary_engine.improvement_population:
                await self.evolutionary_engine.initialize_population()
                
            # Run one evolution cycle
            results = await self.evolutionary_engine.evolve_generation()
            
            if results:
                self.logger.info(f"Evolutionary improvement completed: {results}")
                
        except Exception as e:
            self.logger.error(f"Evolutionary improvement error: {e}")
            
    async def _perform_cleanup(self):
        """Clean up old logs, cache, and temporary files."""
        cleanup_stats = {
            "logs_cleaned": 0,
            "cache_cleaned": 0,
            "temp_cleaned": 0
        }
        
        # Clean old logs (older than 7 days)
        log_dir = Path("logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    if (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days > 7:
                        log_file.unlink()
                        cleanup_stats["logs_cleaned"] += 1
                except:
                    pass
                    
        self.logger.info(f"Cleanup completed: {cleanup_stats}")
        
    async def _perform_deep_analysis(self):
        """Perform deep system analysis using AI."""
        # Gather comprehensive system state
        # Convert deques to lists for JSON serialization
        health_metrics = {}
        for key, value in self.health_monitor.metrics.items():
            if isinstance(value, deque):
                health_metrics[key] = list(value)
            elif isinstance(value, defaultdict):
                health_metrics[key] = dict(value)
            else:
                health_metrics[key] = value
                
        system_state = {
            "health_metrics": health_metrics,
            "performance_history": self.performance_history[-50:],
            "optimization_history": self.optimization_history[-20:],
            "tool_statistics": self._get_tool_statistics(),
            "learning_summary": await self.improvement_orchestrator.get_learning_summary()
        }
        
        # Ask AI for insights
        prompt = f"""Analyze this system state and provide insights for improvement:

System State:
{json.dumps(system_state, indent=2, default=str)}

Provide specific, actionable recommendations for:
1. Performance improvements
2. Tool optimizations
3. Error reduction strategies
4. User experience enhancements

Format as JSON with 'recommendations' array.
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(
                prompt,
                model="gemini"
            )
            
            if response.get('success') and response.get('content'):
                # Extract and apply recommendations
                content = response['content']
                json_match = content.find('{')
                if json_match != -1:
                    recommendations = json.loads(content[json_match:])
                    await self._apply_ai_recommendations(recommendations)
                    
        except Exception as e:
            self.logger.error(f"Deep analysis error: {e}")
            
    async def _test_system_performance(self) -> Dict[str, Any]:
        """Test system performance with a sample operation."""
        try:
            # Simple test: list tools
            start = time.time()
            tools = self.tool_system.get_tools_for_ai()
            duration = time.time() - start
            
            return {
                "success": True,
                "operation": "list_tools",
                "duration": duration,
                "tool_count": len(tools)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _trigger_performance_optimization(self):
        """Trigger performance optimization when slowdown detected."""
        self.logger.warning("Performance degradation detected, triggering optimization")
        
        # Clear caches
        if hasattr(self.tool_system, '_tool_cache'):
            self.tool_system._tool_cache.clear()
            
        # Reduce concurrent operations
        if hasattr(self, 'max_concurrent'):
            self.max_concurrent = max(1, self.max_concurrent - 1)
            
    async def _handle_health_issues(self, issues: List[Dict[str, Any]]):
        """Handle detected health issues."""
        for issue in issues:
            if issue["type"] == "high_memory":
                # Trigger garbage collection
                import gc
                gc.collect()
                
            elif issue["type"] == "high_error_rate":
                # Increase logging for debugging
                logging.getLogger().setLevel(logging.DEBUG)
                
    async def _handle_critical_issues(self, issues: List[Dict[str, Any]]):
        """Handle critical system issues."""
        self.logger.error(f"Critical issues detected: {issues}")
        
        # Implement emergency measures
        for issue in issues:
            if issue["type"] == "high_error_rate":
                # Disable problematic features temporarily
                self.logger.warning("Disabling non-essential features due to high error rate")
                
    def _get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics."""
        total_tools = len(self.tool_system.tools)
        total_usage = sum(t.usage_count for t in self.tool_system.tools.values())
        
        # Find most and least used tools
        tool_usage = [(name, tool.usage_count) for name, tool in self.tool_system.tools.items()]
        tool_usage.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_tools": total_tools,
            "total_usage": total_usage,
            "most_used": tool_usage[:5] if tool_usage else [],
            "least_used": tool_usage[-5:] if len(tool_usage) > 5 else []
        }
        
    async def _apply_ai_recommendations(self, recommendations: Dict[str, Any]):
        """Apply recommendations from AI analysis."""
        if "recommendations" in recommendations:
            for rec in recommendations["recommendations"]:
                self.logger.info(f"Applying recommendation: {rec.get('title', 'Unknown')}")
                # Store recommendations for manual review
                # Actual implementation would depend on recommendation type


async def start_background_daemon():
    """Start the background intelligence daemon."""
    daemon = BackgroundIntelligenceDaemon()
    
    try:
        await daemon.start()
    except KeyboardInterrupt:
        await daemon.stop()
        

if __name__ == "__main__":
    # Run the daemon
    asyncio.run(start_background_daemon())