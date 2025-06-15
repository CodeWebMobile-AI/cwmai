"""
Workflow Executor for Production Orchestrator

Executes individual workflow components with proper error handling and reporting.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import logging

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_config import ProductionConfig


class WorkflowExecutor:
    """Executes workflow components with error handling and reporting."""
    
    def __init__(self, config: ProductionConfig):
        """Initialize workflow executor.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.execution_history: List[Dict[str, Any]] = []
        
    async def execute_task_cycle(self) -> Dict[str, Any]:
        """Execute the task management cycle.
        
        Returns:
            Execution results
        """
        self.logger.info("Starting task management cycle")
        start_time = datetime.now(timezone.utc)
        results = {
            'cycle': 'task_management',
            'start_time': start_time.isoformat(),
            'operations': [],
            'errors': []
        }
        
        try:
            # 1. Analyze existing tasks
            analyze_result = await self._run_script('task_analyzer.py')
            results['operations'].append({
                'name': 'analyze_tasks',
                'success': analyze_result[0],
                'output': analyze_result[1][:500] if analyze_result[0] else analyze_result[1]
            })
            
            # 2. Generate new tasks
            env = {
                'TASK_FOCUS': 'auto',
                'MAX_TASKS': '5'
            }
            generate_result = await self._run_script('task_manager.py', ['generate'], env)
            results['operations'].append({
                'name': 'generate_tasks',
                'success': generate_result[0],
                'output': generate_result[1][:500] if generate_result[0] else generate_result[1]
            })
            
            # 3. Review completed tasks
            review_result = await self._run_script('task_manager.py', ['review'])
            results['operations'].append({
                'name': 'review_tasks',
                'success': review_result[0],
                'output': review_result[1][:500] if review_result[0] else review_result[1]
            })
            
            # 4. Update task priorities
            prioritize_result = await self._run_script('task_manager.py', ['prioritize'])
            results['operations'].append({
                'name': 'prioritize_tasks',
                'success': prioritize_result[0],
                'output': prioritize_result[1][:500] if prioritize_result[0] else prioritize_result[1]
            })
            
            # 5. Create task report
            report_result = await self._run_script('task_manager.py', ['report'])
            results['operations'].append({
                'name': 'create_report',
                'success': report_result[0],
                'output': report_result[1][:500] if report_result[0] else report_result[1]
            })
            
        except Exception as e:
            self.logger.error(f"Error in task cycle: {e}")
            results['errors'].append(str(e))
            
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Store in history
        self.execution_history.append(results)
        
        return results
        
    async def execute_main_cycle(self) -> Dict[str, Any]:
        """Execute the main AI cycle.
        
        Returns:
            Execution results
        """
        self.logger.info("Starting main AI cycle")
        start_time = datetime.now(timezone.utc)
        results = {
            'cycle': 'main_ai_cycle',
            'start_time': start_time.isoformat(),
            'operations': [],
            'errors': []
        }
        
        try:
            # 1. Gather context using AI brain
            # Note: Context gathering is now integrated into main_cycle.py
            # which uses the AI brain's gather_context method internally
            results['operations'].append({
                'name': 'gather_context',
                'success': True,
                'output': 'Context gathering integrated into main cycle via AI brain'
            })
            
            # 2. Execute main cycle
            env = {'FORCE_ACTION': 'auto'}
            main_result = await self._run_script('main_cycle.py', env=env)
            results['operations'].append({
                'name': 'main_cycle',
                'success': main_result[0],
                'output': main_result[1][:500] if main_result[0] else main_result[1]
            })
            
            # 3. Create performance report
            report_result = await self._run_script('create_report.py')
            results['operations'].append({
                'name': 'performance_report',
                'success': report_result[0],
                'output': report_result[1][:500] if report_result[0] else report_result[1]
            })
            
        except Exception as e:
            self.logger.error(f"Error in main cycle: {e}")
            results['errors'].append(str(e))
            
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Store in history
        self.execution_history.append(results)
        
        return results
        
    async def execute_god_mode_cycle(self, intensity: str = 'balanced') -> Dict[str, Any]:
        """Execute the God Mode cycle.
        
        Args:
            intensity: God mode intensity level
            
        Returns:
            Execution results
        """
        self.logger.info(f"Starting God Mode cycle (intensity: {intensity})")
        start_time = datetime.now(timezone.utc)
        results = {
            'cycle': 'god_mode',
            'start_time': start_time.isoformat(),
            'intensity': intensity,
            'operations': [],
            'errors': []
        }
        
        try:
            # Use the existing dynamic god mode controller
            from ai_brain_factory import AIBrainFactory
            from dynamic_god_mode_controller import DynamicGodModeController
            from god_mode_controller import GodModeConfig, IntensityLevel
            
            # Initialize AI Brain
            ai_brain = AIBrainFactory.create_for_production()
            
            # Create configuration
            config = GodModeConfig(
                intensity=IntensityLevel(intensity),
                enable_self_modification=self.config.enable_self_modification,
                enable_multi_repo=True,
                enable_predictive=True,
                enable_quantum=False,
                max_parallel_operations=self.config.max_parallel_operations,
                safety_threshold=self.config.safety_threshold
            )
            
            # Initialize controller
            controller = DynamicGodModeController(config, ai_brain)
            
            # Run one cycle
            cycle_result = await controller.run_god_mode_cycle()
            
            results['operations'] = cycle_result.get('operations', [])
            results['errors'] = cycle_result.get('errors', [])
            results['metrics'] = cycle_result.get('metrics', {})
            results['learnings'] = cycle_result.get('learnings', [])
            results['recommendations'] = cycle_result.get('recommendations', {})
            
        except Exception as e:
            self.logger.error(f"Error in God Mode cycle: {e}")
            results['errors'].append(str(e))
            
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Store in history
        self.execution_history.append(results)
        
        return results
        
    async def execute_monitoring_cycle(self) -> Dict[str, Any]:
        """Execute the monitoring cycle.
        
        Returns:
            Execution results
        """
        self.logger.info("Starting monitoring cycle")
        start_time = datetime.now(timezone.utc)
        results = {
            'cycle': 'monitoring',
            'start_time': start_time.isoformat(),
            'operations': [],
            'errors': [],
            'health_status': {}
        }
        
        try:
            # 1. Check system health
            health_check = await self._check_system_health()
            results['health_status'] = health_check
            results['operations'].append({
                'name': 'health_check',
                'success': health_check.get('healthy', False),
                'output': health_check
            })
            
            # 2. Analyze performance
            performance = await self._analyze_performance()
            results['operations'].append({
                'name': 'performance_analysis',
                'success': True,
                'output': performance
            })
            
            # 3. Create monitoring report
            report = await self._create_monitoring_report(health_check, performance)
            results['operations'].append({
                'name': 'monitoring_report',
                'success': True,
                'output': report
            })
            
            # 4. Update dashboard if configured
            if os.path.exists('scripts/update_task_dashboard.py'):
                dashboard_result = await self._run_script('update_task_dashboard.py')
                results['operations'].append({
                    'name': 'update_dashboard',
                    'success': dashboard_result[0],
                    'output': dashboard_result[1][:500] if dashboard_result[0] else dashboard_result[1]
                })
                
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            results['errors'].append(str(e))
            
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Store in history
        self.execution_history.append(results)
        
        return results
        
    async def _run_script(self, script_name: str, args: List[str] = None, 
                         env: Dict[str, str] = None) -> Tuple[bool, str]:
        """Run a Python script with error handling.
        
        Args:
            script_name: Name of the script to run
            args: Optional command line arguments
            env: Optional environment variables
            
        Returns:
            Tuple of (success, output)
        """
        script_path = os.path.join('scripts', script_name)
        
        if not os.path.exists(script_path):
            return False, f"Script not found: {script_path}"
            
        # Prepare environment
        full_env = os.environ.copy()
        full_env.update({
            'GITHUB_TOKEN': self.config.github_token,
            'ANTHROPIC_API_KEY': self.config.anthropic_api_key,
            'OPENAI_API_KEY': self.config.openai_api_key,
            'GEMINI_API_KEY': self.config.gemini_api_key,
            'DEEPSEEK_API_KEY': self.config.deepseek_api_key
        })
        
        if env:
            full_env.update(env)
            
        # Prepare command
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
            
        try:
            # Run script
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env
            )
            
            # Wait for completion with timeout
            timeout = self.config.task_cycle.max_duration_seconds
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            # Check result
            if process.returncode == 0:
                return True, stdout.decode('utf-8')
            else:
                return False, stderr.decode('utf-8')
                
        except asyncio.TimeoutError:
            return False, f"Script timeout after {timeout} seconds"
        except Exception as e:
            return False, f"Error running script: {str(e)}"
            
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health.
        
        Returns:
            Health status
        """
        health = {
            'healthy': True,
            'checks': {}
        }
        
        # Check state files
        state_files = ['system_state.json', 'task_state.json', 'task_history.json']
        for file in state_files:
            exists = os.path.exists(file)
            health['checks'][file] = exists
            if not exists:
                health['healthy'] = False
                
        # Check API keys
        health['checks']['api_keys'] = {
            'anthropic': bool(self.config.anthropic_api_key),
            'github': bool(self.config.github_token),
            'openai': bool(self.config.openai_api_key)
        }
        
        if not self.config.anthropic_api_key:
            health['healthy'] = False
            
        # Check recent execution history
        if self.execution_history:
            recent_failures = sum(
                1 for exec in self.execution_history[-10:]
                if exec.get('errors')
            )
            health['checks']['recent_failures'] = recent_failures
            if recent_failures > 5:
                health['healthy'] = False
                
        return health
        
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance.
        
        Returns:
            Performance metrics
        """
        performance = {
            'cycle_stats': {},
            'success_rates': {},
            'average_durations': {}
        }
        
        # Analyze execution history by cycle type
        for cycle_type in ['task_management', 'main_ai_cycle', 'god_mode', 'monitoring']:
            cycle_execs = [
                exec for exec in self.execution_history
                if exec.get('cycle') == cycle_type
            ]
            
            if cycle_execs:
                # Success rate
                successes = sum(
                    1 for exec in cycle_execs
                    if not exec.get('errors')
                )
                performance['success_rates'][cycle_type] = successes / len(cycle_execs)
                
                # Average duration
                durations = [
                    exec.get('duration_seconds', 0)
                    for exec in cycle_execs
                ]
                performance['average_durations'][cycle_type] = sum(durations) / len(durations)
                
                # Cycle count
                performance['cycle_stats'][cycle_type] = len(cycle_execs)
                
        return performance
        
    async def _create_monitoring_report(self, health: Dict[str, Any], 
                                      performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring report.
        
        Args:
            health: Health check results
            performance: Performance metrics
            
        Returns:
            Monitoring report
        """
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'health': health,
            'performance': performance,
            'recommendations': []
        }
        
        # Generate recommendations
        if not health['healthy']:
            report['recommendations'].append("System health issues detected - review failures")
            
        for cycle, rate in performance.get('success_rates', {}).items():
            if rate < 0.8:
                report['recommendations'].append(
                    f"Low success rate for {cycle} ({rate:.1%}) - investigate failures"
                )
                
        return report