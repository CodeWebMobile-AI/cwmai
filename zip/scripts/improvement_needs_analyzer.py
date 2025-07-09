"""
Improvement Needs Analyzer

Analyzes the system to determine what improvements would be most valuable.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import ast
import logging
from datetime import datetime, timezone, timedelta

from scripts.state_manager import StateManager
from scripts.tool_health_checker import ToolHealthChecker
from scripts.code_analyzer import CodeAnalyzer


class ImprovementNeedsAnalyzer:
    """Analyzes system to identify high-value improvement opportunities."""
    
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.state_manager = state_manager
        self.health_checker = ToolHealthChecker()
        self.code_analyzer = CodeAnalyzer()
        
    async def analyze_improvement_needs(self) -> List[Dict[str, Any]]:
        """
        Analyze the system to identify improvement opportunities.
        
        Returns:
            List of improvement opportunities ranked by priority
        """
        needs = []
        
        # 1. Analyze tool health
        health_needs = await self._analyze_tool_health()
        needs.extend(health_needs)
        
        # 2. Analyze workflow gaps
        workflow_needs = await self._analyze_workflow_gaps()
        needs.extend(workflow_needs)
        
        # 3. Analyze error patterns
        error_needs = await self._analyze_error_patterns()
        needs.extend(error_needs)
        
        # 4. Analyze performance bottlenecks
        perf_needs = await self._analyze_performance()
        needs.extend(perf_needs)
        
        # 5. Analyze missing capabilities
        capability_needs = await self._analyze_missing_capabilities()
        needs.extend(capability_needs)
        
        # 6. Analyze code quality issues
        quality_needs = await self._analyze_code_quality()
        needs.extend(quality_needs)
        
        # Sort by priority (descending)
        needs.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return needs[:10]  # Top 10 needs
        
    async def _analyze_tool_health(self) -> List[Dict[str, Any]]:
        """Analyze tool health to find improvement opportunities."""
        needs = []
        
        try:
            # Run health check
            health_report = await self.health_checker.run_health_check()
            
            # Analyze issues
            issues_by_type = defaultdict(list)
            for issue in health_report.get('issues', []):
                issues_by_type[issue['type']].append(issue)
                
            # High priority: Tools with critical issues
            critical_tools = set()
            for issue in health_report.get('issues', []):
                if issue.get('severity') == 'high' and issue.get('status') != 'fixed':
                    critical_tools.add(issue.get('tool'))
                    
            if critical_tools:
                needs.append({
                    'type': 'error_fix',
                    'category': 'tool_health',
                    'priority': 0.9,
                    'description': f'Fix critical issues in {len(critical_tools)} tools',
                    'details': {
                        'tools': list(critical_tools),
                        'issue_types': list(issues_by_type.keys())
                    }
                })
                
            # Medium priority: Tools with repeated issues
            if len(issues_by_type.get('null_safety', [])) > 5:
                needs.append({
                    'type': 'optimization',
                    'category': 'null_safety',
                    'priority': 0.7,
                    'description': 'Implement comprehensive null safety checks',
                    'details': {
                        'affected_tools': len(issues_by_type['null_safety']),
                        'pattern': 'NoneType errors'
                    }
                })
                
        except Exception as e:
            self.logger.error(f"Error analyzing tool health: {e}")
            
        return needs
        
    async def _analyze_workflow_gaps(self) -> List[Dict[str, Any]]:
        """Analyze missing or inefficient workflows."""
        needs = []
        
        state = self.state_manager.get_state()
        workflows = state.get('workflows', [])
        repositories = state.get('projects', {})
        
        # Check for missing critical workflows
        existing_workflow_types = {w.get('type', '') for w in workflows if isinstance(w, dict)}
        
        critical_workflows = {
            'security_scan': 'Regular security vulnerability scanning',
            'performance_monitor': 'Continuous performance monitoring',
            'dependency_update': 'Automated dependency updates',
            'code_review': 'Automated code review process',
            'backup': 'Regular backup and recovery',
            'deployment': 'Automated deployment pipeline'
        }
        
        for workflow_type, description in critical_workflows.items():
            if workflow_type not in existing_workflow_types:
                needs.append({
                    'type': 'workflow',
                    'category': 'missing_workflow',
                    'priority': 0.8,
                    'description': f'Create {description} workflow',
                    'details': {
                        'workflow_type': workflow_type,
                        'repositories': len(repositories)
                    }
                })
                
        # Check for repository-specific needs
        repos_without_ci = []
        for repo_name, repo_data in repositories.items():
            # Simple heuristic: repos with many issues might need better automation
            if repo_data.get('metrics', {}).get('issues_open', 0) > 10:
                repos_without_ci.append(repo_name)
                
        if repos_without_ci:
            needs.append({
                'type': 'workflow',
                'category': 'automation',
                'priority': 0.75,
                'description': 'Create issue automation workflow',
                'details': {
                    'repositories': repos_without_ci[:3],  # Top 3
                    'total_issues': sum(repositories[r]['metrics']['issues_open'] for r in repos_without_ci)
                }
            })
            
        return needs
        
    async def _analyze_error_patterns(self) -> List[Dict[str, Any]]:
        """Analyze error patterns from logs and state."""
        needs = []
        
        # Check learning history for patterns
        state = self.state_manager.get_state()
        learning_history = state.get('learning_history', [])
        
        # All entries show 0 improvements - this is a problem!
        if all(entry.get('analysis_summary', {}).get('improvement_count', 0) == 0 
               for entry in learning_history[-10:]):
            needs.append({
                'type': 'error_fix',
                'category': 'tracking',
                'priority': 0.95,
                'description': 'Fix improvement tracking system',
                'details': {
                    'issue': 'No improvements being tracked',
                    'entries_checked': len(learning_history[-10:])
                }
            })
            
        # Check for error patterns in logs
        log_path = Path("logs")
        if log_path.exists():
            recent_errors = await self._analyze_log_errors(log_path)
            if recent_errors:
                needs.append({
                    'type': 'error_fix',
                    'category': 'recurring_errors',
                    'priority': 0.85,
                    'description': 'Fix recurring system errors',
                    'details': recent_errors
                })
                
        return needs
        
    async def _analyze_performance(self) -> List[Dict[str, Any]]:
        """Analyze system performance bottlenecks."""
        needs = []
        
        state = self.state_manager.get_state()
        metrics = state.get('metrics', {})
        
        # Check resource efficiency
        if metrics.get('resource_efficiency', 1.0) < 0.75:
            needs.append({
                'type': 'optimization',
                'category': 'performance',
                'priority': 0.7,
                'description': 'Optimize resource usage',
                'details': {
                    'current_efficiency': metrics.get('resource_efficiency'),
                    'cpu_usage': metrics.get('cpu_usage'),
                    'memory_usage': metrics.get('memory_usage')
                }
            })
            
        # Check for slow tools
        # This would require timing data which we don't have yet
        # But we can add timing to tools as an improvement!
        needs.append({
            'type': 'tool',
            'category': 'monitoring',
            'priority': 0.6,
            'description': 'Create tool performance monitor',
            'details': {
                'purpose': 'Track execution time of all tools',
                'features': ['timing', 'memory usage', 'error rates']
            }
        })
        
        return needs
        
    async def _analyze_missing_capabilities(self) -> List[Dict[str, Any]]:
        """Identify missing system capabilities."""
        needs = []
        
        # Check existing tools
        tools_dir = Path("scripts/custom_tools")
        existing_tools = {f.stem for f in tools_dir.glob("*.py")}
        
        # Essential tools that should exist
        essential_tools = {
            'database_manager': 'Manage database connections and queries',
            'cache_manager': 'Handle caching for performance',
            'rate_limiter': 'Prevent API rate limit issues',
            'notification_system': 'Send alerts and notifications',
            'backup_manager': 'Handle system backups',
            'metrics_collector': 'Collect detailed system metrics',
            'api_monitor': 'Monitor external API health',
            'data_validator': 'Validate data integrity'
        }
        
        for tool_name, description in essential_tools.items():
            if tool_name not in existing_tools:
                needs.append({
                    'type': 'tool',
                    'category': 'missing_capability',
                    'priority': 0.65,
                    'description': f'Create {tool_name}',
                    'details': {
                        'purpose': description,
                        'tool_name': tool_name
                    }
                })
                
        return needs
        
    async def _analyze_code_quality(self) -> List[Dict[str, Any]]:
        """Analyze code quality issues."""
        needs = []
        
        # Sample a few Python files
        sample_files = list(Path("scripts").glob("*.py"))[:10]
        
        quality_issues = defaultdict(int)
        for file_path in sample_files:
            try:
                analysis = await self.code_analyzer.analyze_file(str(file_path))
                for issue in analysis.get('issues', []):
                    quality_issues[issue['type']] += 1
            except:
                pass
                
        if quality_issues:
            top_issue = max(quality_issues.items(), key=lambda x: x[1])
            needs.append({
                'type': 'optimization',
                'category': 'code_quality',
                'priority': 0.5,
                'description': f'Fix {top_issue[0]} issues across codebase',
                'details': {
                    'issue_counts': dict(quality_issues),
                    'files_analyzed': len(sample_files)
                }
            })
            
        return needs
        
    async def _analyze_log_errors(self, log_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze error patterns in logs."""
        error_patterns = defaultdict(int)
        
        # Read recent log files
        log_files = sorted(log_path.glob("*.log"), key=lambda x: x.stat().st_mtime)[-3:]
        
        for log_file in log_files:
            try:
                content = log_file.read_text()
                for line in content.split('\n')[-1000:]:  # Last 1000 lines
                    if 'ERROR' in line or 'Exception' in line:
                        # Extract error type
                        if 'AttributeError' in line:
                            error_patterns['AttributeError'] += 1
                        elif 'KeyError' in line:
                            error_patterns['KeyError'] += 1
                        elif 'TypeError' in line:
                            error_patterns['TypeError'] += 1
                        elif 'ImportError' in line:
                            error_patterns['ImportError'] += 1
                        else:
                            error_patterns['Other'] += 1
            except:
                pass
                
        if error_patterns:
            return {
                'error_counts': dict(error_patterns),
                'log_files_analyzed': len(log_files),
                'most_common': max(error_patterns.items(), key=lambda x: x[1])[0]
            }
            
        return None
        
    def suggest_improvement(self, need: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest a specific improvement based on an identified need.
        
        Returns:
            Improvement specification
        """
        imp_type = need['type']
        category = need['category']
        
        if imp_type == 'tool':
            if category == 'monitoring':
                return {
                    'type': 'tool',
                    'name': need['details']['tool_name'] or 'performance_monitor',
                    'description': need['description'],
                    'implementation': 'Create tool with timing decorators and metrics collection'
                }
            elif category == 'missing_capability':
                return {
                    'type': 'tool',
                    'name': need['details']['tool_name'],
                    'description': need['details']['purpose'],
                    'implementation': f'Implement {need["details"]["purpose"]}'
                }
                
        elif imp_type == 'workflow':
            return {
                'type': 'workflow',
                'name': f'{category}_workflow',
                'description': need['description'],
                'steps': self._generate_workflow_steps(category, need['details'])
            }
            
        elif imp_type == 'error_fix':
            return {
                'type': 'error_fix',
                'target': need['details'].get('tools', ['system'])[0],
                'issue': need['details'].get('issue', category),
                'fix_approach': self._determine_fix_approach(category, need['details'])
            }
            
        elif imp_type == 'optimization':
            return {
                'type': 'optimization',
                'target': category,
                'description': need['description'],
                'approach': self._determine_optimization_approach(category, need['details'])
            }
            
        return need
        
    def _generate_workflow_steps(self, category: str, details: Dict) -> List[Dict]:
        """Generate workflow steps based on category."""
        if category == 'security_scan':
            return [
                {'tool': 'security_vulnerability_scanner', 'params': {'depth': 'full'}},
                {'tool': 'create_issue', 'params': {'labels': ['security']}}
            ]
        elif category == 'automation':
            return [
                {'tool': 'get_repository_issues', 'params': {'state': 'open'}},
                {'tool': 'analyze_issue_patterns', 'params': {}},
                {'tool': 'auto_label_issues', 'params': {}}
            ]
        else:
            return [
                {'tool': 'analyze_repository', 'params': {}},
                {'tool': 'generate_report', 'params': {}}
            ]
            
    def _determine_fix_approach(self, category: str, details: Dict) -> str:
        """Determine approach for fixing issues."""
        if category == 'tracking':
            return 'Update state tracking to properly count improvements'
        elif category == 'null_safety':
            return 'Add comprehensive null checks to affected functions'
        elif category == 'tool_health':
            return 'Fix method calls and add error handling'
        else:
            return 'Analyze and fix root cause'
            
    def _determine_optimization_approach(self, category: str, details: Dict) -> str:
        """Determine optimization approach."""
        if category == 'performance':
            return 'Implement caching and optimize algorithms'
        elif category == 'code_quality':
            return 'Refactor to reduce complexity and improve maintainability'
        else:
            return 'Analyze bottlenecks and optimize'