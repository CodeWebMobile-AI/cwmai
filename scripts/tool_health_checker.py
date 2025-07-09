"""
Tool Health Checker

Proactive system that periodically checks all tools for potential issues
and reports problems before they cause failures.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime, timezone

from scripts.logger import get_logger
from scripts.state_manager import StateManager
from scripts.tool_calling_system import ToolCallingSystem
from scripts.tool_error_monitor import ToolErrorMonitor


class ToolHealthChecker:
    """Proactively checks tools for common issues."""
    
    def __init__(self):
        self.logger = get_logger('ToolHealthChecker')
        self.state_manager = StateManager()
        self.tool_system = ToolCallingSystem()
        self.error_monitor = ToolErrorMonitor()
        self.tools_dir = Path("scripts/custom_tools")
        
        # Common issues to check for
        self.common_issues = {
            'null_safety': self._check_null_safety,
            'method_existence': self._check_method_existence,
            'import_validity': self._check_import_validity,
            'parameter_types': self._check_parameter_types,
            'error_handling': self._check_error_handling,
        }
        
    async def run_health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check on all tools."""
        print("Starting tool health check...")
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_tools': 0,
            'healthy_tools': 0,
            'issues_found': [],
            'auto_fixed': 0,
            'recommendations': []
        }
        
        # Get all tools
        all_tools = list(self.tools_dir.glob("*.py"))
        results['total_tools'] = len(all_tools)
        
        for tool_path in all_tools:
            if tool_path.name.startswith('_'):
                continue
                
            tool_name = tool_path.stem
            tool_issues = await self._check_tool_health(tool_name, tool_path)
            
            if not tool_issues:
                results['healthy_tools'] += 1
            else:
                results['issues_found'].extend(tool_issues)
                
                # Try to auto-fix issues
                for issue in tool_issues:
                    if issue['auto_fixable']:
                        fixed = await self._attempt_auto_fix(tool_name, tool_path, issue)
                        if fixed:
                            results['auto_fixed'] += 1
                            issue['status'] = 'fixed'
                        else:
                            issue['status'] = 'failed_to_fix'
                    else:
                        issue['status'] = 'manual_fix_required'
                        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['issues_found'])
        
        # Save results
        self._save_health_report(results)
        
        return results
        
    async def _check_tool_health(self, tool_name: str, tool_path: Path) -> List[Dict[str, Any]]:
        """Check a single tool for common issues."""
        issues = []
        
        try:
            content = tool_path.read_text()
            
            for issue_type, check_func in self.common_issues.items():
                issue = await check_func(tool_name, content)
                if issue:
                    issues.append({
                        'tool': tool_name,
                        'type': issue_type,
                        'description': issue['description'],
                        'severity': issue.get('severity', 'medium'),
                        'auto_fixable': issue.get('auto_fixable', False),
                        'fix_action': issue.get('fix_action', None),
                        'line_numbers': issue.get('line_numbers', [])
                    })
                    
        except Exception as e:
            issues.append({
                'tool': tool_name,
                'type': 'read_error',
                'description': f"Failed to read tool file: {e}",
                'severity': 'high',
                'auto_fixable': False
            })
            
        return issues
        
    async def _check_null_safety(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """Check for potential NoneType errors."""
        # Pattern to find potential null pointer issues
        patterns = [
            # Direct attribute access without null check
            (r'(\w+)\["(\w+)"\]\.(\w+)\(', "Direct dictionary access without null check"),
            # Chained attribute access without checks
            (r'(\w+)\.(\w+)\.(\w+)\.(\w+)', "Long chain of attribute access without null checks"),
            # Missing get() with default
            (r'(\w+)\["(\w+)"\]\s*\.lower\(\)', "String method on dictionary access without null check"),
        ]
        
        issues_found = []
        lines = content.split('\n')
        
        for pattern, description in patterns:
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    # Check if there's a guard clause nearby
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 2)
                    context = lines[context_start:context_end]
                    
                    # Simple heuristic: check if there's an if statement checking for None
                    has_guard = any('if ' in ctx_line and ('is not None' in ctx_line or 'and ' in ctx_line) 
                                   for ctx_line in context)
                    
                    if not has_guard:
                        issues_found.append({
                            'line': i + 1,
                            'code': line.strip(),
                            'issue': description
                        })
                        
        if issues_found:
            return {
                'description': f"Found {len(issues_found)} potential NoneType errors",
                'severity': 'high',
                'auto_fixable': True,
                'fix_action': 'add_null_checks',
                'line_numbers': [issue['line'] for issue in issues_found],
                'details': issues_found
            }
            
        return None
        
    async def _check_method_existence(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """Check for calls to non-existent methods."""
        # Common incorrect method calls
        incorrect_methods = {
            'HTTPAIClient': {
                'generate': 'generate_enhanced_response',
                'get_response': 'generate_enhanced_response'
            },
            'TaskManager': {
                'create_task': 'add_task',
                'update_task': 'update_task_status'
            },
            'StateManager': {
                'get_state': 'load_state',
                'set_state': 'save_state'
            }
        }
        
        issues_found = []
        
        for class_name, methods in incorrect_methods.items():
            if class_name in content:
                for wrong_method, correct_method in methods.items():
                    pattern = rf'\.{wrong_method}\s*\('
                    matches = list(re.finditer(pattern, content))
                    
                    if matches:
                        lines = content.split('\n')
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issues_found.append({
                                'line': line_num,
                                'wrong_method': wrong_method,
                                'correct_method': correct_method,
                                'class': class_name
                            })
                            
        if issues_found:
            return {
                'description': f"Found {len(issues_found)} incorrect method calls",
                'severity': 'high',
                'auto_fixable': True,
                'fix_action': 'update_method_calls',
                'line_numbers': [issue['line'] for issue in issues_found],
                'details': issues_found
            }
            
        return None
        
    async def _check_import_validity(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """Check for invalid or outdated imports."""
        # Known import mappings
        import_mappings = {
            'scripts.tool_executor': 'scripts.tool_calling_system',
            'scripts.ai_manager': 'scripts.ai_brain',
            'scripts.task_tracker': 'scripts.task_manager',
        }
        
        issues_found = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for old_import, new_import in import_mappings.items():
                if old_import in line and line.strip().startswith(('import', 'from')):
                    issues_found.append({
                        'line': i + 1,
                        'old_import': old_import,
                        'new_import': new_import,
                        'code': line.strip()
                    })
                    
        if issues_found:
            return {
                'description': f"Found {len(issues_found)} outdated imports",
                'severity': 'high',
                'auto_fixable': True,
                'fix_action': 'update_imports',
                'line_numbers': [issue['line'] for issue in issues_found],
                'details': issues_found
            }
            
        return None
        
    async def _check_parameter_types(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """Check for incorrect parameter types in function calls."""
        # Common parameter issues
        issues_found = []
        
        # Check for string parameters that should be Path objects
        path_pattern = r'Path\("([^"]+)"\)'
        if 'pathlib' in content and path_pattern in content:
            # Check if Path is imported
            if 'from pathlib import Path' not in content and 'import pathlib' not in content:
                issues_found.append({
                    'type': 'missing_import',
                    'description': 'Using Path without importing it',
                    'fix': 'from pathlib import Path'
                })
                
        if issues_found:
            return {
                'description': f"Found {len(issues_found)} parameter type issues",
                'severity': 'medium',
                'auto_fixable': True,
                'fix_action': 'fix_parameter_types',
                'details': issues_found
            }
            
        return None
        
    async def _check_error_handling(self, tool_name: str, content: str) -> Optional[Dict[str, Any]]:
        """Check for proper error handling."""
        # Check if the main function has try-except
        main_func_pattern = r'async def ' + tool_name + r'\s*\([^)]*\)\s*:'
        match = re.search(main_func_pattern, content)
        
        if match:
            # Find the function body
            func_start = match.end()
            lines = content[func_start:].split('\n')
            
            # Simple check: is there a try-except in the first level of the function?
            has_error_handling = False
            indent_level = None
            
            for line in lines:
                if line.strip() == '':
                    continue
                    
                # Determine base indent
                if indent_level is None and line.strip():
                    indent_level = len(line) - len(line.lstrip())
                    
                # Check current indent
                current_indent = len(line) - len(line.lstrip())
                
                # If we're back at function level or less, we're done with the function
                if current_indent <= 0 and line.strip():
                    break
                    
                if line.strip().startswith('try:'):
                    has_error_handling = True
                    break
                    
            if not has_error_handling:
                return {
                    'description': 'Main function lacks error handling',
                    'severity': 'medium',
                    'auto_fixable': False,
                    'recommendation': 'Add try-except block to handle errors gracefully'
                }
                
        return None
        
    async def _attempt_auto_fix(self, tool_name: str, tool_path: Path, issue: Dict[str, Any]) -> bool:
        """Attempt to automatically fix an issue."""
        fix_action = issue.get('fix_action')
        
        if fix_action == 'add_null_checks':
            # Use the error monitor's null safety fix
            return await self.error_monitor._fix_nonetype_error(tool_path, 'lower')
            
        elif fix_action == 'update_method_calls':
            # Fix incorrect method calls
            for detail in issue.get('details', []):
                success = await self.error_monitor._update_method_calls(
                    tool_path,
                    detail['wrong_method'],
                    detail['correct_method']
                )
                if not success:
                    return False
            return True
            
        elif fix_action == 'update_imports':
            # Fix outdated imports
            for detail in issue.get('details', []):
                success = await self.error_monitor._update_tool_imports(
                    tool_path,
                    detail['old_import'].replace('scripts.', ''),
                    detail['new_import'].replace('scripts.', '')
                )
                if not success:
                    return False
            return True
            
        elif fix_action == 'fix_parameter_types':
            # Add missing imports
            for detail in issue.get('details', []):
                if detail.get('type') == 'missing_import':
                    success = await self.error_monitor._add_missing_import(
                        tool_path,
                        detail['fix']
                    )
                    if not success:
                        return False
            return True
            
        return False
        
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on found issues."""
        recommendations = []
        
        # Count issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue['type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
        if issue_types.get('null_safety', 0) > 3:
            recommendations.append(
                "Consider implementing a standard null-safety pattern across all tools"
            )
            
        if issue_types.get('method_existence', 0) > 0:
            recommendations.append(
                "Update tool documentation to reflect current API methods"
            )
            
        if issue_types.get('import_validity', 0) > 0:
            recommendations.append(
                "Run a full import audit to update all outdated module references"
            )
            
        if len(issues) > 10:
            recommendations.append(
                "Consider running the health check more frequently to catch issues early"
            )
            
        return recommendations
        
    def _save_health_report(self, results: Dict[str, Any]):
        """Save the health check report."""
        report_path = Path("scripts/tool_health_reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"health_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Health report saved to {report_file}")
        
        # Also update the state with the latest results
        state = self.state_manager.load_state()
        state['latest_health_check'] = results
        self.state_manager.save_state()


async def run_periodic_health_check(interval_minutes: int = 30):
    """Run health checks periodically."""
    checker = ToolHealthChecker()
    logger = get_logger('PeriodicHealthCheck')
    
    while True:
        try:
            logger.info("Running periodic health check...")
            results = await checker.run_health_check()
            
            logger.info(f"Health check complete: {results['healthy_tools']}/{results['total_tools']} tools healthy")
            
            if results['auto_fixed'] > 0:
                logger.info(f"Auto-fixed {results['auto_fixed']} issues")
                
            if results['issues_found']:
                manual_fixes = [i for i in results['issues_found'] if i['status'] == 'manual_fix_required']
                if manual_fixes:
                    logger.warning(f"{len(manual_fixes)} issues require manual intervention")
                    
        except Exception as e:
            logger.error(f"Error in periodic health check: {e}")
            
        # Wait for next check
        await asyncio.sleep(interval_minutes * 60)


if __name__ == "__main__":
    # Run a one-time health check
    async def main():
        checker = ToolHealthChecker()
        results = await checker.run_health_check()
        
        print("\n=== Tool Health Check Results ===")
        print(f"Total tools: {results['total_tools']}")
        print(f"Healthy tools: {results['healthy_tools']}")
        print(f"Issues found: {len(results['issues_found'])}")
        print(f"Auto-fixed: {results['auto_fixed']}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"- {rec}")
                
        if results['issues_found']:
            print("\nRemaining issues:")
            for issue in results['issues_found']:
                if issue['status'] != 'fixed':
                    print(f"- {issue['tool']}: {issue['description']} ({issue['status']})")
    
    asyncio.run(main())