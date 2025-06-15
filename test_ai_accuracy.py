#!/usr/bin/env python3
"""
Test Suite for AI Accuracy
Ensures the conversational AI provides accurate information from the system
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem
from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager
from scripts.conversational_ai_assistant import ConversationalAIAssistant


class AIAccuracyTester:
    """Tests that AI responses match actual system data."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.task_manager = TaskManager()
        self.tool_system = ToolCallingSystem()
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all accuracy tests."""
        print("🧪 Starting AI Accuracy Tests\n")
        
        # Load current state
        state = self.state_manager.load_state()
        
        tests = [
            self.test_repository_count,
            self.test_repository_details,
            self.test_task_count,
            self.test_system_status,
            self.test_health_checks,
            self.test_ai_created_tools,
            self.test_edge_cases
        ]
        
        for test in tests:
            await test()
            
        # Print summary
        self._print_summary()
        
    async def test_repository_count(self):
        """Test repository counting accuracy."""
        print("📊 Testing Repository Count...")
        
        # Get actual data
        state = self.state_manager.load_state()
        actual_repos = state.get('projects', {})
        actual_count = len(actual_repos)
        
        # Get AI response
        result = await self.tool_system.call_tool('count_repositories')
        
        if result.get('success'):
            ai_data = result.get('result', {})
            ai_count = ai_data.get('total', 0)
            
            # Verify accuracy
            if ai_count == actual_count:
                self._record_pass("Repository Count", f"Correctly reported {ai_count} repositories")
            else:
                self._record_fail("Repository Count", 
                                f"AI reported {ai_count} but actual is {actual_count}")
                
            # Check breakdown accuracy
            actual_active = sum(1 for r in actual_repos.values() if not r.get('archived', False))
            actual_archived = sum(1 for r in actual_repos.values() if r.get('archived', False))
            
            ai_active = ai_data.get('breakdown', {}).get('by_status', {}).get('active', 0)
            ai_archived = ai_data.get('breakdown', {}).get('by_status', {}).get('archived', 0)
            
            if ai_active == actual_active and ai_archived == actual_archived:
                self._record_pass("Repository Status Breakdown", 
                                f"Correctly reported {ai_active} active, {ai_archived} archived")
            else:
                self._record_fail("Repository Status Breakdown",
                                f"AI: {ai_active} active, {ai_archived} archived | "
                                f"Actual: {actual_active} active, {actual_archived} archived")
        else:
            self._record_fail("Repository Count", f"Tool call failed: {result.get('error')}")
            
    async def test_repository_details(self):
        """Test repository listing accuracy."""
        print("📋 Testing Repository Details...")
        
        # Get actual data
        state = self.state_manager.load_state()
        actual_repos = state.get('projects', {})
        
        # Get AI response
        result = await self.tool_system.call_tool('get_repositories', limit=20)
        
        if result.get('success'):
            ai_repos = result.get('result', [])
            
            # Check if all repositories are listed
            ai_repo_names = {r['name'] for r in ai_repos}
            actual_repo_names = set(actual_repos.keys())
            
            missing = actual_repo_names - ai_repo_names
            extra = ai_repo_names - actual_repo_names
            
            if not missing and not extra:
                self._record_pass("Repository Listing", 
                                f"All {len(actual_repos)} repositories correctly listed")
            else:
                issues = []
                if missing:
                    issues.append(f"Missing: {missing}")
                if extra:
                    issues.append(f"Extra: {extra}")
                self._record_fail("Repository Listing", " | ".join(issues))
                
            # Verify repository details
            for ai_repo in ai_repos[:3]:  # Check first 3
                name = ai_repo['name']
                if name in actual_repos:
                    actual = actual_repos[name]
                    if (ai_repo.get('stars') == actual.get('stars', 0) and
                        ai_repo.get('language') == actual.get('language', '')):
                        self._record_pass(f"Repo Details: {name}", "Details match")
                    else:
                        self._record_fail(f"Repo Details: {name}", 
                                        f"Mismatch - AI: {ai_repo} | Actual: {actual}")
        else:
            self._record_fail("Repository Details", f"Tool call failed: {result.get('error')}")
            
    async def test_task_count(self):
        """Test task counting accuracy."""
        print("📝 Testing Task Count...")
        
        # Get actual data directly from task state
        task_state = self.task_manager.state
        tasks = list(task_state.get('tasks', {}).values())
        actual_count = len(tasks)
        
        # Count by status
        actual_by_status = {}
        for task in tasks:
            status = task.get('status', 'unknown')
            actual_by_status[status] = actual_by_status.get(status, 0) + 1
            
        # Get AI response
        result = await self.tool_system.call_tool('count_tasks')
        
        if result.get('success'):
            ai_data = result.get('result', {})
            ai_count = ai_data.get('total', 0)
            
            if ai_count == actual_count:
                self._record_pass("Task Count", f"Correctly reported {ai_count} tasks")
            else:
                self._record_fail("Task Count", 
                                f"AI reported {ai_count} but actual is {actual_count}")
                                
            # Check status breakdown
            ai_by_status = ai_data.get('counts', {})
            if ai_by_status == actual_by_status:
                self._record_pass("Task Status Breakdown", "Status counts match")
            else:
                self._record_fail("Task Status Breakdown",
                                f"AI: {ai_by_status} | Actual: {actual_by_status}")
        else:
            self._record_fail("Task Count", f"Tool call failed: {result.get('error')}")
            
    async def test_system_status(self):
        """Test system status accuracy."""
        print("💻 Testing System Status...")
        
        result = await self.tool_system.call_tool('get_system_status')
        
        if result.get('success'):
            ai_status = result.get('result', {})
            
            # Verify health status
            if 'healthy' in ai_status:
                self._record_pass("System Health Status", 
                                f"Reported health: {ai_status['healthy']}")
            else:
                self._record_fail("System Health Status", "No health status reported")
                
            # Verify task metrics match
            task_metrics = ai_status.get('tasks', {})
            task_state = self.task_manager.state
            actual_tasks = list(task_state.get('tasks', {}).values())
            actual_active = len([t for t in actual_tasks if t.get('status') == 'active'])
            actual_pending = len([t for t in actual_tasks if t.get('status') == 'pending'])
            
            if (task_metrics.get('active') == actual_active and 
                task_metrics.get('pending') == actual_pending):
                self._record_pass("System Task Metrics", "Task counts accurate")
            else:
                self._record_fail("System Task Metrics",
                                f"AI: {task_metrics} | Actual: active={actual_active}, pending={actual_pending}")
        else:
            self._record_fail("System Status", f"Tool call failed: {result.get('error')}")
            
    async def test_health_checks(self):
        """Test health check accuracy."""
        print("🏥 Testing Health Checks...")
        
        # Repository health check
        result = await self.tool_system.call_tool('repository_health_check')
        
        if result.get('success'):
            health_data = result.get('result', {})
            total_checked = health_data.get('total_checked', 0)
            
            state = self.state_manager.load_state()
            actual_repos = len(state.get('projects', {}))
            
            if total_checked == actual_repos:
                self._record_pass("Repository Health Check", 
                                f"Checked all {total_checked} repositories")
            else:
                self._record_fail("Repository Health Check",
                                f"Checked {total_checked} but have {actual_repos} repos")
        else:
            self._record_fail("Repository Health Check", f"Tool call failed: {result.get('error')}")
            
        # AI health dashboard
        result = await self.tool_system.call_tool('ai_health_dashboard')
        
        if result.get('success'):
            self._record_pass("AI Health Dashboard", "Successfully generated dashboard")
        else:
            self._record_fail("AI Health Dashboard", f"Tool call failed: {result.get('error')}")
            
    async def test_ai_created_tools(self):
        """Test AI's ability to report on self-created tools."""
        print("🤖 Testing AI Created Tools Reporting...")
        
        result = await self.tool_system.call_tool('get_tool_usage_stats')
        
        if result.get('success'):
            stats = result.get('result', {})
            ai_created = stats.get('ai_created_tools', 0)
            total_tools = stats.get('total_tools', 0)
            
            # Verify counts are reasonable
            if total_tools > 0:
                self._record_pass("Tool Statistics", 
                                f"Reported {total_tools} total tools, {ai_created} AI-created")
            else:
                self._record_fail("Tool Statistics", "No tools reported")
        else:
            self._record_fail("Tool Statistics", f"Tool call failed: {result.get('error')}")
            
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("🔧 Testing Edge Cases...")
        
        # Test with invalid parameters
        result = await self.tool_system.call_tool('get_repositories', limit=-1)
        if result.get('success'):
            self._record_pass("Invalid Parameter Handling", "Handled negative limit gracefully")
        else:
            self._record_fail("Invalid Parameter Handling", "Failed on negative limit")
            
        # Test non-existent tool behavior
        result = await self.tool_system.call_tool('totally_fake_tool')
        if 'not found' in str(result.get('error', '')).lower():
            self._record_pass("Non-existent Tool", "Correctly reported tool not found")
        else:
            self._record_fail("Non-existent Tool", "Unexpected response for fake tool")
            
    def _record_pass(self, test_name: str, message: str):
        """Record a passing test."""
        self.test_results.append({
            'name': test_name,
            'passed': True,
            'message': message
        })
        print(f"  ✅ {test_name}: {message}")
        
    def _record_fail(self, test_name: str, message: str):
        """Record a failing test."""
        self.test_results.append({
            'name': test_name,
            'passed': False,
            'message': message
        })
        print(f"  ❌ {test_name}: {message}")
        
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['name']}: {result['message']}")
                    
        # Save results to file
        with open('ai_accuracy_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': passed/total*100
                },
                'results': self.test_results
            }, f, indent=2)
            
        print(f"\n💾 Results saved to ai_accuracy_test_results.json")


async def main():
    """Run the accuracy tests."""
    tester = AIAccuracyTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())