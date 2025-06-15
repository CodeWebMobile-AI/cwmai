#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autonomous Tool System
Tests tool creation, validation, and execution in detail
"""

import asyncio
import sys
import json
import inspect
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.tool_calling_system import ToolCallingSystem


class ToolValidator:
    """Validates generated tools before they're added to the system."""
    
    @staticmethod
    def validate_code_syntax(code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax and return issues."""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, issues
        return True, issues
    
    @staticmethod
    def check_code_quality(code: str, tool_name: str) -> List[str]:
        """Check for common issues in generated code."""
        issues = []
        lines = code.split('\n')
        
        # Check for problematic patterns
        for i, line in enumerate(lines, 1):
            # Check for main() function
            if 'def main(' in line or 'async def main(' in line:
                issues.append(f"Line {i}: Contains main() function (should be removed)")
            
            # Check for __main__ block
            if "if __name__ == '__main__':" in line:
                issues.append(f"Line {i}: Contains __main__ block (should be removed)")
            
            # Check for self parameter in the main tool function
            if f'def {tool_name}(self' in line or f'async def {tool_name}(self' in line:
                issues.append(f"Line {i}: Tool function has 'self' parameter (should be removed)")
            
            # Check for hardcoded paths
            if '/home/' in line or 'C:\\' in line:
                issues.append(f"Line {i}: Contains hardcoded path")
            
            # Check for missing error handling
            if 'open(' in line and 'try:' not in code[:code.find(line)]:
                issues.append(f"Line {i}: File operation without error handling")
        
        # Check for required elements
        if f'def {tool_name}' not in code and f'async def {tool_name}' not in code:
            issues.append(f"Missing main function: {tool_name}")
        
        if '__description__' not in code:
            issues.append("Missing __description__ variable")
        
        return issues
    
    @staticmethod
    async def test_tool_execution(tool_file: Path, tool_name: str) -> Tuple[bool, str]:
        """Test if the tool can be loaded and executed."""
        try:
            # Dynamically import and test the tool
            import importlib.util
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the tool function exists
            if not hasattr(module, tool_name):
                return False, f"Module doesn't have function '{tool_name}'"
            
            func = getattr(module, tool_name)
            
            # Check function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            if params and params[0] == 'self':
                return False, "Function expects 'self' as first parameter"
            
            # Try to execute the function with no parameters
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            
            # Check if result is properly formatted
            if not isinstance(result, dict):
                return False, f"Function returned {type(result).__name__}, expected dict"
            
            return True, "Tool executed successfully"
            
        except Exception as e:
            return False, f"Execution error: {str(e)}"


async def test_tool_creation_scenarios():
    """Test various tool creation scenarios."""
    print("🧪 Testing Autonomous Tool Creation System\n")
    
    # Initialize components
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    tool_system = ToolCallingSystem()
    validator = ToolValidator()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Simple Counter Tool",
            "query": "count all markdown files in the system",
            "tool_name": "count_markdown_files",
            "expected_behavior": "Should create a tool that counts .md files"
        },
        {
            "name": "Analysis Tool", 
            "query": "analyze python code complexity",
            "tool_name": "analyze_python_complexity",
            "expected_behavior": "Should create a code analysis tool"
        },
        {
            "name": "Report Generator",
            "query": "generate weekly development report",
            "tool_name": "generate_weekly_report",
            "expected_behavior": "Should create a report generation tool"
        },
        {
            "name": "Monitoring Tool",
            "query": "monitor repository activity",
            "tool_name": "monitor_repository_activity", 
            "expected_behavior": "Should create an activity monitoring tool"
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"📋 Test: {scenario['name']}")
        print(f"Query: '{scenario['query']}'")
        print(f"Expected tool: {scenario['tool_name']}")
        print('='*60)
        
        # Clean up any existing tool
        tool_file = Path(f"scripts/custom_tools/{scenario['tool_name']}.py")
        if tool_file.exists():
            tool_file.unlink()
            print("✓ Cleaned up existing tool")
        
        # Remove from loaded tools
        if scenario['tool_name'] in tool_system.tools:
            del tool_system.tools[scenario['tool_name']]
        
        test_result = {
            "scenario": scenario['name'],
            "tool_name": scenario['tool_name'],
            "created": False,
            "syntax_valid": False,
            "quality_issues": [],
            "execution_success": False,
            "execution_message": "",
            "final_status": "FAILED"
        }
        
        try:
            # Step 1: Trigger tool creation
            print("\n1️⃣ Triggering tool creation...")
            response = await assistant.handle_conversation(scenario['query'])
            print(f"   Response: {response[:150]}...")
            
            # Step 2: Check if tool was created
            if tool_file.exists():
                test_result["created"] = True
                print(f"\n✅ Tool file created: {tool_file}")
                
                # Step 3: Validate code syntax
                print("\n2️⃣ Validating code syntax...")
                code = tool_file.read_text()
                syntax_valid, syntax_issues = validator.validate_code_syntax(code)
                test_result["syntax_valid"] = syntax_valid
                
                if syntax_valid:
                    print("   ✅ Syntax is valid")
                else:
                    print(f"   ❌ Syntax errors: {syntax_issues}")
                    test_result["quality_issues"].extend(syntax_issues)
                
                # Step 4: Check code quality
                print("\n3️⃣ Checking code quality...")
                quality_issues = validator.check_code_quality(code, scenario['tool_name'])
                test_result["quality_issues"].extend(quality_issues)
                
                if not quality_issues:
                    print("   ✅ No quality issues found")
                else:
                    print("   ⚠️  Quality issues found:")
                    for issue in quality_issues:
                        print(f"      - {issue}")
                
                # Step 5: Test tool execution
                print("\n4️⃣ Testing tool execution...")
                exec_success, exec_message = await validator.test_tool_execution(
                    tool_file, scenario['tool_name']
                )
                test_result["execution_success"] = exec_success
                test_result["execution_message"] = exec_message
                
                if exec_success:
                    print(f"   ✅ {exec_message}")
                else:
                    print(f"   ❌ {exec_message}")
                
                # Step 6: Show generated function signature
                print("\n5️⃣ Generated function signature:")
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if f'def {scenario["tool_name"]}' in line:
                        print(f"   {line.strip()}")
                        if i+1 < len(lines) and lines[i+1].strip().startswith('"""'):
                            print(f"   {lines[i+1].strip()}")
                        break
                
                # Determine final status
                if (test_result["created"] and 
                    test_result["syntax_valid"] and 
                    len(test_result["quality_issues"]) == 0 and 
                    test_result["execution_success"]):
                    test_result["final_status"] = "PASSED"
                elif test_result["created"] and test_result["syntax_valid"]:
                    test_result["final_status"] = "PARTIAL"
                
            else:
                print("\n❌ No tool file was created")
                
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            test_result["execution_message"] = str(e)
        
        results.append(test_result)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r["final_status"] == "PASSED")
    partial = sum(1 for r in results if r["final_status"] == "PARTIAL")
    failed = sum(1 for r in results if r["final_status"] == "FAILED")
    
    print(f"\nTotal scenarios: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"⚠️  Partial: {partial}")
    print(f"❌ Failed: {failed}")
    
    print("\nDetailed Results:")
    for result in results:
        status_emoji = "✅" if result["final_status"] == "PASSED" else "⚠️" if result["final_status"] == "PARTIAL" else "❌"
        print(f"\n{status_emoji} {result['scenario']}:")
        print(f"   Tool: {result['tool_name']}")
        print(f"   Created: {'Yes' if result['created'] else 'No'}")
        print(f"   Syntax Valid: {'Yes' if result['syntax_valid'] else 'No'}")
        print(f"   Quality Issues: {len(result['quality_issues'])}")
        print(f"   Execution: {'Success' if result['execution_success'] else 'Failed'}")
        if result['execution_message']:
            print(f"   Message: {result['execution_message']}")
    
    # Save detailed report
    report_file = Path("autonomous_tool_test_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": asyncio.get_event_loop().time(),
            "summary": {
                "total": len(results),
                "passed": passed,
                "partial": partial,
                "failed": failed
            },
            "results": results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return results


async def test_tool_update_mechanism():
    """Test if existing tools can be updated/improved."""
    print("\n\n🔄 Testing Tool Update Mechanism\n")
    
    tool_system = ToolCallingSystem()
    
    # Test updating an existing tool
    print("Testing enhancement of existing tool...")
    
    try:
        result = await tool_system._enhance_tool(
            tool_name="count_repositories",
            enhancement="Add ability to filter by programming language and show more detailed statistics"
        )
        
        if result.get('success'):
            print("✅ Tool enhancement successful")
            print(f"   {result.get('message')}")
        else:
            print("❌ Tool enhancement failed")
            print(f"   {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error during enhancement: {str(e)}")


async def test_error_recovery():
    """Test how the system handles and recovers from errors."""
    print("\n\n🛡️ Testing Error Recovery\n")
    
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    # Test scenarios that might cause errors
    error_scenarios = [
        {
            "name": "Invalid Tool Name",
            "query": "create a tool with spaces in name",
            "expected": "Should handle invalid tool names gracefully"
        },
        {
            "name": "Complex Requirements",
            "query": "create a tool that accesses external APIs without credentials",
            "expected": "Should handle impossible requirements"
        },
        {
            "name": "Conflicting Tool",
            "query": "create a new count_repositories tool",
            "expected": "Should detect existing tool and not create duplicate"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n📋 Test: {scenario['name']}")
        print(f"Query: '{scenario['query']}'")
        print(f"Expected: {scenario['expected']}")
        
        try:
            response = await assistant.handle_conversation(scenario['query'])
            print(f"Response: {response[:200]}...")
            print("✅ Handled gracefully")
        except Exception as e:
            print(f"❌ Unhandled error: {str(e)}")


async def main():
    """Run all tests."""
    # Run main test suite
    results = await test_tool_creation_scenarios()
    
    # Test update mechanism
    await test_tool_update_mechanism()
    
    # Test error recovery
    await test_error_recovery()
    
    print("\n\n✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())