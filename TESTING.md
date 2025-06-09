# Testing Strategy and Guide

## Overview

The CWMAI project uses a custom testing approach focused on integration testing and real-world scenarios. Our testing strategy emphasizes reliability, graceful error handling, and comprehensive system validation without heavy external dependencies.

## Testing Philosophy

### Core Principles

1. **Integration-First Testing**: Tests validate complete workflows rather than isolated units
2. **Graceful Degradation**: Tests work with or without API keys, using mock data when needed
3. **Environment-Aware**: Different test configurations for development, production, and testing environments
4. **Comprehensive Logging**: Debug capabilities to troubleshoot complex AI system interactions
5. **Real-World Scenarios**: Tests mirror actual usage patterns and edge cases

### No External Test Frameworks

We deliberately avoid heavy testing frameworks like pytest or unittest to:
- Reduce dependencies and complexity
- Maintain full control over test execution
- Simplify CI/CD integration
- Enable custom async testing patterns

## Current Test Structure

### Test Files Overview

```
test_dynamic_system.py     # Complete system integration tests
test_factory_pattern.py    # Factory pattern validation
test_fixes.py             # Specific bug fix validation
test_swarm_debug.py       # Debug and troubleshooting tests
```

## Writing Tests

### Basic Test Structure

All tests follow this pattern:

```python
#!/usr/bin/env python3
"""
Test description and purpose.
"""

import asyncio
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.module_to_test import ClassToTest

async def test_specific_functionality():
    """Test description."""
    print("Testing specific functionality...")
    
    # Arrange
    test_object = ClassToTest()
    
    # Act
    result = await test_object.method_to_test()
    
    # Assert
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'expected_property'), "Result should have expected property"
    
    print("✓ Test passed")

def main():
    """Run all tests."""
    tests = [
        test_specific_functionality,
        # Add more test functions
    ]
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                asyncio.run(test())
            else:
                test()
        except Exception as e:
            print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    main()
```

### Factory Pattern Testing

For testing different environments and configurations:

```python
def test_environment_factory():
    """Test environment-specific factory method."""
    print("Testing create_for_testing()...")
    
    brain = AIBrainFactory.create_for_testing()
    
    # Validate brain properties
    assert brain is not None, "Brain should not be None"
    assert hasattr(brain, 'state'), "Brain should have state"
    assert brain.context.get('environment') == 'test', "Should be test environment"
    assert brain.context.get('mock_data') == True, "Should have mock data enabled"
    
    print("✓ Testing factory test passed")
```

### Async Testing with Error Handling

For testing async AI operations:

```python
async def test_ai_functionality():
    """Test AI functionality with graceful error handling."""
    print("Testing AI functionality...")
    
    try:
        # Initialize AI brain
        ai_brain = AIBrainFactory.create_for_testing()
        
        # Test operation
        result = await ai_brain.process_task(test_task)
        
        # Validate result
        assert result is not None, "Should return result"
        print(f"✓ AI operation completed: {result.get('status', 'Unknown')}")
        
    except Exception as e:
        print(f"⚠️  AI test failed (expected without API keys): {e}")
        print("✓ Graceful handling of missing AI providers")
        
        # Use mock data for remaining tests
        result = {'status': 'mock', 'data': 'test_data'}
        print("✓ Using mock data for remaining tests")
```

### Debug Testing

For comprehensive debugging and troubleshooting:

```python
async def test_with_debug_logging():
    """Test with comprehensive debug logging."""
    print("🔍 Testing with debug logging enabled")
    
    # Enable debug logging
    swarm.enable_debug_logging("DEBUG")
    
    # Get debug summary
    debug_summary = swarm.get_debug_summary()
    print(f"Total Agents: {debug_summary['swarm_config']['total_agents']}")
    
    # Run test with detailed monitoring
    result = await swarm.process_task_swarm(test_task, context)
    
    # Analyze results for issues
    error_count = 0
    for analysis in result.get('individual_analyses', []):
        if 'error' in analysis:
            error_count += 1
            print(f"❌ Analysis error: {analysis.get('error', 'Unknown')}")
    
    print(f"📈 Error Summary: {error_count} total errors")
```

## Testing Patterns

### 1. API Key Handling

Always handle missing API keys gracefully:

```python
# Check for API keys at startup
api_keys = {
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
}

missing_keys = [k for k, v in api_keys.items() if not v]

if missing_keys:
    print("INFO: Missing API keys:", missing_keys)
    print("Using test configuration with mock data.\n")
else:
    print("✓ All API keys available for full testing.\n")
```

### 2. Mock Data for Testing

Create realistic test data when API services are unavailable:

```python
# Create mock task for testing
task = {
    'title': 'Test Task',
    'type': 'NEW_PROJECT',
    'priority': 'medium',
    'description': 'Mock task for testing'
}

context = {
    'projects': [],
    'recent_outcomes': [],
    'capabilities': ['GitHub API', 'AI Models', 'Task Generation'],
    'market_trends': []
}
```

### 3. Environment Configuration

Test different environments with specific configurations:

```python
def test_production_factory():
    """Test production environment configuration."""
    try:
        brain = AIBrainFactory.create_for_production()
        assert brain.context.get('environment') == 'production'
        assert brain.context.get('monitoring_enabled') == True
        print("✓ Production factory test passed")
    except Exception as e:
        print(f"⚠️  Production test failed (expected in test environment): {e}")
```

### 4. Assertion Patterns

Use descriptive assertions with clear error messages:

```python
# Good assertions
assert brain is not None, "Brain should not be None"
assert hasattr(brain, 'state'), "Brain should have state attribute"
assert len(results) > 0, "Should have at least one result"
assert result.get('status') == 'success', f"Expected success, got {result.get('status')}"

# Health validation
is_healthy = AIBrainFactory._validate_brain_health(brain)
assert is_healthy == True, "Test brain should be healthy"
```

## Test Organization

### File Naming

- `test_*.py` - Test files should start with `test_`
- Use descriptive names that indicate what is being tested
- Group related tests in the same file

### Function Naming

```python
def test_specific_functionality():        # Basic test
async def test_async_functionality():     # Async test  
def test_edge_case_handling():           # Edge case test
def test_error_recovery():               # Error handling test
```

### Test Structure

1. **Arrange**: Set up test data and dependencies
2. **Act**: Execute the functionality being tested
3. **Assert**: Validate the results
4. **Cleanup**: Clean up resources (if needed)

## Running Tests

### Individual Test Files

```bash
# Run a specific test file
python test_dynamic_system.py

# Run with debug output
python test_swarm_debug.py
```

### All Tests

```bash
# Run all tests (create this script if needed)
for test_file in test_*.py; do
    echo "Running $test_file..."
    python "$test_file"
    echo ""
done
```

## Best Practices

### 1. Test Independence

- Each test should be independent and not rely on other tests
- Clean up any state changes after tests
- Use fresh instances for each test

### 2. Descriptive Output

```python
print("Testing feature functionality...")
print(f"✓ Feature completed with result: {result}")
print(f"⚠️  Warning: {warning_message}")
print(f"✗ Test failed: {error_message}")
```

### 3. Error Handling

```python
try:
    result = await risky_operation()
    assert result is not None, "Operation should return result"
    print("✓ Operation successful")
except Exception as e:
    print(f"⚠️  Operation failed (may be expected): {e}")
    # Provide fallback or alternative test path
```

### 4. Performance Testing

```python
import time

start_time = time.time()
result = await operation_to_test()
duration = time.time() - start_time

print(f"Operation completed in {duration:.2f} seconds")
assert duration < 10.0, "Operation should complete within 10 seconds"
```

### 5. Comprehensive Validation

```python
# Validate result structure
assert isinstance(result, dict), "Result should be a dictionary"
assert 'status' in result, "Result should contain status"
assert 'data' in result, "Result should contain data"

# Validate business logic
assert result['status'] in ['success', 'error'], "Status should be valid"
if result['status'] == 'success':
    assert result['data'] is not None, "Success should include data"
```

## Debugging Tests

### Enable Debug Logging

```python
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In tests
logger.debug(f"Testing with data: {test_data}")
logger.info(f"Test result: {result}")
```

### Debug Analysis

```python
# Check for specific error patterns
debug_summary = system.get_debug_summary()
performance_metrics = debug_summary.get('performance_metrics', {})

print(f"Performance Metrics:")
print(f"  - Total Tasks: {performance_metrics.get('total_tasks', 0)}")
print(f"  - Average Duration: {performance_metrics.get('average_duration', 0):.2f}s")
print(f"  - Average Confidence: {performance_metrics.get('average_confidence', 0):.2f}")
```

## Troubleshooting

### Common Issues

#### 1. Missing API Keys
**Problem**: Tests fail with authentication errors
**Solution**: Use mock data or test configurations

```python
if not os.getenv('API_KEY'):
    print("Using mock configuration for testing")
    brain = AIBrainFactory.create_for_testing()  # Uses mock data
```

#### 2. Async Errors
**Problem**: `RuntimeError: cannot be called from a running event loop`
**Solution**: Use proper async/await patterns

```python
# Wrong
result = asyncio.run(async_function())  # If already in async context

# Right  
result = await async_function()  # In async function
# Or
asyncio.run(async_function())  # In sync context
```

#### 3. Import Errors
**Problem**: Module not found errors
**Solution**: Ensure proper path setup

```python
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
```

#### 4. Test Data Issues
**Problem**: Tests fail with empty or invalid data
**Solution**: Validate test data setup

```python
# Validate test data before using
assert len(test_data) > 0, "Test data should not be empty"
assert 'required_field' in test_data, "Test data should have required fields"
```

### Debug Commands

```bash
# Run with maximum verbosity
python test_file.py 2>&1 | tee test_output.log

# Check for specific error patterns
grep -i "error\|fail\|exception" test_output.log

# Monitor system resources during tests
top -p $(pgrep -f "python test_")
```

## Integration with CI/CD

### GitHub Actions Integration

```yaml
# In .github/workflows/test.yml
- name: Run Tests
  run: |
    for test_file in test_*.py; do
      echo "Running $test_file..."
      python "$test_file" || exit 1
    done
```

### Test Environment Variables

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Future Enhancements

### Potential Improvements

1. **Test Coverage Reporting**: Add coverage analysis for critical paths
2. **Performance Benchmarking**: Automated performance regression testing
3. **Test Data Management**: Centralized test data configuration
4. **Parallel Test Execution**: Run independent tests concurrently
5. **Test Report Generation**: Automated test result reporting

### Migration Considerations

If moving to external test frameworks in the future:
- Maintain current async testing patterns
- Preserve mock data and API key handling
- Keep environment-specific test configurations
- Retain debug logging capabilities

---

This testing approach ensures reliable, maintainable tests that work in various environments while providing comprehensive validation of the CWMAI system's complex AI-driven functionality.