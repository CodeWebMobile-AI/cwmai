# Testing Strategy and Guide

## Overview

This document outlines the testing strategy, approach, and best practices for the CWMAI (CodeWebMobile-AI) project. Our testing framework focuses on reliability, environment-specific validation, and comprehensive coverage of AI-driven components.

## Testing Philosophy

### Core Principles

1. **Environment-Aware Testing**: Tests should adapt to different environments (development, production, testing)
2. **Graceful Degradation**: Tests should handle missing dependencies (API keys, external services) gracefully
3. **Real vs Mock Testing**: Use real AI services when available, fallback to mock data when not
4. **Comprehensive Coverage**: Test integration workflows, individual components, and edge cases
5. **Debug-First Approach**: Include comprehensive logging and debugging capabilities

### Testing Types

| Test Type | Purpose | Examples |
|-----------|---------|----------|
| **Integration Tests** | Test complete workflows end-to-end | `test_dynamic_system.py` |
| **Unit Tests** | Test individual components and classes | `test_factory_pattern.py` |
| **Feature Tests** | Test specific features and bug fixes | `test_fixes.py` |
| **Debug Tests** | Test with enhanced logging for troubleshooting | `test_swarm_debug.py` |

## Testing Framework

### Technology Stack

- **Core Framework**: Python's built-in `assert` statements
- **Async Testing**: `asyncio.run()` for testing async components
- **Mocking Strategy**: Factory pattern with environment-specific configurations
- **Logging**: Built-in `logging` module with debug levels
- **No External Dependencies**: No pytest, unittest, or specialized testing frameworks

### Project Structure

```
/
├── test_dynamic_system.py      # Integration tests
├── test_factory_pattern.py     # Unit tests
├── test_fixes.py               # Feature tests
├── test_swarm_debug.py         # Debug tests
└── scripts/
    ├── ai_brain_factory.py     # Test configurations
    └── ...
```

## Writing Tests

### Basic Test Structure

```python
#!/usr/bin/env python3
"""
Test description here.
"""

import asyncio
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.your_module import YourClass


def test_your_function():
    """Test a simple function."""
    print("Testing your_function()...")
    
    # Setup
    instance = YourClass()
    
    # Test
    result = instance.your_function()
    
    # Assertions
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"
    
    print("✓ Your function test passed")


async def test_async_function():
    """Test an async function."""
    print("Testing async_function()...")
    
    # Setup
    instance = YourClass()
    
    # Test
    result = await instance.async_function()
    
    # Assertions
    assert result is not None, "Result should not be None"
    
    print("✓ Async function test passed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("YOUR TEST SUITE")
    print("=" * 80)
    
    # Run sync tests
    test_your_function()
    
    # Run async tests
    asyncio.run(test_async_function())
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
```

### Environment-Aware Testing

Use the AIBrainFactory for environment-specific test configurations:

```python
from scripts.ai_brain_factory import AIBrainFactory

def test_with_environment_detection():
    """Test with proper environment configuration."""
    # Use factory for environment-specific setup
    ai_brain = AIBrainFactory.create_for_testing()
    
    # Validate environment setup
    assert ai_brain.context.get('environment') == 'test'
    assert ai_brain.context.get('mock_data') == True
    
    # Your test logic here
```

### Async Testing Patterns

```python
async def test_ai_components():
    """Test AI components with proper async handling."""
    try:
        # Initialize components
        controller = YourController()
        
        # Test with timeout
        result = await asyncio.wait_for(
            controller.process_something(),
            timeout=30.0
        )
        
        assert result is not None
        
    except asyncio.TimeoutError:
        print("⚠️ Test timed out (expected with slow AI responses)")
    except Exception as e:
        print(f"⚠️ Test failed (may be expected without API keys): {e}")
        # Continue with graceful degradation
```

### Error Handling and Graceful Degradation

```python
def test_with_missing_dependencies():
    """Test behavior when dependencies are missing."""
    try:
        # Attempt full functionality
        result = full_function_with_dependencies()
        print("✓ Full functionality test passed")
        
    except MissingDependencyError as e:
        print(f"⚠️ Expected failure without dependencies: {e}")
        
        # Test fallback behavior
        fallback_result = fallback_function()
        assert fallback_result is not None
        print("✓ Fallback functionality test passed")
```

## Testing Best Practices

### 1. API Key Management

```python
# Check for API keys at test start
def check_api_availability():
    """Check which APIs are available for testing."""
    api_keys = {
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
    }
    
    missing_keys = [k for k, v in api_keys.items() if not v]
    
    if missing_keys:
        print("INFO: Missing API keys:", missing_keys)
        print("Using test configuration with mock data.\n")
        return False
    else:
        print("✓ All API keys available for full testing.\n")
        return True
```

### 2. Comprehensive Assertions

```python
def validate_task_structure(task):
    """Validate task has required structure."""
    assert task is not None, "Task should not be None"
    assert isinstance(task, dict), "Task should be a dictionary"
    
    # Check required fields
    required_fields = ['title', 'type', 'priority']
    for field in required_fields:
        assert field in task, f"Task should have '{field}' field"
        assert task[field], f"Task '{field}' should not be empty"
    
    # Check field types
    assert isinstance(task['title'], str), "Task title should be string"
    assert task['priority'] in ['low', 'medium', 'high'], "Priority should be valid"
```

### 3. Debug-Friendly Testing

```python
def test_with_debug_info():
    """Test with comprehensive debug information."""
    print(f"Test started at: {datetime.now().isoformat()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    try:
        # Your test logic
        result = your_function()
        
        # Debug output
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()
        raise
```

### 4. Performance Testing

```python
import time

async def test_performance():
    """Test performance with timing."""
    start_time = time.time()
    
    result = await your_async_function()
    
    duration = time.time() - start_time
    print(f"Function completed in {duration:.2f} seconds")
    
    # Performance assertions
    assert duration < 60.0, "Function should complete within 60 seconds"
    assert result is not None, "Function should return a result"
```

## Advanced Testing Patterns

### Factory Pattern Testing

Test different environment configurations:

```python
def test_all_factory_methods():
    """Test all factory methods work correctly."""
    factories = [
        ('workflow', AIBrainFactory.create_for_workflow),
        ('production', AIBrainFactory.create_for_production),
        ('testing', AIBrainFactory.create_for_testing),
        ('development', AIBrainFactory.create_for_development),
        ('fallback', AIBrainFactory.create_minimal_fallback)
    ]
    
    for name, factory_method in factories:
        try:
            brain = factory_method()
            assert brain is not None, f"{name} factory should return brain"
            print(f"✓ {name} factory test passed")
        except Exception as e:
            print(f"⚠️ {name} factory test failed: {e}")
```

### Swarm Intelligence Testing

Test AI swarm with debug logging:

```python
async def test_swarm_with_logging():
    """Test swarm intelligence with comprehensive logging."""
    # Create swarm with debug logging
    swarm = DynamicSwarmIntelligence(ai_brain)
    swarm.enable_debug_logging("DEBUG")
    
    # Test task
    test_task = {
        'id': 'test_task',
        'type': 'NEW_PROJECT',
        'description': 'Test project description',
        'priority': 'medium'
    }
    
    # Run analysis with error handling
    try:
        result = await swarm.process_task_swarm(test_task, context)
        
        # Validate result structure
        assert 'collective_review' in result
        assert 'individual_analyses' in result
        assert 'duration_seconds' in result
        
        print(f"✓ Swarm analysis completed in {result['duration_seconds']:.2f}s")
        
    except Exception as e:
        print(f"Swarm analysis error: {e}")
        # Get debug information
        debug_summary = swarm.get_debug_summary()
        print(f"Debug info: {debug_summary}")
        raise
```

## Running Tests

### Individual Tests

```bash
# Run specific test
python test_dynamic_system.py

# Run with Python path
python -m pytest test_factory_pattern.py  # If using pytest

# Run with output
python test_swarm_debug.py 2>&1 | tee test_output.log
```

### All Tests

```bash
# Run all tests in sequence
python test_dynamic_system.py
python test_factory_pattern.py
python test_fixes.py
python test_swarm_debug.py

# Or create a test runner script
python run_all_tests.py
```

### Test Runner Script

```python
#!/usr/bin/env python3
"""Run all tests in sequence."""

import subprocess
import sys

def run_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                               capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to run {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    tests = [
        'test_dynamic_system.py',
        'test_factory_pattern.py', 
        'test_fixes.py',
        'test_swarm_debug.py'
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if run_test(test):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

## Troubleshooting

### Common Issues

#### 1. "list index out of range" Errors

**Symptoms**: Array access errors in swarm intelligence or AI response parsing.

**Debugging**:
```python
# Enable debug logging
swarm.enable_debug_logging("DEBUG")

# Check for empty response lists
if not responses:
    print("WARNING: Empty responses list detected")
    return default_response

# Validate array access
if index < len(array):
    value = array[index]
else:
    print(f"Index {index} out of range for array of length {len(array)}")
```

#### 2. API Key Issues

**Symptoms**: Authentication failures or missing provider errors.

**Solution**:
```python
# Check API key availability
def validate_api_keys():
    required_keys = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY']
    missing = [key for key in required_keys if not os.getenv(key)]
    
    if missing:
        print(f"Missing API keys: {missing}")
        print("Tests will use mock data.")
        return False
    return True
```

#### 3. Async/Await Issues

**Symptoms**: RuntimeError about event loops or hanging tests.

**Solution**:
```python
# Proper async test structure
async def test_async_function():
    """Test async function properly."""
    try:
        # Use timeout for safety
        result = await asyncio.wait_for(
            your_async_function(),
            timeout=30.0
        )
        assert result is not None
        
    except asyncio.TimeoutError:
        print("Function timed out - may need optimization")
        raise
    except Exception as e:
        print(f"Async error: {e}")
        raise

# Run async tests
if __name__ == "__main__":
    asyncio.run(test_async_function())
```

#### 4. Module Import Issues

**Symptoms**: ImportError or ModuleNotFoundError.

**Solution**:
```python
# Add proper path handling
import sys
import os

# Add scripts directory to Python path
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Then import modules
from ai_brain_factory import AIBrainFactory
```

### Debug Logging Setup

```python
import logging

def setup_test_logging():
    """Setup comprehensive logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_debug.log')
        ]
    )
    
    # Set specific loggers
    logging.getLogger('swarm_intelligence').setLevel(logging.DEBUG)
    logging.getLogger('ai_brain').setLevel(logging.INFO)
```

### Performance Monitoring

```python
import time
import psutil

class PerformanceMonitor:
    """Monitor test performance."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        
    def stop(self):
        """Stop monitoring and report."""
        duration = time.time() - self.start_time
        end_memory = psutil.Process().memory_info().rss
        memory_delta = end_memory - self.start_memory
        
        print(f"Test duration: {duration:.2f} seconds")
        print(f"Memory usage: {memory_delta / 1024 / 1024:.2f} MB")
        
        return {
            'duration': duration,
            'memory_delta': memory_delta
        }

# Usage in tests
def test_with_monitoring():
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Your test code
    result = your_function()
    
    metrics = monitor.stop()
    assert metrics['duration'] < 60.0, "Test should complete quickly"
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python test_dynamic_system.py
        python test_factory_pattern.py
        python test_fixes.py
        python test_swarm_debug.py
```

## Best Practices Summary

1. **Always test with and without API keys** - Ensure graceful degradation
2. **Use descriptive test names** - Make test purpose clear
3. **Include setup and teardown** - Clean state between tests
4. **Test error conditions** - Not just happy paths
5. **Use timeouts for async tests** - Prevent hanging tests
6. **Log comprehensively** - Make debugging easier
7. **Validate all return types** - Ensure consistent interfaces
8. **Test environment variations** - Development, testing, production
9. **Mock external dependencies** - Make tests reliable
10. **Performance monitoring** - Track test execution time

---

This testing strategy ensures reliable, maintainable, and comprehensive test coverage for the CWMAI AI system while providing clear guidance for writing effective tests.