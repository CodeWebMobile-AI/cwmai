# Enhanced Tool System

This document describes the enhanced capabilities added to the CWMAI tool system, including smart import resolution, multi-tool orchestration, tool evolution, and semantic tool matching.

## Overview

The enhanced tool system provides four major improvements:

1. **Dependency Resolver** - Automatically fixes import issues in generated tools
2. **Multi-Tool Orchestrator** - Handles complex queries requiring multiple tools
3. **Tool Evolution** - Tools learn and improve from usage patterns
4. **Semantic Tool Matcher** - Prevents duplicate tools and finds existing capabilities

## Components

### 1. Dependency Resolver (`scripts/dependency_resolver.py`)

**Purpose:** Automatically resolve and fix import dependencies in auto-generated tools.

**Key Features:**
- Analyzes code to detect required imports
- Identifies undefined names that need imports
- Suggests appropriate imports based on project structure
- Fixes import paths and adds missing imports
- Generates fallback implementations for missing modules

**Usage Example:**
```python
from scripts.dependency_resolver import DependencyResolver

resolver = DependencyResolver()

# Fix imports in code
fixed_code = resolver.fix_import_paths(tool_code)

# Analyze imports
imports = resolver.analyze_imports(code)

# Get import suggestions
undefined = resolver.analyze_undefined_names(code)
suggestions = resolver.suggest_imports(undefined)
```

### 2. Multi-Tool Orchestrator (`scripts/multi_tool_orchestrator.py`)

**Purpose:** Execute complex workflows involving multiple tools in coordination.

**Key Features:**
- Query decomposition into sub-tasks
- Automatic tool selection for each task
- Multiple execution strategies (sequential, parallel, pipeline)
- Dependency management between tools
- Result aggregation from multiple tools

**Execution Strategies:**
- **Sequential**: Tools run one after another in order
- **Parallel**: Independent tools run simultaneously
- **Pipeline**: Output of one tool feeds into the next
- **Conditional**: Tools run based on conditions
- **Iterative**: Tools run in loops

**Usage Example:**
```python
from scripts.tool_calling_system import ToolCallingSystem

tool_system = ToolCallingSystem()

# Handle a complex query
result = await tool_system.call_tool(
    "handle_complex_query",
    query="Analyze all repositories for security issues and create GitHub issues"
)
```

### 3. Tool Evolution (`scripts/tool_evolution.py`)

**Purpose:** Enable tools to learn from usage and automatically improve over time.

**Key Features:**
- Tracks performance metrics (success rate, execution time, errors)
- Identifies patterns in tool failures
- Suggests code improvements based on usage
- Automatically applies improvements with validation
- Maintains performance history

**Tracked Metrics:**
- Success/failure counts
- Average execution time
- Error patterns and frequencies
- Parameter usage patterns
- Performance trends

**Improvement Types:**
- Import fixes for missing dependencies
- Performance optimizations
- Enhanced error handling
- Code refactoring
- Parameter validation

**Usage Example:**
```python
# Tools are automatically tracked during execution
result = await tool_system.call_tool("some_tool", param="value")

# Manually trigger evolution
evolution_result = await tool_system.call_tool("evolve_tool", tool_name="some_tool")
```

### 4. Semantic Tool Matcher (`scripts/semantic_tool_matcher.py`)

**Purpose:** Find existing tools that match queries and prevent duplicate tool creation.

**Key Features:**
- Semantic similarity matching using TF-IDF
- Capability-based tool discovery
- Prevents duplicate tool creation
- Suggests tool compositions for complex tasks
- Updates tool index based on usage

**Matching Criteria:**
- Tool name and description
- Action verbs (create, analyze, find, etc.)
- Keywords and parameters
- Input/output types
- Usage examples

**Usage Example:**
```python
# Find similar tools
matches = await tool_system.call_tool(
    "find_similar_tools",
    query="scan code for bugs"
)

# Automatic matching during tool creation
# If you try to create "scan_repositories_for_bugs"
# It will suggest using "analyze_repository" instead
```

## Integration with Existing System

The enhanced components integrate seamlessly with the existing `ToolCallingSystem`:

1. **During Tool Creation**: 
   - Dependency resolver fixes imports automatically
   - Semantic matcher prevents duplicates

2. **During Tool Execution**:
   - Performance is tracked for evolution
   - Errors are analyzed for improvement patterns

3. **For Complex Queries**:
   - Multi-tool orchestrator decomposes and coordinates
   - Results are aggregated intelligently

## Benefits

1. **Reduced Errors**: Import issues are automatically resolved
2. **No Duplicates**: Existing tools are reused when possible
3. **Better Performance**: Tools improve over time
4. **Complex Workflows**: Multiple tools work together seamlessly
5. **Self-Improving**: System gets better with usage

## Example Workflows

### Security Audit Workflow
```python
query = "Analyze all repositories for security vulnerabilities and create GitHub issues"

# This automatically:
# 1. Scans repositories (analyze_repository tool)
# 2. Identifies vulnerabilities (security_scanner tool)
# 3. Creates issues (create_issue tool)
# 4. Generates report (report_generator tool)
```

### Code Quality Improvement
```python
query = "Find all Python files with TODO comments and create tasks for them"

# This automatically:
# 1. Searches code (search_code tool)
# 2. Extracts TODOs (pattern matching)
# 3. Creates tasks (task creation tools)
# 4. Summarizes findings
```

## Configuration

The enhanced tools are automatically available if the required dependencies are installed:

```python
# The system checks for enhanced components
if ENHANCED_TOOLS_AVAILABLE:
    # All enhanced features are enabled
    # Including new tools: handle_complex_query, evolve_tool, find_similar_tools
```

## Performance Considerations

1. **Semantic Matching**: First tool load builds index (one-time cost)
2. **Evolution Tracking**: Minimal overhead per tool execution
3. **Multi-Tool Orchestration**: Parallel execution when possible
4. **Import Resolution**: Only runs on tool creation

## Future Enhancements

1. **Cross-Tool Learning**: Tools learn from similar tools' improvements
2. **Predictive Tool Creation**: Anticipate needed tools based on usage
3. **Advanced Workflows**: Visual workflow designer
4. **Tool Versioning**: Track and rollback tool changes
5. **Distributed Execution**: Run tools across multiple workers