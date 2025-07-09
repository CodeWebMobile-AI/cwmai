# Tool Integration Plan

## Current Structure Analysis

### Tool Loading System
The system automatically loads tools from `scripts/custom_tools/` directory:
1. **Consolidated Tools** (8) - Already in place and working
2. **Specialized Tools** (17) - Need to be integrated properly

### Integration Strategy

## 1. Tool Organization Structure

Create subdirectories to organize tools by category:

```
scripts/custom_tools/
├── core/                    # Consolidated tools (8)
│   ├── universal_counter.py
│   ├── repository_manager.py
│   ├── architecture_analyzer.py
│   ├── code_analyzer.py
│   ├── file_manager.py
│   ├── system_controller.py
│   ├── smart_creator.py
│   └── system_monitor.py
│
├── monetization/           # Business & Monetization tools (8)
│   ├── analyze_project_monetization.py
│   ├── discover_business_opportunities.py
│   ├── evaluate_project_opportunities.py
│   ├── generate_monetization_strategy.py
│   ├── create_market_analysis.py
│   ├── generate_business_architecture.py
│   ├── portfolio_monetization_report.py
│   └── create_monetized_project.py
│
└── repository/            # Repository Control tools (9)
    ├── manage_local_repository.py
    ├── repository_file_manager.py
    ├── git_repository_operations.py
    ├── code_generator_for_repos.py
    ├── dependency_manager.py
    ├── project_builder.py
    ├── development_server_manager.py
    ├── database_operations.py
    └── environment_config_manager.py
```

## 2. Tool Calling System Updates

### A. Update Tool Discovery
Modify `_load_custom_tools()` to support subdirectories:

```python
def _load_custom_tools(self):
    """Load custom tools from organized directories."""
    custom_tools_dir = Path("scripts/custom_tools")
    
    # Load from root directory (backward compatibility)
    self._load_tools_from_directory(custom_tools_dir, category="general")
    
    # Load from subdirectories
    for subdir in custom_tools_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('_'):
            self._load_tools_from_directory(subdir, category=subdir.name)
```

### B. Tool Categorization
Add category metadata to tools:

```python
class ToolDefinition:
    def __init__(self, name: str, func: Callable, description: str, 
                 parameters: Dict[str, Any], examples: List[str] = None,
                 category: str = "general"):  # Add category
        # ... existing code ...
        self.category = category
```

## 3. Tool Registry Enhancement

Create a tool registry that maps tool capabilities:

```python
TOOL_REGISTRY = {
    "counting": ["universal_counter"],
    "repository": ["repository_manager", "manage_local_repository"],
    "architecture": ["architecture_analyzer", "generate_business_architecture"],
    "analysis": ["code_analyzer", "analyze_project_monetization"],
    "files": ["file_manager", "repository_file_manager"],
    "system": ["system_controller"],
    "creation": ["smart_creator", "create_monetized_project"],
    "monitoring": ["system_monitor"],
    "monetization": [
        "analyze_project_monetization",
        "discover_business_opportunities",
        "evaluate_project_opportunities",
        "generate_monetization_strategy",
        "create_market_analysis",
        "portfolio_monetization_report"
    ],
    "development": [
        "git_repository_operations",
        "code_generator_for_repos",
        "dependency_manager",
        "project_builder",
        "development_server_manager",
        "database_operations",
        "environment_config_manager"
    ]
}
```

## 4. Integration Workflow

### Step 1: Create Tool Wrapper
Create a unified interface for specialized tools:

```python
class SpecializedToolWrapper:
    """Wrapper for specialized tools to ensure consistent interface."""
    
    def __init__(self, tool_path: Path):
        self.tool_path = tool_path
        self.category = tool_path.parent.name
        
    async def execute(self, operation: str = None, **kwargs):
        """Unified execution interface."""
        # Load and execute the tool
        pass
```

### Step 2: Tool Discovery Enhancement
Enhance the AI's ability to discover appropriate tools:

```python
async def find_best_tool(self, task_description: str) -> str:
    """Find the best tool for a given task using AI."""
    # Check consolidated tools first
    # Then check specialized tools based on task keywords
    # Use semantic matching for complex queries
```

### Step 3: Tool Chaining
Enable tools to work together:

```python
async def execute_tool_chain(self, tools: List[str], data_flow: Dict):
    """Execute multiple tools in sequence with data flow."""
    results = {}
    for tool in tools:
        result = await self.call_tool(tool, **data_flow)
        results[tool] = result
        # Update data_flow with results for next tool
    return results
```

## 5. Migration Steps

1. **Reorganize Tools** (Phase 1)
   - Create subdirectories
   - Move tools to appropriate categories
   - Update imports and paths

2. **Update Tool Loading** (Phase 2)
   - Modify `_load_custom_tools()` method
   - Add category support
   - Test loading from subdirectories

3. **Enhance Discovery** (Phase 3)
   - Implement tool registry
   - Add semantic matching
   - Create tool recommendation system

4. **Enable Tool Chaining** (Phase 4)
   - Implement workflow system
   - Create data flow mechanisms
   - Test complex multi-tool operations

## 6. Usage Examples

### Example 1: Monetization Analysis
```python
# AI can now use specialized monetization tools
result = await tool_system.call_tool(
    "analyze_project_monetization",
    project_path="/path/to/project"
)
```

### Example 2: Repository Management Chain
```python
# Chain multiple repository tools
workflow = [
    ("manage_local_repository", {"action": "create"}),
    ("git_repository_operations", {"action": "init"}),
    ("environment_config_manager", {"action": "setup"}),
    ("dependency_manager", {"action": "install"})
]
result = await tool_system.execute_workflow(workflow)
```

### Example 3: Intelligent Tool Selection
```python
# AI automatically selects best tool
task = "I need to analyze the business potential of my Laravel project"
best_tool = await tool_system.find_best_tool(task)
# Returns: "analyze_project_monetization"
```

## 7. Benefits

1. **Better Organization** - Tools grouped by functionality
2. **Easier Discovery** - AI can find tools by category
3. **Powerful Workflows** - Chain tools for complex operations
4. **Backward Compatible** - Existing code continues to work
5. **Scalable** - Easy to add new tool categories

## 8. Implementation Priority

1. **High Priority**:
   - Tool reorganization
   - Update tool loading system
   - Basic category support

2. **Medium Priority**:
   - Tool registry implementation
   - Semantic tool matching
   - Basic workflow support

3. **Low Priority**:
   - Advanced tool chaining
   - Complex workflow orchestration
   - Tool recommendation system