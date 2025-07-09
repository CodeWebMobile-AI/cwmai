#!/usr/bin/env python3
"""
Script to integrate enhanced tool loading into the existing system
"""

import os
from pathlib import Path

def create_tool_calling_patch():
    """Create a patch to integrate enhanced tool loading."""
    
    patch_content = '''
# Add to imports section (after line 20)
from scripts.tool_loader_enhanced import EnhancedToolLoader

# Add to __init__ method (after line 100)
        # Initialize enhanced tool loader
        self.enhanced_loader = EnhancedToolLoader(self)
        self.tool_categories = {}

# Replace _load_custom_tools method (line 1908)
    def _load_custom_tools(self):
        """Load custom tools using enhanced loader with category support."""
        try:
            # Use enhanced loader
            loaded_counts = self.enhanced_loader.load_all_tools()
            
            # Store category information
            self.tool_categories = self.enhanced_loader.tool_categories
            
            # Log results
            for category, count in loaded_counts.items():
                self.logger.info(f"Loaded {count} tools in category: {category}")
                
            total_tools = sum(loaded_counts.values())
            self.logger.info(f"Total custom tools loaded: {total_tools}")
            
        except Exception as e:
            self.logger.error(f"Error loading custom tools: {str(e)}")
            # Fallback to original method
            self._load_custom_tools_legacy()

# Add new methods for category support
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a specific category."""
        if hasattr(self, 'enhanced_loader'):
            return self.enhanced_loader.get_tools_by_category(category)
        return []
        
    def find_tools_for_task(self, task_description: str) -> List[Tuple[str, str, float]]:
        """Find best tools for a given task."""
        if hasattr(self, 'enhanced_loader'):
            return self.enhanced_loader.find_tools_for_task(task_description)
        return []
        
    def get_tool_categories(self) -> Dict[str, str]:
        """Get all available tool categories."""
        if hasattr(self, 'enhanced_loader'):
            return self.enhanced_loader.get_all_categories()
        return {}
'''
    
    # Save patch file
    with open('tool_calling_system_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("✅ Created patch file: tool_calling_system_patch.txt")
    print("\nTo apply the patch:")
    print("1. Manually add the imports to tool_calling_system.py")
    print("2. Add initialization code to __init__ method")
    print("3. Replace _load_custom_tools method")
    print("4. Add the new category support methods")

def create_example_usage():
    """Create example usage for the enhanced tool system."""
    
    example_content = '''#!/usr/bin/env python3
"""
Example usage of enhanced tool system with categories
"""

import asyncio
from scripts.tool_calling_system import ToolCallingSystem

async def demonstrate_enhanced_tools():
    """Demonstrate enhanced tool features."""
    
    # Initialize tool system
    tool_system = ToolCallingSystem()
    
    print("=== Enhanced Tool System Demo ===\\n")
    
    # 1. Show available categories
    print("1. Available Tool Categories:")
    categories = tool_system.get_tool_categories()
    for category, description in categories.items():
        tools = tool_system.get_tools_by_category(category)
        print(f"   {category}: {description} ({len(tools)} tools)")
    
    # 2. Find tools for specific tasks
    print("\\n2. Finding Tools for Tasks:")
    
    tasks = [
        "I need to analyze the business potential of my project",
        "Help me manage my git repository",
        "Count the number of Python files in my project",
        "Create a monetization strategy for my SaaS"
    ]
    
    for task in tasks:
        print(f"\\n   Task: '{task}'")
        matches = tool_system.find_tools_for_task(task)
        for tool_name, category, score in matches[:3]:
            print(f"      - {tool_name} ({category}) - relevance: {score}")
    
    # 3. Use specialized monetization tool
    print("\\n3. Using Specialized Monetization Tool:")
    
    if "analyze_project_monetization" in tool_system.tools:
        result = await tool_system.call_tool(
            "analyze_project_monetization",
            project_path=".",
            analysis_type="comprehensive"
        )
        print(f"   Result: {result}")
    
    # 4. Chain repository tools
    print("\\n4. Chaining Repository Tools:")
    
    workflow = [
        ("git_repository_operations", {"action": "status"}),
        ("dependency_manager", {"action": "check"}),
        ("environment_config_manager", {"action": "validate"})
    ]
    
    for tool_name, params in workflow:
        if tool_name in tool_system.tools:
            print(f"   Running: {tool_name}")
            result = await tool_system.call_tool(tool_name, **params)
            print(f"   Result: {result}")

async def demonstrate_tool_discovery():
    """Show how AI can discover tools more intelligently."""
    
    tool_system = ToolCallingSystem()
    
    print("\\n=== Intelligent Tool Discovery ===\\n")
    
    # Complex queries that need the right tool
    queries = [
        {
            "task": "What's the total addressable market for my Laravel marketplace?",
            "expected_tool": "create_market_analysis"
        },
        {
            "task": "Set up a new project with database and dependencies",
            "expected_tools": ["project_builder", "database_operations", "dependency_manager"]
        },
        {
            "task": "Analyze code quality and suggest monetization opportunities",
            "expected_tools": ["code_analyzer", "analyze_project_monetization"]
        }
    ]
    
    for query in queries:
        print(f"Query: {query['task']}")
        matches = tool_system.find_tools_for_task(query['task'])
        
        print("   Recommended tools:")
        for tool, category, score in matches[:5]:
            print(f"   - {tool} ({category})")
        print()

if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demonstrate_enhanced_tools())
    asyncio.run(demonstrate_tool_discovery())
'''
    
    with open('enhanced_tools_demo.py', 'w') as f:
        f.write(example_content)
    
    print("✅ Created example file: enhanced_tools_demo.py")

def main():
    """Main integration process."""
    print("Enhanced Tool Integration Helper")
    print("=" * 50)
    
    # Check current structure
    tools_dir = Path("scripts/custom_tools")
    
    print("\\nCurrent tool structure:")
    print(f"Total tools: {len(list(tools_dir.glob('*.py')))}")
    
    subdirs = [d for d in tools_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"Categories found: {len(subdirs)}")
        for subdir in subdirs:
            tool_count = len(list(subdir.glob('*.py')))
            print(f"  - {subdir.name}: {tool_count} tools")
    else:
        print("No category directories found yet")
    
    print("\\nCreating integration files...")
    
    # Create patch file
    create_tool_calling_patch()
    
    # Create example usage
    create_example_usage()
    
    print("\\n✅ Integration files created!")
    print("\\nNext steps:")
    print("1. Run: python3 reorganize_tools.py")
    print("2. Apply patches from tool_calling_system_patch.txt")
    print("3. Test with: python3 enhanced_tools_demo.py")

if __name__ == "__main__":
    main()