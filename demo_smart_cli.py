#!/usr/bin/env python3
"""
Demo of Smart CWMAI CLI

Shows the full capabilities of the intelligent natural language interface.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.smart_natural_language_interface import SmartNaturalLanguageInterface
from scripts.ai_brain import IntelligentAIBrain


async def demo_smart_features():
    """Demonstrate smart CLI features."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              🧠 Smart CWMAI CLI Demo 🧠                       ║
║                                                               ║
║  Demonstrating the most intelligent features                  ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Create smart interface
    print("🔧 Initializing smart interface with all features enabled...")
    
    ai_brain = IntelligentAIBrain(enable_round_robin=True)
    interface = SmartNaturalLanguageInterface(
        ai_brain=ai_brain,
        enable_learning=True,
        enable_multi_model=True,
        enable_plugins=True
    )
    
    await interface.initialize()
    print("✅ Smart interface ready!\n")
    
    # Demo scenarios
    demos = [
        {
            'title': '1. Natural Language Understanding',
            'commands': [
                "What's the status of my projects?",
                "Create an issue for the auth service about users reporting that login is taking too long",
                "Find me some AI tools that can help with code review"
            ]
        },
        {
            'title': '2. Complex Multi-Step Operations',
            'commands': [
                "Search for JavaScript testing frameworks and then create a comparison task",
                "Find all repositories with security issues and create a plan to fix them"
            ]
        },
        {
            'title': '3. Intelligent Automation',
            'commands': [
                "Every morning at 9am, check for new issues and create tasks for critical ones",
                "When a new issue is created, automatically analyze it and suggest a solution"
            ]
        },
        {
            'title': '4. Smart Visualizations',
            'commands': [
                "Show me a chart of task completion over the last week",
                "Visualize the distribution of issues by project"
            ]
        },
        {
            'title': '5. Context-Aware Explanations',
            'commands': [
                "Explain how the task management system works",
                "Why did my last command fail?"
            ]
        },
        {
            'title': '6. Market-Aware Architecture',
            'commands': [
                "Design an architecture for a SaaS platform that competes with Notion",
                "What's trending in developer tools right now?"
            ]
        },
        {
            'title': '7. Learning and Adaptation',
            'commands': [
                "Do my usual morning routine",
                "Create another issue like the last one but for the payment service"
            ]
        }
    ]
    
    # Run demos
    for demo in demos[:3]:  # Run first 3 demos to keep it short
        print(f"\n{'='*60}")
        print(f"🎯 {demo['title']}")
        print('='*60)
        
        for command in demo['commands']:
            print(f"\n💬 User: {command}")
            
            try:
                result = await interface.process_input(command)
                
                # Display key results
                print(f"🤖 Assistant:")
                
                if result.get('explanation'):
                    print(f"   {result['explanation']}")
                
                if result.get('success'):
                    action = result.get('action', 'unknown')
                    
                    if action == 'status_displayed':
                        stats = result.get('stats', {})
                        print(f"   Projects: {stats.get('total_projects', 0)}")
                        print(f"   Active Tasks: {stats.get('active_tasks', 0)}")
                        print(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
                    
                    elif action == 'issue_created':
                        print(f"   Created issue in {result.get('repository')}")
                        print(f"   Enhancements: {', '.join(result.get('enhancements', []))}")
                    
                    elif action == 'search_completed':
                        print(f"   Found {result.get('total_found', 0)} results")
                        print(f"   Sources: {', '.join(result.get('sources', []))}")
                    
                    elif action.startswith('plugin_'):
                        plugin_name = result.get('plugin_name')
                        print(f"   Used {plugin_name} plugin")
                        
                        if plugin_name == 'automation':
                            workflow = result.get('plugin_data', {}).get('workflow', {})
                            print(f"   Created {workflow.get('type')} workflow")
                        elif plugin_name == 'visualization':
                            print(f"   Generated {result.get('plugin_data', {}).get('type')} chart")
                
                if result.get('suggestions'):
                    print(f"\n   💡 Suggestions:")
                    for suggestion in result['suggestions'][:2]:
                        print(f"      • {suggestion}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            await asyncio.sleep(1)  # Brief pause between commands
    
    # Show learning summary
    print(f"\n\n{'='*60}")
    print("📊 Learning Summary")
    print('='*60)
    
    if interface.context.command_patterns:
        print("\n🧠 Learned Patterns:")
        for pattern, count in sorted(interface.context.command_patterns.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {pattern}: used {count} times")
    
    if interface.context.current_project:
        print(f"\n📁 Current Project Context: {interface.context.current_project}")
    
    print("\n✨ The system has learned from this session and will be even smarter next time!")
    
    # Save the model
    interface._save_user_model()
    print("\n💾 User model saved for future sessions")


async def main():
    """Run the demo."""
    try:
        await demo_smart_features()
        print("\n\n🎉 Demo completed! The Smart CWMAI CLI is ready for use.")
        print("Run 'python run_smart_cli.py' to start using it interactively.")
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    
    asyncio.run(main())