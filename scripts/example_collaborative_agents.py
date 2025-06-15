"""
Example: Using Collaborative Multi-Agent Systems

Demonstrates how to use the specialized agents with the enhanced coordinator
to solve a complex software engineering task.
"""

import asyncio
import json
from datetime import datetime, timezone

from work_item_types import WorkItem, TaskPriority
from ai_brain import IntelligentAIBrain
from swarm_intelligence import RealSwarmIntelligence
from enhanced_agent_coordinator import EnhancedAgentCoordinator
from agent_factory import AgentFactory


async def demonstrate_collaborative_agents():
    """Demonstrate the collaborative multi-agent system."""
    
    # Initialize AI brain (would use actual AI client in production)
    ai_brain = IntelligentAIBrain()
    
    # Initialize swarm intelligence (optional - can work without it)
    swarm = RealSwarmIntelligence(ai_brain=ai_brain, num_agents=5)
    
    # Create enhanced coordinator
    coordinator = EnhancedAgentCoordinator(
        ai_brain=ai_brain,
        swarm_intelligence=swarm  # Can be None for specialized agents only
    )
    
    # Create a complex work item
    work_item = WorkItem(
        id="task_001",
        task_type="feature_implementation",
        title="Implement User Authentication System",
        description="""
        Create a secure user authentication system with the following requirements:
        - Support for email/password and OAuth (Google, GitHub)
        - JWT token-based authentication
        - Role-based access control (Admin, User, Guest)
        - Password reset functionality
        - Account verification via email
        - Rate limiting for login attempts
        - Secure session management
        - API endpoints for all auth operations
        """,
        priority=TaskPriority.HIGH,
        estimated_cycles=10,
        metadata={
            'requirements': [
                'Must follow OWASP security guidelines',
                'Should be scalable to 100k users',
                'API response time < 200ms',
                'Support for MFA in future'
            ],
            'tech_stack': ['PHP', 'Laravel', 'MySQL', 'Redis']
        }
    )
    
    print("🚀 Starting Collaborative Multi-Agent Analysis")
    print(f"📋 Task: {work_item.title}")
    print(f"📝 Description: {work_item.description[:100]}...")
    print("-" * 80)
    
    # Coordinate the work item
    result = await coordinator.coordinate_work_item(work_item)
    
    # Display results
    print("\n📊 Coordination Results:")
    print(f"⏱️  Coordination Time: {result['coordination_time']:.2f} seconds")
    
    # Show specialized agent results
    print("\n🤖 Specialized Agent Results:")
    for agent_type, agent_result in result['specialized_agent_results'].items():
        print(f"\n  {agent_type.upper()} Agent:")
        print(f"    ✓ Success: {agent_result['success']}")
        print(f"    📈 Confidence: {agent_result['confidence']:.2%}")
        print(f"    💡 Insights: {len(agent_result.get('insights', []))}")
        print(f"    📝 Recommendations: {len(agent_result.get('recommendations', []))}")
        print(f"    📦 Artifacts Created: {len(agent_result.get('artifacts_created', []))}")
        
        # Show some insights
        for insight in agent_result.get('insights', [])[:2]:
            print(f"      - {insight}")
    
    # Show consensus
    if result.get('consensus'):
        consensus = result['consensus']
        print("\n🤝 Consensus Building:")
        print(f"  📊 Confidence Level: {consensus['confidence_level']:.2%}")
        print(f"  ✅ Key Agreements: {len(consensus['key_agreements'])}")
        print(f"  ⚠️  Conflicts: {len(consensus['conflicts'])}")
        
        # Show unified recommendations
        print("\n  🎯 Unified Recommendations:")
        for rec in consensus['unified_recommendations'][:3]:
            print(f"    - {rec['recommendation']}")
            print(f"      (Supported by: {', '.join(rec['sources'])})")
    
    # Show final recommendation
    if result.get('final_recommendation'):
        final = result['final_recommendation']
        print("\n📋 Final Recommendation:")
        print(f"  Summary: {final['summary']}")
        print(f"  Confidence: {final['confidence']:.2%}")
        
        print("\n  🚀 Next Steps:")
        for step in final['next_steps'][:5]:
            print(f"    [{step['priority'].upper()}] {step['action']}")
        
        if final.get('risks'):
            print("\n  ⚠️  Identified Risks:")
            for risk in final['risks']:
                print(f"    - {risk}")
    
    # Show coordination status
    print("\n📊 Coordination Status:")
    status = coordinator.get_coordination_status()
    print(f"  Tasks Coordinated: {status['coordinator_metrics']['tasks_coordinated']}")
    print(f"  Agent Collaborations: {status['coordinator_metrics']['agent_collaborations']}")
    print(f"  Mode: {status['coordination_mode']}")
    
    # Show agent factory status
    factory_status = status['factory_status']
    print(f"\n🏭 Agent Factory Status:")
    print(f"  Active Agents: {factory_status['active_agents']}")
    print(f"  Available Types: {', '.join(factory_status['available_agent_types'])}")
    
    return result


async def demonstrate_agent_team_composition():
    """Demonstrate how different task types get different agent teams."""
    
    ai_brain = IntelligentAIBrain()
    factory = AgentFactory(ai_brain)
    
    task_types = [
        'feature_implementation',
        'security_audit', 
        'bug_fix',
        'documentation',
        'optimization'
    ]
    
    print("\n🏭 Agent Team Composition by Task Type:")
    print("-" * 60)
    
    for task_type in task_types:
        team = factory.create_agent_team(task_type)
        print(f"\n📋 {task_type}:")
        print(f"   Team: {[agent.agent_type for agent in team]}")
        
        # Clean up
        for agent in team:
            factory.retire_agent(agent.agent_id)


async def demonstrate_adversarial_review():
    """Demonstrate how agents review each other's work adversarially."""
    
    ai_brain = IntelligentAIBrain()
    factory = AgentFactory(ai_brain)
    
    # Create agents
    code_agent = factory.create_agent('coder')
    test_agent = factory.create_agent('tester')
    security_agent = factory.create_agent('security')
    
    print("\n🔍 Adversarial Review Process:")
    print("-" * 60)
    
    # Simulate code artifact
    code_artifact = {
        'files': [{
            'filename': 'auth.py',
            'content': 'def login(username, password): pass'
        }]
    }
    
    # Each agent reviews the code
    print("\n📝 Code Agent created authentication code")
    
    # Test agent review
    test_review = await test_agent.review_artifact(
        'main_code', code_artifact, code_agent.agent_id, None
    )
    print(f"\n🧪 Test Agent Review:")
    print(f"   Approval: {test_review['approval']}")
    for feedback in test_review['feedback']:
        print(f"   - {feedback}")
    
    # Security agent review
    security_review = await security_agent.review_artifact(
        'main_code', code_artifact, code_agent.agent_id, None
    )
    print(f"\n🔒 Security Agent Review:")
    print(f"   Approval: {security_review['approval']}")
    for feedback in security_review['feedback']:
        print(f"   - {feedback}")
    
    print("\n💡 Key Insight: TestAgent and SecurityAgent don't approve code without proper validation!")


async def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("🤖 Collaborative Multi-Agent Systems Demonstration")
    print("=" * 80)
    
    # Demo 1: Full coordination
    await demonstrate_collaborative_agents()
    
    # Demo 2: Team composition
    await demonstrate_agent_team_composition()
    
    # Demo 3: Adversarial review
    await demonstrate_adversarial_review()
    
    print("\n✅ Demonstration Complete!")


if __name__ == "__main__":
    asyncio.run(main())