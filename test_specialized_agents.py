"""
Test Specialized Agents

Demonstrates usage of the specialized agent system.
"""

import asyncio
import logging
from datetime import datetime, timezone

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.ai_brain import IntelligentAIBrain
from scripts.agent_factory import AgentFactory
from scripts.enhanced_agent_coordinator import EnhancedAgentCoordinator
from scripts.base_agent import AgentContext


async def test_individual_agents():
    """Test individual specialized agents."""
    print("\n=== Testing Individual Specialized Agents ===\n")
    
    # Initialize AI brain
    ai_brain = IntelligentAIBrain({}, {})
    
    # Create agent factory
    factory = AgentFactory(ai_brain)
    
    # Test 1: Planner Agent
    print("1. Testing Planner Agent:")
    planner = factory.create_agent('planner', 'test_planner')
    
    planning_task = WorkItem(
        id="test_plan_001",
        task_type="planning",
        title="Design User Authentication System",
        description="Create a comprehensive plan for implementing a secure user authentication system with OAuth2, 2FA, and session management.",
        priority=TaskPriority.HIGH,
        repository="auth-service",
        estimated_cycles=5
    )
    
    context = AgentContext(work_item=planning_task)
    result = await planner.execute(context)
    
    print(f"   - Success: {result.success}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Subtasks created: {result.artifacts.get('subtask_count', 0)}")
    print(f"   - Recommendations: {result.recommendations[:2]}")
    
    # Test 2: Code Agent
    print("\n2. Testing Code Agent:")
    coder = factory.create_agent('code', 'test_coder')
    
    coding_task = WorkItem(
        id="test_code_001",
        task_type="implementation",
        title="Implement JWT Token Handler",
        description="Create a JWT token handler class with methods for generating, validating, and refreshing tokens. Include proper error handling and security best practices.",
        priority=TaskPriority.HIGH,
        repository="auth-service",
        estimated_cycles=3
    )
    
    context = AgentContext(work_item=coding_task)
    result = await coder.execute(context)
    
    print(f"   - Success: {result.success}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Lines of code: {result.artifacts.get('lines_of_code', 0)}")
    print(f"   - Files created: {result.artifacts.get('files_modified', 0)}")
    
    # Test 3: Test Agent
    print("\n3. Testing Test Agent:")
    tester = factory.create_agent('test', 'test_tester')
    
    testing_task = WorkItem(
        id="test_qa_001",
        task_type="testing",
        title="Create Test Suite for JWT Handler",
        description="Generate comprehensive unit and integration tests for the JWT token handler, including edge cases and security tests.",
        priority=TaskPriority.MEDIUM,
        repository="auth-service",
        estimated_cycles=2
    )
    
    context = AgentContext(work_item=testing_task)
    result = await tester.execute(context)
    
    print(f"   - Success: {result.success}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Test count: {result.artifacts.get('test_count', 0)}")
    print(f"   - Coverage estimate: {result.artifacts.get('coverage_estimate', 0)}%")
    
    # Test 4: Security Agent
    print("\n4. Testing Security Agent:")
    security = factory.create_agent('security', 'test_security')
    
    security_task = WorkItem(
        id="test_sec_001",
        task_type="security_audit",
        title="Security Audit for Authentication System",
        description="Perform security audit of the authentication system, checking for OWASP Top 10 vulnerabilities and recommending hardening measures.",
        priority=TaskPriority.CRITICAL,
        repository="auth-service",
        estimated_cycles=4
    )
    
    context = AgentContext(work_item=security_task)
    result = await security.execute(context)
    
    print(f"   - Success: {result.success}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Vulnerabilities found: {result.artifacts.get('vulnerabilities_found', 0)}")
    print(f"   - Critical issues: {result.artifacts.get('critical_issues', 0)}")
    
    # Test 5: Documentation Agent
    print("\n5. Testing Documentation Agent:")
    docs = factory.create_agent('docs', 'test_docs')
    
    docs_task = WorkItem(
        id="test_docs_001",
        task_type="documentation",
        title="Create API Documentation",
        description="Generate comprehensive API documentation for the authentication service endpoints, including examples and error codes.",
        priority=TaskPriority.MEDIUM,
        repository="auth-service",
        estimated_cycles=2
    )
    
    context = AgentContext(work_item=docs_task)
    result = await docs.execute(context)
    
    print(f"   - Success: {result.success}")
    print(f"   - Confidence: {result.confidence:.2%}")
    print(f"   - Documentation files: {result.artifacts.get('doc_files', 0)}")
    print(f"   - Word count: {result.artifacts.get('word_count', 0)}")
    
    # Show team performance
    print("\n=== Team Performance Report ===")
    report = factory.get_team_performance_report()
    print(f"Total agents: {report['total_agents']}")
    print(f"Total tasks completed: {report['overall_metrics']['total_tasks']}")
    print(f"Average success rate: {report['overall_metrics']['average_success_rate']:.2%}")


async def test_agent_coordination():
    """Test coordinated agent execution."""
    print("\n\n=== Testing Agent Coordination ===\n")
    
    # Initialize coordinator
    ai_brain = IntelligentAIBrain({}, {})
    coordinator = EnhancedAgentCoordinator(ai_brain)
    
    # Test complex task requiring multiple agents
    complex_task = WorkItem(
        id="test_coord_001",
        task_type="feature_implementation",
        title="Implement Complete User Registration Flow",
        description="""
        Implement a complete user registration flow including:
        - User registration API endpoint
        - Email verification system
        - Password strength validation
        - Rate limiting
        - Comprehensive tests
        - Security audit
        - API documentation
        
        The system should follow security best practices and be production-ready.
        """,
        priority=TaskPriority.CRITICAL,
        repository="user-service",
        estimated_cycles=10
    )
    
    print(f"Coordinating task: {complex_task.title}")
    print(f"Description: {complex_task.description[:100]}...")
    
    # Execute coordination
    result = await coordinator.coordinate_work_item(complex_task)
    
    print(f"\nCoordination Results:")
    print(f"- Success: {result.success}")
    print(f"- Duration: {result.duration_seconds:.2f} seconds")
    print(f"- Primary agent: {result.primary_agent}")
    print(f"- Supporting agents: {result.supporting_agents}")
    print(f"- Overall confidence: {result.consensus.get('confidence', 0):.2%}")
    
    print(f"\nConsensus Recommendations:")
    for i, rec in enumerate(result.consensus.get('recommendations', [])[:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nNext Steps:")
    for i, step in enumerate(result.final_output.get('next_steps', [])[:3], 1):
        print(f"  {i}. {step}")
    
    print(f"\nArtifacts Generated:")
    artifacts = result.final_output.get('artifacts', {})
    for key, value in artifacts.items():
        if value > 0:
            print(f"  - {key}: {value}")
    
    # Get coordination report
    print("\n=== Coordination Report ===")
    report = coordinator.get_coordination_report()
    metrics = report['coordination_metrics']
    print(f"Total coordinations: {metrics['total_coordinations']}")
    print(f"Success rate: {metrics['successful_coordinations'] / max(1, metrics['total_coordinations']):.2%}")
    print(f"Average agents per task: {metrics['average_agents_per_task']:.1f}")
    print(f"Average duration: {metrics['average_duration']:.2f} seconds")
    
    # Cleanup
    await coordinator.shutdown()


async def test_agent_collaboration():
    """Test agent collaboration capabilities."""
    print("\n\n=== Testing Agent Collaboration ===\n")
    
    # Initialize
    ai_brain = IntelligentAIBrain({}, {})
    factory = AgentFactory(ai_brain)
    
    # Create a team
    team_config = {
        'planner': {'count': 1},
        'code': {'count': 2},
        'test': {'count': 1},
        'security': {'count': 1}
    }
    
    team = factory.create_agent_team(team_config)
    print(f"Created team with {len(team)} agents")
    
    # Create a task requiring collaboration
    collab_task = WorkItem(
        id="test_collab_001",
        task_type="refactoring",
        title="Refactor Legacy Payment System",
        description="Refactor the legacy payment processing system to use modern patterns, improve security, and add comprehensive tests.",
        priority=TaskPriority.HIGH,
        repository="payment-service",
        estimated_cycles=8
    )
    
    context = AgentContext(work_item=collab_task)
    
    # Get planner's perspective
    planner = team['planner_team_1']
    planner_analysis = await planner.analyze(context)
    
    print(f"Planner Analysis:")
    print(f"- Confidence: {planner_analysis.get('confidence', 0):.2%}")
    print(f"- Complexity: {planner_analysis.get('complexity', 'unknown')}")
    
    # Have other agents collaborate
    other_agents = [agent for aid, agent in team.items() if aid != 'planner_team_1']
    collaboration = await planner.collaborate(context, other_agents)
    
    print(f"\nCollaboration Results:")
    print(f"- Participating agents: {collaboration['participating_agents']}")
    print(f"- Duration: {collaboration['duration_seconds']:.2f} seconds")
    
    print(f"\nSynthesized Insights:")
    synthesis = collaboration.get('synthesis', {})
    if 'common_themes' in synthesis:
        print(f"- Common themes: {synthesis['common_themes']}")
    if 'conflicts' in synthesis:
        print(f"- Conflicts: {synthesis['conflicts']}")
    if 'recommendations' in synthesis:
        print(f"- Recommendations: {synthesis['recommendations'][:3]}")


async def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test individual agents
        await test_individual_agents()
        
        # Test coordination
        await test_agent_coordination()
        
        # Test collaboration
        await test_agent_collaboration()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())