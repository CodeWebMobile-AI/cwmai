"""
Test the new dynamic portfolio analysis system.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from scripts.ai_brain import IntelligentAIBrain
from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.market_research_engine import MarketResearchEngine
from scripts.portfolio_intelligence import PortfolioIntelligence
from scripts.project_outcome_tracker import ProjectOutcomeTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_market_research():
    """Test the market research engine."""
    print("\n🔬 Testing Market Research Engine...")
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create market research engine
    market_research = MarketResearchEngine(ai_brain)
    
    # Discover market trends
    print("\n📊 Discovering market trends...")
    trends = await market_research.discover_market_trends()
    
    print(f"\nDiscovered {len(trends)} market trends:")
    for trend in trends[:3]:  # Show first 3
        print(f"\n- {trend.trend_name}")
        print(f"  Category: {trend.category}")
        print(f"  Demand: {trend.demand_level}")
        print(f"  Opportunity Score: {trend.opportunity_score:.2f}")
        print(f"  Technologies: {', '.join(trend.technologies[:3])}")
        print(f"  Problem Space: {trend.problem_space[:100]}...")
    
    # Test project opportunity generation
    print("\n\n💡 Generating project opportunities...")
    
    # Mock portfolio for testing
    mock_portfolio = {
        "test-cms": {
            "name": "test-cms",
            "description": "A content management system",
            "language": "JavaScript",
            "topics": ["cms", "nodejs", "react"],
            "recent_activity": {"recent_commits": 10}
        },
        "test-api": {
            "name": "test-api", 
            "description": "RESTful API service",
            "language": "Python",
            "topics": ["api", "fastapi", "postgresql"],
            "recent_activity": {"recent_commits": 5}
        }
    }
    
    opportunities = await market_research.generate_project_opportunities(
        mock_portfolio,
        max_opportunities=3
    )
    
    print(f"\nGenerated {len(opportunities)} project opportunities:")
    for opp in opportunities:
        print(f"\n📦 {opp.title}")
        print(f"   Problem: {opp.problem_statement[:100]}...")
        print(f"   Solution: {opp.solution_approach[:100]}...")
        tech_stack_preview = opp.tech_stack[:3] if isinstance(opp.tech_stack, list) else [str(opp.tech_stack)]
        print(f"   Tech Stack: {', '.join(tech_stack_preview)}")
        print(f"   Market Demand: {opp.market_demand:.2f}")
        print(f"   Innovation Score: {opp.innovation_score:.2f}")
    
    # Get research stats
    stats = market_research.get_research_stats()
    print(f"\n📈 Research Stats: {json.dumps(stats, indent=2)}")


async def test_portfolio_intelligence():
    """Test the portfolio intelligence system."""
    print("\n\n🧠 Testing Portfolio Intelligence...")
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create portfolio intelligence
    portfolio_intel = PortfolioIntelligence(ai_brain)
    
    # Analyze mock portfolio
    mock_portfolio = {
        "ai-assistant": {
            "name": "ai-assistant",
            "description": "AI-powered coding assistant using GPT-4",
            "language": "Python",
            "topics": ["ai", "gpt4", "automation"],
            "recent_activity": {"recent_commits": 25},
            "health_score": 85
        },
        "react-dashboard": {
            "name": "react-dashboard",
            "description": "Real-time analytics dashboard built with React",
            "language": "TypeScript", 
            "topics": ["react", "dashboard", "analytics"],
            "recent_activity": {"recent_commits": 15},
            "health_score": 90
        },
        "mobile-tracker": {
            "name": "mobile-tracker",
            "description": "Mobile app for fitness tracking",
            "language": "Swift",
            "topics": ["ios", "mobile", "health"],
            "recent_activity": {"recent_commits": 8},
            "health_score": 75
        }
    }
    
    print("\n🔍 Analyzing portfolio...")
    insights = await portfolio_intel.analyze_portfolio(mock_portfolio, force_refresh=True)
    
    print(f"\n📊 Portfolio Insights:")
    print(f"   Total Projects: {insights.total_projects}")
    print(f"   Technology Coverage: {dict(list(insights.technology_coverage.items())[:5])}")
    print(f"   Domain Coverage: {insights.domain_coverage}")
    print(f"   Innovation Leaders: {insights.innovation_leaders}")
    
    if insights.strategic_gaps:
        print(f"\n🎯 Strategic Gaps:")
        for gap in insights.strategic_gaps[:2]:
            print(f"   - {gap.get('gap_type')}: {gap.get('description', 'N/A')[:100]}...")
    
    if insights.growth_recommendations:
        print(f"\n📈 Growth Recommendations:")
        for rec in insights.growth_recommendations[:2]:
            print(f"   - {rec.get('title', 'N/A')}")
            print(f"     {rec.get('description', 'N/A')[:100]}...")
    
    # Test project synergies
    print("\n\n🔗 Testing Project Synergies...")
    synergies = portfolio_intel.get_project_synergies("ai-assistant")
    
    if synergies:
        print(f"Synergies for 'ai-assistant':")
        for syn in synergies:
            print(f"   - With {syn['project']}: Score {syn['synergy_score']:.2f}")
            print(f"     Reasons: {', '.join(syn['reasons'])}")


async def test_intelligent_work_finder():
    """Test the updated intelligent work finder."""
    print("\n\n🔎 Testing Intelligent Work Finder...")
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create system state
    system_state = {
        'projects': {
            "test-project-1": {
                "name": "test-project-1",
                "description": "E-commerce platform with AI recommendations",
                "language": "JavaScript",
                "topics": ["ecommerce", "ai", "nodejs"],
                "recent_activity": {"recent_commits": 20},
                "health_score": 88
            }
        },
        'repositories': {},
        'system_performance': {
            'total_cycles': 100,
            'failed_actions': 5
        }
    }
    
    # Create work finder
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    print("\n🔍 Discovering work opportunities...")
    work_items = await work_finder.discover_work(max_items=5)
    
    print(f"\nDiscovered {len(work_items)} work items:")
    for item in work_items:
        print(f"\n📋 {item.title}")
        print(f"   Type: {item.task_type}")
        print(f"   Priority: {item.priority.name}")
        print(f"   Description: {item.description[:100]}...")
        
        if item.metadata:
            if 'problem_statement' in item.metadata:
                print(f"   Problem: {item.metadata['problem_statement'][:80]}...")
            if 'tech_stack' in item.metadata:
                print(f"   Tech Stack: {', '.join(item.metadata['tech_stack'][:3])}")
    
    # Get discovery stats
    stats = work_finder.get_discovery_stats()
    print(f"\n📊 Discovery Stats: {json.dumps(stats, indent=2)}")


async def test_outcome_tracking():
    """Test the project outcome tracking system."""
    print("\n\n📈 Testing Project Outcome Tracking...")
    
    # Create AI brain
    ai_brain = IntelligentAIBrain()
    
    # Create outcome tracker
    tracker = ProjectOutcomeTracker(ai_brain, "test_outcomes.json")
    
    # Track a new project
    print("\n📝 Tracking new project...")
    await tracker.track_new_project(
        project_id="test-proj-123",
        project_name="AI Task Scheduler",
        project_metadata={
            'project_type': 'productivity',
            'tech_stack': ['Python', 'FastAPI', 'React'],
            'problem_statement': 'Developers waste time on manual task scheduling',
            'target_market': 'Software developers',
            'market_demand': 0.8,
            'innovation_score': 0.7,
            'complexity': 'medium'
        }
    )
    
    # Simulate project progress
    print("\n📊 Updating project metrics...")
    await tracker.update_project_metrics(
        "test-proj-123",
        {
            'health_score': 0.85,
            'commits_count': 45,
            'contributors_count': 3,
            'issues_closed': 12,
            'features_implemented': 8
        }
    )
    
    # Evaluate outcome
    print("\n🎯 Evaluating project outcome...")
    evaluation = await tracker.evaluate_project_outcome("test-proj-123")
    
    if 'error' not in evaluation:
        print(f"   Success Score: {evaluation.get('success_score', 0):.2f}")
        eval_details = evaluation.get('evaluation', {})
        print(f"   Success Factors: {', '.join(eval_details.get('success_factors', [])[:3])}")
        print(f"   Lessons Learned: {', '.join(eval_details.get('lessons_learned', [])[:2])}")
    
    # Generate learning insights
    print("\n🧠 Generating learning insights...")
    insights = await tracker.generate_learning_insights()
    
    print(f"Generated {len(insights)} insights:")
    for insight in insights[:3]:
        print(f"\n   - {insight.description}")
        print(f"     Confidence: {insight.confidence:.2f}")
        print(f"     Recommendations: {', '.join(insight.recommendations[:2])}")
    
    # Test recommendations for new project
    print("\n\n💡 Getting recommendations for new project...")
    recommendations = tracker.get_recommendations_for_new_project(
        project_type='productivity',
        tech_stack=['Python', 'React'],
        target_market='Software developers'
    )
    
    print(f"   Predicted Success Rate: {recommendations['predicted_success_rate']:.2f}")
    print(f"   Risk Factors: {', '.join(recommendations['risk_factors'][:2])}")
    print(f"   Success Factors: {', '.join(recommendations['success_factors'][:2])}")
    
    # Clean up test file
    import os
    if os.path.exists("test_outcomes.json"):
        os.remove("test_outcomes.json")


async def main():
    """Run all tests."""
    print("🚀 Testing Dynamic Portfolio Analysis System")
    print("=" * 60)
    
    try:
        # Test each component
        await test_market_research()
        await test_portfolio_intelligence()
        await test_intelligent_work_finder()
        await test_outcome_tracking()
        
        print("\n\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())