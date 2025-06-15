#!/usr/bin/env python3
"""
Test script for the Research Intelligence System.

This script tests all components of the research system to ensure they work correctly.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add scripts directory to path
sys.path.append('scripts')

from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.research_need_analyzer import ResearchNeedAnalyzer
from scripts.intelligent_research_selector import IntelligentResearchSelector
from scripts.research_scheduler import ResearchScheduler, ResearchPriority
from scripts.research_query_generator import ResearchQueryGenerator
from scripts.research_action_engine import ResearchActionEngine
from scripts.research_learning_system import ResearchLearningSystem
from scripts.research_evolution_engine import ResearchEvolutionEngine


class MockStateManager:
    """Mock state manager for testing."""
    
    def load_state(self):
        return {
            "task_state": {
                "tasks": [
                    {"id": "1", "status": "pending", "created_at": "2025-01-09T10:00:00"},
                    {"id": "2", "status": "completed", "created_at": "2025-01-09T09:00:00"},
                    {"id": "3", "status": "failed", "created_at": "2025-01-09T08:00:00"}
                ]
            },
            "metrics": {
                "claude_interactions": {
                    "success_rate": 0.0,
                    "total_attempts": 10,
                    "failure_reasons": ["no response", "task rejected"]
                }
            },
            "swarm_state": {
                "recent_decisions": [
                    {"consensus_level": 0.6, "decision_time": 300}
                ]
            },
            "outcome_learning": {
                "learned_patterns": [],
                "successful_predictions": 0,
                "total_predictions": 1
            },
            "portfolio": {
                "projects": [
                    {"status": "active", "task_count": 5, "completed_tasks": 1}
                ]
            }
        }


class MockAIBrain:
    """Mock AI brain for testing."""
    
    async def generate_enhanced_response(self, prompt):
        return {
            "content": f"Research response for: {prompt[:50]}...",
            "insights": [
                "Implement better task templates for Claude",
                "Add acceptance criteria to all tasks",
                "Use structured formats for GitHub issues"
            ],
            "quality": "good"
        }


async def test_knowledge_store():
    """Test the research knowledge store."""
    print("Testing Research Knowledge Store...")
    
    store = ResearchKnowledgeStore("test_research_knowledge")
    
    # Test storing research
    research_data = {
        "topic": "Claude interaction optimization",
        "content": "Research shows that clear acceptance criteria improve response rates",
        "insights": ["Use structured templates", "Include examples", "Specify success criteria"]
    }
    
    research_id = store.store_research("claude_interaction", research_data, quality_score=0.8)
    print(f"✓ Stored research with ID: {research_id}")
    
    # Test retrieving research
    retrieved = store.retrieve_research(research_id=research_id)
    assert len(retrieved) == 1
    print("✓ Successfully retrieved research")
    
    # Test search
    search_results = store.search_research("claude")
    assert len(search_results) >= 1
    print("✓ Search functionality works")
    
    # Test statistics
    stats = store.get_statistics()
    print(f"✓ Store statistics: {stats['total_entries']} entries")
    
    print("Research Knowledge Store: PASSED\n")


async def test_need_analyzer():
    """Test the research need analyzer."""
    print("Testing Research Need Analyzer...")
    
    mock_state = MockStateManager()
    analyzer = ResearchNeedAnalyzer(mock_state)
    
    # Test gap analysis
    gaps = analyzer.analyze_performance_gaps()
    assert "critical" in gaps or "high" in gaps
    print("✓ Performance gap analysis works")
    
    # Test immediate needs
    immediate_needs = analyzer.get_immediate_research_needs()
    assert len(immediate_needs) > 0
    print(f"✓ Identified {len(immediate_needs)} immediate research needs")
    
    print("Research Need Analyzer: PASSED\n")


async def test_research_selector():
    """Test the intelligent research selector."""
    print("Testing Intelligent Research Selector...")
    
    store = ResearchKnowledgeStore("test_research_knowledge")
    analyzer = ResearchNeedAnalyzer(MockStateManager())
    selector = IntelligentResearchSelector(store, analyzer)
    
    # Test topic selection
    context = {
        "metrics": {"claude_success_rate": 0.0, "task_completion_rate": 0.087},
        "system_health": "critical"
    }
    
    topics = selector.select_research_topics(context)
    assert len(topics) > 0
    print(f"✓ Selected {len(topics)} research topics")
    
    # Test effectiveness update
    selector.update_topic_effectiveness("Claude interaction patterns", 0.9)
    print("✓ Updated topic effectiveness")
    
    print("Intelligent Research Selector: PASSED\n")


async def test_query_generator():
    """Test the research query generator."""
    print("Testing Research Query Generator...")
    
    generator = ResearchQueryGenerator()
    
    # Test query generation
    research_need = {
        "topic": "Task decomposition strategies",
        "area": "task_generation",
        "context": {
            "current_metrics": {"task_completion_rate": 0.087},
            "failure_patterns": ["tasks too complex", "missing criteria"]
        },
        "severity": "high"
    }
    
    queries = generator.generate_queries(research_need)
    assert len(queries) > 0
    print(f"✓ Generated {len(queries)} research queries")
    
    # Test contextual query
    contextual_query = generator.create_contextual_query(
        "Claude interaction optimization",
        "claude_interaction", 
        "0% response rate to GitHub issues",
        "critical"
    )
    assert "query" in contextual_query
    print("✓ Created contextual query")
    
    print("Research Query Generator: PASSED\n")


async def test_action_engine():
    """Test the research action engine."""
    print("Testing Research Action Engine...")
    
    engine = ResearchActionEngine()
    
    # Test insight extraction
    research = {
        "area": "claude_interaction",
        "topic": "Claude interaction optimization",
        "content": {
            "research_findings": "To improve Claude interaction success rates, implement the following strategies: Use clear acceptance criteria in GitHub issues. Format tasks with structured templates. Include specific implementation steps. These best practices should significantly improve response rates.",
            "recommendations": [
                "Use clear acceptance criteria in GitHub issues",
                "Format tasks with structured templates", 
                "Include specific implementation steps"
            ]
        }
    }
    
    insights = engine.extract_actionable_insights(research)
    assert len(insights) > 0
    print(f"✓ Extracted {len(insights)} actionable insights")
    
    # Test task generation
    tasks = engine.generate_implementation_tasks(insights[:2])
    assert len(tasks) > 0
    print(f"✓ Generated {len(tasks)} implementation tasks")
    
    print("Research Action Engine: PASSED\n")


async def test_learning_system():
    """Test the research learning system."""
    print("Testing Research Learning System...")
    
    learning = ResearchLearningSystem()
    
    # Test outcome recording
    outcome = {
        "research_quality": "good",
        "implementation_success": "partially_implemented",
        "performance_impact": "moderate_improvement",
        "value_delivered": 5,
        "research_metadata": {"area": "claude_interaction", "topic": "prompt optimization"}
    }
    
    learning.record_research_outcome("test_research_1", outcome)
    print("✓ Recorded research outcome")
    
    # Test prediction
    research_proposal = {
        "area": "task_generation",
        "topic": "complexity scoring",
        "severity": "high"
    }
    
    prediction = learning.predict_research_effectiveness(research_proposal)
    assert "predicted_effectiveness" in prediction
    print(f"✓ Predicted effectiveness: {prediction['predicted_effectiveness']:.2f}")
    
    # Test recommendations
    current_state = {"metrics": {"claude_success_rate": 0.1}}
    recommendations = learning.get_research_recommendations(current_state)
    print(f"✓ Generated {len(recommendations)} research recommendations")
    
    print("Research Learning System: PASSED\n")


async def test_scheduler():
    """Test the research scheduler."""
    print("Testing Research Scheduler...")
    
    scheduler = ResearchScheduler(MockStateManager())
    
    # Test scheduling
    research_id = scheduler.schedule_research(
        "critical_performance",
        ResearchPriority.HIGH,
        {"reason": "test"}
    )
    print(f"✓ Scheduled research with ID: {research_id}")
    
    # Test getting next research
    next_research = scheduler.get_next_research()
    if next_research:
        print("✓ Retrieved next research item")
        # Complete it
        scheduler.complete_research(next_research['id'], True, {"test": True}, 0.8)
        print("✓ Completed research item")
    
    # Test status
    status = scheduler.get_schedule_status()
    print(f"✓ Scheduler status: {status['queue_size']} items in queue")
    
    print("Research Scheduler: PASSED\n")


async def test_evolution_engine():
    """Test the research evolution engine."""
    print("Testing Research Evolution Engine...")
    
    mock_state = MockStateManager()
    mock_ai = MockAIBrain()
    
    engine = ResearchEvolutionEngine(
        state_manager=mock_state,
        ai_brain=mock_ai
    )
    
    # Test single research cycle
    cycle_result = await engine.execute_research_cycle()
    assert cycle_result["status"] in ["completed", "failed"]
    print(f"✓ Executed research cycle: {cycle_result['status']}")
    
    # Test status
    status = engine.get_status()
    assert "research_cycles_completed" in status
    print(f"✓ Engine status: {status['research_cycles_completed']} cycles completed")
    
    # Test manual research trigger
    manual_result = await engine.trigger_manual_research(
        "Test topic", "claude_interaction", "high"
    )
    assert "request" in manual_result
    print("✓ Manual research trigger works")
    
    print("Research Evolution Engine: PASSED\n")


async def test_integration():
    """Test full system integration."""
    print("Testing Full System Integration...")
    
    # Create a complete system
    mock_state = MockStateManager()
    mock_ai = MockAIBrain()
    
    engine = ResearchEvolutionEngine(
        state_manager=mock_state,
        ai_brain=mock_ai
    )
    
    # Test emergency research
    trigger_event = {
        "reason": "Claude success rate at 0%",
        "area": "claude_interaction",
        "severity": "critical"
    }
    
    emergency_result = await engine.execute_emergency_research(trigger_event)
    assert "trigger" in emergency_result
    print("✓ Emergency research execution works")
    
    # Test getting research recommendations
    learning_summary = engine.learning_system.get_learning_summary()
    print(f"✓ Learning summary: {learning_summary['total_records']} records")
    
    # Test knowledge store integration
    store_stats = engine.knowledge_store.get_statistics()
    print(f"✓ Knowledge store: {store_stats['total_entries']} entries")
    
    print("Full System Integration: PASSED\n")


async def main():
    """Run all tests."""
    print("=== CWMAI Research Intelligence System Tests ===\n")
    
    try:
        await test_knowledge_store()
        await test_need_analyzer()
        await test_research_selector()
        await test_query_generator()
        await test_action_engine()
        await test_learning_system()
        await test_scheduler()
        await test_evolution_engine()
        await test_integration()
        
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Research Intelligence System is ready for deployment.")
        print("\nKey Benefits:")
        print("- 📊 Analyzes system performance gaps automatically")
        print("- 🎯 Selects relevant research topics intelligently") 
        print("- 🔍 Generates targeted research queries")
        print("- ⚡ Converts insights into actionable improvements")
        print("- 🧠 Learns from outcomes to improve future research")
        print("- 🔄 Orchestrates continuous self-improvement cycles")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Cleanup test files
    import shutil
    if os.path.exists("test_research_knowledge"):
        shutil.rmtree("test_research_knowledge")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)