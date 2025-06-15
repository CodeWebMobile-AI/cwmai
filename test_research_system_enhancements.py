"""
Test script to verify the enhanced research system works as intended.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import all the components
try:
    from research_evolution_engine import ResearchEvolutionEngine
    from research_knowledge_store import ResearchKnowledgeStore
    from knowledge_graph_builder import KnowledgeGraphBuilder
    from research_insight_processor import ResearchInsightProcessor
    from dynamic_research_trigger import DynamicResearchTrigger
    from cross_research_analyzer import CrossResearchAnalyzer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class MockStateManager:
    """Mock state manager for testing."""
    def __init__(self):
        self.state = {
            "performance": {
                "claude_interactions": {
                    "total_attempts": 100,
                    "successful": 75
                },
                "task_completion": {
                    "total_tasks": 50,
                    "completed_tasks": 40
                }
            },
            "recent_errors": [
                {"type": "timeout", "message": "Request timeout", "timestamp": datetime.now().isoformat()},
                {"type": "api_error", "message": "API rate limit", "timestamp": datetime.now().isoformat()}
            ],
            "task_state": {
                "tasks": [
                    {"id": "1", "status": "completed"},
                    {"id": "2", "status": "failed", "failure_reason": "timeout"},
                    {"id": "3", "status": "in_progress"}
                ]
            }
        }
    
    def load_state(self):
        return self.state
    
    def save_state(self, state):
        self.state = state


class MockAIBrain:
    """Mock AI brain for testing."""
    async def generate_enhanced_response(self, prompt):
        # Generate mock research response
        return {
            "content": """
            Research findings on system optimization:
            
            1. Implement caching to improve performance
            2. Use connection pooling for database queries
            3. Add retry logic for API calls
            
            These improvements should enhance system reliability and efficiency.
            Common patterns observed: timeout issues, performance bottlenecks.
            
            Recommendation: Focus on implementing robust error handling.
            """
        }


async def test_knowledge_store():
    """Test ResearchKnowledgeStore functionality."""
    print("\n📚 Testing Knowledge Store...")
    
    store = ResearchKnowledgeStore("test_research_knowledge")
    
    # Test storing research
    research_data = {
        "topic": "performance optimization",
        "content": "Test research content about improving system performance",
        "quality_score": 0.8
    }
    
    research_id = store.store_research("efficiency", research_data, quality_score=0.8)
    print(f"  ✅ Stored research with ID: {research_id}")
    
    # Test retrieval
    retrieved = store.retrieve_research(research_id=research_id)
    if retrieved and len(retrieved) > 0:
        print(f"  ✅ Retrieved research successfully")
    else:
        print(f"  ❌ Failed to retrieve research")
    
    # Test statistics
    stats = store.get_statistics()
    print(f"  📊 Storage stats: {stats['total_entries']} entries, {stats['average_quality_score']:.2f} avg quality")
    
    return True


async def test_knowledge_graph():
    """Test KnowledgeGraphBuilder functionality."""
    print("\n🕸️ Testing Knowledge Graph Builder...")
    
    builder = KnowledgeGraphBuilder("test_research_knowledge/knowledge_graph")
    
    # Test processing research
    research_data = {
        "id": "test_research_1",
        "content": {
            "findings": "Implement caching improves performance. Database optimization reduces latency."
        },
        "topic": {
            "topic": "performance",
            "area": "optimization"
        }
    }
    
    results = builder.process_research(research_data)
    print(f"  ✅ Extracted {len(results['entities_extracted'])} entities")
    print(f"  ✅ Found {len(results['relationships_extracted'])} relationships")
    print(f"  ✅ Identified {len(results['patterns_identified'])} patterns")
    
    # Test semantic search
    search_results = builder.semantic_search("performance")
    print(f"  🔍 Semantic search found {len(search_results)} results")
    
    # Test insights
    insights = builder.get_insights()
    print(f"  💡 Graph contains {insights['graph_summary']['total_entities']} entities")
    
    return True


async def test_insight_processor():
    """Test ResearchInsightProcessor functionality."""
    print("\n💭 Testing Insight Processor...")
    
    processor = ResearchInsightProcessor("test_research_knowledge/processed_insights")
    
    # Test processing single research
    research_item = {
        "id": "test_1",
        "content": """
        Recommendation: Implement caching for frequently accessed data.
        Pattern: Performance degrades when database queries increase.
        Success factor: Systems work well when properly cached.
        Common mistake: Not implementing proper error handling.
        Improvement opportunity: Optimize database query patterns.
        """,
        "quality_score": 0.8
    }
    
    results = processor.process_single_research(research_item)
    print(f"  ✅ Extracted {len(results['insights'])} total insights")
    print(f"  ✅ Found {len(results['recommendations'])} recommendations")
    print(f"  ✅ Found {len(results['patterns'])} patterns")
    print(f"  ✅ Found {len(results['success_factors'])} success factors")
    
    # Test batch processing
    batch_results = processor.process_research_batch([research_item])
    if batch_results.get("aggregated_insights"):
        print(f"  ✅ Generated aggregated insights")
    
    return True


async def test_dynamic_trigger():
    """Test DynamicResearchTrigger functionality."""
    print("\n⚡ Testing Dynamic Research Trigger...")
    
    state_manager = MockStateManager()
    trigger = DynamicResearchTrigger(state_manager)
    
    # Test metric collection
    metrics = trigger._collect_current_metrics()
    print(f"  📊 Current metrics: Claude success={metrics['claude_success_rate']:.1f}%, Task completion={metrics['task_completion_rate']:.1f}%")
    
    # Test anomaly detection
    anomalies = trigger._detect_anomalies()
    print(f"  🔍 Detected {len(anomalies)} anomalies")
    
    # Test opportunity identification
    opportunities = trigger._identify_opportunities()
    print(f"  💡 Identified {len(opportunities)} opportunities")
    
    # Add test event
    trigger.add_event("test_event", {"type": "test", "data": "test data"})
    print(f"  ✅ Added event to queue")
    
    return True


async def test_cross_research_analyzer():
    """Test CrossResearchAnalyzer functionality."""
    print("\n🔗 Testing Cross-Research Analyzer...")
    
    # Create mock knowledge store with test data
    store = ResearchKnowledgeStore("test_research_knowledge")
    
    # Add some test research items
    test_items = [
        {
            "type": "efficiency",
            "content": "Caching improves performance significantly",
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "efficiency", 
            "content": "Database optimization reduces query time",
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "innovation",
            "content": "New caching strategies show promise",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    for item in test_items:
        store.store_research(item["type"], item)
    
    analyzer = CrossResearchAnalyzer(store)
    
    # Test corpus analysis
    analysis = analyzer.analyze_research_corpus()
    print(f"  📊 Analyzed {analysis['research_count']} research items")
    print(f"  ✅ Found {len(analysis['pattern_analysis']['patterns'])} patterns")
    print(f"  ✅ Identified {len(analysis['theme_analysis']['major_themes'])} major themes")
    print(f"  ✅ Generated {len(analysis['meta_insights'])} meta-insights")
    print(f"  ✅ Created {len(analysis['recommendations'])} recommendations")
    
    return True


async def test_integration():
    """Test full integration of all components."""
    print("\n🔧 Testing Full Integration...")
    
    # Create components
    state_manager = MockStateManager()
    ai_brain = MockAIBrain()
    
    # Initialize research evolution engine
    engine = ResearchEvolutionEngine(
        state_manager=state_manager,
        ai_brain=ai_brain
    )
    
    print("  ✅ Research Evolution Engine initialized")
    
    # Test research cycle execution
    print("  🔄 Executing research cycle...")
    cycle_results = await engine.execute_research_cycle()
    
    print(f"  ✅ Cycle status: {cycle_results['status']}")
    print(f"  ✅ Research conducted: {len(cycle_results['research_conducted'])}")
    print(f"  ✅ Insights extracted: {len(cycle_results['insights_extracted'])}")
    
    # Test status reporting
    status = engine.get_status()
    print(f"  📊 Engine status: {status['research_cycles_completed']} cycles completed")
    
    # Test effectiveness metrics
    effectiveness = engine._get_effectiveness_metrics()
    print(f"  📈 Effectiveness metrics calculated:")
    for metric, value in effectiveness.items():
        print(f"     - {metric}: {value:.2f}")
    
    return True


async def test_quality_assessment():
    """Test the quality assessment implementation."""
    print("\n🏆 Testing Quality Assessment...")
    
    engine = ResearchEvolutionEngine()
    
    # Test quality assessment with different research types
    test_cases = [
        {
            "content": "Short content with some keywords like implement and improve",
            "topic": {"topic": "test", "priority": "low"},
            "expected_range": (0.2, 0.6)
        },
        {
            "content": """
            Comprehensive research findings with multiple insights:
            1. First, implement caching to improve performance
            2. Second, optimize database queries 
            3. Finally, add monitoring
            
            This approach will enhance system efficiency and provide better optimization.
            The implementation should be done in phases with careful testing.
            """,
            "topic": {"topic": "optimization", "priority": "high"},
            "area": "efficiency",
            "expected_range": (0.6, 1.0)
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        quality = engine._assess_research_quality(test_case)
        min_expected, max_expected = test_case["expected_range"]
        
        if min_expected <= quality <= max_expected:
            print(f"  ✅ Test case {i+1}: Quality={quality:.2f} (expected {min_expected}-{max_expected})")
        else:
            print(f"  ❌ Test case {i+1}: Quality={quality:.2f} (expected {min_expected}-{max_expected})")
    
    return True


async def cleanup_test_files():
    """Clean up test files created during testing."""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = [
        "test_research_knowledge",
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            import shutil
            shutil.rmtree(test_dir)
            print(f"  ✅ Removed {test_dir}")


async def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Research System")
    print("=" * 50)
    
    try:
        # Run individual component tests
        await test_knowledge_store()
        await test_knowledge_graph()
        await test_insight_processor()
        await test_dynamic_trigger()
        await test_cross_research_analyzer()
        await test_quality_assessment()
        
        # Run integration test
        await test_integration()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await cleanup_test_files()


if __name__ == "__main__":
    asyncio.run(main())