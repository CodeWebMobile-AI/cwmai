"""
Test Smart Task Generation System

This script demonstrates the enhanced intelligent task generation system
with all the new features integrated.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_smart_task_generation():
    """Test the enhanced task generation system."""
    logger.info("=== Starting Smart Task Generation Test ===")
    
    # Import components
    from scripts.ai_brain import IntelligentAIBrain
    from scripts.intelligent_task_generator import IntelligentTaskGenerator
    from scripts.progressive_task_generator import ProgressiveTaskGenerator
    from scripts.smart_context_aggregator import SmartContextAggregator
    from scripts.predictive_task_engine import PredictiveTaskEngine
    from scripts.task_intelligence_dashboard import TaskIntelligenceDashboard
    from scripts.complexity_analyzer import ComplexityAnalyzer
    from scripts.hierarchical_task_manager import HierarchicalTaskManager
    
    # Initialize AI brain
    logger.info("Initializing AI brain...")
    ai_brain = IntelligentAIBrain()
    
    # Initialize context aggregator
    logger.info("Initializing smart context aggregator...")
    context_aggregator = SmartContextAggregator()
    
    # Initialize predictive engine
    logger.info("Initializing predictive task engine...")
    predictive_engine = PredictiveTaskEngine()
    
    # Initialize task generators with enhanced features
    logger.info("Initializing enhanced task generators...")
    
    # Mock charter system
    class MockCharterSystem:
        async def get_current_charter(self):
            return {
                'TASK_TYPES': ['FEATURE', 'BUG_FIX', 'TESTING', 'DOCUMENTATION'],
                'PROJECT_METHODOLOGY': 'Agile with AI-driven prioritization',
                'DECISION_PRINCIPLES': ['Value-driven', 'Data-informed', 'Adaptive']
            }
    
    charter_system = MockCharterSystem()
    
    # Initialize generators
    task_generator = IntelligentTaskGenerator(
        ai_brain=ai_brain,
        charter_system=charter_system,
        context_aggregator=context_aggregator,
        predictive_engine=predictive_engine
    )
    
    # Initialize progressive generator
    complexity_analyzer = ComplexityAnalyzer(ai_brain)
    hierarchical_manager = HierarchicalTaskManager()
    progressive_generator = ProgressiveTaskGenerator(
        ai_brain=ai_brain,
        hierarchical_manager=hierarchical_manager,
        complexity_analyzer=complexity_analyzer,
        context_aggregator=context_aggregator,
        predictive_engine=predictive_engine
    )
    
    # Initialize dashboard
    logger.info("Initializing task intelligence dashboard...")
    dashboard = TaskIntelligenceDashboard(
        task_generator=task_generator,
        progressive_generator=progressive_generator,
        predictive_engine=predictive_engine,
        context_aggregator=context_aggregator,
        output_dir="test_dashboard_output"
    )
    
    # Test 1: Gather comprehensive context
    logger.info("\n=== Test 1: Comprehensive Context Gathering ===")
    context = await context_aggregator.gather_comprehensive_context()
    logger.info(f"Context quality score: {context.quality_score:.2%}")
    logger.info(f"Repository health scores: {len(context.repository_health)}")
    logger.info(f"Cross-repo patterns found: {len(context.cross_repo_patterns)}")
    logger.info(f"External signals: {len(context.external_signals)}")
    
    # Test 2: Predictive task generation
    logger.info("\n=== Test 2: Predictive Task Generation ===")
    
    # Convert context for predictive engine
    context_dict = {
        'repository_health': context.repository_health,
        'recent_activities': context.recent_activities,
        'system_capabilities': context.system_capabilities
    }
    
    predictions = await predictive_engine.predict_next_tasks(context_dict)
    logger.info(f"Generated {len(predictions)} task predictions")
    
    for i, pred in enumerate(predictions[:3]):
        logger.info(f"\nPrediction {i+1}:")
        logger.info(f"  Type: {pred.task_type}")
        logger.info(f"  Repository: {pred.repository}")
        logger.info(f"  Urgency: {pred.urgency}")
        logger.info(f"  Confidence: {pred.confidence:.2%}")
        logger.info(f"  Triggers: {', '.join(pred.trigger_factors)}")
    
    # Test 3: Generate smart tasks
    logger.info("\n=== Test 3: Smart Task Generation ===")
    
    # Create test context
    test_context = {
        'projects': [
            {
                'name': 'test-api-service',
                'full_name': 'org/test-api-service',
                'health_score': 65,
                'language': 'Python',
                'metrics': {'issues_open': 5}
            },
            {
                'name': 'test-frontend',
                'full_name': 'org/test-frontend',
                'health_score': 82,
                'language': 'React',
                'metrics': {'issues_open': 2}
            }
        ],
        'recent_tasks': [],
        'capabilities': ['testing', 'documentation', 'code_analysis'],
        'market_trends': ['AI integration', 'Security focus', 'Performance optimization']
    }
    
    # Generate multiple diverse tasks
    tasks = await task_generator.generate_multiple_tasks(test_context, count=5)
    logger.info(f"Generated {len(tasks)} diverse tasks")
    
    for i, task in enumerate(tasks):
        logger.info(f"\nTask {i+1}:")
        logger.info(f"  Title: {task.get('title', 'No title')}")
        logger.info(f"  Type: {task.get('type', 'Unknown')}")
        logger.info(f"  Repository: {task.get('repository', 'None')}")
        logger.info(f"  Priority: {task.get('priority', 'medium')}")
        logger.info(f"  Generation reason: {task.get('generation_context', {}).get('generation_reason', 'Unknown')}")
        
        # Check for enhanced features
        if 'complexity_analysis' in task:
            logger.info(f"  Complexity: {task['complexity_analysis']['level']}")
        if 'decomposition' in task:
            logger.info(f"  Decomposed into: {task['decomposition']['sub_task_count']} sub-tasks")
    
    # Test 4: Progressive task generation
    logger.info("\n=== Test 4: Progressive Task Generation ===")
    
    # Simulate a completed task
    completed_task = {
        'id': 'task_001',
        'type': 'FEATURE',
        'title': 'Implement user authentication',
        'repository': 'test-api-service',
        'status': 'completed',
        'completion_time': 4.5
    }
    
    from scripts.progressive_task_generator import ProgressionContext
    prog_context = ProgressionContext(
        completed_task=completed_task,
        repository_context={'name': 'test-api-service', 'health': 0.65},
        project_state={'phase': 'development'},
        recent_patterns=['authentication', 'api'],
        current_priorities=['security', 'user_experience'],
        ai_agent_capacity={'available': True},
        processing_constraints={}
    )
    
    next_tasks = await progressive_generator.generate_next_tasks(completed_task, prog_context)
    logger.info(f"Generated {len(next_tasks)} follow-up task suggestions")
    
    for i, suggestion in enumerate(next_tasks):
        logger.info(f"\nSuggestion {i+1}:")
        logger.info(f"  Title: {suggestion.title}")
        logger.info(f"  Type: {suggestion.task_type}")
        logger.info(f"  Relationship: {suggestion.relationship.value}")
        logger.info(f"  Confidence: {suggestion.confidence:.2%}")
        logger.info(f"  Reason: {suggestion.trigger_reason}")
    
    # Test 5: Train predictive models (with mock data)
    logger.info("\n=== Test 5: Predictive Model Training ===")
    
    # Create mock historical data
    historical_tasks = []
    for i in range(100):
        historical_tasks.append({
            'id': f'hist_{i}',
            'type': ['FEATURE', 'BUG_FIX', 'TESTING', 'DOCUMENTATION'][i % 4],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'status': 'completed' if i % 3 != 0 else 'failed',
            'priority': ['high', 'medium', 'low'][i % 3],
            'value_score': 0.5 + (i % 10) / 20,
            'repository': f'repo_{i % 5}',
            'requirements': ['req1', 'req2'] if i % 2 == 0 else ['req1'],
            'estimated_hours': 2 + (i % 8)
        })
    
    # Train models
    performance = await predictive_engine.train_models(historical_tasks, [])
    logger.info(f"Model training complete: {performance}")
    
    # Test 6: Generate dashboard
    logger.info("\n=== Test 6: Task Intelligence Dashboard ===")
    
    dashboard_data = await dashboard.generate_dashboard()
    logger.info(f"Dashboard generated with visualizations:")
    for viz_name, viz_path in dashboard_data['visualizations'].items():
        logger.info(f"  - {viz_name}: {viz_path}")
    
    logger.info(f"\nDashboard Summary:")
    summary = dashboard_data['summary']
    logger.info(f"  Status: {summary['overall_status']}")
    logger.info(f"  Key Metrics: {json.dumps(summary['key_metrics'], indent=2)}")
    
    if summary['insights']:
        logger.info(f"  Insights:")
        for insight in summary['insights']:
            logger.info(f"    - {insight}")
    
    if summary['recommendations']:
        logger.info(f"  Recommendations:")
        for rec in summary['recommendations']:
            logger.info(f"    - {rec}")
    
    # Test 7: Demonstrate cross-repository awareness
    logger.info("\n=== Test 7: Cross-Repository Intelligence ===")
    
    # Check cross-repo patterns
    if context.cross_repo_patterns:
        logger.info("Cross-repository patterns detected:")
        for pattern in context.cross_repo_patterns[:3]:
            logger.info(f"  Pattern: {pattern['pattern']}")
            logger.info(f"  Repositories: {', '.join(pattern['repositories'])}")
            logger.info(f"  Confidence: {pattern['confidence']:.2%}")
            logger.info(f"  Opportunity: {pattern.get('opportunity', 'Unknown')}")
    
    # Test 8: Early warning detection
    logger.info("\n=== Test 8: Early Warning System ===")
    
    warnings = await predictive_engine.detect_early_warnings(context_dict, historical_tasks)
    logger.info(f"Detected {len(warnings)} early warnings")
    
    for warning in warnings[:3]:
        logger.info(f"\nWarning: {warning.warning_type}")
        logger.info(f"  Severity: {warning.severity}")
        logger.info(f"  Affected: {', '.join(warning.affected_components)}")
        logger.info(f"  Time to impact: {warning.time_to_impact}")
        logger.info(f"  Recommended actions:")
        for action in warning.recommended_actions:
            logger.info(f"    - {action}")
    
    # Final analytics
    logger.info("\n=== Final System Analytics ===")
    
    gen_analytics = task_generator.get_generation_analytics()
    logger.info(f"Total tasks generated: {gen_analytics['total_generated']}")
    logger.info(f"Intelligence features status: {json.dumps(gen_analytics['intelligence_features'], indent=2)}")
    
    if 'prediction_confidence' in gen_analytics:
        logger.info(f"Prediction confidence: {gen_analytics['prediction_confidence']['overall']:.2%}")
    
    prog_analytics = progressive_generator.get_progression_analytics()
    logger.info(f"Total progression patterns: {prog_analytics['total_patterns']}")
    logger.info(f"Cross-project insights: {json.dumps(prog_analytics.get('cross_project_insights', {}), indent=2)}")
    
    logger.info("\n=== Smart Task Generation Test Complete ===")
    
    # Save test results
    test_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tests_run': 8,
        'status': 'success',
        'key_findings': {
            'context_quality': context.quality_score,
            'predictions_generated': len(predictions),
            'tasks_generated': len(tasks),
            'follow_up_suggestions': len(next_tasks),
            'warnings_detected': len(warnings),
            'dashboard_status': summary['overall_status']
        }
    }
    
    with open('test_results_smart_generation.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"\nTest results saved to test_results_smart_generation.json")

if __name__ == "__main__":
    asyncio.run(test_smart_task_generation())