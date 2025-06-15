#!/usr/bin/env python3
"""
Test script for the Enhanced Intelligent Work Finder
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from scripts.enhanced_intelligent_work_finder import EnhancedIntelligentWorkFinder
from scripts.ai_brain import IntelligentAIBrain
from scripts.work_item_types import WorkItem, TaskPriority


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_enhanced_work_finder():
    """Test the enhanced intelligent work finder."""
    
    # Create AI brain instance
    ai_brain = IntelligentAIBrain({})
    
    # Create test system state
    system_state = {
        'projects': {
            'test-cms-project': {
                'description': 'A content management system built with Laravel',
                'language': 'PHP',
                'topics': ['cms', 'laravel', 'php'],
                'health_score': 75,
                'recent_activity': {
                    'recent_commits': 5,
                    'last_commit_date': '2024-01-10T10:00:00Z'
                }
            },
            'ai-analytics-tool': {
                'description': 'AI-powered analytics dashboard',
                'language': 'TypeScript', 
                'topics': ['ai', 'analytics', 'typescript'],
                'health_score': 90,
                'recent_activity': {
                    'recent_commits': 12,
                    'last_commit_date': '2024-01-12T15:00:00Z'
                }
            }
        },
        'system_performance': {
            'failed_actions': 2,
            'total_cycles': 100,
            'error_rate': 0.02,
            'learning_metrics': {
                'resource_efficiency': 0.85
            }
        }
    }
    
    # Create enhanced work finder
    finder = EnhancedIntelligentWorkFinder(ai_brain, system_state)
    
    logger.info("🧪 Testing Enhanced Intelligent Work Finder")
    logger.info("=" * 60)
    
    # Test 1: Discover work with active projects
    logger.info("\n📋 Test 1: Discovering work with active projects")
    work_items = await finder.discover_work(max_items=5, current_workload=2)
    
    logger.info(f"Discovered {len(work_items)} work items:")
    for item in work_items:
        logger.info(f"\n  📌 {item.title}")
        logger.info(f"     Type: {item.task_type}")
        logger.info(f"     Priority: {item.priority.name}")
        logger.info(f"     Repository: {item.repository}")
        logger.info(f"     Predicted Value: {item.metadata.get('predicted_value', 'N/A')}")
        logger.info(f"     AI Generated: {item.metadata.get('ai_generated', False)}")
        logger.info(f"     Creative: {item.metadata.get('creative', False)}")
    
    # Test 2: Test with no active projects (bootstrap mode)
    logger.info("\n📋 Test 2: Testing bootstrap mode (no active projects)")
    empty_state = {'projects': {}, 'system_performance': {}}
    finder_bootstrap = EnhancedIntelligentWorkFinder(ai_brain, empty_state)
    
    bootstrap_items = await finder_bootstrap.discover_work(max_items=3, current_workload=0)
    
    logger.info(f"Bootstrap generated {len(bootstrap_items)} new project ideas:")
    for item in bootstrap_items:
        logger.info(f"\n  🚀 {item.title}")
        logger.info(f"     Type: {item.task_type}")
        if 'project_name' in item.metadata:
            logger.info(f"     Project Name: {item.metadata['project_name']}")
    
    # Test 3: Simulate task outcomes and learning
    logger.info("\n📋 Test 3: Testing learning system")
    
    # Record some task outcomes
    if work_items:
        # Simulate successful task
        await finder.record_task_outcome(work_items[0], {
            'success': True,
            'value_created': 8.5,
            'execution_time': 120,
            'issue_number': 123
        })
        
        # Simulate failed task
        if len(work_items) > 1:
            await finder.record_task_outcome(work_items[1], {
                'success': False,
                'error': 'Failed to complete task',
                'execution_time': 30
            })
    
    # Test 4: Get discovery statistics
    logger.info("\n📋 Test 4: Discovery Statistics")
    stats = finder.get_discovery_stats()
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # Test 5: Test duplicate detection
    logger.info("\n📋 Test 5: Testing intelligent deduplication")
    
    # Try to discover work again - should filter duplicates intelligently
    work_items_2 = await finder.discover_work(max_items=5, current_workload=3)
    
    logger.info(f"Second discovery found {len(work_items_2)} items (after deduplication)")
    
    # Compare with first discovery
    first_titles = {item.title for item in work_items}
    second_titles = {item.title for item in work_items_2}
    duplicates = first_titles.intersection(second_titles)
    
    if duplicates:
        logger.info(f"⚠️  Found {len(duplicates)} potential duplicates: {duplicates}")
    else:
        logger.info("✅ No exact duplicates found - deduplication working!")
    
    # Test 6: Test high workload scenario
    logger.info("\n📋 Test 6: Testing high workload adaptation")
    high_load_items = await finder.discover_work(max_items=5, current_workload=8)
    
    logger.info(f"High workload discovery: {len(high_load_items)} items")
    priorities = [item.priority.name for item in high_load_items]
    logger.info(f"Priority distribution: {priorities}")
    
    logger.info("\n✅ All tests completed!")
    
    # Summary comparison
    logger.info("\n📊 Comparison: Old vs Enhanced Work Finder")
    logger.info("=" * 60)
    logger.info("Old Work Finder:")
    logger.info("  - Hard-coded task templates")
    logger.info("  - Random selection from lists")
    logger.info("  - No learning or adaptation")
    logger.info("  - Static cooldown periods")
    logger.info("  - No value prediction")
    
    logger.info("\nEnhanced Work Finder:")
    logger.info("  - AI-driven task discovery")
    logger.info("  - Context-aware generation")
    logger.info("  - Learning from outcomes")
    logger.info("  - Dynamic priority and cooldowns")
    logger.info("  - Value prediction and optimization")
    logger.info("  - Market trend awareness")
    logger.info("  - Creative task generation")


if __name__ == "__main__":
    asyncio.run(test_enhanced_work_finder())