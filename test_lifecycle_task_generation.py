#!/usr/bin/env python3
"""Test script to demonstrate lifecycle-aware task generation."""

import asyncio
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager
from scripts.dynamic_charter import DynamicCharterSystem


async def test_lifecycle_analysis():
    """Test repository lifecycle analysis and task generation."""
    print("\n=== Testing Lifecycle-Aware Task Generation ===\n")
    
    # Initialize components
    state_manager = StateManager()
    system_state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(system_state, {})
    charter_system = DynamicCharterSystem(ai_brain)
    
    # Initialize repository analyzer with AI brain
    analyzer = RepositoryAnalyzer(ai_brain=ai_brain)
    
    # Initialize task generator
    task_generator = IntelligentTaskGenerator(
        ai_brain=ai_brain,
        charter_system=charter_system
    )
    
    # Test repositories at different lifecycle stages
    test_repos = [
        "CodeWebMobile-AI/modern-ai-dashboard",  # New project
        "CodeWebMobile-AI/moderncms-with-ai-powered-content-recommendations",  # Active development
        "CodeWebMobile-AI/ai-creative-studio"  # More mature
    ]
    
    for repo_name in test_repos:
        print(f"\n{'='*60}")
        print(f"Analyzing repository: {repo_name}")
        print('='*60)
        
        try:
            # Analyze repository
            repo_analysis = await analyzer.analyze_repository(repo_name)
            
            # Check if lifecycle analysis was performed
            if 'lifecycle_analysis' in repo_analysis:
                lifecycle = repo_analysis['lifecycle_analysis']
                print(f"\nLifecycle Stage: {lifecycle.get('current_stage', 'Unknown')}")
                print(f"Stage Confidence: {lifecycle.get('stage_confidence', 0):.1%}")
                
                # Show stage indicators
                indicators = lifecycle.get('stage_indicators', {})
                print(f"\nKey Indicators:")
                print(f"  - Repository Age: {indicators.get('repository_age_days', 0)} days")
                print(f"  - Commit Frequency: {indicators.get('commit_frequency_per_week', 0):.1f}/week")
                print(f"  - Documentation Score: {indicators.get('documentation_score', 0):.1%}")
                print(f"  - Test Coverage Score: {indicators.get('test_coverage_score', 0):.1%}")
                print(f"  - CI/CD Maturity: {indicators.get('ci_cd_maturity', 0):.1%}")
                
                # Show appropriate task types
                print(f"\nAppropriate Task Types for Stage:")
                for task_type in lifecycle.get('appropriate_task_types', [])[:5]:
                    print(f"  - {task_type}")
                
                # Show transition plan
                transition = lifecycle.get('transition_plan', {})
                if transition.get('next_stage'):
                    print(f"\nTransition to Next Stage ({transition['next_stage']}):")
                    print(f"  - Current Readiness: {transition.get('current_readiness', 0):.1%}")
                    print(f"  - Required Tasks:")
                    for task in transition.get('required_tasks', [])[:3]:
                        print(f"    • {task}")
                
                # Show lifecycle insights
                insights = lifecycle.get('lifecycle_insights', {})
                if insights:
                    print(f"\nLifecycle Insights:")
                    print(f"  - Health Assessment: {insights.get('health_assessment', 'Unknown')}")
                    print(f"  - Growth Trajectory: {insights.get('growth_trajectory', 'Unknown')}")
                    if insights.get('risk_factors'):
                        print(f"  - Risk Factors:")
                        for risk in insights['risk_factors'][:3]:
                            print(f"    • {risk}")
            else:
                print("\nNo lifecycle analysis available")
            
            # Generate a task for this repository
            print(f"\n--- Generating Task for {repo_name} ---")
            
            context = {
                'projects': {repo_name: repo_analysis},
                'active_projects': [repo_name]
            }
            
            task = await task_generator.generate_task_for_repository(
                repo_name, repo_analysis, context
            )
            
            if not task.get('skip'):
                print(f"\nGenerated Task:")
                print(f"  Type: {task.get('type')}")
                print(f"  Title: {task.get('title')}")
                print(f"  Priority: {task.get('priority')}")
                
                # Show lifecycle metadata
                lifecycle_meta = task.get('lifecycle_metadata', {})
                if lifecycle_meta:
                    print(f"\nLifecycle Metadata:")
                    print(f"  - Current Stage: {lifecycle_meta.get('current_stage')}")
                    print(f"  - Appropriate for Stage: {lifecycle_meta.get('appropriate_for_stage')}")
                    print(f"  - Helps Transition: {lifecycle_meta.get('helps_transition')}")
                
                # Show how task helps progression
                if task.get('stage_progression_value'):
                    print(f"\nStage Progression Value:")
                    print(f"  {task['stage_progression_value']}")
            else:
                print(f"\nTask generation skipped: {task.get('reason')}")
                
        except Exception as e:
            print(f"\nError analyzing {repo_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n=== Testing Project Planning ===\n")
    
    # Test project planning for one repository
    if test_repos:
        try:
            from scripts.project_planner import ProjectPlanner
            
            repo_name = test_repos[0]
            print(f"Creating project roadmap for: {repo_name}")
            
            # Get fresh analysis
            repo_analysis = await analyzer.analyze_repository(repo_name)
            
            # Initialize planner
            planner = ProjectPlanner(ai_brain=ai_brain)
            
            # Create roadmap
            roadmap = await planner.create_project_roadmap(repo_analysis)
            
            print(f"\nProject Roadmap:")
            print(f"  Project: {roadmap.project_name}")
            print(f"  Type: {roadmap.project_type}")
            print(f"  Current Stage: {roadmap.current_stage.value}")
            print(f"  Target Stage: {roadmap.target_stage.value}")
            print(f"  Estimated Duration: {roadmap.estimated_duration_days} days")
            
            print(f"\nMilestones ({len(roadmap.milestones)}):")
            for milestone in roadmap.milestones[:5]:
                print(f"  - {milestone.name}")
                print(f"    Priority: {milestone.priority}")
                print(f"    Stage: {milestone.stage.value}")
                if milestone.target_date:
                    days_until = (milestone.target_date - datetime.now(milestone.target_date.tzinfo)).days
                    print(f"    Target: {days_until} days from now")
            
            print(f"\nProject Phases:")
            for phase in roadmap.phases:
                print(f"  - {phase['name']} ({phase['duration_days']} days)")
                print(f"    Focus: {phase['focus']}")
            
            if roadmap.key_risks:
                print(f"\nKey Risks:")
                for risk in roadmap.key_risks:
                    print(f"  • {risk}")
            
            print(f"\nSuccess Metrics:")
            for metric, target in roadmap.success_metrics.items():
                print(f"  - {metric}: {target}")
                
        except Exception as e:
            print(f"\nError creating project roadmap: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Run all tests."""
    print("Starting Lifecycle-Aware Task Generation Tests")
    print(f"Timestamp: {datetime.now()}")
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"ANTHROPIC_API_KEY exists: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    print(f"GITHUB_TOKEN exists: {bool(os.getenv('GITHUB_TOKEN'))}")
    
    if not os.getenv('GITHUB_TOKEN'):
        print("\nWARNING: GITHUB_TOKEN not set. Repository analysis will fail.")
        print("Please set GITHUB_TOKEN environment variable.")
        return
    
    await test_lifecycle_analysis()
    
    print("\n\n=== Summary ===")
    print("The lifecycle-aware task generation system:")
    print("1. Analyzes repository lifecycle stage (inception → mature)")
    print("2. Generates stage-appropriate tasks")
    print("3. Plans transitions to the next stage")
    print("4. Creates project roadmaps with milestones")
    print("5. Considers lifecycle context in task prioritization")


if __name__ == "__main__":
    asyncio.run(main())