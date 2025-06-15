#!/usr/bin/env python3
"""
Test the dynamic project generation system.

This script tests that NEW_PROJECT tasks are generated based on real market research
rather than hardcoded ideas.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.dynamic_charter import DynamicCharter as DynamicCharterSystem
from scripts.ai_brain import AIBrain
from scripts.project_creator import ProjectCreator


async def test_new_project_generation():
    """Test that NEW_PROJECT tasks use dynamic research."""
    print("\n=== Testing Dynamic Project Generation ===\n")
    
    # Initialize components
    ai_brain = AIBrain()
    charter_system = DynamicCharterSystem(ai_brain)
    task_generator = IntelligentTaskGenerator(ai_brain, charter_system)
    
    # Create context that triggers portfolio expansion
    context = {
        'projects': [],  # Empty portfolio
        'recent_tasks': [],
        'capabilities': ['project_creation', 'ai_integration'],
        'market_trends': [
            'AI automation growing 40% yearly',
            'Small businesses need efficiency tools',
            'Remote work continues to expand'
        ]
    }
    
    print("1. Testing task generation with empty portfolio...")
    
    # Analyze system needs
    charter = await charter_system.get_current_charter()
    need_analysis = await task_generator._analyze_system_needs(context, charter)
    
    print(f"\nNeed Analysis:")
    print(f"- Type: {need_analysis.get('need_type')}")
    print(f"- Description: {need_analysis.get('specific_need')}")
    print(f"- Priority: {need_analysis.get('priority')}")
    
    # Generate task based on need
    if need_analysis.get('need_type') == 'portfolio_expansion':
        print("\n2. Generating NEW_PROJECT task with market research...")
        
        task = await task_generator._generate_new_project_task(context)
        
        print(f"\nGenerated Task:")
        print(f"- Type: {task.get('type')}")
        print(f"- Title: {task.get('title')}")
        print(f"- Description: {task.get('description', '')[:200]}...")
        
        # Check for research metadata
        if task.get('metadata', {}).get('research_based'):
            print("\n✅ Task is based on market research!")
            research = task['metadata'].get('research_result', {})
            print(f"- Problem: {research.get('problem', 'N/A')}")
            print(f"- Target Audience: {research.get('target_audience', 'N/A')}")
            print(f"- Monetization: {research.get('monetization', 'N/A')}")
        else:
            print("\n❌ Task is not based on research")
        
        # Check for hardcoded content
        hardcoded_terms = [
            'Business Analytics Dashboard',
            'Customer Engagement Mobile App',
            'Content Management System',
            'API Gateway Service',
            'E-Commerce Marketplace',
            'Team Collaboration Suite'
        ]
        
        description = task.get('description', '').lower()
        title = task.get('title', '').lower()
        
        found_hardcoded = False
        for term in hardcoded_terms:
            if term.lower() in description or term.lower() in title:
                print(f"\n⚠️ Found hardcoded term: {term}")
                found_hardcoded = True
        
        if not found_hardcoded:
            print("\n✅ No hardcoded project ideas found!")
        
        return task
    else:
        print(f"\n⚠️ System identified different need: {need_analysis.get('need_type')}")
        return None


async def test_project_creator_research():
    """Test that ProjectCreator uses research for project details."""
    print("\n\n=== Testing Project Creator Research ===\n")
    
    # Initialize project creator
    github_token = os.getenv('GITHUB_TOKEN', 'dummy_token')
    ai_brain = AIBrain()
    project_creator = ProjectCreator(github_token, ai_brain)
    
    # Create a dummy task
    task = {
        'type': 'NEW_PROJECT',
        'title': 'Create Solution for Daily Problem',
        'description': 'Build an application that solves a real problem',
        'metadata': {}
    }
    
    print("1. Testing project detail generation with research...")
    
    # Generate project details
    details = await project_creator._generate_project_details(task)
    
    print(f"\nGenerated Project Details:")
    print(f"- Name: {details.get('name')}")
    print(f"- Description: {details.get('description')}")
    
    # Check for research-based fields
    research_fields = ['problem_statement', 'target_audience', 'monetization_strategy', 'market_validation']
    has_research = all(field in details for field in research_fields)
    
    if has_research:
        print("\n✅ Project details include research-based fields:")
        print(f"- Problem Statement: {details.get('problem_statement', '')[:100]}...")
        print(f"- Target Audience: {details.get('target_audience', '')[:100]}...")
        print(f"- Monetization: {details.get('monetization_strategy', '')[:100]}...")
    else:
        print("\n❌ Project details missing research fields")
    
    # Test architecture generation
    print("\n2. Testing architecture generation...")
    
    architecture = await project_creator._generate_project_architecture(details)
    
    if architecture:
        print("\n✅ Architecture generated successfully!")
        print(f"- Has Design System: {'design_system' in architecture}")
        print(f"- Has Foundation: {'foundational_architecture' in architecture}")
        print(f"- Has Roadmap: {'feature_implementation_roadmap' in architecture}")
    else:
        print("\n❌ Architecture generation failed")
    
    return details, architecture


async def main():
    """Run all tests."""
    print("Starting Dynamic Project Generation Tests...")
    print("=" * 50)
    
    # Test 1: Task generation
    task = await test_new_project_generation()
    
    # Test 2: Project creator
    if task:
        details, architecture = await test_project_creator_research()
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    
    # Summary
    print("\n=== Summary ===")
    print("✅ Dynamic project generation is working if:")
    print("   - Tasks are based on market research")
    print("   - No hardcoded project ideas are found")
    print("   - Project details include real problem statements")
    print("   - Architecture is generated comprehensively")


if __name__ == "__main__":
    asyncio.run(main())