#!/usr/bin/env python3
"""Test that all three documentation files are created"""

import os
import sys
import json
from pathlib import Path

# Add the zip folder to Python path
sys.path.insert(0, '/Users/fortressarigo/PhpstormProjects/cwmai/zip')

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.project_creator import ProjectCreator

# Mock project details that would normally come from AI
MOCK_PROJECT_DETAILS = {
    "name": "smart-inventory-tracker",
    "description": "An AI-powered inventory management system for small businesses",
    "problem_statement": "Small businesses struggle with inventory management, leading to stockouts and overstock situations that cost them money",
    "target_audience": "Small retail businesses with 1-50 locations",
    "core_entities": ["Product", "Inventory", "Supplier", "Order", "Alert"],
    "monetization_strategy": "SaaS subscription model with tiered pricing based on inventory size and number of locations",
    "initial_features": [
        "Real-time inventory tracking",
        "AI-powered demand forecasting",
        "Automated reorder alerts",
        "Supplier management",
        "Multi-location support"
    ],
    "tech_stack": "Laravel backend with React frontend, PostgreSQL database, Redis for caching",
    "architecture": {
        "design_system": {
            "suggested_font": {
                "font_name": "Inter",
                "font_stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "google_font_link": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
                "rationale": "Clean, modern font that's highly readable for data-heavy interfaces"
            },
            "color_palette": {
                "rationale": "Professional color scheme that conveys trust and efficiency",
                "primary": {"hex": "#2563EB", "name": "Business Blue", "usage": "Primary actions and branding"},
                "secondary": {"hex": "#7C3AED", "name": "Smart Purple", "usage": "AI features and insights"},
                "accent": {"hex": "#10B981", "name": "Success Green", "usage": "Positive states and growth"},
                "danger": {"hex": "#EF4444", "name": "Alert Red", "usage": "Low stock warnings"}
            }
        },
        "foundational_architecture": {
            "overview": {
                "section_title": "System Overview",
                "content": "Microservices architecture with API gateway, separate services for inventory, analytics, and notifications"
            },
            "data_layer": {
                "section_title": "Data Architecture",
                "content": "PostgreSQL for transactional data, TimescaleDB for time-series inventory data, Redis for real-time updates"
            }
        },
        "feature_implementation_roadmap": [
            {
                "feature_name": "Real-time Inventory Tracking",
                "description": "Track inventory levels across all locations in real-time",
                "required_db_changes": ["inventory_transactions table", "inventory_snapshots table"],
                "new_api_endpoints": ["GET /api/inventory/current", "POST /api/inventory/update"],
                "real_time_events": ["inventory.updated", "inventory.low_stock"]
            }
        ]
    },
    "market_research": {
        "market_size": {
            "tam": "$10B global inventory management software market",
            "sam": "$2B small business segment",
            "som": "$100M addressable market in first 3 years",
            "growth_rate": "15% annual growth"
        },
        "monetization_models": [
            {
                "model_name": "Starter Plan",
                "type": "Subscription",
                "pricing_structure": "$49/month",
                "target_segment": "Single location businesses",
                "features": ["Up to 1,000 SKUs", "Basic analytics", "Email alerts"]
            },
            {
                "model_name": "Growth Plan",
                "type": "Subscription",
                "pricing_structure": "$149/month",
                "target_segment": "2-10 locations",
                "features": ["Up to 10,000 SKUs", "Advanced analytics", "API access", "Priority support"]
            },
            {
                "model_name": "Enterprise Plan",
                "type": "Custom pricing",
                "pricing_structure": "Starting at $499/month",
                "target_segment": "10+ locations",
                "features": ["Unlimited SKUs", "Custom integrations", "Dedicated support", "SLA"]
            }
        ]
    }
}

def test_documentation_creation():
    """Test creating a repository and checking all documentation files"""
    
    print("🚀 Testing three documentation files creation...\n")
    
    # Create simulator
    simulator = LocalGitHubSimulator()
    
    # Create repository with template
    print("1️⃣ Creating repository with template...")
    repo = simulator.create_repository(
        name="test-docs-project",
        description="Testing documentation generation",
        organization="CodeWebMobile-AI",
        template="react-starter-kit"
    )
    print(f"✅ Repository created at: {repo.path}\n")
    
    # Create project creator
    creator = ProjectCreator(github_token=None, ai_brain=None)
    creator.storage_mode = "local"  # Force local mode
    
    # Run customization (which should create all 3 MD files)
    print("2️⃣ Running project customization...")
    
    # Mock the AI response to avoid actual API calls
    async def mock_generate_readme(repo, details):
        return f"""# {details['name']}

## Overview
{details['description']}

## Problem Statement
{details['problem_statement']}

## Target Audience
{details['target_audience']}

## Features
{chr(10).join(f'- {feature}' for feature in details.get('initial_features', []))}

## Tech Stack
{details.get('tech_stack', 'Modern web stack')}

## Getting Started

### Prerequisites
- Node.js 18+
- PHP 8.2+
- PostgreSQL 14+
- Redis

### Installation
1. Clone the repository
2. Install dependencies: `composer install && npm install`
3. Copy .env.example to .env
4. Generate app key: `php artisan key:generate`
5. Run migrations: `php artisan migrate`
6. Start development server: `npm run dev`

## License
MIT
"""
    
    # Temporarily replace the async method
    import asyncio
    creator._generate_readme = mock_generate_readme
    
    # Run customization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(
            creator._customize_project(repo, MOCK_PROJECT_DETAILS)
        )
        print(f"Customization results: {results}\n")
    finally:
        loop.close()
    
    # Check if all three files were created
    print("3️⃣ Checking documentation files...")
    repo_path = Path(repo.path)
    
    expected_files = {
        "README.md": "📖",
        "ARCHITECTURE.md": "🏗️",
        "MONETIZATION_STRATEGY.md": "💰"
    }
    
    all_created = True
    for filename, emoji in expected_files.items():
        file_path = repo_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"{emoji} ✅ {filename} exists ({size} bytes)")
            
            # Show first few lines
            with open(file_path, 'r') as f:
                lines = f.readlines()[:5]
                print(f"   Preview: {lines[0].strip()}")
                print()
        else:
            print(f"{emoji} ❌ {filename} missing!")
            all_created = False
    
    # Summary
    print("\n📊 Summary:")
    if all_created:
        print("✅ All three documentation files were created successfully!")
    else:
        print("❌ Some documentation files are missing.")
    
    return all_created

if __name__ == "__main__":
    success = test_documentation_creation()
    sys.exit(0 if success else 1)