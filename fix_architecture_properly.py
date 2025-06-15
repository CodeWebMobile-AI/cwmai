#!/usr/bin/env python3
"""
Fix architecture documents to match the quality of new project generation.
Uses the same detailed approach as project_creator.py
"""

import os
import asyncio
import json
from github import Github
from scripts.http_ai_client import HTTPAIClient
from typing import Dict, Any


async def analyze_project_purpose(repo_name: str, description: str, ai_client: HTTPAIClient) -> Dict[str, Any]:
    """Analyze the project to understand its business purpose and context."""
    
    prompt = f"""
    Analyze this project and extract its business context:
    
    Repository Name: {repo_name}
    Description: {description}
    
    The repository name is HIGHLY DESCRIPTIVE. Use it to understand what problem this project solves.
    
    Based on the name and description, provide:
    1. Problem Statement: What specific problem does this solve?
    2. Target Audience: Who are the primary users? Be specific (e.g., "Small business owners managing inventory", not just "businesses")
    3. Key Features: List 5-7 specific features this project would have based on its purpose
    4. Core Entities: What are the main data entities (e.g., User, Document, Summary, etc.)
    5. Monetization Strategy: How could this project make money?
    
    Examples of good analysis:
    - "summarize-ai-mobile" → AI document summarization for busy professionals on mobile devices
    - "hydra-health-ai" → Personalized hydration tracking for fitness enthusiasts
    - "review-sentry" → Automated review management for small business reputation
    
    Return as JSON:
    {{
        "problem_statement": "specific problem description",
        "target_audience": "detailed target user description",
        "key_features": ["feature1", "feature2", ...],
        "core_entities": ["Entity1", "Entity2", ...],
        "monetization_strategy": "how it makes money"
    }}
    """
    
    response = await ai_client.generate_enhanced_response(prompt, model="gemini-2.0-flash")
    
    # Parse response
    if isinstance(response, dict):
        content = response.get('content', '')
    else:
        content = str(response)
    
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback
    return {
        "problem_statement": f"Solution for {repo_name.replace('-', ' ')}",
        "target_audience": "Users seeking efficient solutions",
        "key_features": ["Core functionality", "User management", "Data processing"],
        "core_entities": ["User", "Data", "Result"],
        "monetization_strategy": "Subscription model"
    }


async def generate_detailed_architecture(repo_name: str, project_context: Dict[str, Any], ai_client: HTTPAIClient) -> str:
    """Generate a detailed architecture document matching new project quality."""
    
    # Use the same prompt structure as project_creator.py
    prompt = f"""
    You are a distinguished CTO, pragmatic Principal Engineer, and skilled UI/UX Designer working on an architecture document.
    
    Generate a COMPREHENSIVE architecture document for this project:
    
    Project: {repo_name}
    Problem Statement: {project_context['problem_statement']}
    Target Audience: {project_context['target_audience']}
    Key Features: {json.dumps(project_context['key_features'])}
    Core Entities: {json.dumps(project_context['core_entities'])}
    
    Technical Mandates:
    1. Backend MUST use Laravel 11+ (PHP 8.3) patterns
    2. Frontend MUST use React 19 with TypeScript
    3. MUST include Tailwind CSS and shadcn/ui
    4. MUST be production-ready and scalable
    
    Create an architecture specification that includes:
    
    1. Design System:
       - Typography: Specific font recommendation with rationale
       - Color Palette: Primary, secondary, accent colors with hex codes and usage
       - UI Components: List of required components
    
    2. Core Architecture (4 detailed sections):
       - Core Components & Rationale
       - Database Schema Design (with actual tables and relationships)
       - API Design & Key Endpoints (RESTful, with specific routes)
       - Frontend Structure (component hierarchy)
    
    3. Feature Implementation Roadmap:
       For each key feature, specify:
       - Feature name and description
       - Required database changes
       - Backend components needed
       - Frontend components needed
       - API endpoints
       - Estimated complexity
    
    4. Non-Functional Requirements:
       - Performance targets
       - Security considerations
       - Scalability approach
       - Monitoring strategy
    
    Return as JSON matching this exact schema:
    {{
      "title": "Architecture for {repo_name}",
      "description": "Detailed architecture blueprint",
      "problem_statement": "{project_context['problem_statement']}",
      "target_audience": "{project_context['target_audience']}",
      "core_entities": {json.dumps(project_context['core_entities'])},
      "design_system": {{
        "typography": {{
          "font": "Font name",
          "font_stack": "Full font stack",
          "google_fonts_url": "https://fonts.google.com/...",
          "rationale": "Why this font fits the project"
        }},
        "color_palette": {{
          "primary": {{"hex": "#HEX", "usage": "Primary actions and branding"}},
          "secondary": {{"hex": "#HEX", "usage": "Secondary elements"}},
          "accent": {{"hex": "#HEX", "usage": "Highlights and CTAs"}},
          "success": {{"hex": "#HEX", "usage": "Success states"}},
          "warning": {{"hex": "#HEX", "usage": "Warning states"}},
          "error": {{"hex": "#HEX", "usage": "Error states"}},
          "background": {{"hex": "#HEX", "usage": "Main background"}},
          "surface": {{"hex": "#HEX", "usage": "Card/component backgrounds"}},
          "text_primary": {{"hex": "#HEX", "usage": "Main text"}},
          "text_secondary": {{"hex": "#HEX", "usage": "Secondary text"}}
        }}
      }},
      "foundational_architecture": {{
        "core_components": {{
          "section_title": "1. Core Components & Rationale",
          "content": "Detailed description of core components and why they're needed"
        }},
        "database_schema": {{
          "section_title": "2. Database Schema Design", 
          "content": "Detailed database design with tables, fields, and relationships"
        }},
        "api_design": {{
          "section_title": "3. API Design & Key Endpoints",
          "content": "RESTful API design with specific endpoints and their purposes"
        }},
        "frontend_structure": {{
          "section_title": "4. Frontend Structure",
          "content": "Component hierarchy and state management approach"
        }}
      }},
      "feature_implementation_roadmap": [
        {{
          "feature_name": "Feature name",
          "description": "What this feature does",
          "priority": "high|medium|low",
          "database_changes": ["migration details"],
          "backend_components": ["component names"],
          "frontend_components": ["component names"],
          "api_endpoints": ["GET /api/...", "POST /api/..."],
          "complexity": "low|medium|high"
        }}
      ]
    }}
    
    Be SPECIFIC. For {repo_name}, think about what actual features, database tables, and API endpoints would be needed.
    """
    
    response = await ai_client.generate_enhanced_response(prompt, model="gemini-2.0-flash")
    
    # Parse response
    if isinstance(response, dict):
        content = response.get('content', '')
    else:
        content = str(response)
    
    try:
        # Extract JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            architecture = json.loads(json_match.group())
            
            # Format as markdown like project_creator does
            return format_architecture_document(architecture)
    except Exception as e:
        print(f"Error parsing architecture: {e}")
    
    return None


def format_architecture_document(architecture: Dict[str, Any]) -> str:
    """Format architecture data as a markdown document."""
    
    md = f"""# {architecture.get('title', 'Project Architecture')}

## Overview
{architecture.get('description', '')}

### Problem Statement
{architecture.get('problem_statement', '')}

### Target Audience
{architecture.get('target_audience', '')}

### Core Entities
"""
    
    # Add entities
    for entity in architecture.get('core_entities', []):
        md += f"- {entity}\n"
    
    # Design System
    design = architecture.get('design_system', {})
    typography = design.get('typography', {})
    colors = design.get('color_palette', {})
    
    md += f"""

## Design System

### Typography
- **Font**: {typography.get('font', 'Inter')}
- **Font Stack**: `{typography.get('font_stack', "'Inter', system-ui, sans-serif")}`
- **Google Fonts**: {typography.get('google_fonts_url', 'https://fonts.google.com/specimen/Inter')}
- **Rationale**: {typography.get('rationale', 'Clean and modern typeface')}

### Color Palette
"""
    
    # Add colors
    for color_name, color_data in colors.items():
        if isinstance(color_data, dict):
            md += f"- **{color_name.replace('_', ' ').title()}**: `{color_data.get('hex', '#000000')}` - {color_data.get('usage', '')}\n"
    
    # Foundational Architecture
    md += "\n## Foundational Architecture\n"
    
    foundation = architecture.get('foundational_architecture', {})
    for section_key, section_data in foundation.items():
        if isinstance(section_data, dict):
            md += f"\n### {section_data.get('section_title', section_key)}\n\n"
            md += f"{section_data.get('content', '')}\n"
    
    # Feature Roadmap
    md += "\n## Feature Implementation Roadmap\n"
    
    for feature in architecture.get('feature_implementation_roadmap', []):
        md += f"\n### {feature.get('feature_name', 'Feature')}\n"
        md += f"**Description**: {feature.get('description', '')}\n"
        md += f"**Priority**: {feature.get('priority', 'medium')}\n"
        md += f"**Complexity**: {feature.get('complexity', 'medium')}\n\n"
        
        if feature.get('database_changes'):
            md += "**Database Changes**:\n"
            for change in feature['database_changes']:
                md += f"- {change}\n"
        
        if feature.get('api_endpoints'):
            md += "\n**API Endpoints**:\n"
            for endpoint in feature['api_endpoints']:
                md += f"- `{endpoint}`\n"
        
        if feature.get('frontend_components'):
            md += "\n**Frontend Components**:\n"
            for component in feature['frontend_components']:
                md += f"- {component}\n"
    
    return md


async def fix_repository_architecture(repo, ai_client: HTTPAIClient):
    """Fix a single repository's architecture."""
    try:
        print(f"\n🏗️  Fixing architecture for {repo.name}...")
        
        # Step 1: Analyze project purpose
        project_context = await analyze_project_purpose(
            repo.name,
            repo.description or f"{repo.name} application",
            ai_client
        )
        
        print(f"  📊 Analyzed project context:")
        print(f"     Problem: {project_context['problem_statement'][:60]}...")
        print(f"     Audience: {project_context['target_audience'][:60]}...")
        
        # Step 2: Generate detailed architecture
        architecture_content = await generate_detailed_architecture(
            repo.name,
            project_context,
            ai_client
        )
        
        if not architecture_content:
            print(f"  ❌ Failed to generate architecture")
            return False
        
        # Step 3: Update the file
        try:
            arch_file = repo.get_contents("ARCHITECTURE.md")
            repo.update_file(
                "ARCHITECTURE.md",
                f"Update architecture with detailed project-specific content",
                architecture_content,
                arch_file.sha,
                branch="main"
            )
            print(f"  ✅ Updated ARCHITECTURE.md")
        except:
            repo.create_file(
                "ARCHITECTURE.md",
                f"Add detailed architecture documentation",
                architecture_content,
                branch="main"
            )
            print(f"  ✅ Created ARCHITECTURE.md")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    """Fix architecture for specific repositories."""
    
    # Target the repositories mentioned
    repos_to_fix = [
        "summarize-ai-mobile",
        # Add more as needed
    ]
    
    print("🚀 Fixing architecture documents to match new project quality...\n")
    
    # Initialize
    github_token = os.environ.get('GITHUB_TOKEN')
    g = Github(github_token)
    ai_client = HTTPAIClient()
    
    for repo_name in repos_to_fix:
        try:
            repo = g.get_repo(f"CodeWebMobile-AI/{repo_name}")
            await fix_repository_architecture(repo, ai_client)
            await asyncio.sleep(2)  # Rate limit
        except Exception as e:
            print(f"❌ Error with {repo_name}: {e}")
    
    print("\n✅ Architecture fix complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())