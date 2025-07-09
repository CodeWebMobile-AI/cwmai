"""
AI-Generated Tool: create_monetized_project
Description: Creates new project with complete monetization setup and documentation
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess

from scripts.ai_brain import IntelligentAIBrain
from scripts.enhanced_intelligent_task_generator import EnhancedIntelligentTaskGenerator
from scripts.business_market_research_engine import BusinessMarketResearchEngine
from scripts.dynamic_charter import DynamicCharter
from scripts.project_outcome_tracker import ProjectOutcomeTracker
from scripts.state_manager import StateManager

__description__ = "Create a new project with comprehensive monetization strategy and business documentation"
__parameters__ = {
    "project_type": {
        "type": "string",
        "required": False,
        "description": "Type of project to create (e.g., 'B2B SaaS', 'marketplace', 'API service')"
    },
    "target_market": {
        "type": "string",
        "required": False,
        "default": "small businesses",
        "description": "Primary target market for the project"
    },
    "target_revenue": {
        "type": "integer",
        "required": False,
        "default": 10000,
        "description": "Target monthly revenue (MRR)"
    },
    "architecture": {
        "type": "string",
        "required": False,
        "default": "laravel-react",
        "description": "Technical architecture (laravel-react, nextjs, django-vue)"
    },
    "create_repository": {
        "type": "boolean",
        "required": False,
        "default": True,
        "description": "Actually create the project repository and files"
    }
}
__examples__ = [
    "create_monetized_project()",
    "create_monetized_project(project_type='B2B SaaS', target_revenue=50000)",
    "create_monetized_project(target_market='developers', architecture='nextjs')"
]

async def create_monetized_project(**kwargs) -> Dict[str, Any]:
    """
    Create a new project with complete monetization setup.
    
    This tool discovers a business opportunity, validates it, and creates
    a full project with monetization strategy, market analysis, technical
    architecture, and initial codebase.
    """
    try:
        # Initialize components
        ai_brain = IntelligentAIBrain()
        task_generator = EnhancedIntelligentTaskGenerator(
            ai_brain,
            DynamicCharter(ai_brain),
            ProjectOutcomeTracker(ai_brain)
        )
        market_research = BusinessMarketResearchEngine(ai_brain)
        state_manager = StateManager()
        
        # Get parameters
        project_type = kwargs.get('project_type')
        target_market = kwargs.get('target_market', 'small businesses')
        target_revenue = kwargs.get('target_revenue', 10000)
        architecture = kwargs.get('architecture', 'laravel-react')
        create_repository = kwargs.get('create_repository', True)
        
        # Get existing portfolio
        try:
            state = state_manager.load_state()
            existing_portfolio = state.get('repositories', {})
        except:
            existing_portfolio = {}
        
        print("Discovering and validating business opportunity...")
        
        # Discover business opportunities
        opportunities = await market_research.discover_business_opportunities(
            existing_portfolio,
            max_opportunities=3
        )
        
        if not opportunities:
            return {
                "status": "error",
                "message": "No viable business opportunities found"
            }
        
        # Select best opportunity or use provided criteria
        selected_opportunity = None
        
        if project_type:
            # Find opportunity matching project type
            for opp in opportunities:
                if project_type and hasattr(opp, 'monetization_model') and hasattr(opp.monetization_model, 'model_type'):
                    if project_type.lower() in str(opp.monetization_model.model_type).lower():
                        selected_opportunity = opp
                        break
        
        if not selected_opportunity:
            # Use highest scoring opportunity
            selected_opportunity = opportunities[0]
        
        print(f"Selected opportunity: {selected_opportunity.name}")
        
        # Generate complete project task
        context = {
            'active_projects': existing_portfolio,
            'user_request': f"Create {selected_opportunity.name}",
            'preferences': {
                'architecture': architecture,
                'target_revenue': target_revenue
            }
        }
        
        # Create the task (this generates all documentation)
        task = await task_generator._generate_new_project_task(context)
        
        if not task:
            return {
                "status": "error",
                "message": "Failed to generate project task"
            }
        
        # Extract project details
        project_name = task['details']['project_name']
        project_slug = project_name.lower().replace(' ', '-') if project_name else 'unnamed-project'
        
        result = {
            "status": "success",
            "project_created": project_name,
            "opportunity": {
                "name": selected_opportunity.name,
                "tagline": selected_opportunity.tagline,
                "problem": selected_opportunity.problem_statement,
                "solution": selected_opportunity.solution_description,
                "score": selected_opportunity.total_score
            },
            "monetization": {
                "model": selected_opportunity.monetization_model.model_type,
                "target_arpu": selected_opportunity.monetization_model.target_arpu,
                "ltv_cac_ratio": f"{selected_opportunity.monetization_model.ltv_cac_ratio:.1f}:1",
                "revenue_projection_year_1": f"${selected_opportunity.get_first_year_revenue_projection():,.0f}"
            },
            "implementation": {
                "architecture": architecture,
                "complexity": selected_opportunity.implementation_complexity,
                "time_to_market": f"{selected_opportunity.time_to_market} days",
                "time_to_revenue": f"{selected_opportunity.time_to_revenue} days"
            },
            "documents_generated": []
        }
        
        # Create the actual project if requested
        if create_repository:
            project_path = Path(f"local_repos/repositories/CodeWebMobile-AI/{project_slug}")
            
            if project_path.exists():
                result['message'] = f"Project directory already exists: {project_path}"
            else:
                # Create project directory
                project_path.mkdir(parents=True, exist_ok=True)
                
                # Save all generated documents
                files_created = []
                
                for filename, content in task.get('files_to_create', {}).items():
                    if filename == 'implementation_plan':
                        # Skip implementation plan (it's metadata)
                        continue
                    
                    file_path = project_path / filename
                    
                    # Ensure content is string
                    if isinstance(content, dict):
                        content = json.dumps(content, indent=2)
                    
                    file_path.write_text(content)
                    files_created.append(str(file_path))
                    result['documents_generated'].append(filename)
                
                # Initialize git repository
                try:
                    subprocess.run(['git', 'init'], cwd=project_path, check=True)
                    
                    # Create .gitignore
                    gitignore_content = """# Dependencies
node_modules/
vendor/

# Environment
.env
.env.local

# IDE
.idea/
.vscode/

# Build
dist/
build/
*.log

# OS
.DS_Store
Thumbs.db
"""
                    (project_path / '.gitignore').write_text(gitignore_content)
                    
                    # Initial commit
                    subprocess.run(['git', 'add', '.'], cwd=project_path, check=True)
                    subprocess.run(
                        ['git', 'commit', '-m', f'Initial commit: {project_name} with monetization strategy'],
                        cwd=project_path,
                        check=True
                    )
                    
                    result['repository_initialized'] = True
                except:
                    result['repository_initialized'] = False
                
                # Add to state
                try:
                    state = state_manager.load_state()
                    if 'repositories' not in state:
                        state['repositories'] = {}
                    
                    state['repositories'][project_slug] = {
                        'name': project_name,
                        'path': str(project_path),
                        'created_at': task.get('metadata', {}).get('created_at'),
                        'monetization_model': selected_opportunity.monetization_model.model_type,
                        'target_revenue': target_revenue,
                        'architecture': architecture
                    }
                    
                    state_manager.state = state
                    state_manager.save_state()
                except:
                    pass
                
                result['project_path'] = str(project_path)
                result['files_created'] = len(files_created)
        
        # Add next steps
        result['next_steps'] = [
            f"cd {project_slug}",
            "Review monetization_strategy.md for revenue model",
            "Check architecture.md for technical implementation",
            "Read market_analysis.md for competitive insights",
            f"Start development with target of ${target_revenue}/month MRR"
        ]
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating monetized project: {str(e)}"
        }

# Test function
async def main():
    # Test project creation
    result = await create_monetized_project(
        project_type="B2B SaaS",
        target_market="small businesses",
        target_revenue=25000,
        architecture="laravel-react"
    )
    
    print(json.dumps(result, indent=2))
    
    if result['status'] == 'success':
        print(f"\nProject created successfully!")
        print(f"Path: {result.get('project_path', 'Not created')}")
        print(f"Documents: {', '.join(result['documents_generated'])}")

if __name__ == "__main__":
    asyncio.run(main())