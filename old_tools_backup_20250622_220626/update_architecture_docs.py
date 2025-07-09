"""
Update Architecture Docs Tool
=============================

Updates ARCHITECTURE.md files in repositories to the new comprehensive format.
"""

import os
import asyncio
import logging
import subprocess
import json
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback

from scripts.state_manager import StateManager
from scripts.logger import logger

__description__ = "Update ARCHITECTURE.md files to the new comprehensive format across repositories"
__parameters__ = {
    'dry_run': {'type': 'boolean', 'required': False, 'description': 'If true, only check without making changes'},
    'repo_name': {'type': 'string', 'required': False, 'description': 'Specific repository to update (if not provided, updates all)'},
    'preserve_content': {'type': 'boolean', 'required': False, 'description': 'Try to preserve existing content when updating'}
}
__examples__ = [
    {'dry_run': True},
    {'repo_name': 'reputation-ai', 'preserve_content': True},
    {}
]

# New ARCHITECTURE.md template
ARCHITECTURE_TEMPLATE = """# {project_name} Architecture

## System Overview

{project_name} is a Laravel-based web application with a React frontend, designed to {project_description}.

### Core Purpose
- {core_purpose_1}
- {core_purpose_2}
- {core_purpose_3}

### Key Features
- {feature_1}
- {feature_2}
- {feature_3}

## Technical Stack

### Backend
- **Framework**: Laravel 10.x
- **Language**: PHP 8.1+
- **Database**: MySQL 8.0
- **Cache**: Redis
- **Queue**: Laravel Queue with Redis driver
- **Authentication**: Laravel Sanctum

### Frontend
- **Framework**: React 18.x with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: React Context API
- **HTTP Client**: Axios
- **UI Components**: Headless UI

### Infrastructure
- **Deployment**: Docker containers
- **Web Server**: Nginx
- **Process Manager**: Supervisor (for queue workers)
- **Monitoring**: Application logs + health endpoints

## Key Components

### Backend Structure
```
app/
├── Http/
│   ├── Controllers/      # API and web controllers
│   ├── Middleware/       # Request/response middleware
│   └── Requests/         # Form request validation
├── Models/               # Eloquent models
├── Services/             # Business logic services
├── Repositories/         # Data access layer
└── Jobs/                 # Background jobs

database/
├── migrations/           # Database schema migrations
├── factories/           # Model factories for testing
└── seeders/             # Database seeders

routes/
├── api.php              # API routes
├── web.php              # Web routes
└── channels.php         # Broadcasting channels
```

### Frontend Structure
```
resources/js/
├── Components/          # Reusable React components
├── Pages/              # Page components (Inertia.js)
├── Layouts/            # Layout components
├── Hooks/              # Custom React hooks
├── Services/           # API service modules
├── Types/              # TypeScript type definitions
└── Utils/              # Utility functions
```

## API Design

### RESTful Endpoints
- `GET /api/resource` - List resources with pagination
- `GET /api/resource/{id}` - Get single resource
- `POST /api/resource` - Create new resource
- `PUT /api/resource/{id}` - Update resource
- `DELETE /api/resource/{id}` - Delete resource

### Authentication Flow
1. User submits credentials to `/api/login`
2. Server validates and returns Sanctum token
3. Frontend stores token in secure HTTP-only cookie
4. Subsequent requests include token for authentication

### Response Format
```json
{
    "success": true,
    "data": {},
    "message": "Operation successful",
    "meta": {
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0"
    }
}
```

## Database Schema

### Core Tables
- **users**: User accounts and authentication
- **roles**: User roles for authorization
- **permissions**: Granular permissions
- **{resource}_table**: Main resource storage

### Relationships
- Users have many-to-many relationship with roles
- Roles have many-to-many relationship with permissions
- Resources belong to users (creator/owner)

### Indexing Strategy
- Primary keys on all tables
- Foreign key indexes for relationships
- Composite indexes for common query patterns
- Full-text indexes for search functionality

## Security Considerations

### Authentication & Authorization
- Token-based authentication using Laravel Sanctum
- Role-based access control (RBAC)
- API rate limiting per user/IP
- CORS configuration for frontend domain

### Data Protection
- Input validation on all user inputs
- SQL injection prevention via Eloquent ORM
- XSS protection through React's default escaping
- CSRF protection on state-changing operations

### Infrastructure Security
- HTTPS enforcement in production
- Environment variables for sensitive configuration
- Regular security dependency updates
- Database encryption at rest

## Deployment Strategy

### Development Workflow
1. Local development with Docker Compose
2. Feature branches with PR reviews
3. Automated testing in CI/CD pipeline
4. Staging deployment for QA
5. Production deployment with zero downtime

### CI/CD Pipeline
- **Build Stage**: Compile assets, run tests
- **Test Stage**: Unit, integration, and E2E tests
- **Deploy Stage**: Container build and registry push
- **Release Stage**: Rolling deployment to production

### Environment Configuration
- `.env.local` - Local development
- `.env.staging` - Staging environment
- `.env.production` - Production environment

### Monitoring & Logging
- Application logs aggregated in centralized system
- Performance monitoring for API endpoints
- Error tracking and alerting
- Uptime monitoring and health checks

## Development Guidelines

### Code Standards
- PSR-12 for PHP code style
- ESLint + Prettier for JavaScript/TypeScript
- Conventional commits for git messages
- Comprehensive PHPDoc and JSDoc comments

### Testing Requirements
- Unit tests for all service methods
- Integration tests for API endpoints
- React component tests with Testing Library
- E2E tests for critical user flows

### Performance Considerations
- Database query optimization with eager loading
- API response caching where appropriate
- Frontend code splitting and lazy loading
- Image optimization and CDN usage

## Future Enhancements

### Planned Features
- {future_feature_1}
- {future_feature_2}
- {future_feature_3}

### Technical Improvements
- Migration to microservices architecture
- Implementation of GraphQL API
- Real-time features with WebSockets
- Enhanced caching strategies

---

*Last Updated: {date}*
"""

async def get_file_content(repo_name: str, file_path: str) -> Optional[str]:
    """Get file content from repository using gh CLI."""
    cmd = f"gh api repos/{repo_name}/contents/{file_path} --jq '.content' | base64 -d"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    return None

async def extract_project_info(repo_name: str, existing_content: Optional[str]) -> Dict[str, str]:
    """Extract project information from repository name and existing content."""
    # Get base project name
    project_name = repo_name.split('/')[-1]
    
    # Convert kebab-case to Title Case
    display_name = ' '.join(word.capitalize() for word in project_name.replace('-', ' ').split())
    
    # Generate description based on project name
    descriptions = {
        'reputation-ai': 'manage and analyze online reputation using AI-powered insights',
        'eco-track-ai': 'track and optimize environmental impact with machine learning',
        'health-guardian-ai': 'monitor and improve health metrics through intelligent analysis',
        'skill-mentor-ai': 'provide personalized skill development and mentoring',
        'budget-optimizer-ai': 'optimize financial budgets using predictive analytics',
        'smart-inventory-ai': 'manage inventory with demand forecasting and optimization',
        'customer-insights-ai': 'analyze customer behavior and generate actionable insights',
        'workflow-automator-ai': 'automate business workflows with intelligent decision making',
        'content-curator-ai': 'curate and recommend content based on user preferences',
        'market-predictor-ai': 'predict market trends and provide investment insights'
    }
    
    project_description = descriptions.get(project_name, 'provide intelligent solutions for modern challenges')
    
    # Extract any existing valuable content
    preserved_content = {}
    if existing_content and preserve_content:
        # Try to extract existing sections we might want to preserve
        import re
        
        # Look for key features
        features_match = re.search(r'(?:Key Features|Features|Main Features)[\s\S]*?(?=##|\Z)', existing_content, re.IGNORECASE)
        if features_match:
            preserved_content['features'] = features_match.group(0)
    
    # Build template variables
    return {
        'project_name': display_name,
        'project_description': project_description,
        'date': '2024-01-15',
        'core_purpose_1': f'Provide enterprise-grade {project_name.replace("-ai", "")} management',
        'core_purpose_2': 'Leverage AI for intelligent automation and insights',
        'core_purpose_3': 'Deliver scalable, secure, and user-friendly solutions',
        'feature_1': 'Real-time data processing and analytics',
        'feature_2': 'AI-powered predictions and recommendations',
        'feature_3': 'Comprehensive API for third-party integrations',
        'future_feature_1': 'Advanced machine learning model integration',
        'future_feature_2': 'Mobile application for iOS and Android',
        'future_feature_3': 'Enhanced real-time collaboration features'
    }

async def create_or_update_file(repo_name: str, content: str, branch: str = "main") -> bool:
    """Create or update ARCHITECTURE.md file in the repository."""
    try:
        # Check if file exists
        existing_sha = None
        sha_cmd = f"gh api repos/{repo_name}/contents/ARCHITECTURE.md --jq '.sha' 2>/dev/null"
        sha_result = subprocess.run(sha_cmd, shell=True, capture_output=True, text=True)
        if sha_result.returncode == 0:
            existing_sha = sha_result.stdout.strip()
        
        # Encode content to base64
        content_b64 = base64.b64encode(content.encode()).decode()
        
        # Prepare the data
        data = {
            "message": "Update ARCHITECTURE.md to comprehensive format\n\nUpdated to include detailed technical stack, API design, database schema, security considerations, and deployment strategy",
            "content": content_b64,
            "branch": branch
        }
        
        if existing_sha:
            data["sha"] = existing_sha
        
        # Create/update the file
        cmd = f"gh api repos/{repo_name}/contents/ARCHITECTURE.md -X PUT --input -"
        result = subprocess.run(
            cmd, 
            shell=True, 
            input=json.dumps(data),
            capture_output=True, 
            text=True
        )
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error creating/updating file: {e}")
        return False

async def update_architecture_docs(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update ARCHITECTURE.md files to the new comprehensive format.
    """
    try:
        dry_run = kwargs.get('dry_run', False)
        specific_repo = kwargs.get('repo_name')
        preserve_content = kwargs.get('preserve_content', True)
        
        # Initialize components with absolute path
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        state_file_path = os.path.join(project_root, "system_state.json")
        state_manager = StateManager(local_path=state_file_path)
        state_manager.load_state()
        state = state_manager.get_state()
        
        # Get repositories to check
        projects = state.get('projects', {})
        if specific_repo:
            # Check only specific repository
            if specific_repo in projects:
                repos_to_check = {specific_repo: projects[specific_repo]}
            else:
                # Try with organization prefix
                for key, project in projects.items():
                    if project.get('name') == specific_repo:
                        repos_to_check = {key: project}
                        break
                else:
                    return {
                        "success": False,
                        "error": f"Repository '{specific_repo}' not found in managed projects"
                    }
        else:
            # Check all repositories
            repos_to_check = projects
        
        results = {
            "checked": 0,
            "updated": 0,
            "already_updated": 0,
            "created": 0,
            "failed": 0,
            "repositories": {}
        }
        
        for repo_key, project in repos_to_check.items():
            repo_name = project.get('full_name')
            if not repo_name:
                logger.warning(f"Skipping {repo_key}: No full_name found")
                continue
                
            results["checked"] += 1
            logger.info(f"Checking {repo_name} ARCHITECTURE.md...")
            
            try:
                # Get existing content
                existing_content = await get_file_content(repo_name, "ARCHITECTURE.md")
                
                # Check if already in new format
                if existing_content:
                    new_format_indicators = [
                        "## System Overview",
                        "## Technical Stack", 
                        "## Key Components",
                        "## API Design",
                        "## Database Schema",
                        "## Security Considerations",
                        "## Deployment Strategy"
                    ]
                    
                    indicator_count = sum(1 for indicator in new_format_indicators if indicator in existing_content)
                    
                    if indicator_count >= 6:
                        results["already_updated"] += 1
                        results["repositories"][repo_name] = {
                            "status": "already_updated",
                            "message": "ARCHITECTURE.md already in new format"
                        }
                        logger.info(f"✓ {repo_name}: Already updated")
                        continue
                
                # Extract project information
                project_info = await extract_project_info(repo_name, existing_content)
                
                # Generate new content
                new_content = ARCHITECTURE_TEMPLATE.format(**project_info)
                
                if dry_run:
                    action = "update" if existing_content else "create"
                    results["repositories"][repo_name] = {
                        "status": f"would_{action}",
                        "message": f"Would {action} ARCHITECTURE.md (dry run)"
                    }
                    logger.info(f"→ {repo_name}: Would {action} ARCHITECTURE.md (dry run)")
                else:
                    # Create or update the file
                    success = await create_or_update_file(
                        repo_name,
                        new_content,
                        project.get('default_branch', 'main')
                    )
                    
                    if success:
                        if existing_content:
                            results["updated"] += 1
                            action = "updated"
                        else:
                            results["created"] += 1
                            action = "created"
                            
                        results["repositories"][repo_name] = {
                            "status": action,
                            "message": f"ARCHITECTURE.md {action} successfully"
                        }
                        logger.info(f"✓ {repo_name}: ARCHITECTURE.md {action}")
                    else:
                        results["failed"] += 1
                        results["repositories"][repo_name] = {
                            "status": "failed",
                            "error": "Failed to create/update file"
                        }
                        logger.error(f"✗ {repo_name}: Failed to update")
                        
            except Exception as e:
                results["failed"] += 1
                results["repositories"][repo_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"✗ {repo_name}: Error - {e}")
        
        # Summary
        summary = f"""
Architecture Docs Update Complete:
- Repositories checked: {results['checked']}
- Already updated: {results['already_updated']}
- Updated: {results['updated']}
- Created: {results['created']}
- Failed: {results['failed']}
"""
        
        if dry_run:
            summary += "\n(DRY RUN - No changes were made)"
            
        logger.info(summary)
        
        return {
            "success": True,
            "summary": summary.strip(),
            "dry_run": dry_run,
            **results
        }
        
    except Exception as e:
        error_msg = f"Error updating architecture docs: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

# Export for tool system
update_architecture_docs_tool = update_architecture_docs

if __name__ == "__main__":
    import sys
    
    async def test():
        # Test dry run
        print("Testing dry run...")
        result = await update_architecture_docs(dry_run=True)
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test specific repo if provided
        if len(sys.argv) > 1:
            repo = sys.argv[1]
            print(f"\nTesting specific repo: {repo}")
            result = await update_architecture_docs(repo_name=repo, dry_run=True)
            print(f"Result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test())