"""
Project Creator

Creates new projects by forking Laravel React starter kit.
Handles all GitHub operations and project customization.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import asyncio
from github import Github, GithubException
import base64
from scripts.task_manager import TaskManager


class ProjectCreator:
    """Create projects dynamically from Laravel React starter kit."""
    
    STARTER_KIT_REPO = "laravel/react-starter-kit"
    ORGANIZATION = "CodeWebMobile-AI"
    
    def __init__(self, github_token: str, ai_brain=None):
        """Initialize with GitHub client and AI brain.
        
        Args:
            github_token: GitHub personal access token
            ai_brain: AI brain for intelligent customization
        """
        self.github = Github(github_token)
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        self.created_projects = []
        self.task_manager = TaskManager(github_token)
        
    async def create_project(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create project by forking starter kit and customizing.
        
        Args:
            task: NEW_PROJECT task with details
            
        Returns:
            Creation result with repo details
        """
        self.logger.info(f"Creating project for task: {task.get('title', 'Unknown')}")
        
        try:
            # Generate project details using AI
            project_details = await self._generate_project_details(task)
            
            # Check if project already exists
            if self._project_exists(project_details['name']):
                return {
                    'success': False,
                    'error': f"Project {project_details['name']} already exists"
                }
            
            # Fork the starter kit
            forked_repo = await self._fork_starter_kit(project_details['name'])
            
            if not forked_repo:
                return {
                    'success': False,
                    'error': 'Failed to fork starter kit'
                }
            
            # Customize the project
            customization_result = await self._customize_project(
                forked_repo, 
                project_details
            )
            
            # Create initial project structure
            structure_result = await self._create_initial_structure(
                forked_repo,
                project_details
            )
            
            # Create initial issues/tasks
            issues_created = await self._create_initial_issues(
                forked_repo,
                project_details
            )
            
            # Set up project settings
            await self._configure_project_settings(forked_repo, project_details)
            
            # Record creation
            creation_record = {
                'success': True,
                'repo_url': forked_repo.html_url,
                'repo_name': forked_repo.name,
                'project_name': project_details['name'],
                'description': project_details['description'],
                'customizations': customization_result,
                'initial_issues': issues_created,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.created_projects.append(creation_record)
            
            return creation_record
            
        except Exception as e:
            self.logger.error(f"Project creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_project_details(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project details using AI.
        
        Args:
            task: Task details
            
        Returns:
            Project details
        """
        if not self.ai_brain:
            # Fallback to extracting from task
            return self._extract_project_details(task)
            
        prompt = f"""
        Generate detailed project specifications for this NEW_PROJECT task.
        
        Task Details:
        {json.dumps(task, indent=2)}
        
        The project will be created by forking the Laravel React starter kit which provides:
        - Laravel 10+ backend with Sanctum authentication
        - React 18+ with TypeScript frontend
        - Tailwind CSS for styling
        - PostgreSQL database
        - Redis for caching and queues
        - Docker development environment
        - GitHub Actions CI/CD
        
        Generate project details including:
        
        1. name: Valid GitHub repository name (lowercase, hyphens, no spaces)
        2. description: Clear, concise project description (max 200 chars)
        3. customizations: Specific changes needed from base starter kit:
           - Additional packages to install (composer/npm)
           - Configuration changes
           - Initial features to implement
        4. initial_features: First 5 features to build (specific and actionable)
        5. tech_additions: Any additional technologies/services needed
        6. readme_sections: Custom sections for README
        7. environment_variables: Additional env vars needed
        8. database_schema: Initial schema considerations
        
        Ensure the name is unique, descriptive, and follows GitHub naming conventions.
        Format as JSON.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        details = self._parse_json_response(response)
        
        # Ensure required fields
        return self._ensure_project_details(details, task)
    
    def _extract_project_details(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract project details from task without AI.
        
        Args:
            task: Task details
            
        Returns:
            Basic project details
        """
        title = task.get('title', 'New Project')
        
        # Generate name from title
        name = title.lower()
        name = name.replace('[test]', '').replace('[swarm]', '')
        name = name.replace(' ', '-')
        name = ''.join(c for c in name if c.isalnum() or c == '-')
        name = name.strip('-')[:50]  # GitHub limit
        
        return {
            'name': name or 'new-project',
            'description': task.get('description', '')[:200],
            'customizations': {
                'packages': [],
                'configuration': [],
                'features': task.get('requirements', [])[:5]
            },
            'initial_features': task.get('requirements', [])[:5],
            'tech_additions': [],
            'readme_sections': [],
            'environment_variables': [],
            'database_schema': {}
        }
    
    def _ensure_project_details(self, details: Dict[str, Any], 
                               task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure project details have all required fields.
        
        Args:
            details: Generated details
            task: Original task
            
        Returns:
            Complete project details
        """
        # Ensure name
        if not details.get('name'):
            details['name'] = self._extract_project_details(task)['name']
            
        # Ensure description
        if not details.get('description'):
            details['description'] = task.get('description', '')[:200]
            
        # Ensure other fields
        defaults = {
            'customizations': {},
            'initial_features': [],
            'tech_additions': [],
            'readme_sections': [],
            'environment_variables': [],
            'database_schema': {}
        }
        
        for key, default in defaults.items():
            if key not in details:
                details[key] = default
                
        return details
    
    def _project_exists(self, name: str) -> bool:
        """Check if project already exists in organization.
        
        Args:
            name: Project name
            
        Returns:
            True if exists
        """
        try:
            org = self.github.get_organization(self.ORGANIZATION)
            org.get_repo(name)
            return True
        except GithubException:
            return False
    
    async def _fork_starter_kit(self, project_name: str):
        """Fork the Laravel React starter kit.
        
        Args:
            project_name: Name for the new project
            
        Returns:
            Forked repository object or None
        """
        try:
            # Get starter kit repo
            starter_repo = self.github.get_repo(self.STARTER_KIT_REPO)
            
            # Get organization
            org = self.github.get_organization(self.ORGANIZATION)
            
            # Fork to organization
            self.logger.info(f"Forking {self.STARTER_KIT_REPO} to {self.ORGANIZATION}/{project_name}")
            
            # GitHub doesn't allow direct org forking via API, so we need to:
            # 1. Create a new repo
            # 2. Push starter kit contents to it
            
            # Create new repository
            new_repo = org.create_repo(
                name=project_name,
                description=f"Project created from Laravel React starter kit",
                private=False,
                has_issues=True,
                has_wiki=True,
                has_downloads=True,
                auto_init=False  # Don't auto-init, we'll push from starter
            )
            
            # Clone starter kit contents
            # In a real implementation, this would use git operations
            # For now, we'll copy key files
            await self._copy_starter_contents(starter_repo, new_repo)
            
            return new_repo
            
        except Exception as e:
            self.logger.error(f"Failed to fork starter kit: {e}")
            return None
    
    async def _copy_starter_contents(self, source_repo, target_repo):
        """Copy contents from starter kit to new repo.
        
        Args:
            source_repo: Source repository (starter kit)
            target_repo: Target repository (new project)
        """
        try:
            # Get default branch
            default_branch = source_repo.default_branch
            
            # Get contents of root directory
            contents = source_repo.get_contents("", ref=default_branch)
            
            # Copy each file/directory
            while contents:
                file_content = contents.pop(0)
                
                if file_content.type == "dir":
                    # Get directory contents and add to queue
                    contents.extend(
                        source_repo.get_contents(file_content.path, ref=default_branch)
                    )
                else:
                    # Copy file
                    try:
                        target_repo.create_file(
                            path=file_content.path,
                            message=f"Initial commit: Add {file_content.path}",
                            content=file_content.decoded_content,
                            branch=target_repo.default_branch
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to copy {file_content.path}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to copy starter contents: {e}")
    
    async def _customize_project(self, repo, details: Dict[str, Any]) -> Dict[str, Any]:
        """Customize the forked project with AI-generated content.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            Customization results
        """
        results = {
            'readme_updated': False,
            'package_json_updated': False,
            'env_example_updated': False,
            'custom_files_created': []
        }
        
        # Update README
        readme_content = await self._generate_readme(repo, details)
        if readme_content:
            try:
                readme_file = repo.get_contents("README.md")
                repo.update_file(
                    "README.md",
                    f"Customize README for {details['name']}",
                    readme_content,
                    readme_file.sha
                )
                results['readme_updated'] = True
            except Exception as e:
                self.logger.error(f"Failed to update README: {e}")
                
        # Update package.json with project name
        try:
            package_file = repo.get_contents("package.json")
            package_data = json.loads(package_file.decoded_content)
            package_data['name'] = details['name']
            package_data['description'] = details['description']
            
            repo.update_file(
                "package.json",
                f"Update package.json for {details['name']}",
                json.dumps(package_data, indent=2),
                package_file.sha
            )
            results['package_json_updated'] = True
        except Exception as e:
            self.logger.error(f"Failed to update package.json: {e}")
            
        # Update .env.example if needed
        if details.get('environment_variables'):
            env_content = await self._generate_env_example(details)
            if env_content:
                try:
                    env_file = repo.get_contents(".env.example")
                    updated_env = env_file.decoded_content.decode() + "\n" + env_content
                    
                    repo.update_file(
                        ".env.example",
                        "Add project-specific environment variables",
                        updated_env,
                        env_file.sha
                    )
                    results['env_example_updated'] = True
                except Exception as e:
                    self.logger.error(f"Failed to update .env.example: {e}")
                    
        return results
    
    async def _generate_readme(self, repo, details: Dict[str, Any]) -> Optional[str]:
        """Generate customized README content.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            README content or None
        """
        if not self.ai_brain:
            return self._generate_basic_readme(details)
            
        prompt = f"""
        Generate a comprehensive README.md for this Laravel React project.
        
        Project Details:
        {json.dumps(details, indent=2)}
        
        The project is based on Laravel React starter kit with:
        - Laravel 10+ backend with Sanctum auth
        - React 18+ with TypeScript
        - Tailwind CSS
        - PostgreSQL database
        - Redis caching
        - Docker development setup
        
        Include these sections:
        1. Project title and description
        2. Features (initial features from details)
        3. Tech Stack
        4. Prerequisites
        5. Installation steps
        6. Environment setup
        7. Development workflow
        8. Testing
        9. Deployment
        10. Contributing guidelines
        11. Any custom sections from details
        
        Make it professional, clear, and specific to this project.
        Use Markdown formatting.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return response.get('content', '')
    
    def _generate_basic_readme(self, details: Dict[str, Any]) -> str:
        """Generate basic README without AI.
        
        Args:
            details: Project details
            
        Returns:
            Basic README content
        """
        features_list = '\n'.join([f"- {f}" for f in details.get('initial_features', [])])
        
        return f"""# {details['name']}

{details['description']}

## Features

{features_list}

## Tech Stack

- **Backend**: Laravel 10+ with Sanctum authentication
- **Frontend**: React 18+ with TypeScript
- **Styling**: Tailwind CSS
- **Database**: PostgreSQL
- **Caching**: Redis
- **Development**: Docker

## Prerequisites

- Docker and Docker Compose
- Node.js 18+
- PHP 8.1+
- Composer

## Installation

1. Clone the repository:
```bash
git clone https://github.com/{self.ORGANIZATION}/{details['name']}.git
cd {details['name']}
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Start Docker containers:
```bash
docker-compose up -d
```

4. Install dependencies:
```bash
docker-compose exec app composer install
docker-compose exec app npm install
```

5. Generate application key:
```bash
docker-compose exec app php artisan key:generate
```

6. Run migrations:
```bash
docker-compose exec app php artisan migrate
```

7. Start development servers:
```bash
docker-compose exec app npm run dev
```

## Development

Access the application at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Testing

Run tests with:
```bash
docker-compose exec app php artisan test
docker-compose exec app npm test
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the MIT License.
"""
    
    async def _generate_env_example(self, details: Dict[str, Any]) -> str:
        """Generate additional environment variables.
        
        Args:
            details: Project details
            
        Returns:
            Environment variables content
        """
        env_vars = details.get('environment_variables', [])
        
        if not env_vars:
            return ""
            
        content = "\n# Project-specific environment variables\n"
        
        for var in env_vars:
            if isinstance(var, dict):
                content += f"{var.get('name', 'VAR')}={var.get('default', '')}\n"
            else:
                content += f"{var}=\n"
                
        return content
    
    async def _create_initial_structure(self, repo, 
                                       details: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial project structure and files.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            Structure creation results
        """
        results = {
            'directories_created': [],
            'files_created': []
        }
        
        # Create any project-specific directories/files based on features
        # This is where you'd add custom components, services, etc.
        
        return results
    
    async def _create_initial_issues(self, repo, 
                                    details: Dict[str, Any]) -> List[int]:
        """Create initial GitHub issues for project features.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            List of created issue numbers
        """
        issue_numbers = []
        
        # Create setup issue using centralized method
        setup_description = """## Initial Setup Tasks

- [ ] Configure environment variables
- [ ] Set up database schema
- [ ] Configure authentication
- [ ] Set up CI/CD pipeline
- [ ] Add project documentation
- [ ] Configure production deployment

This issue tracks the initial setup of the project."""
        
        # Use task manager's centralized method to ensure @claude mention
        self.task_manager.repo = repo
        setup_issue_number = self.task_manager.create_ai_task_issue(
            title="Initial Project Setup",
            description=setup_description,
            labels=['setup'],
            priority="high",
            task_type="setup"
        )
        
        # Get the issue object for compatibility
        setup_issue = repo.get_issue(setup_issue_number) if setup_issue_number else None
        issue_numbers.append(setup_issue.number)
        
        # Create feature issues using centralized method
        for i, feature in enumerate(details.get('initial_features', [])[:5]):
            feature_description = f"""## Feature Implementation

Implement the following feature: {feature}

### Acceptance Criteria
- Feature is fully functional
- Tests are written and passing
- Documentation is updated
- Code follows project standards

### Technical Considerations
- Follow Laravel/React best practices
- Ensure responsive design
- Consider performance implications"""
            
            feature_issue_number = self.task_manager.create_ai_task_issue(
                title=f"Implement: {feature}",
                description=feature_description,
                labels=['feature', 'enhancement'],
                priority="medium",
                task_type="feature"
            )
            
            if feature_issue_number:
                issue_numbers.append(feature_issue_number)
            
        return issue_numbers
    
    async def _configure_project_settings(self, repo, 
                                         details: Dict[str, Any]) -> None:
        """Configure repository settings.
        
        Args:
            repo: Repository object  
            details: Project details
        """
        try:
            # Update repository description
            repo.edit(
                description=details['description'],
                has_wiki=True,
                has_issues=True,
                has_projects=True
            )
            
            # Add topics/tags
            topics = ['laravel', 'react', 'typescript', 'fullstack']
            
            # Add project-specific topics based on features
            if 'auth' in str(details.get('initial_features', [])).lower():
                topics.append('authentication')
            if 'api' in str(details.get('initial_features', [])).lower():
                topics.append('rest-api')
                
            repo.replace_topics(topics)
            
        except Exception as e:
            self.logger.error(f"Failed to configure project settings: {e}")
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed JSON or empty dict
        """
        content = response.get('content', '')
        
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        return {}
    
    def get_created_projects(self) -> List[Dict[str, Any]]:
        """Get list of projects created by this instance.
        
        Returns:
            List of created project records
        """
        return self.created_projects
    
    async def update_project(self, repo_url: str, updates: Dict[str, Any]) -> bool:
        """Update an existing project with new features or changes.
        
        Args:
            repo_url: Repository URL
            updates: Updates to apply
            
        Returns:
            Success status
        """
        try:
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1]
            org_name = repo_url.split('/')[-2]
            
            repo = self.github.get_repo(f"{org_name}/{repo_name}")
            
            # Apply updates based on type
            if 'new_features' in updates:
                for feature in updates['new_features']:
                    repo.create_issue(
                        title=f"Feature: {feature}",
                        body="Feature requested by AI orchestrator",
                        labels=['feature']
                    )
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update project: {e}")
            return False