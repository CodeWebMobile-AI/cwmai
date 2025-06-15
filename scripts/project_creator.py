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
import os
import aiohttp


class ProjectCreator:
    """Create diverse, innovative projects based on market needs."""
    
    # Single starter kit option
    STARTER_KITS = {
        'laravel-react': 'laravel/react-starter-kit'
    }
    ORGANIZATION = "CodeWebMobile-AI"
    
    def __init__(self, github_token: str, ai_brain=None):
        """Initialize with GitHub client and AI brain.
        
        Args:
            github_token: GitHub personal access token
            ai_brain: AI brain for intelligent customization
        """
        self.github = Github(github_token)
        self.github_token = github_token
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
            tech_stack = project_details.get('tech_stack', project_details.get('tech_additions', ''))
            self.project_description = project_details.get('description', '')
            forked_repo = await self._fork_starter_kit(project_details['name'], tech_stack)
            
            if not forked_repo:
                return {
                    'success': False,
                    'error': 'Failed to fork starter kit'
                }
            
            # Customize the project
            self.logger.info(f"Starting project customization for {project_details['name']}")
            customization_result = await self._customize_project(
                forked_repo, 
                project_details
            )
            self.logger.info(f"Customization result: {customization_result}")
            
            # Create initial project structure
            self.logger.info(f"Creating initial structure for {project_details['name']}")
            structure_result = await self._create_initial_structure(
                forked_repo,
                project_details
            )
            self.logger.info(f"Structure result: {structure_result}")
            
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
        """Generate project details using AI after market research.
        
        Args:
            task: Task details
            
        Returns:
            Project details
        """
        # Check if task already has architecture and project details in metadata
        metadata = task.get('metadata', {})
        self.logger.info(f"Task metadata keys: {list(metadata.keys())}")
        self.logger.info(f"Has architecture: {bool(metadata.get('architecture'))}")
        self.logger.info(f"Has selected_project: {bool(metadata.get('selected_project'))}")
        
        if metadata.get('architecture') and metadata.get('selected_project'):
            self.logger.info("✅ Using pre-generated architecture and project details from task metadata")
            
            selected_project = metadata['selected_project']
            architecture = metadata['architecture']
            
            # Convert to project details format
            return {
                'name': self._sanitize_project_name(selected_project.get('project_name', 'new-project')),
                'description': selected_project.get('project_goal', '')[:200],
                'problem_statement': selected_project.get('problem_solved', ''),
                'target_audience': selected_project.get('target_audience', ''),
                'monetization_strategy': selected_project.get('monetization_strategy', ''),
                'initial_features': selected_project.get('key_features', []),
                'core_entities': selected_project.get('core_entities', []),
                'architecture': architecture,
                'customizations': {
                    'packages': [],
                    'configuration': [],
                    'features': selected_project.get('key_features', [])
                },
                'tech_additions': [],
                'readme_sections': [],
                'environment_variables': [],
                'database_schema': architecture.get('foundational_architecture', {}).get('database_schema', {})
            }
        
        self.logger.warning("⚠️ No pre-generated metadata found - falling back to AI generation")
        
        if not self.ai_brain:
            raise ValueError("AI Brain is required for project generation - cannot proceed without AI")
        
        # First, research real-world problems and opportunities
        self.logger.info("🔍 Starting market research for project generation")
        research_results = await self._research_market_opportunities()
        
        if not research_results:
            raise ValueError("Market research failed - cannot generate project without real-world insights")
        
        # Generate project based on research
        prompt = f"""
        Based on real market research, generate a project that solves an actual problem.
        
        Market Research Results:
        {json.dumps(research_results, indent=2)}
        
        Task Context:
        {json.dumps(task, indent=2)}
        
        The project will be created by forking the Laravel React starter kit which provides:
        - Laravel 11+ backend with Sanctum authentication
        - React 18+ with TypeScript frontend  
        - Tailwind CSS for styling
        - MySQL database (NOT PostgreSQL)
        - Redis for caching, queues, and real-time updates
        - Laravel Echo Server (local) for WebSockets (NO Pusher)
        - Docker development environment
        - GitHub Actions CI/CD
        
        Generate project details that:
        1. Addresses a real problem identified in the research
        2. Has clear monetization potential (24/7 revenue)
        3. Solves everyday problems for real people
        4. Can scale to serve many users
        
        Include:
        1. name: Valid GitHub repository name (lowercase, hyphens, no spaces)
        2. description: Clear, concise project description (max 200 chars)
        3. problem_statement: The real-world problem being solved
        4. target_audience: Who will use this and why
        5. monetization_strategy: How this will generate revenue 24/7
        6. customizations: Specific changes needed from base starter kit
        7. initial_features: First 5 features to build (specific and actionable)
        8. tech_additions: Any additional technologies/services needed
        9. readme_sections: Custom sections for README
        10. environment_variables: Additional env vars needed
        11. database_schema: Initial schema considerations
        12. market_validation: Evidence this solution is needed
        
        Ensure the project is innovative, practical, and has real business value.
        Format as JSON.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        details = self._parse_json_response(response)
        
        # Ensure required fields
        return self._ensure_project_details(details, task)
    
    async def _research_market_opportunities(self) -> Dict[str, Any]:
        """Research real-world problems and market opportunities using AI brain.
        
        Returns:
            Market research results
        """
        try:
            # Prepare research prompt
            research_prompt = """
            Research current market opportunities and real-world problems that need solving.
            Focus on:
            
            1. Problems people face daily that technology could solve
            2. Inefficiencies in common business processes  
            3. Gaps in current software solutions
            4. Emerging trends that create new needs
            5. Services that could generate passive income 24/7
            
            Consider these criteria:
            - Must solve real problems for real people
            - Should have clear monetization potential
            - Can operate with minimal human intervention
            - Scalable to serve many users
            - Not overly saturated in the market
            
            Research areas:
            - Small business operations
            - Personal productivity
            - Health and wellness
            - Education and learning
            - Financial management
            - Community and social needs
            - Environmental solutions
            - Remote work challenges
            
            Provide 5 specific problem/opportunity pairs with:
            - problem_description: Clear description of the problem
            - affected_audience: Who experiences this problem
            - current_solutions: What exists now and why it's insufficient
            - opportunity: How technology could solve this better
            - monetization_model: How to generate revenue
            - market_size: Estimated audience size
            - competition_level: Low/Medium/High
            
            Format as JSON array.
            """
            
            # Use AI brain to get research
            response = await self.ai_brain.generate_enhanced_response(research_prompt, model='gemini')
            
            if response and 'content' in response:
                content = response['content']
                
                # Parse JSON from response
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    opportunities = json.loads(json_match.group())
                    return {
                        'opportunities': opportunities,
                        'research_timestamp': datetime.now(timezone.utc).isoformat(),
                        'source': 'ai_brain_gemini'
                    }
            
            # If parsing fails, raise error
            raise ValueError("Failed to parse market research response - AI generation required")
                        
        except Exception as e:
            self.logger.error(f"Market research failed: {e}")
            raise
    
    
    
    
    def _ensure_project_details(self, details: Dict[str, Any], 
                               task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure project details have all required fields.
        
        Args:
            details: Generated details
            task: Original task
            
        Returns:
            Complete project details
        """
        # Required fields that must be present
        required_fields = ['name', 'description', 'problem_statement', 'initial_features']
        missing_fields = [field for field in required_fields if not details.get(field)]
        
        if missing_fields:
            raise ValueError(f"AI generation incomplete - missing required fields: {', '.join(missing_fields)}")
            
        # Ensure other fields have at least empty values
        defaults = {
            'customizations': {},
            'tech_additions': [],
            'readme_sections': [],
            'environment_variables': [],
            'database_schema': {}
        }
        
        for key, default in defaults.items():
            if key not in details:
                details[key] = default
                
        return details
    
    def _generate_problem_statement(self, details: Dict[str, Any]) -> str:
        """Get problem statement from project details.
        
        Args:
            details: Project details
            
        Returns:
            Problem statement text
        """
        if not details.get('problem_statement'):
            raise ValueError("Problem statement is required but was not generated by AI")
        
        return details['problem_statement']
    
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
    
    async def _fork_starter_kit(self, project_name: str, tech_stack: str = None):
        """Fork the Laravel React starter kit.
        
        Args:
            project_name: Name for the new project
            tech_stack: Technology stack to use (ignored - always uses Laravel React)
            
        Returns:
            Forked repository object or None
        """
        try:
            # Always use Laravel React starter kit
            starter_kit_key = 'laravel-react'
            starter_kit_repo = self.STARTER_KITS['laravel-react']
            
            # Get starter kit repo
            starter_repo = self.github.get_repo(starter_kit_repo)
            
            # Get organization
            org = self.github.get_organization(self.ORGANIZATION)
            
            # Fork to organization
            self.logger.info(f"Forking {starter_kit_repo} to {self.ORGANIZATION}/{project_name}")
            
            # GitHub doesn't allow direct org forking via API, so we need to:
            # 1. Create a new repo
            # 2. Push starter kit contents to it
            
            # Create new repository
            new_repo = org.create_repo(
                name=project_name,
                description=self.project_description or f"Project created from {starter_kit_key} starter kit",
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
        """Copy contents from starter kit to new repo using git commands.
        
        Args:
            source_repo: Source repository (starter kit)
            target_repo: Target repository (new project)
        """
        import subprocess
        import tempfile
        import shutil
        
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="starter_kit_")
            self.logger.info(f"Using temporary directory: {temp_dir}")
            
            # Clone the starter kit
            clone_url = source_repo.clone_url
            self.logger.info(f"Cloning starter kit from {clone_url}")
            
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, temp_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to clone starter kit: {result.stderr}")
                raise Exception(f"Git clone failed: {result.stderr}")
            
            # Remove the .git directory to start fresh
            git_dir = os.path.join(temp_dir, ".git")
            if os.path.exists(git_dir):
                shutil.rmtree(git_dir)
            
            # Initialize new git repository
            os.chdir(temp_dir)
            subprocess.run(["git", "init"], check=True)
            
            # Configure git user for commits
            subprocess.run(["git", "config", "user.email", "ai@codewebmobile.com"], check=True)
            subprocess.run(["git", "config", "user.name", "CodeWebMobile AI"], check=True)
            
            # Add all files
            subprocess.run(["git", "add", "."], check=True)
            
            # Make initial commit
            subprocess.run(
                ["git", "commit", "-m", f"Initial commit from {source_repo.name} starter kit"],
                check=True
            )
            
            # Add the new repository as remote
            # Use token authentication in URL
            target_clone_url = target_repo.clone_url
            if target_clone_url.startswith("https://"):
                # Insert token after https://
                target_clone_url = target_clone_url.replace(
                    "https://github.com/", 
                    f"https://{self.github_token}@github.com/"
                )
            
            self.logger.info(f"Setting remote origin to {target_repo.full_name}")
            subprocess.run(
                ["git", "remote", "add", "origin", target_clone_url],
                check=True
            )
            
            # Push to the new repository
            self.logger.info(f"Pushing to {target_repo.full_name}")
            push_result = subprocess.run(
                ["git", "push", "-u", "origin", "main"],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for push
            )
            
            if push_result.returncode != 0:
                # Try with master branch if main fails
                self.logger.warning("Push to main failed, trying master branch")
                subprocess.run(["git", "branch", "-m", "master"], check=True)
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", "master"],
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout for push
                )
                
                if push_result.returncode != 0:
                    self.logger.error(f"Failed to push: {push_result.stderr}")
                    raise Exception(f"Git push failed: {push_result.stderr}")
            
            self.logger.info("Successfully pushed starter kit contents to new repository")
            
            # Give GitHub more time to process the push and make files available
            self.logger.info("Waiting for GitHub to process the push and make files available...")
            await asyncio.sleep(5)  # Increased from 3 to 5 seconds
            
            # Verify the repository has content with retry
            max_retries = 3
            for retry in range(max_retries):
                try:
                    contents = target_repo.get_contents("")
                    self.logger.info(f"✅ Repository now has {len(contents)} items in root")
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        self.logger.warning(f"Repository not ready yet (attempt {retry + 1}/{max_retries}), waiting...")
                        await asyncio.sleep(2)
                    else:
                        self.logger.warning(f"Could not verify repository contents after {max_retries} attempts: {e}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git command failed: {e}")
            self.logger.error(f"Command: {e.cmd}")
            self.logger.error(f"Return code: {e.returncode}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to copy starter contents: {e}")
            raise
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    # Change out of temp directory first
                    os.chdir("/tmp")
                    shutil.rmtree(temp_dir)
                    self.logger.info("Cleaned up temporary directory")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp directory: {e}")
    
    async def _generate_project_architecture(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive project architecture using AI.
        
        Args:
            details: Project details including problem statement
            
        Returns:
            Complete architecture specification
        """
        architecture_prompt = f"""
        You are a distinguished Chief Technology Officer (CTO), a pragmatic Principal Engineer, and a skilled UI/UX Designer. 
        Your expertise lies in creating scalable, secure, and user-friendly web applications using the Laravel and React/Inertia.js ecosystem with TypeScript.

        **Primary Directive:**
        Leverage your most current knowledge, including best practices, security standards, and package releases up to June 2025. 
        All frontend code must be written in TypeScript. All design choices must adhere to accessibility standards (WCAG AA). 
        Where a specific item is recommended, provide a brief justification and a link to its source.

        **Project Context:**
        * **Starting Point:** "The project will be bootstrapped using the official 'laravel/react-starter-kit', which uses TypeScript. All React components will have a .tsx extension."
        * **Technical Mandates:**
          * **Real-time Requirement:** "For any features that benefit from real-time updates, the architecture MUST use Redis and the 'laravel-echo-server' running locally."
        * **Non-Functional Requirements:**
          * **Testing Strategy:** "The architecture MUST include a comprehensive testing strategy (Pest for backend, Vitest & RTL for frontend)."
          * **Security Posture:** "A 'security-first' approach is mandatory, addressing OWASP Top 10 vulnerabilities."
          * **Observability:** "The plan must include a strategy for logging (e.g., 'stderr' channel) and monitoring key metrics."
        * **Design System Requirements:**
          * **Typography & Colors:** "Generate a simple, professional design system. The color palette must be accessible and logically structured. The typography must be clean and highly readable, sourced from Google Fonts."
        * **Project Name:** "{details.get('name', 'project')}"
        * **Project Goal:** "{details.get('problem_statement', details.get('description', 'Build a web application'))}"
        * **Core Entities:** "{', '.join(self._extract_core_entities(details))}"
        * **Key Features:** "{json.dumps(details.get('initial_features', []), indent=2)}"
        * **Expected Scale:** "Start with a small user base, but design with a clear path for significant scaling."
        * **Team Skills:** "The team is proficient with Laravel but intermediate with React and TypeScript."

        **Task:**
        Generate a **complete, production-grade System Architecture, Design System, and Feature Implementation Roadmap**. 
        The document must provide a foundational engineering architecture AND a foundational design system, plus a detailed plan for implementing each "Key Feature".

        The final output must be a single JSON object that strictly adheres to the following tool schema.

        **Tool Schema (JSON Output):**
        {{
          "title": "Full-Stack Blueprint for {details.get('name', 'project')}",
          "description": "A comprehensive architecture, design system, and feature roadmap.",
          "design_system": {{
            "suggested_font": {{
              "font_name": "e.g., Inter",
              "google_font_link": "A link to the Google Fonts page for the selected font.",
              "font_stack": "The CSS font-family stack (e.g., 'Inter', sans-serif).",
              "rationale": "A brief explanation of why this font was chosen (e.g., for its excellent readability at various sizes)."
            }},
            "color_palette": {{
              "rationale": "A brief explanation of the color theory behind the palette choice (e.g., 'A complementary palette chosen for its balance and professional feel').",
              "primary": {{ "name": "Primary", "hex": "#RRGGBB", "usage": "Main brand color, used for headers and primary actions." }},
              "secondary": {{ "name": "Secondary", "hex": "#RRGGBB", "usage": "Supporting color for secondary elements." }},
              "accent": {{ "name": "Accent", "hex": "#RRGGBB", "usage": "Used for call-to-action buttons and highlights." }},
              "neutral_text": {{ "name": "Neutral Text", "hex": "#RRGGBB", "usage": "Primary text color for high contrast and readability." }},
              "neutral_background": {{ "name": "Neutral Background", "hex": "#RRGGBB", "usage": "Main background color for content areas." }},
              "neutral_border": {{ "name": "Neutral Border", "hex": "#RRGGBB", "usage": "For card borders, dividers, and form inputs." }},
              "success": {{ "name": "Success", "hex": "#RRGGBB", "usage": "For success messages and confirmation." }},
              "warning": {{ "name": "Warning", "hex": "#RRGGBB", "usage": "For warnings and non-critical alerts." }},
              "danger": {{ "name": "Danger", "hex": "#RRGGBB", "usage": "For error messages and destructive actions." }}
            }}
          }},
          "foundational_architecture": {{
            "core_components": {{ "section_title": "1. Core Components & Rationale", "content": "..." }},
            "database_schema": {{ "section_title": "2. Database Schema Design", "content": "..." }},
            "api_design": {{ "section_title": "3. API Design & Key Endpoints", "content": "..." }},
            "frontend_structure": {{ "section_title": "4. Frontend Structure (TypeScript)", "content": "..." }},
            "real_time_architecture": {{ "section_title": "5. Real-time & Events Architecture", "content": "..." }},
            "auth_flow": {{ "section_title": "6. Authentication & Authorization Flow", "content": "..." }},
            "deployment_plan": {{ "section_title": "7. Deployment & Scalability Plan", "content": "..." }},
            "testing_strategy": {{ "section_title": "8. Testing Strategy", "content": "..." }},
            "security_hardening_plan": {{ "section_title": "9. Security & Hardening Plan", "content": "..." }},
            "logging_and_observability": {{ "section_title": "10. Logging & Observability", "content": "..." }}
          }},
          "feature_implementation_roadmap": [
            {{
              "feature_name": "Example: Real-time User Notifications",
              "description": "...",
              "required_db_changes": [ ... ],
              "impacted_backend_components": [ ... ],
              "impacted_frontend_components": [ ... ],
              "new_api_endpoints": [ ... ],
              "real_time_events": [ ... ],
              "suggested_tests": [ ... ]
            }}
          ]
        }}
        """
        
        try:
            # Use AI brain with Gemini for architecture generation
            response = await self.ai_brain.generate_enhanced_response(architecture_prompt, model='gemini')
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.error(f"Architecture generation failed: {e}")
            return {}
    
    def _extract_core_entities(self, details: Dict[str, Any]) -> List[str]:
        """Extract core entities from project details.
        
        Args:
            details: Project details
            
        Returns:
            List of core entities
        """
        # Extract from features and description
        entities = set()
        
        # Common entity patterns
        entity_keywords = [
            'user', 'customer', 'product', 'order', 'payment', 'subscription',
            'post', 'article', 'comment', 'category', 'tag', 'media',
            'notification', 'message', 'report', 'analytics', 'dashboard',
            'account', 'profile', 'settings', 'team', 'organization'
        ]
        
        # Check features
        for feature in details.get('initial_features', []):
            feature_lower = feature.lower()
            for keyword in entity_keywords:
                if keyword in feature_lower:
                    entities.add(keyword.capitalize())
        
        # Check description
        description = details.get('description', '').lower()
        for keyword in entity_keywords:
            if keyword in description:
                entities.add(keyword.capitalize())
        
        # Ensure we have at least some core entities
        if not entities:
            entities = {'User', 'Resource', 'Setting'}
            
        return sorted(list(entities))
    
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
        
        # Generate architecture if not already done
        if 'architecture' not in details:
            architecture = await self._generate_project_architecture(details)
            if architecture:
                details['architecture'] = architecture
                results['architecture_generated'] = True
        
        # Update README with architecture
        self.logger.info(f"📝 Generating README for {details['name']}")
        self.logger.info(f"Project has architecture: {bool(details.get('architecture'))}")
        self.logger.info(f"Project has problem_statement: {bool(details.get('problem_statement'))}")
        readme_content = await self._generate_readme(repo, details)
        if readme_content:
            self.logger.info(f"✅ README generated, length: {len(readme_content)} chars")
            self.logger.info(f"README preview (first 200 chars): {readme_content[:200]}...")
            # Try to update README with retry logic
            max_retries = 3
            for retry in range(max_retries):
                try:
                    readme_file = repo.get_contents("README.md")
                    self.logger.info(f"Current README SHA: {readme_file.sha}")
                    repo.update_file(
                        "README.md",
                        f"Customize README for {details['name']}",
                        readme_content,
                        readme_file.sha
                    )
                    results['readme_updated'] = True
                    self.logger.info("✅ README successfully updated in GitHub!")
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        self.logger.warning(f"README update attempt {retry + 1}/{max_retries} failed: {e}")
                        self.logger.info("Waiting before retry...")
                        await asyncio.sleep(2)
                        # Refresh repository object
                        repo = self.github.get_repo(repo.full_name)
                    else:
                        self.logger.error(f"❌ Failed to update README after {max_retries} attempts: {e}")
                        self.logger.error(f"Error type: {type(e).__name__}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            self.logger.warning("⚠️ No README content generated")
                
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
        
        # Save architecture document
        if details.get('architecture'):
            try:
                self.logger.info(f"📐 Creating ARCHITECTURE.md for {details['name']}")
                architecture_content = self._format_architecture_document(details)
                self.logger.info(f"Architecture document length: {len(architecture_content)} chars")
                repo.create_file(
                    "ARCHITECTURE.md",
                    "Add project architecture documentation",
                    architecture_content
                )
                results['architecture_saved'] = True
                self.logger.info("✅ ARCHITECTURE.md saved successfully!")
            except Exception as e:
                self.logger.error(f"❌ Failed to save architecture document: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
        return results
    
    def _format_architecture_document(self, details: Dict[str, Any]) -> str:
        """Format architecture as a comprehensive markdown document.
        
        Args:
            details: Project details including architecture
            
        Returns:
            Formatted architecture document
        """
        architecture = details.get('architecture', {})
        
        # Build the architecture document
        doc = f"""# {details.get('name', 'Project')} Architecture

## Overview
{details.get('description', '')}

### Problem Statement
{details.get('problem_statement', '')}

### Target Audience
{details.get('target_audience', '')}

### Core Entities
{', '.join(details.get('core_entities', []))}

## Design System
"""
        
        # Add design system details
        if architecture.get('design_system'):
            design = architecture['design_system']
            
            # Font information
            if design.get('suggested_font'):
                font = design['suggested_font']
                doc += f"""
### Typography
- **Font**: {font.get('font_name', 'Not specified')}
- **Font Stack**: `{font.get('font_stack', '')}`
- **Google Fonts**: {font.get('google_font_link', '')}
- **Rationale**: {font.get('rationale', '')}
"""
            
            # Color palette
            if design.get('color_palette'):
                doc += "\n### Color Palette\n"
                colors = design['color_palette']
                doc += f"{colors.get('rationale', '')}\n\n"
                
                for color_key in ['primary', 'secondary', 'accent', 'neutral_text', 
                                'neutral_background', 'neutral_border', 'success', 'warning', 'danger']:
                    if color_key in colors:
                        color = colors[color_key]
                        doc += f"- **{color.get('name', color_key)}**: `{color.get('hex', '')}` - {color.get('usage', '')}\n"
        
        # Add foundational architecture
        doc += "\n## System Architecture\n"
        
        if architecture.get('foundational_architecture'):
            foundation = architecture['foundational_architecture']
            
            for section_key, section in foundation.items():
                if isinstance(section, dict) and 'section_title' in section:
                    doc += f"\n### {section['section_title']}\n"
                    doc += f"{section.get('content', '')}\n"
        
        # Add feature implementation roadmap
        doc += "\n## Feature Implementation Roadmap\n"
        
        if architecture.get('feature_implementation_roadmap'):
            for i, feature in enumerate(architecture['feature_implementation_roadmap'], 1):
                doc += f"\n### {i}. {feature.get('feature_name', 'Feature')}\n"
                doc += f"{feature.get('description', '')}\n\n"
                
                if feature.get('required_db_changes'):
                    doc += "**Database Changes:**\n"
                    for change in feature['required_db_changes']:
                        doc += f"- {change}\n"
                    doc += "\n"
                
                if feature.get('impacted_backend_components'):
                    doc += "**Backend Components:**\n"
                    for component in feature['impacted_backend_components']:
                        doc += f"- {component}\n"
                    doc += "\n"
                
                if feature.get('impacted_frontend_components'):
                    doc += "**Frontend Components:**\n"
                    for component in feature['impacted_frontend_components']:
                        doc += f"- {component}\n"
                    doc += "\n"
                
                if feature.get('new_api_endpoints'):
                    doc += "**API Endpoints:**\n"
                    for endpoint in feature['new_api_endpoints']:
                        doc += f"- {endpoint}\n"
                    doc += "\n"
                
                if feature.get('real_time_events'):
                    doc += "**Real-time Events:**\n"
                    for event in feature['real_time_events']:
                        doc += f"- {event}\n"
                    doc += "\n"
                
                if feature.get('suggested_tests'):
                    doc += "**Testing Requirements:**\n"
                    for test in feature['suggested_tests']:
                        doc += f"- {test}\n"
                    doc += "\n"
        
        # Add metadata
        doc += f"""
## Metadata
- **Generated**: {datetime.now(timezone.utc).isoformat()}
- **Project Type**: Laravel React Starter Kit
- **Architecture Version**: 1.0.0

---
*This architecture document is maintained by the CodeWebMobile AI system and should be the source of truth for all development decisions.*
"""
        
        return doc
    
    async def _generate_readme(self, repo, details: Dict[str, Any]) -> Optional[str]:
        """Generate customized README content.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            README content or None
        """
        if not self.ai_brain:
            raise ValueError("AI Brain is required for README generation")
            
        # Prepare architecture details if available
        architecture_section = ""
        if details.get('architecture'):
            architecture_section = f"Architecture Details:\n{json.dumps(details.get('architecture', {}), indent=2)}\n\n"
        
        prompt = f"""
        Generate a comprehensive README.md for this project that CLEARLY EXPLAINS:
        1. WHAT the project does (the specific problem it solves)
        2. WHO it's for (target users)
        3. WHY it's needed (the pain point addressed)
        4. HOW it works (key features and workflow)
        
        Project Details:
        {json.dumps(details, indent=2)}
        
        Tech Stack (use what's specified in details, or suggest modern alternatives):
        {details.get('tech_stack', 'Modern web technologies')}
        
        {architecture_section}
        
        Include these sections:
        1. Project title and description
        2. Problem Statement (from details)
        3. Target Audience and Value Proposition
        4. Features (initial features from details)
        5. Tech Stack
        6. Architecture Overview (if available)
        7. Prerequisites
        8. Installation steps
        9. Environment setup
        10. Development workflow
        11. Testing
        12. Deployment
        13. Monetization Strategy (if available)
        14. Contributing guidelines
        15. Any custom sections from details
        
        Make it professional, clear, and specific to this project.
        Use Markdown formatting.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        content = response.get('content', '')
        
        if not content:
            raise ValueError("AI failed to generate README content")
            
        return content
    
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
            details: Project details including architecture
            
        Returns:
            List of created issue numbers
        """
        issue_numbers = []
        
        # Extract architecture details if available
        architecture = details.get('architecture', {})
        core_entities = details.get('core_entities', ['Post', 'Category'])
        api_design = architecture.get('foundational_architecture', {}).get('api_design', {})
        auth_flow = architecture.get('foundational_architecture', {}).get('auth_flow', {})
        
        # Build context-aware setup description
        setup_description = f"""## Initial Project Setup

Based on the project architecture for **{details.get('name', 'the project')}**, complete the following setup tasks:

1. **Configure API authentication**
   - [ ] Set up Sanctum/API authentication middleware as per the auth flow design
   - [ ] Create auth controllers following the project's authentication architecture
   - [ ] Add protected API routes based on the API design specifications
   - [ ] Implement role-based access control if specified in architecture

2. **Create initial data models**
   - [ ] Add core models: {', '.join(core_entities[:3]) if core_entities else 'Post, Category'}
   - [ ] Create corresponding migrations following the database schema design
   - [ ] Set up model relationships as defined in the architecture
   - [ ] Add resource controllers with proper API versioning

3. **Update starter kit dashboard component**
   - [ ] Customize the dashboard to display {details.get('name', 'project')} specific metrics
   - [ ] Add data display widgets using shadcn/ui components
   - [ ] Integrate with API endpoints defined in the architecture
   - [ ] Update sidebar navigation to match project features
   - [ ] Apply the project's design system (colors, fonts, etc.)

4. **Set up API integration**
   - [ ] Create API service layer following the architecture's API design
   - [ ] Add axios interceptors for auth tokens and error handling
   - [ ] Create custom hooks for data fetching patterns
   - [ ] Implement caching strategy if specified in architecture
   - [ ] Add real-time capabilities using Redis/Echo if required

5. **Configure development helpers**
   - [ ] Add environment variables for all services mentioned in architecture
   - [ ] Create database seeders for {', '.join(core_entities[:2]) if core_entities else 'demo data'}
   - [ ] Add TypeScript types matching the API response formats
   - [ ] Set up testing framework (Pest for backend, Vitest for frontend)
   - [ ] Configure ESLint/Prettier with project standards

**Architecture Context:**
- Target Audience: {details.get('target_audience', 'General users')}
- Key Features: {', '.join(details.get('initial_features', ['Core CRUD operations'])[:3])}
- Tech Stack: Laravel 11+, React 19, TypeScript, Tailwind CSS 4, Redis

This issue tracks the comprehensive initial setup aligned with the project's specific architecture and requirements."""
        
        # Use task manager's centralized method to ensure @claude mention
        setup_issue_number = self.task_manager.create_ai_task_issue(
            title="Initial Project Setup",
            description=setup_description,
            labels=['setup', 'api', 'frontend', 'database'],
            priority="high",
            task_type="setup",
            repository=repo.full_name  # Pass the target repository
        )
        
        # Get the issue object for compatibility
        setup_issue = repo.get_issue(setup_issue_number) if setup_issue_number else None
        issue_numbers.append(setup_issue.number)
        
        # Create feature issues using centralized method
        architecture = details.get('architecture', {})
        feature_roadmap = architecture.get('feature_implementation_roadmap', [])
        
        for i, feature in enumerate(details.get('initial_features', [])[:5]):
            # Find matching feature details from roadmap
            feature_details = None
            for roadmap_item in feature_roadmap:
                if feature.lower() in roadmap_item.get('feature_name', '').lower() or \
                   roadmap_item.get('feature_name', '').lower() in feature.lower():
                    feature_details = roadmap_item
                    break
            
            # Build architecture-aware feature description
            if feature_details:
                feature_description = f"""## Feature Implementation

Implement: **{feature}**

### Feature Overview
{feature_details.get('description', f'Implement the {feature} functionality as specified in the project architecture.')}

### Required Database Changes
{chr(10).join(f'- [ ] {change}' for change in feature_details.get('required_db_changes', ['Review and implement necessary database schema changes']))}

### Backend Implementation
**Impacted Components:**
{chr(10).join(f'- {component}' for component in feature_details.get('impacted_backend_components', ['Review architecture for specific components']))}

**New API Endpoints:**
{chr(10).join(f'- [ ] {endpoint}' for endpoint in feature_details.get('new_api_endpoints', ['Implement RESTful endpoints as per architecture']))}

### Frontend Implementation
**Impacted Components:**
{chr(10).join(f'- {component}' for component in feature_details.get('impacted_frontend_components', ['Review architecture for specific components']))}

### Real-time Features
{chr(10).join(f'- [ ] {event}' for event in feature_details.get('real_time_events', [])) if feature_details.get('real_time_events') else '- No real-time events required for this feature'}

### Testing Requirements
{chr(10).join(f'- [ ] {test}' for test in feature_details.get('suggested_tests', ['Unit tests', 'Integration tests', 'Feature tests']))}

### Technical Stack Context
- **Backend:** Laravel 11+ with {details.get('name', 'project')}-specific configurations
- **Frontend:** React 19 + TypeScript following the project's component structure
- **API:** RESTful with Sanctum authentication
- **Database:** MySQL with migrations
- **Real-time:** Redis + Laravel Echo (if applicable)"""
            else:
                # Fallback to generic but still context-aware description
                feature_description = f"""## Feature Implementation

Implement: **{feature}**

### Feature Overview
Implement the {feature} functionality as part of the {details.get('name', 'project')} application.

### Implementation Guidelines
Based on the project architecture for {details.get('name', 'project')}:

**Backend Tasks:**
- [ ] Create necessary models and migrations
- [ ] Implement API controllers and routes
- [ ] Add service layer if complex business logic
- [ ] Implement proper validation and authorization
- [ ] Follow the established API design patterns

**Frontend Tasks:**
- [ ] Create TypeScript interfaces for data types
- [ ] Build React components using shadcn/ui
- [ ] Implement data fetching with custom hooks
- [ ] Add proper error handling and loading states
- [ ] Ensure responsive design with Tailwind CSS

**Testing Requirements:**
- [ ] Write Pest tests for backend functionality
- [ ] Add Vitest tests for React components
- [ ] Include integration tests for API endpoints
- [ ] Test error scenarios and edge cases

### Architecture Alignment
- Follow the authentication flow established in the project
- Use the defined database schema patterns
- Maintain consistency with existing API design
- Apply the project's design system (colors, typography)"""
            
            feature_issue_number = self.task_manager.create_ai_task_issue(
                title=f"Implement: {feature}",
                description=feature_description,
                labels=['feature', 'enhancement'],
                priority="medium",
                task_type="feature",
                repository=repo.full_name  # Pass the target repository
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
            Parsed JSON
        """
        content = response.get('content', '')
        
        if not content:
            raise ValueError("AI response is empty - no content to parse")
        
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in AI response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in AI response: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse AI response: {e}")
    
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
    
    def _sanitize_project_name(self, name: str) -> str:
        """Sanitize project name for GitHub repository naming.
        
        Args:
            name: Raw project name
            
        Returns:
            Sanitized name suitable for GitHub repo
        """
        # Convert to lowercase
        name = name.lower()
        
        # Replace spaces and special characters with hyphens
        import re
        name = re.sub(r'[^a-z0-9-]', '-', name)
        
        # Remove multiple consecutive hyphens
        name = re.sub(r'-+', '-', name)
        
        # Remove leading/trailing hyphens
        name = name.strip('-')
        
        # Ensure it's not empty and within GitHub's limit
        if not name:
            name = 'new-project'
        
        # GitHub repo name limit is 100 characters
        return name[:100]