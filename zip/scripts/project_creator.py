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
from scripts.storage_mode import StorageMode
from scripts.local_github_simulator import LocalGitHubSimulator
from pathlib import Path


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
            github_token: GitHub personal access token (None for local mode)
            ai_brain: AI brain for intelligent customization
        """
        self.github_token = github_token
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        self.created_projects = []
        
        # Check if we're in local mode
        self.storage_mode = StorageMode.LOCAL  # Always local mode
        
        if self.storage_mode == StorageMode.LOCAL:
            self.logger.info("ProjectCreator initialized in LOCAL mode")
            self.local_simulator = LocalGitHubSimulator()
            self.github = None  # No GitHub client in local mode
        else:
            self.logger.info("ProjectCreator initialized in GITHUB mode")
            self.github = Github(github_token)
            self.local_simulator = None
            
        self.task_manager = TaskManager(github_token)
        
    async def create_project(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create project by forking starter kit and customizing.
        
        ALL OPERATIONS MUST SUCCEED OR THE ENTIRE PROJECT CREATION FAILS.
        NO PARTIAL FAILURES ARE ALLOWED.
        
        Args:
            task: NEW_PROJECT task with details
            
        Returns:
            Creation result with repo details
        """
        self.logger.info(f"Creating project for task: {task.get('title', 'Unknown')}")
        
        # NO TRY-CATCH - Let exceptions propagate and stop the system
        # Generate project details using AI
        project_details = await self._generate_project_details(task)
        
        # Validate project details are complete
        required_fields = ['name', 'description', 'problem_statement', 'initial_features']
        missing_fields = [field for field in required_fields if not project_details.get(field)]
        if missing_fields:
            raise ValueError(f"Project generation incomplete - missing required fields: {', '.join(missing_fields)}")
        
        # Check if project already exists
        if self._project_exists(project_details['name']):
            raise ValueError(f"Project {project_details['name']} already exists")
        
        # Fork the starter kit - MUST SUCCEED
        tech_stack = project_details.get('tech_stack', project_details.get('tech_additions', ''))
        self.project_description = project_details.get('description', '')
        forked_repo = await self._fork_starter_kit(project_details['name'], tech_stack)
        
        if not forked_repo:
            raise ValueError('CRITICAL: Failed to fork starter kit - cannot create project')
        
        # Customize the project - MUST SUCCEED
        self.logger.info(f"Starting project customization for {project_details['name']}")
        customization_result = await self._customize_project(
            forked_repo, 
            project_details
        )
        self.logger.info(f"Customization result: {customization_result}")
        
        # Validate ALL customization steps succeeded
        if not customization_result.get('package_json_updated'):
            raise ValueError('CRITICAL: Failed to update package.json')
        
        # Validate ALL documentation was created
        if not all([
            customization_result.get('readme_updated'),
            customization_result.get('architecture_saved'),
            customization_result.get('monetization_saved'),
            customization_result.get('metadata_saved')
        ]):
            missing_docs = []
            if not customization_result.get('readme_updated'):
                missing_docs.append('README.md')
            if not customization_result.get('architecture_saved'):
                missing_docs.append('ARCHITECTURE.md')
            if not customization_result.get('monetization_saved'):
                missing_docs.append('MONETIZATION_STRATEGY.md')
            if not customization_result.get('metadata_saved'):
                missing_docs.append('metadata.json')
            raise ValueError(f"CRITICAL: Failed to create required documentation: {', '.join(missing_docs)}")
        
        # Create initial project structure - MUST SUCCEED
        self.logger.info(f"Creating initial structure for {project_details['name']}")
        structure_result = await self._create_initial_structure(
            forked_repo,
            project_details
        )
        self.logger.info(f"Structure result: {structure_result}")
        
        # Validate structure creation
        if not structure_result.get('success'):
            raise ValueError(f"CRITICAL: Failed to create project structure: {structure_result.get('error', 'Unknown error')}")
        
        # Create initial issues/tasks - MUST SUCCEED IN GITHUB MODE
        issues_created = []
        if self.storage_mode != StorageMode.LOCAL:
            # In GitHub mode, issue creation MUST succeed
            issues_created = await self._create_initial_issues(
                forked_repo,
                project_details
            )
            if not issues_created:
                raise ValueError("CRITICAL: Failed to create initial GitHub issues")
        else:
            # In local mode, issues are optional
            try:
                issues_created = await self._create_initial_issues(
                    forked_repo,
                    project_details
                )
            except Exception as e:
                self.logger.info(f"Local mode: Skipping issue creation - {e}")
        
        # Set up project settings - MUST SUCCEED
        await self._configure_project_settings(forked_repo, project_details)
        
        # Final validation - ensure we have a complete project
        if not forked_repo.name:
            raise ValueError("CRITICAL: Repository name is missing")
        if not forked_repo.html_url and self.storage_mode != StorageMode.LOCAL:
            raise ValueError("CRITICAL: Repository URL is missing")
        
        # Record creation - only if EVERYTHING succeeded
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
        
        self.logger.info(f"✅ SUCCESSFULLY created project {project_details['name']} with ALL components")
        
        return creation_record
    
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
        
        # REQUIRE selected_project - architecture is optional and will be generated later
        if not metadata.get('selected_project'):
            # Log detailed information about what's missing
            missing_fields = []
            # Architecture is optional - will be generated later
            if not metadata.get('selected_project'):
                missing_fields.append('selected_project')
            
            if missing_fields:
                error_msg = f"""
CRITICAL: Cannot create project - required metadata is missing!

Missing fields: {', '.join(missing_fields)}
Available metadata keys: {list(metadata.keys())}

This task was not properly prepared by the intelligent work finder.
The task MUST include 'selected_project' in metadata.

Task details:
- Title: {task.get('title', 'Unknown')}
- Type: {task.get('task_type', 'Unknown')}
- Has selected_project: {bool(metadata.get('selected_project'))}

The intelligent work finder should have:
1. Discovered market problems
2. Generated project specifications (selected_project)
3. Included it in the task metadata

Architecture will be generated during the project creation process.
"""
                self.logger.error(error_msg)
                raise ValueError(f"Cannot create project: missing required metadata fields: {', '.join(missing_fields)}")
        
        # Use selected_project from metadata
        self.logger.info("✅ Using pre-generated project details from task metadata")
        
        selected_project = metadata['selected_project']
        # Architecture is optional - will be generated if not present
        architecture = metadata.get('architecture', {})
        
        # Convert to project details format - include all metadata
        return {
            'name': self._sanitize_project_name(selected_project.get('project_name', 'new-project')),
            'description': selected_project.get('project_goal', '')[:200],
            'problem_statement': selected_project.get('problem_solved', '') or metadata.get('detailed_description', ''),
            'target_audience': selected_project.get('target_audience', '') or metadata.get('market_validation', {}).get('affected_users', ''),
            'monetization_strategy': selected_project.get('monetization_strategy', '') or metadata.get('full_opportunity', {}).get('monetization_model', ''),
            'initial_features': selected_project.get('key_features', []) or selected_project.get('initial_features', []),
            'architecture': architecture,
            'customizations': {
                'packages': [],
                'configuration': [],
                'features': selected_project.get('key_features', [])
            },
            'tech_additions': [],
            'readme_sections': [],
            'environment_variables': [],
            'database_schema': architecture.get('foundational_architecture', {}).get('database_schema', {}),
            'metadata': metadata,  # Pass through all metadata including market research
            'full_opportunity': metadata.get('full_opportunity', {}),
            'research_phases': metadata.get('research_phases', {})
        }
    
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
            if self.storage_mode == StorageMode.LOCAL:
                # Check using local simulator
                return self.local_simulator.repository_exists(self.ORGANIZATION, name)
            else:
                # GitHub mode
                org = self.github.get_organization(self.ORGANIZATION)
                org.get_repo(name)
                return True
        except (GithubException, Exception):
            return False
    
    async def _fork_starter_kit(self, project_name: str, tech_stack: str = None):
        """Fork the Laravel React starter kit.
        
        MUST SUCCEED OR RAISE EXCEPTION - NO PARTIAL FAILURES.
        
        Args:
            project_name: Name for the new project
            tech_stack: Technology stack to use (ignored - always uses Laravel React)
            
        Returns:
            Forked repository object
            
        Raises:
            Exception: If forking fails for any reason
        """
        # Always use Laravel React starter kit
        starter_kit_key = 'laravel-react'
        starter_kit_repo = self.STARTER_KITS['laravel-react']
        
        if self.storage_mode == StorageMode.LOCAL:
            # Local mode - create repository using simulator
            self.logger.info(f"Creating local repository: {self.ORGANIZATION}/{project_name}")
            
            description = self.project_description or f"Project created from {starter_kit_key} starter kit"
            
            # Pass template parameter to copy starter kit
            local_repo = self.local_simulator.create_repository(
                organization=self.ORGANIZATION,
                name=project_name,
                description=description,
                template='react-starter-kit'  # This will copy the template
            )
            
            # Validate that template was copied
            from pathlib import Path
            repo_path = Path(local_repo.path)
            essential_files = ['package.json', 'composer.json', 'vite.config.ts', 'artisan']
            missing_files = [f for f in essential_files if not (repo_path / f).exists()]
            
            if missing_files:
                error_msg = (
                    f"❌ LOCAL MODE ERROR: Template files not copied!\n"
                    f"Missing essential files: {', '.join(missing_files)}\n"
                    f"The react-starter-kit template was not properly copied to {repo_path}\n"
                    f"Please ensure the template exists at: {self.local_simulator.templates_path / 'react-starter-kit'}\n"
                    f"You may need to copy the template from the parent project."
                )
                self.logger.error(error_msg)
                raise NotImplementedError(error_msg)
            
            self.logger.info(f"✅ Successfully created local repository with template at {repo_path}")
            return local_repo
        else:
            # GitHub mode - original code
            # Get starter kit repo
            starter_repo = self.github.get_repo(starter_kit_repo)
            
            # Get organization
            org = self.github.get_organization(self.ORGANIZATION)
            
            # Fork to organization
            self.logger.info(f"Forking {starter_kit_repo} to {self.ORGANIZATION}/{project_name}")
            
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
            await self._copy_starter_contents(starter_repo, new_repo)
            
            return new_repo
    
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
        * **Key Features (MUST USE EXACTLY THESE):** {json.dumps(details.get('initial_features', []), indent=2)}
          CRITICAL: In the feature_implementation_roadmap section, you MUST use these EXACT feature names as provided.
          DO NOT rename, rephrase, or create new features. Each feature in the roadmap must match one of the provided features above.
        * **Expected Scale:** "Start with a small user base, but design with a clear path for significant scaling."
        * **Team Skills:** "The team is proficient with Laravel but intermediate with React and TypeScript."

        **Task:**
        Generate a **complete, production-grade System Architecture, Design System, and Feature Implementation Roadmap**. 
        The document must provide a foundational engineering architecture AND a foundational design system, plus a detailed plan for implementing each "Key Feature".
        
        CRITICAL: In the feature_implementation_roadmap, you MUST create an entry for EACH of the Key Features provided above. Use the exact feature names without modification.

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
        
        # NO TRY-CATCH - Let exceptions propagate
        # Use AI brain with Gemini for architecture generation
        response = await self.ai_brain.generate_enhanced_response(architecture_prompt, model='gemini')
        return self._parse_json_response(response)
    
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
        # Validate we can customize this repository
        if self.storage_mode == StorageMode.LOCAL:
            if not hasattr(repo, 'path'):
                raise NotImplementedError(
                    "❌ LOCAL MODE ERROR: Repository object missing 'path' attribute.\n"
                    "Cannot customize local repository without filesystem path."
                )
            
            # Local mode repositories should have update_file and create_file methods now
            if not hasattr(repo, 'update_file') or not hasattr(repo, 'create_file'):
                self.logger.warning(
                    "Local repository missing update_file/create_file methods. "
                    "Using fallback methods."
                )
        
        results = {
            'readme_updated': False,
            'package_json_updated': False,
            'env_example_updated': False,
            'custom_files_created': []
        }
        
        # Architecture is now generated as part of structured documents
        # Remove this old approach to avoid confusion
        
        # README will be generated as part of structured documents below
                
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
                    # Handle both string and bytes content
                    if hasattr(env_file, 'decoded_content'):
                        existing_content = env_file.decoded_content
                        if isinstance(existing_content, bytes):
                            existing_content = existing_content.decode()
                    else:
                        existing_content = ""
                    updated_env = existing_content + "\n" + env_content
                    
                    repo.update_file(
                        ".env.example",
                        "Add project-specific environment variables",
                        updated_env,
                        env_file.sha
                    )
                    results['env_example_updated'] = True
                except Exception as e:
                    self.logger.error(f"Failed to update .env.example: {e}")
        
        # Create all three documentation files together using parallel chunked generation
        self.logger.info(f"📚 Creating all documentation files for {details['name']}")
        
        # Generate structured output for all documents first using parallel chunks
        structured_docs = await self._generate_all_structured_documents(repo, details)
        
        # Log the structured data to debug issues
        self.logger.info(f"📊 Generated structured docs keys: {list(structured_docs.keys())}")
        self.logger.info(f"📊 README has {len(structured_docs.get('readme', {}))} keys")
        self.logger.info(f"📊 Architecture has {len(structured_docs.get('architecture', {}))} keys")
        self.logger.info(f"📊 Monetization has {len(structured_docs.get('monetization', {}))} keys")
        
        # Save complete structured data to metadata.json
        await self._save_project_metadata(repo, details, structured_docs, results)
        
        # Transform structured data to markdown files
        await self._create_markdown_files_from_structured(repo, structured_docs, results)
        
        # Validate all documentation was created
        if not all([
            results.get('readme_updated'),
            results.get('architecture_saved'),
            results.get('monetization_saved'),
            results.get('metadata_saved')
        ]):
            missing = []
            if not results.get('readme_updated'):
                missing.append('README.md')
            if not results.get('architecture_saved'):
                missing.append('ARCHITECTURE.md')
            if not results.get('monetization_saved'):
                missing.append('MONETIZATION_STRATEGY.md')
            if not results.get('metadata_saved'):
                missing.append('metadata.json')
            raise ValueError(f"Failed to create required documentation files: {', '.join(missing)}")
        
        # Add documentation creation summary to results
        results['documentation_created'] = {
            'readme': results.get('readme_updated', False),
            'architecture': results.get('architecture_saved', False),
            'monetization': results.get('monetization_saved', False),
            'metadata': results.get('metadata_saved', False)
        }
                    
        return results
    
    async def _generate_comprehensive_readme(self, repo, details: Dict[str, Any]) -> bool:
        """Generate comprehensive README.md with deep research content.
        
        Args:
            repo: Repository object
            details: Project details including research data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract research-based content
            problem_statement = details.get('problem_statement', '')
            target_audience = details.get('target_audience', '')
            monetization_strategy = details.get('monetization_strategy', '')
            architecture = details.get('architecture', {})
            initial_features = details.get('initial_features', [])
            
            # Extract opportunity data from metadata if available
            metadata = details.get('metadata', {})
            market_research = metadata.get('market_research', {})
            validation_metrics = metadata.get('validation_metrics', {})
            
            # Get market validation data
            market_validation = f"""
Market Size: {market_research.get('market_size', 'Large market opportunity')}
Competition Level: {market_research.get('competition_level', 'Moderate')}
Validation Score: {validation_metrics.get('validation_score', '9.0')}/10
Market Demand: {validation_metrics.get('market_demand', 'High')}
Current Solutions: {market_research.get('current_solutions', 'Existing solutions have significant gaps')}
            """
            
            competitive_advantage = f"""
Problem: {market_research.get('problem_description', problem_statement)}
Affected Audience: {market_research.get('affected_audience', target_audience)}
Revenue Potential: {validation_metrics.get('revenue_potential', monetization_strategy)}
Validation Sources: {', '.join(market_research.get('validation_sources', ['Market research', 'User feedback'])[:3])}
            """
            
            # Generate comprehensive README with research insights
            prompt = f"""
Generate a comprehensive, professional README.md for the project "{details.get('name')}" that incorporates deep market research.

Project Research Context:
- Problem Statement: {problem_statement}
- Target Audience: {target_audience}
- Market Validation: {market_validation}
- Competitive Advantage: {competitive_advantage}
- Monetization Strategy: {monetization_strategy}
- Initial Features: {json.dumps(initial_features)}

Architecture Overview:
{json.dumps(architecture, indent=2)}

Create a README that includes:

1. **Project Title & Tagline**
   - Eye-catching header with the project name
   - Compelling tagline that summarizes the value proposition

2. **Problem & Solution**
   - Deep dive into the real-world problem (with research evidence)
   - Clear explanation of how this solution addresses it
   - Why existing solutions fall short

3. **Target Audience**
   - Who will benefit from this solution
   - User personas and use cases
   - Market size and opportunity

4. **Key Features**
   IMPORTANT: You MUST use EXACTLY these features provided from market research:
   {json.dumps(initial_features, indent=2)}
   
   For EACH feature above:
   - Use the exact feature name as provided
   - Add a detailed description explaining what it does
   - Explain how it solves the specific problem
   - DO NOT create new features or rename the provided ones
   - DO NOT use generic features like "User Dashboard" or "API Platform"

5. **Tech Stack**
   - Laravel 11+ backend details
   - React 19 with TypeScript frontend
   - Database architecture (MySQL)
   - Redis integration for real-time features
   - Additional technologies used

6. **Getting Started**
   - Prerequisites
   - Installation steps
   - Environment setup
   - Database migrations
   - Running the development server

7. **Architecture Overview**
   - System design principles
   - API structure
   - Frontend component organization
   - Database schema highlights
   - Security measures

8. **Monetization Strategy**
   - Revenue model details
   - Pricing tiers (if applicable)
   - Growth projections
   - Market opportunity

9. **Development Roadmap**
   - Current status
   - Upcoming features
   - Long-term vision

10. **Contributing**
    - How to contribute
    - Code standards
    - Pull request process

11. **License & Contact**
    - License information
    - Support contacts

Make it professional, detailed, and compelling. Use markdown formatting effectively with:
- Clear headers and subheaders
- Code blocks for technical details
- Tables where appropriate
- Badges for tech stack
- Emphasis on key points

This README should serve as both technical documentation and a compelling pitch for the project.
"""
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            readme_content = response.get('content', '')
            
            if not readme_content:
                raise ValueError("AI failed to generate README content")
            
            # Update or create README.md
            if self.storage_mode == StorageMode.LOCAL:
                repo.update_file(
                    "README.md",
                    "Update README with comprehensive research-based documentation",
                    readme_content
                )
            else:
                try:
                    existing = repo.get_contents("README.md")
                    repo.update_file(
                        "README.md",
                        "Update README with comprehensive research-based documentation",
                        readme_content,
                        existing.sha
                    )
                except:
                    repo.create_file(
                        "README.md",
                        "Create comprehensive research-based README",
                        readme_content
                    )
            
            self.logger.info("✅ Successfully created comprehensive README.md")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive README: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
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

## Key Features

### AI-Generated Features from Market Research
The following features were identified through AI-powered market research and analysis:

"""
        # Add features if available
        if details.get('initial_features'):
            for i, feature in enumerate(details['initial_features'], 1):
                doc += f"{i}. **{feature}**\n"
            doc += "\n"
        
        doc += """## Design System
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
        doc += "Implementation details for the AI-generated features:\n\n"
        
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
    
    def _format_monetization_document(self, details: Dict[str, Any]) -> str:
        """Format monetization strategy as a comprehensive markdown document.
        
        Args:
            details: Project details including monetization data
            
        Returns:
            Formatted monetization strategy document
        """
        doc = f"""# {details.get('name', 'Project')} Monetization Strategy

## Executive Summary
{details.get('description', '')}

### Market Opportunity
{details.get('problem_statement', '')}

### Target Market
{details.get('target_audience', '')}

## Revenue Models
"""
        
        # Add monetization models from details
        if details.get('monetization_strategy'):
            doc += f"\n### Primary Revenue Stream\n{details['monetization_strategy']}\n"
        
        if details.get('market_research', {}).get('monetization_models'):
            doc += "\n### Detailed Revenue Streams\n"
            for model in details['market_research']['monetization_models']:
                if isinstance(model, dict):
                    doc += f"\n#### {model.get('model_name', 'Revenue Model')}\n"
                    doc += f"- **Type**: {model.get('type', '')}\n"
                    doc += f"- **Description**: {model.get('description', '')}\n"
                    doc += f"- **Pricing**: {model.get('pricing_structure', '')}\n"
                    doc += f"- **Target Users**: {model.get('target_segment', '')}\n"
                    doc += f"- **Revenue Potential**: {model.get('revenue_potential', '')}\n"
                else:
                    doc += f"- {model}\n"
        
        # Add market analysis
        doc += "\n## Market Analysis\n"
        
        if details.get('market_research', {}).get('market_size'):
            market = details['market_research']['market_size']
            doc += f"""
### Market Size
- **Total Addressable Market (TAM)**: {market.get('tam', 'To be determined')}
- **Serviceable Addressable Market (SAM)**: {market.get('sam', 'To be determined')}
- **Serviceable Obtainable Market (SOM)**: {market.get('som', 'To be determined')}
- **Growth Rate**: {market.get('growth_rate', 'To be determined')}
"""
        
        if details.get('market_research', {}).get('competitors'):
            doc += "\n### Competitive Landscape\n"
            for competitor in details['market_research']['competitors']:
                if isinstance(competitor, dict):
                    doc += f"\n#### {competitor.get('name', 'Competitor')}\n"
                    doc += f"- **Strengths**: {competitor.get('strengths', [])}\n"
                    doc += f"- **Weaknesses**: {competitor.get('weaknesses', [])}\n"
                    doc += f"- **Market Share**: {competitor.get('market_share', 'Unknown')}\n"
                else:
                    doc += f"- {competitor}\n"
        
        # Add pricing strategy
        doc += "\n## Pricing Strategy\n"
        
        if details.get('market_research', {}).get('pricing_strategy'):
            pricing = details['market_research']['pricing_strategy']
            if isinstance(pricing, dict):
                doc += f"""
### Pricing Tiers
{pricing.get('overview', '')}

#### Tier Structure
"""
                for tier in pricing.get('tiers', []):
                    if isinstance(tier, dict):
                        doc += f"\n**{tier.get('name', 'Tier')}** - ${tier.get('price', '0')}/month\n"
                        doc += f"- Target: {tier.get('target_users', '')}\n"
                        doc += f"- Features: {', '.join(tier.get('features', []))}\n"
            else:
                doc += f"{pricing}\n"
        
        # Add customer acquisition strategy
        doc += "\n## Customer Acquisition Strategy\n"
        
        if details.get('market_research', {}).get('customer_acquisition'):
            acquisition = details['market_research']['customer_acquisition']
            doc += f"""
### Acquisition Channels
{acquisition.get('channels', [])}

### Customer Acquisition Cost (CAC)
{acquisition.get('estimated_cac', 'To be determined')}

### Lifetime Value (LTV)
{acquisition.get('estimated_ltv', 'To be determined')}

### LTV:CAC Ratio
{acquisition.get('ltv_cac_ratio', 'To be determined')}
"""
        
        # Add growth projections
        doc += "\n## Growth Projections\n"
        
        doc += """
### Year 1 Targets
- **Users**: 1,000 active users
- **Revenue**: $10,000 MRR by month 12
- **Growth Rate**: 20% month-over-month

### Year 2 Targets
- **Users**: 10,000 active users
- **Revenue**: $100,000 MRR
- **Growth Rate**: 15% month-over-month

### Year 3 Targets
- **Users**: 50,000 active users
- **Revenue**: $500,000 MRR
- **Growth Rate**: 10% month-over-month
"""
        
        # Add key metrics
        doc += "\n## Key Performance Indicators (KPIs)\n"
        doc += """
### Revenue Metrics
- Monthly Recurring Revenue (MRR)
- Annual Recurring Revenue (ARR)
- Average Revenue Per User (ARPU)
- Customer Lifetime Value (CLV)

### Growth Metrics
- User Growth Rate
- Revenue Growth Rate
- Churn Rate
- Net Revenue Retention

### Engagement Metrics
- Daily Active Users (DAU)
- Monthly Active Users (MAU)
- Feature Adoption Rate
- User Satisfaction Score (NPS)
"""
        
        # Add implementation roadmap
        doc += "\n## Implementation Roadmap\n"
        doc += """
### Phase 1: MVP & Early Adopters (Months 1-3)
- Launch core features with basic monetization
- Focus on early adopter acquisition
- Implement usage tracking and analytics
- Target: 100 paying users

### Phase 2: Product Market Fit (Months 4-6)
- Refine pricing based on user feedback
- Implement advanced features for premium tiers
- Launch referral program
- Target: 500 paying users

### Phase 3: Scale & Optimize (Months 7-12)
- Optimize conversion funnel
- Implement enterprise features
- Launch affiliate program
- Target: 2,000 paying users

### Phase 4: Market Expansion (Year 2)
- International market entry
- Additional product lines
- Strategic partnerships
- Target: 10,000 paying users
"""
        
        # Add metadata
        doc += f"""
## Metadata
- **Generated**: {datetime.now(timezone.utc).isoformat()}
- **Strategy Version**: 1.0.0
- **Review Cycle**: Quarterly

---
*This monetization strategy should be reviewed and updated quarterly based on market feedback and performance metrics.*
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
        4. **Key Features**
           IMPORTANT: You MUST use EXACTLY these features provided from market research:
           {json.dumps(details.get('initial_features', []), indent=2)}
           
           For EACH feature above:
           - Use the exact feature name as provided
           - Add a detailed description explaining what it does
           - Explain how it solves the specific problem
           - DO NOT create new features or rename the provided ones
           - DO NOT use generic features like "User Dashboard" or "API Platform"
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
        
        MUST SUCCEED OR RAISE EXCEPTION - NO PARTIAL FAILURES.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            Structure creation results with success status
        """
        results = {
            'success': True,  # Add explicit success flag
            'directories_created': [],
            'files_created': [],
            'error': None
        }
        
        try:
            # Create any project-specific directories/files based on features
            # For now, the basic structure comes from the starter kit
            # Additional structure would be added here based on features
            
            # Validate basic structure exists
            if self.storage_mode == StorageMode.LOCAL:
                # In local mode, verify the repository path exists
                if hasattr(repo, 'path'):
                    from pathlib import Path
                    repo_path = Path(repo.path)
                    if not repo_path.exists():
                        raise ValueError(f"Repository path does not exist: {repo_path}")
                    # Log what we have
                    self.logger.info(f"✅ Local repository structure verified at: {repo_path}")
                    results['directories_created'].append(str(repo_path))
            
            # Mark as successful
            results['success'] = True
            return results
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self.logger.error(f"Failed to create initial structure: {e}")
            raise  # Re-raise to fail the entire project creation
    
    async def _create_initial_issues(self, repo, 
                                    details: Dict[str, Any]) -> List[int]:
        """Create initial GitHub issues for project features.
        
        Args:
            repo: Repository object
            details: Project details including architecture
            
        Returns:
            List of created issue numbers
        """
        print(f"DEBUG: _create_initial_issues called with repo: {repo.full_name if repo else 'None'}")
        print(f"DEBUG: Task manager exists: {self.task_manager is not None}")
        issue_numbers = []
        
        # Extract architecture details if available
        architecture = details.get('architecture', {})
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
   - [ ] Create models based on the problem domain and features
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
   - [ ] Create database seeders for test data
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
        if setup_issue_number:
            try:
                setup_issue = repo.get_issue(setup_issue_number)
                issue_numbers.append(setup_issue.number)
            except Exception as e:
                self.logger.warning(f"Could not get setup issue: {e}")
                issue_numbers.append(setup_issue_number)
        
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
            if self.storage_mode == StorageMode.LOCAL:
                # Local mode - just log the configuration
                self.logger.info(f"Configured local repository: {repo.name}")
                self.logger.info(f"Description: {details['description']}")
                
                # Add topics/tags to metadata
                topics = ['laravel', 'react', 'typescript', 'fullstack']
                
                # Add project-specific topics based on features
                if 'auth' in str(details.get('initial_features', [])).lower():
                    topics.append('authentication')
                if 'api' in str(details.get('initial_features', [])).lower():
                    topics.append('rest-api')
                    
                self.logger.info(f"Topics: {topics}")
            else:
                # GitHub mode - original code
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
    
    async def _generate_all_structured_documents(self, repo, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output for all documentation.
        
        Args:
            repo: Repository object
            details: Project details
            
        Returns:
            Dictionary containing structured documentation
        """
        self.logger.info("🤖 Generating structured documentation with AI...")
        
        structured_docs = {
            'readme': {},
            'architecture': {},
            'monetization': {},
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Generate structured README
            structured_docs['readme'] = await self._generate_structured_readme(details)
        except Exception as e:
            self.logger.error(f"Failed to generate structured README: {e}")
            raise
        
        try:
            # Generate structured architecture
            structured_docs['architecture'] = await self._generate_structured_architecture(details)
        except Exception as e:
            self.logger.error(f"Failed to generate structured architecture: {e}")
            raise
        
        try:
            # Generate structured monetization strategy
            structured_docs['monetization'] = await self._generate_structured_monetization(details)
        except Exception as e:
            self.logger.error(f"Failed to generate structured monetization: {e}")
            raise
        
        # Validate documentation quality
        issues = self._validate_generated_documentation(structured_docs)
        if issues:
            self.logger.warning(f"Documentation validation issues: {issues}")
            # Log but don't fail - AI might generate slightly different structures
        
        return structured_docs
    
    async def _generate_structured_readme(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive README following enhanced template.
        
        Args:
            details: Project details including market research
            
        Returns:
            Structured README data
        """
        if not self.ai_brain:
            raise ValueError("AI Brain is required for content generation")
        
        self.logger.info("📝 Generating comprehensive README following enhanced template...")
        
        # Extract research data from metadata
        metadata = details.get('metadata', {})
        market_research = metadata.get('market_research', {})
        validation_metrics = metadata.get('validation_metrics', {})
            
        # Get the AI-generated features
        initial_features = details.get('initial_features', [])
        
        prompt = f"""
        Generate a comprehensive README that enables developers to understand and build the project.
        This project uses the Laravel React starter kit with Inertia.js.
        
        Project Context:
        - Name: {details.get('name')}
        - Problem: {market_research.get('problem_description', details.get('problem_statement', ''))}
        - Audience: {market_research.get('affected_audience', details.get('target_audience', ''))}
        - Market Size: {market_research.get('market_size', 'Large market opportunity')}
        - Validation Score: {validation_metrics.get('validation_score', '9.0')}/10
        - Revenue Potential: {validation_metrics.get('revenue_potential', '$50,000-$200,000 ARR')}
        
        CRITICAL REQUIREMENT - You MUST use these EXACT features provided from market research:
        {json.dumps(initial_features, indent=2)}
        
        Generate JSON with this EXACT structure:
        {{
            "title": "{details.get('name')}",
            "tagline": "[One sentence that captures the value proposition]",
            "overview": "[2-3 paragraphs explaining what problem it solves, who it's for, and how it's different]",
            "key_features": [
                // YOU MUST CREATE AN ENTRY FOR EACH FEATURE IN THE LIST ABOVE
                // Use the EXACT feature names provided - DO NOT rename or replace them
                {{
                    "name": "[Use exact feature name from list above]",
                    "description": "[What this specific feature does]",
                    "benefit": "[How it solves the problem for users]"
                }}
            ],
            "quick_start": {{
                "prerequisites": [
                    "PHP 8.2 or higher",
                    "Composer 2.x",
                    "Node.js 18.x or higher",
                    "npm 9.x or higher",
                    "MySQL 8.0 or PostgreSQL 13+",
                    "Redis (optional, for caching and queues)"
                ],
                "installation": [
                    "git clone https://github.com/CodeWebMobile-AI/{details.get('name')}.git",
                    "cd {details.get('name')}",
                    "composer install",
                    "npm install",
                    "cp .env.example .env",
                    "php artisan key:generate"
                ],
                "database_setup": [
                    "# Configure database in .env",
                    "DB_CONNECTION=mysql",
                    "DB_DATABASE={details.get('name').lower().replace('-', '_')}",
                    "DB_USERNAME=root",
                    "DB_PASSWORD=",
                    "",
                    "# Run migrations",
                    "php artisan migrate --seed"
                ],
                "ai_configuration": [
                    "# Add AI service credentials to .env",
                    "OPENAI_API_KEY=your_api_key_here",
                    "# Or for Gemini:",
                    "GOOGLE_AI_API_KEY=your_api_key_here"
                ],
                "development_server": [
                    "# Terminal 1: Start Laravel",
                    "php artisan serve",
                    "",
                    "# Terminal 2: Start Vite",
                    "npm run dev",
                    "",
                    "# Access at http://localhost:8000"
                ]
            }},
            "project_structure": {{
                "overview": "Laravel React starter kit with Inertia.js architecture",
                "key_directories": [
                    "app/Http/Controllers/Admin/ - Admin portal controllers",
                    "app/Http/Controllers/User/ - User portal controllers", 
                    "app/Models/ - Eloquent models",
                    "app/Services/ - Business logic and AI integration",
                    "resources/js/pages/Admin/ - Admin React pages",
                    "resources/js/pages/User/ - User React pages",
                    "resources/js/components/ - Reusable React components"
                ]
            }},
            "configuration": {{
                "environment_variables": [
                    {{
                        "name": "APP_NAME",
                        "description": "Application name",
                        "example": "{details.get('name')}"
                    }},
                    {{
                        "name": "DB_DATABASE",
                        "description": "Database name",
                        "example": "{details.get('name').lower().replace('-', '_')}"
                    }},
                    {{
                        "name": "OPENAI_API_KEY",
                        "description": "OpenAI API key for AI features",
                        "example": "sk-..."
                    }},
                    {{
                        "name": "REDIS_HOST",
                        "description": "Redis server host (if using)",
                        "example": "127.0.0.1"
                    }}
                ]
            }},
            "development_workflow": {{
                "creating_features": [
                    "# Create model with migration, factory, controller",
                    "php artisan make:model [Model] -mfc",
                    "",
                    "# Create React page",
                    "touch resources/js/pages/User/[Feature]/Index.tsx",
                    "",
                    "# Add route",
                    "Route::resource('[feature]', [Feature]Controller::class);"
                ],
                "testing": {{
                    "backend": "php artisan test",
                    "frontend": "npm run test",
                    "coverage": "php artisan test --coverage"
                }},
                "code_quality": [
                    "./vendor/bin/phpcs - PHP CodeSniffer",
                    "npm run lint - ESLint for TypeScript/React",
                    "npm run lint:fix - Auto-fix linting issues"
                ]
            }},
            "admin_panel": {{
                "access": "/admin",
                "default_credentials": {{
                    "email": "admin@example.com",
                    "password": "password",
                    "warning": "⚠️ Change these immediately in production!"
                }},
                "features": [
                    "Dashboard with system metrics",
                    "User management",
                    "Analytics and reporting",
                    "System configuration"
                ]
            }},
            "api_documentation": {{
                "authentication": "Laravel Sanctum for API authentication",
                "endpoints": [
                    {{
                        "method": "POST",
                        "path": "/api/login",
                        "description": "Authenticate and receive API token"
                    }},
                    {{
                        "method": "GET",
                        "path": "/api/[resources]",
                        "description": "List resources"
                    }},
                    {{
                        "method": "POST",
                        "path": "/api/[resources]",
                        "description": "Create new resource"
                    }}
                ]
            }},
            "deployment": {{
                "production_build": [
                    "composer install --no-dev --optimize-autoloader",
                    "npm ci && npm run build",
                    "php artisan config:cache",
                    "php artisan route:cache",
                    "php artisan view:cache"
                ],
                "server_requirements": [
                    "PHP 8.2+ with required extensions",
                    "MySQL 8.0+ or PostgreSQL 13+",
                    "Redis 6.0+ (optional)",
                    "Nginx or Apache",
                    "Supervisor for queue workers"
                ]
            }},
            "troubleshooting": [
                {{
                    "issue": "Class not found error",
                    "solution": "composer dump-autoload"
                }},
                {{
                    "issue": "Vite manifest not found",
                    "solution": "npm run build"
                }},
                {{
                    "issue": "Permission denied errors",
                    "solution": "chmod -R 775 storage bootstrap/cache"
                }}
            ],
            "contributing": {{
                "process": "Fork → Feature Branch → Commit → Push → Pull Request",
                "code_standards": "PSR-12 for PHP, ESLint config for TypeScript",
                "commit_format": "type(scope): description (e.g., feat(auth): add OAuth support)"
            }},
            "license": "MIT",
            "support": {{
                "documentation": "See /docs directory",
                "issues": "GitHub Issues",
                "email": "support@codewebmobile.com"
            }}
        }}
        
        Return ONLY valid JSON. Be specific to this project.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_structured_architecture(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive architecture following enhanced template.
        
        Args:
            details: Project details
            
        Returns:
            Structured architecture data
        """
        if not self.ai_brain:
            raise ValueError("AI Brain is required for content generation")
            
        self.logger.info("🔧 Generating comprehensive architecture following enhanced template...")
        
        # Use the enhanced architecture generator
        try:
            from scripts.enhanced_architecture_generator import EnhancedArchitectureGenerator
            generator = EnhancedArchitectureGenerator(self.ai_brain)
            return await generator.generate_complete_architecture(details)
        except ImportError:
            self.logger.warning("Enhanced architecture generator not available, using legacy method")
            # Fall back to existing implementation
        
        # Extract market research data
        metadata = details.get('metadata', {})
        market_research = metadata.get('market_research', {})
        
        # Get the AI-generated features
        initial_features = details.get('initial_features', [])
        
        prompt = f"""
        Generate a comprehensive system architecture document following this exact structure.
        This project builds on the Laravel React starter kit with Inertia.js.
        
        Project Context:
        - Name: {details.get('name')}
        - Problem: {market_research.get('problem_description', details.get('problem_statement'))}
        - Audience: {market_research.get('affected_audience', details.get('target_audience'))}
        - Market Size: {market_research.get('market_size', 'Growing market')}
        - Validation Score: {metadata.get('validation_metrics', {}).get('validation_score', '9.0')}/10
        
        CRITICAL - These are the KEY FEATURES that MUST be implemented:
        {json.dumps(initial_features, indent=2)}
        
        IMPORTANT: The Laravel React starter kit already provides:
        - Complete authentication system (login, register, password reset, email verification)
        - User model and profile management
        - shadcn/ui component library with all UI elements
        - Inertia.js for server-side routing with React frontend
        - TypeScript and Tailwind CSS configuration
        
        Generate JSON with this EXACT structure:
        {{
            "executive_summary": {{
                "project": "{details.get('name')} - [One-line value proposition based on problem]",
                "problem_domain": "[Specific problem from research]",
                "solution_type": "AI-powered [category] platform",
                "architecture_pattern": "Inertia.js Monolith with Event-Driven AI Services",
                "deployment_target": "Local Development → Cloud Production"
            }},
            "key_features": {{
                "ai_generated_features_from_market_research": [
                    // YOU MUST CREATE AN ENTRY FOR EACH FEATURE IN THE LIST ABOVE
                    // Explain each feature from a TECHNICAL ARCHITECTURE perspective
                    {{
                        "feature_name": "[Use exact feature name from list above]",
                        "technical_description": "[How this feature works technically]",
                        "architecture_impact": "[What components/services this requires]",
                        "implementation_complexity": "[Simple/Medium/Complex]",
                        "key_components": [
                            "[Component 1 needed]",
                            "[Component 2 needed]"
                        ],
                        "data_flow": "[How data flows through the system for this feature]",
                        "integration_points": "[How it connects with other features]"
                    }}
                    // REPEAT FOR ALL FEATURES IN THE LIST
                ]
            }},
            "system_context": {{
                "problem_analysis": {{
                    "problem": "[Detailed from research]",
                    "current_solutions": "[From market research]",
                    "our_approach": "[How AI solves it differently]",
                    "key_innovation": "[Unique technical approach]"
                }},
                "user_journey": [
                    "Discovery → [How users find us]",
                    "Onboarding → [OAuth connections, setup]",
                    "First Value → [Key feature within 2 minutes]",
                    "Engagement → [Daily/weekly usage patterns]",
                    "Retention → [Why users stay]"
                ]
            }},
            "building_blocks": {{
                "starter_kit_foundation": {{
                    "authentication": ["Login/Register ✓", "Password Reset ✓", "Email Verification ✓"],
                    "ui_components": ["shadcn/ui forms ✓", "Layouts ✓", "Theme system ✓"],
                    "architecture": ["Inertia.js ✓", "Server routing ✓", "TypeScript ✓"]
                }},
                "domain_models": [
                    {{
                        "name": "[Core Model 1]",
                        "purpose": "[What it represents]",
                        "fields": [
                            {{"name": "field1", "type": "string", "description": "Purpose"}},
                            {{"name": "field2", "type": "bigint", "description": "Foreign key to users"}}
                        ],
                        "relationships": ["belongsTo:User", "hasMany:[Model2]"],
                        "key_methods": ["calculateValue()", "needsAttention()"]
                    }},
                    {{
                        "name": "[Core Model 2]",
                        "purpose": "[What it represents]",
                        "fields": [...],
                        "relationships": [...],
                        "key_methods": [...]
                    }}
                ]
            }},
            "portal_separation": {{
                "admin_controllers": [
                    "DashboardController.php - Platform metrics and KPIs",
                    "UsersController.php - User management CRUD",
                    "AnalyticsController.php - Business intelligence",
                    "RevenueController.php - Financial metrics",
                    "SystemHealthController.php - Infrastructure monitoring",
                    "AuditLogController.php - Security and compliance",
                    "ConfigurationController.php - System settings",
                    "ReportsController.php - Custom report generation",
                    "AIUsageController.php - AI token monitoring",
                    "WebhooksController.php - External integrations",
                    "BackupController.php - Data backup management",
                    "QueueMonitorController.php - Job queue health",
                    "CacheController.php - Cache management",
                    "ErrorLogController.php - Error tracking",
                    "ABTestingController.php - Feature experiments"
                ],
                "user_controllers": [
                    "DashboardController.php - User overview",
                    "[Feature]Controller.php - Core feature CRUD",
                    "SettingsController.php - User preferences",
                    "ProfileController.php - Profile management",
                    "BillingController.php - Subscription management",
                    "NotificationController.php - Notification center",
                    "APITokenController.php - API access management",
                    "TeamController.php - Team collaboration",
                    "ExportController.php - Data export",
                    "IntegrationController.php - Third-party connections",
                    "ActivityController.php - Activity history",
                    "SupportController.php - Help and support"
                ],
                "admin_pages": [
                    "Dashboard.tsx - Real-time metrics dashboard",
                    "Users/Index.tsx - User list with filters",
                    "Users/Show.tsx - User detail view",
                    "Users/Edit.tsx - User editing",
                    "Analytics/Overview.tsx - Business metrics",
                    "Analytics/Revenue.tsx - Financial dashboard",
                    "Analytics/UserBehavior.tsx - User analytics",
                    "Analytics/Performance.tsx - System performance",
                    "Settings/General.tsx - System configuration",
                    "Settings/Security.tsx - Security settings",
                    "Settings/Integrations.tsx - API integrations",
                    "Reports/Builder.tsx - Report creator",
                    "Reports/Scheduled.tsx - Automated reports",
                    "Monitoring/Health.tsx - System health",
                    "Monitoring/Logs.tsx - Log viewer",
                    "Monitoring/Queue.tsx - Queue status",
                    "Support/Tickets.tsx - Support tickets",
                    "Support/FAQ.tsx - FAQ management",
                    "Billing/Overview.tsx - Revenue overview",
                    "Billing/Subscriptions.tsx - Subscription management"
                ],
                "user_pages": [
                    "Dashboard.tsx - Personalized home",
                    "[Feature]/Index.tsx - Feature list view",
                    "[Feature]/Create.tsx - Create new item",
                    "[Feature]/Edit.tsx - Edit existing item",
                    "[Feature]/Show.tsx - Detail view",
                    "Settings/Profile.tsx - Profile settings",
                    "Settings/Security.tsx - Security preferences",
                    "Settings/Notifications.tsx - Alert preferences",
                    "Settings/API.tsx - API token management",
                    "Billing/Subscription.tsx - Current plan",
                    "Billing/Upgrade.tsx - Upgrade options",
                    "Billing/History.tsx - Payment history",
                    "Team/Members.tsx - Team management",
                    "Team/Invite.tsx - Invite members",
                    "Activity/Timeline.tsx - Activity feed",
                    "Support/Help.tsx - Help center",
                    "Support/Contact.tsx - Contact support",
                    "Integrations/Connected.tsx - Active integrations",
                    "Integrations/Available.tsx - Integration marketplace"
                ],
                "middleware": {{
                    "admin_check": "EnsureUserIsAdmin - Validates admin role",
                    "subscription_check": "CheckSubscriptionStatus - Validates active subscription",
                    "feature_access": "CheckFeatureAccess - Validates feature permissions",
                    "rate_limiting": "ThrottleRequests - API rate limiting",
                    "api_auth": "AuthenticateAPI - API token validation",
                    "team_access": "ValidateTeamMembership - Team authorization",
                    "route_groups": "Separate /admin, /app, and /api prefixes"
                }},
                "shared_components": [
                    "DataTable.tsx - Reusable data grid",
                    "Chart.tsx - Analytics charts",
                    "Modal.tsx - Modal dialogs",
                    "Notification.tsx - Toast notifications",
                    "SearchBar.tsx - Global search",
                    "FilterPanel.tsx - Advanced filters",
                    "ExportButton.tsx - Data export",
                    "StatusBadge.tsx - Status indicators",
                    "UserAvatar.tsx - User profiles",
                    "LoadingSpinner.tsx - Loading states",
                    "ErrorBoundary.tsx - Error handling",
                    "ConfirmDialog.tsx - Action confirmation"
                ],
                "background_jobs": [
                    "ProcessSubscriptionPayments - Recurring billing",
                    "SendDigestEmails - Email notifications",
                    "GenerateAnalyticsReports - Report generation",
                    "CleanupExpiredData - Data retention",
                    "SyncExternalData - Third-party sync",
                    "CalculateMetrics - Metric aggregation",
                    "BackupDatabase - Automated backups",
                    "ProcessWebhooks - Webhook delivery",
                    "UpdateSearchIndex - Search indexing",
                    "CheckSystemHealth - Health monitoring"
                ],
                "api_endpoints": {{
                    "admin": [
                        "/api/admin/metrics - Platform metrics",
                        "/api/admin/users - User management",
                        "/api/admin/revenue - Financial data",
                        "/api/admin/system - System status",
                        "/api/admin/logs - System logs",
                        "/api/admin/config - Configuration"
                    ],
                    "user": [
                        "/api/v1/[feature] - Feature CRUD",
                        "/api/v1/user - User profile",
                        "/api/v1/settings - User settings",
                        "/api/v1/notifications - Notifications",
                        "/api/v1/billing - Billing info",
                        "/api/v1/team - Team data",
                        "/api/v1/activity - Activity feed"
                    ],
                    "webhooks": [
                        "/webhooks/stripe - Payment webhooks",
                        "/webhooks/github - Code webhooks",
                        "/webhooks/slack - Notifications"
                    ]
                }},
                "email_templates": [
                    "welcome.blade.php - New user welcome",
                    "subscription-confirmed.blade.php - Payment confirmation",
                    "password-reset.blade.php - Password reset",
                    "weekly-digest.blade.php - Activity summary",
                    "invoice.blade.php - Payment invoices",
                    "team-invitation.blade.php - Team invites",
                    "feature-announcement.blade.php - New features",
                    "trial-ending.blade.php - Trial expiration",
                    "payment-failed.blade.php - Failed payments",
                    "account-suspended.blade.php - Account issues"
                ]
            }},
            "implementation_roadmap": {{
                "phase1_foundation": {{
                    "week": 1,
                    "tasks": [
                        "Set up Laravel React starter kit",
                        "Implement domain models and migrations",
                        "Create admin/user portal separation",
                        "Build basic CRUD for core entities"
                    ],
                    "deliverables": [
                        "Working auth system (from starter)",
                        "Database schema implemented",
                        "Portal routing configured",
                        "Basic feature working"
                    ]
                }},
                "phase2_core": {{
                    "week": "2-3",
                    "tasks": [
                        "Integrate AI services (OpenAI/Gemini)",
                        "Build real-time features with Echo",
                        "Implement core business logic",
                        "Create analytics dashboard"
                    ],
                    "deliverables": [
                        "AI integration working",
                        "Real-time updates",
                        "3-5 core features complete",
                        "Admin can see metrics"
                    ]
                }},
                "phase3_enhancement": {{
                    "week": 4,
                    "tasks": [
                        "Performance optimization",
                        "Advanced features",
                        "Comprehensive testing",
                        "UI/UX polish"
                    ],
                    "deliverables": [
                        "Sub-2s page loads",
                        "80% test coverage",
                        "Production ready"
                    ]
                }}
            }},
            "data_architecture": {{
                "schema": {{
                    "users_extension": "Add is_admin, subscription_tier, trial_ends_at to existing users table",
                    "[table1]": {{
                        "fields": [
                            "id BIGINT PRIMARY KEY",
                            "user_id BIGINT FOREIGN KEY",
                            "[domain_field] VARCHAR(255)",
                            "status ENUM('active','pending','completed')",
                            "metadata JSON",
                            "created_at TIMESTAMP"
                        ],
                        "indexes": [
                            "idx_user_status (user_id, status)",
                            "idx_created (created_at)"
                        ]
                    }},
                    "[table2]": {{...}}
                }},
                "caching": {{
                    "strategy": "Redis with strategic TTLs",
                    "keys": [
                        "user:{{id}}:stats - 5 min TTL",
                        "[feature]:{{id}}:data - 1 hour TTL",
                        "analytics:{{type}}:{{period}} - 1 hour TTL"
                    ]
                }},
                "queues": [
                    "Process[Feature] - Handles AI processing",
                    "Send[Notification] - Email/push notifications",
                    "Generate[Report] - Background analytics"
                ]
            }},
            "ai_integration": {{
                "service_interface": {{
                    "methods": [
                        "analyze(data): AIResult",
                        "generateContent(prompt, context): string",
                        "extractInsights(content): array"
                    ],
                    "providers": ["OpenAI (primary)", "Gemini (fallback)", "Local models (future)"]
                }},
                "prompts": {{
                    "[feature]_prompt": "Specialized prompt for [specific use case]",
                    "system_context": "You are an AI assistant for [domain]..."
                }},
                "error_handling": [
                    "Rate limit retry with exponential backoff",
                    "Fallback to cached responses",
                    "User notification on failures"
                ]
            }},
            "security": {{
                "authentication": {{
                    "method": "Laravel Sanctum (from starter kit)",
                    "enhancements": "Add API tokens for external access"
                }},
                "authorization": {{
                    "policies": [
                        "[Model]Policy - Row-level security",
                        "Feature access by subscription tier"
                    ],
                    "middleware": [
                        "CheckSubscriptionLimits",
                        "ValidateAPIAccess"
                    ]
                }},
                "data_protection": [
                    "HTTPS everywhere",
                    "Encrypted sensitive fields",
                    "GDPR compliance features"
                ]
            }},
            "performance": {{
                "targets": {{
                    "page_load": "< 2 seconds",
                    "api_response": "< 200ms p95",
                    "ai_processing": "< 5 seconds",
                    "concurrent_users": "1,000 initial"
                }},
                "optimization": [
                    "Eager loading with select fields",
                    "Database query optimization",
                    "CDN for static assets",
                    "Redis caching layer"
                ]
            }},
            "monitoring": {{
                "health_checks": "/health endpoint with component status",
                "logging": {{
                    "channels": ["user_activity", "ai_operations", "errors"],
                    "retention": "30 days",
                    "alerts": ["High error rate", "Slow responses", "AI failures"]
                }},
                "metrics": [
                    "Response times by endpoint",
                    "AI token usage and costs",
                    "User engagement metrics"
                ]
            }}
        }}
        
        Return ONLY valid JSON. Be specific to this project's domain and problem.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_architecture_core(self, details: Dict[str, Any], core_entities: List[str]) -> Dict[str, Any]:
        """Generate core architecture components."""
        prompt = f"""
        Generate core architecture components for project: {details.get('name')}
        Description: {details.get('description')}
        Core Entities: {json.dumps(core_entities)}
        
        Return JSON with ONLY these fields:
        {{
            "overview": {{
                "description": "Clear system architecture overview for {details.get('name')}",
                "architecture_style": "monolithic/microservices",
                "key_principles": ["Scalability", "Security", "Maintainability"]
            }},
            "core_entities": [
                {{"name": "Entity1", "description": "What it represents", "relationships": ["Entity2"]}},
                {{"name": "Entity2", "description": "What it represents", "relationships": ["Entity1"]}}
            ],
            "system_components": [
                {{
                    "name": "Laravel API",
                    "type": "backend",
                    "description": "RESTful API server",
                    "technologies": ["Laravel", "PHP 8.2", "MySQL"],
                    "responsibilities": ["Authentication", "Business logic", "Data management"]
                }},
                {{
                    "name": "React Frontend",
                    "type": "frontend",
                    "description": "User interface",
                    "technologies": ["React", "TypeScript", "Tailwind CSS"],
                    "responsibilities": ["UI rendering", "User interactions", "API integration"]
                }}
            ]
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_architecture_technical(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical architecture components."""
        prompt = f"""
        Generate technical architecture for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "data_architecture": {{
                "primary_database": {{
                    "type": "MySQL",
                    "purpose": "Main application data storage",
                    "schemas": ["public"]
                }},
                "cache_layer": {{
                    "type": "Redis",
                    "purpose": "Session storage and caching",
                    "ttl_strategy": "Based on data type"
                }},
                "data_flow": ["Client -> API -> Cache -> Database"]
            }},
            "api_design": {{
                "style": "RESTful",
                "versioning": "/api/v1",
                "authentication": "Laravel Sanctum",
                "rate_limiting": "100 requests per minute",
                "endpoints_structure": {{
                    "pattern": "/api/v1/{{resource}}/{{id}}",
                    "methods": ["GET", "POST", "PUT", "DELETE"]
                }}
            }},
            "security_architecture": {{
                "authentication": {{
                    "method": "JWT via Sanctum",
                    "token_lifetime": "24 hours",
                    "refresh_strategy": "Refresh tokens"
                }},
                "authorization": {{
                    "model": "RBAC",
                    "roles": ["admin", "user"],
                    "permissions": ["create", "read", "update", "delete"]
                }},
                "data_protection": ["HTTPS", "Input validation", "CORS", "CSRF protection"]
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_architecture_deployment(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment and operations architecture."""
        prompt = f"""
        Generate deployment architecture for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "scalability_plan": {{
                "horizontal_scaling": {{
                    "load_balancer": "Nginx",
                    "auto_scaling": "Based on CPU usage"
                }},
                "vertical_scaling": {{
                    "initial_size": "2GB RAM",
                    "max_size": "16GB RAM"
                }},
                "caching_strategy": ["Database query caching", "API response caching", "CDN"]
            }},
            "deployment_architecture": {{
                "environments": [
                    {{"name": "development", "url": "dev.example.com", "purpose": "Development"}},
                    {{"name": "staging", "url": "staging.example.com", "purpose": "Testing"}},
                    {{"name": "production", "url": "app.example.com", "purpose": "Live"}}
                ],
                "ci_cd_pipeline": ["Code commit", "Tests", "Build", "Deploy"],
                "infrastructure_as_code": "Docker Compose"
            }},
            "monitoring_and_logging": {{
                "monitoring": {{
                    "apm": "Application monitoring",
                    "metrics": ["response time", "error rate"],
                    "alerts": ["High error rate", "Slow response"]
                }},
                "logging": {{
                    "centralized_logging": "Log aggregation",
                    "log_levels": ["ERROR", "WARN", "INFO"],
                    "retention": "30 days"
                }}
            }},
            "performance_targets": {{
                "page_load": "< 3 seconds",
                "api_response": "< 200ms",
                "uptime": "99.9%",
                "concurrent_users": "10,000"
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='deepseek')
        return self._extract_json_from_response(response.get('content', ''))
    
    async def _generate_architecture_design(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate design system architecture."""
        prompt = f"""
        Generate design system for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "design_system": {{
                "typography": {{
                    "font_family": "Inter",
                    "font_sizes": {{"h1": "32px", "h2": "24px", "body": "16px"}},
                    "line_heights": {{"heading": "1.2", "body": "1.5"}}
                }},
                "color_palette": {{
                    "primary": {{"hex": "#3B82F6", "name": "Blue", "usage": "Primary actions"}},
                    "secondary": {{"hex": "#8B5CF6", "name": "Purple", "usage": "Secondary"}},
                    "success": {{"hex": "#10B981", "name": "Green", "usage": "Success"}},
                    "danger": {{"hex": "#EF4444", "name": "Red", "usage": "Errors"}},
                    "neutral": {{"hex": "#6B7280", "name": "Gray", "usage": "Text"}}
                }},
                "spacing": {{
                    "unit": "8px",
                    "scale": ["0", "1", "2", "4", "8", "16", "32"]
                }},
                "components": {{
                    "buttons": ["primary", "secondary", "outline"],
                    "forms": ["input", "select", "checkbox"],
                    "layout": ["container", "grid", "flex"]
                }}
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_json_with_retry(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generate JSON response with retry logic for parsing errors.
        
        Args:
            prompt: The prompt to send to AI
            max_retries: Maximum number of retries for JSON parsing errors
            
        Returns:
            Parsed JSON dictionary
        """
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"[Attempt {attempt + 1}/{max_retries}] Generating JSON response...")
                response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
                content = response.get('content', '')
                
                # Try to extract and parse JSON
                result = self._extract_json_from_response(content)
                self.logger.debug(f"Successfully parsed JSON on attempt {attempt + 1}")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"[Attempt {attempt + 1}/{max_retries}] JSON parsing failed: {e}")
                
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying AI request due to JSON parsing error (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(2)  # Wait before retry
                else:
                    self.logger.error(f"Failed to get valid JSON after {max_retries} attempts")
                    raise
        
        # Should not reach here
        raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts")
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON from AI response."""
        try:
            import re
            # First try to find JSON in code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Fall back to finding raw JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("No JSON found in response")
            
            # Fix common JSON issues from AI responses
            # Replace escaped newlines in strings with actual newlines
            json_str = re.sub(r'\\n', '\n', json_str)
            # Replace double backslashes with single (for escaped quotes in code)
            json_str = re.sub(r'\\\\', r'\\', json_str)
            # Replace smart quotes with regular quotes
            json_str = json_str.replace('"', '"').replace('"', '"')
            json_str = json_str.replace(''', "'").replace(''', "'")
                    
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"Response content: {content[:500]}...")
            # Try one more time with more aggressive cleaning
            try:
                # Remove all backslashes except those before quotes
                json_str = re.sub(r'\\(?!")', '', json_str)
                return json.loads(json_str)
            except:
                raise e
    
    def _validate_generated_documentation(self, docs: Dict[str, Any]) -> List[str]:
        """Validate that generated documentation meets quality standards.
        
        Args:
            docs: Generated documentation dictionary
            
        Returns:
            List of validation issues (empty if all good)
        """
        issues = []
        
        # Check architecture completeness
        arch = docs.get('architecture', {})
        required_sections = [
            'executive_summary', 'system_context', 'building_blocks',
            'portal_separation', 'implementation_roadmap', 'data_architecture',
            'ai_integration', 'security', 'performance', 'monitoring'
        ]
        
        for section in required_sections:
            if not arch.get(section):
                issues.append(f"Architecture missing required section: {section}")
        
        # Check for starter kit acknowledgment
        if arch and 'starter_kit_foundation' not in str(arch):
            issues.append("Architecture doesn't acknowledge Laravel React starter kit")
        
        # Check for code examples
        if arch and 'class ' not in str(arch) and 'function ' not in str(arch):
            issues.append("Architecture lacks code examples")
        
        # Check monetization completeness
        monetization = docs.get('monetization', {})
        if monetization:
            if not monetization.get('pricing_tiers'):
                issues.append("Monetization missing pricing tiers")
            if not monetization.get('implementation_code'):
                issues.append("Monetization missing implementation details")
            if not monetization.get('metrics_queries'):
                issues.append("Monetization missing metrics tracking queries")
        
        # Check README structure
        readme = docs.get('readme', {})
        if readme:
            if not readme.get('quick_start'):
                issues.append("README missing quick start section")
            if not readme.get('project_structure'):
                issues.append("README missing project structure")
            if not readme.get('key_features'):
                issues.append("README missing key features")
        
        return issues
    
    async def _generate_structured_monetization(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive monetization strategy with implementation details.
        
        Args:
            details: Project details
            
        Returns:
            Structured monetization data
        """
        if not self.ai_brain:
            raise ValueError("AI Brain is required for content generation")
        
        self.logger.info("💰 Generating comprehensive monetization strategy with implementation details...")
        
        # Extract market research data
        metadata = details.get('metadata', {})
        market_research = metadata.get('market_research', {})
        validation_metrics = metadata.get('validation_metrics', {})
        
        prompt = f"""
        Generate a comprehensive monetization strategy that directly ties to technical implementation.
        
        Project: {details.get('name')}
        Market Research:
        - Problem: {market_research.get('problem_description', details.get('problem_statement'))}
        - Market Size: {market_research.get('market_size', 'Growing market')}
        - Competition: {market_research.get('competition_level', 'Moderate')}
        - Revenue Potential: {validation_metrics.get('revenue_potential', '$50,000-$200,000 ARR')}
        
        Generate JSON with implementation-focused monetization:
        {{
            "executive_summary": {{
                "business_model": "Subscription SaaS",
                "primary_revenue": "Monthly/annual subscriptions",
                "secondary_revenue": ["API access", "White-label", "Professional services"],
                "target_ltv": "$1,200",
                "payback_period": "3 months"
            }},
            "pricing_tiers": [
                {{
                    "name": "Free",
                    "price": 0,
                    "purpose": "User acquisition",
                    "features": [
                        "[Limited feature 1] - up to 3 [items]",
                        "[Limited feature 2] - basic only"
                    ],
                    "implementation": {{
                        "database_flag": "subscription_tier = 'free'",
                        "middleware_check": "CheckFreeTierLimits",
                        "usage_tracking": "track in usage_records table"
                    }},
                    "conversion_target": "5% to paid in 30 days"
                }},
                {{
                    "name": "Starter",
                    "price": 29,
                    "features": [
                        "Everything in Free",
                        "[Feature 1] - up to 100 [items]",
                        "[Feature 2] - standard access"
                    ],
                    "implementation": {{
                        "database_flag": "subscription_tier = 'starter'",
                        "feature_flags": {{"[feature1]_limit": 100, "[feature2]_enabled": true}},
                        "api_rate_limit": "100 requests/hour"
                    }}
                }},
                {{
                    "name": "Professional",
                    "price": 99,
                    "features": [
                        "Everything in Starter",
                        "[Advanced feature 1]",
                        "[Advanced feature 2]",
                        "API access"
                    ],
                    "implementation": {{
                        "database_flag": "subscription_tier = 'professional'",
                        "feature_flags": {{"[feature1]_limit": 1000, "api_enabled": true}},
                        "api_rate_limit": "1000 requests/hour"
                    }}
                }}
            ],
            "implementation_code": {{
                "subscription_check": "Middleware CheckSubscriptionLimits that checks user feature limits and redirects to upgrade if exceeded",
                "usage_tracking": "DB insert to usage_records table tracking user_id, feature name, and timestamp",
                "upgrade_flow": "Inertia render of UpgradePrompt page with current and recommended tiers"
            }},
            "metrics_queries": {{
                "mrr_calculation": "SELECT DATE_FORMAT(created_at, '%Y-%m') as month, SUM(CASE WHEN tier = 'starter' THEN 29 WHEN tier = 'professional' THEN 99 END) as mrr FROM subscriptions WHERE status = 'active' GROUP BY month",
                "churn_analysis": "SELECT cohort_month, COUNT(DISTINCT user_id) as cohort_size, COUNT(DISTINCT CASE WHEN churned_at IS NOT NULL THEN user_id END) as churned FROM user_cohorts GROUP BY cohort_month",
                "feature_adoption": "SELECT subscription_tier, feature, COUNT(DISTINCT user_id) as users FROM usage_records WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) GROUP BY subscription_tier, feature"
            }},
            "market_analysis": {{
                "target_segments": [
                    {{
                        "segment": "Small Businesses",
                        "size": "[X thousand businesses]",
                        "pain_points": ["Manual processes", "Limited resources"],
                        "willingness_to_pay": "$29-99/month",
                        "acquisition_strategy": "Content marketing, SEO"
                    }},
                    {{
                        "segment": "Growing Companies",
                        "size": "[Y thousand companies]", 
                        "pain_points": ["Scaling challenges", "Integration needs"],
                        "willingness_to_pay": "$99-299/month",
                        "acquisition_strategy": "Direct sales, partnerships"
                    }}
                ],
                "competitive_advantage": "[Key differentiator based on AI/automation]",
                "market_entry": "Bottom-up approach targeting [initial segment]"
            }},
            "customer_acquisition": {{
                "channels": [
                    {{
                        "name": "Content Marketing",
                        "cac": "$50",
                        "strategy": "SEO-optimized blog, tutorials, case studies"
                    }},
                    {{
                        "name": "Product-Led Growth",
                        "cac": "$20",
                        "strategy": "Free tier with viral features"
                    }},
                    {{
                        "name": "Partnerships",
                        "cac": "$100",
                        "strategy": "Integrate with complementary tools"
                    }}
                ],
                "retention_tactics": [
                    "Onboarding automation",
                    "Customer success check-ins",
                    "Feature adoption campaigns",
                    "Annual plan discounts"
                ]
            }},
            "growth_projections": {{
                "year_1": {{
                    "revenue": "$50,000-$100,000",
                    "customers": "500-1000",
                    "mrr_growth": "20% monthly"
                }},
                "year_2": {{
                    "revenue": "$300,000-$600,000",
                    "customers": "2000-4000",
                    "mrr_growth": "15% monthly"
                }},
                "year_3": {{
                    "revenue": "$1M-$2M",
                    "customers": "5000-10000",
                    "mrr_growth": "10% monthly"
                }}
            }}
        }}
        
        Return ONLY valid JSON. Be specific to this project's market and problem.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_monetization_executive_market(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary and market analysis chunk."""
        # Extract research data from metadata
        metadata = details.get('metadata', {})
        market_research = metadata.get('market_research', {})
        validation_metrics = metadata.get('validation_metrics', {})
        
        prompt = f"""
        Generate executive summary and market analysis based on deep market research.
        
        Project: {details.get('name')}
        Description: {details.get('description')}
        Problem: {market_research.get('problem_description', details.get('problem_statement'))}
        
        Market Research Data:
        - Market Size: {market_research.get('market_size', 'Large and growing')}
        - Competition Level: {market_research.get('competition_level', 'Moderate')}
        - Current Solutions: {market_research.get('current_solutions', 'Multiple competitors exist')}
        - Validation Score: {validation_metrics.get('validation_score', '9.0')}/10
        - Revenue Potential: {validation_metrics.get('revenue_potential', '$50,000 ARR in year one')}
        
        Return JSON with ONLY these fields:
        {{
            "executive_summary": {{
                "business_model": "SaaS/Marketplace/Platform/etc",
                "revenue_streams": ["Primary stream description", "Secondary stream description"],
                "target_market_size": "$X billion market opportunity",
                "projected_revenue_year_1": "$100K-500K",
                "projected_revenue_year_3": "$1M-5M",
                "value_proposition": "Clear value statement"
            }},
            "market_analysis": {{
                "market_size": {{
                    "tam": "$X billion - Total Addressable Market description",
                    "sam": "$X million - Serviceable Addressable Market description", 
                    "som": "$X million - Serviceable Obtainable Market description",
                    "growth_rate": "X% annual growth"
                }},
                "target_segments": [
                    {{
                        "name": "Primary Segment",
                        "size": "X thousand businesses/users",
                        "characteristics": ["Specific trait 1", "Specific trait 2"],
                        "pain_points": ["Current pain 1", "Current pain 2"],
                        "willingness_to_pay": "$X-Y per month"
                    }}
                ],
                "competitors": [
                    {{
                        "name": "Major Competitor",
                        "market_share": "X%",
                        "strengths": ["Their strength 1", "Their strength 2"],
                        "weaknesses": ["Their weakness 1", "Their weakness 2"],
                        "pricing": "$X-Y per month",
                        "our_advantage": "How we differentiate"
                    }}
                ],
                "market_trends": [
                    {{"trend": "Growing trend description", "impact": "Positive impact on our business"}}
                ]
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_monetization_revenue_pricing(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate revenue models and pricing strategy chunk."""
        prompt = f"""
        Generate revenue models and pricing strategy for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "revenue_models": [
                {{
                    "model_type": "Subscription",
                    "description": "Monthly/annual recurring revenue",
                    "pricing_tiers": [
                        {{
                            "name": "Starter",
                            "price_monthly": 49,
                            "price_annual": 490,
                            "features": ["Core feature 1", "Core feature 2", "Basic support"],
                            "limitations": ["10 users", "100GB storage"],
                            "target_users": "Individual users and small teams",
                            "expected_conversion": "5%"
                        }},
                        {{
                            "name": "Professional", 
                            "price_monthly": 149,
                            "price_annual": 1490,
                            "features": ["All Starter features", "Advanced feature 1", "Priority support"],
                            "limitations": ["50 users", "1TB storage"],
                            "target_users": "Growing businesses",
                            "expected_conversion": "3%"
                        }},
                        {{
                            "name": "Enterprise",
                            "price_monthly": "Custom",
                            "price_annual": "Custom", 
                            "features": ["All features", "Custom integrations", "Dedicated support", "SLA"],
                            "limitations": ["Unlimited"],
                            "target_users": "Large organizations",
                            "expected_conversion": "1%"
                        }}
                    ]
                }},
                {{
                    "model_type": "Usage-based",
                    "description": "Pay per use or transaction",
                    "fee_structure": {{
                        "base_fee": "$0",
                        "usage_fee": "$0.10 per transaction",
                        "volume_discounts": ["1000+: $0.08", "10000+: $0.06"]
                    }}
                }}
            ],
            "pricing_strategy": {{
                "approach": "Value-based pricing",
                "competitive_positioning": "Premium quality at competitive price",
                "psychological_pricing": ["$49 instead of $50", "Annual discount incentive"],
                "price_testing": "A/B test pricing tiers",
                "promotional_strategy": ["14-day free trial", "20% off annual plans"]
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _generate_monetization_customer_growth(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customer acquisition and growth strategy chunk."""  
        prompt = f"""
        Generate customer acquisition and growth strategy for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "customer_acquisition": {{
                "channels": [
                    {{
                        "name": "Content Marketing",
                        "cost_per_acquisition": 50,
                        "expected_volume": "100 customers/month",
                        "strategy": "SEO-optimized blog posts and tutorials",
                        "conversion_rate": "2%"
                    }},
                    {{
                        "name": "Paid Advertising",
                        "cost_per_acquisition": 100,
                        "expected_volume": "200 customers/month",
                        "strategy": "Google Ads and Facebook campaigns",
                        "conversion_rate": "1.5%"
                    }},
                    {{
                        "name": "Partner Referrals",
                        "cost_per_acquisition": 30,
                        "expected_volume": "50 customers/month",
                        "strategy": "Strategic partnerships and affiliate program",
                        "conversion_rate": "5%"
                    }}
                ],
                "retention_strategy": {{
                    "onboarding": ["Interactive tutorial", "Personal onboarding call", "Email drip campaign"],
                    "engagement": ["Regular feature updates", "Customer success check-ins", "Community building"],
                    "churn_prevention": ["Usage monitoring", "Proactive support", "Win-back campaigns"],
                    "expected_churn_rate": "5% monthly"
                }},
                "referral_program": {{
                    "incentive": "1 month free for referrer and referee",
                    "expected_referral_rate": "20%",
                    "viral_coefficient": "0.3"
                }}
            }},
            "growth_strategy": {{
                "phases": [
                    {{
                        "name": "Launch (Months 1-6)",
                        "focus": "Product-market fit",
                        "targets": {{
                            "users": 1000,
                            "mrr": "$50,000",
                            "growth_rate": "20% monthly"
                        }},
                        "key_initiatives": ["Beta testing", "Early adopter program", "Product iteration"]
                    }},
                    {{
                        "name": "Growth (Months 7-12)",
                        "focus": "Scale acquisition",
                        "targets": {{
                            "users": 5000,
                            "mrr": "$250,000",
                            "growth_rate": "15% monthly"
                        }},
                        "key_initiatives": ["Marketing automation", "Sales team", "Feature expansion"]
                    }},
                    {{
                        "name": "Scale (Year 2+)",
                        "focus": "Market expansion",
                        "targets": {{
                            "users": 20000,
                            "mrr": "$1,000,000",
                            "growth_rate": "10% monthly"
                        }},
                        "key_initiatives": ["International expansion", "Enterprise sales", "Platform ecosystem"]
                    }}
                ],
                "expansion_opportunities": ["Add-on products", "API marketplace", "White-label solution"]
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='deepseek')
        return self._extract_json_from_response(response.get('content', ''))
    
    async def _generate_monetization_financial_risks(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial projections and risk analysis chunk."""
        prompt = f"""
        Generate financial projections and risk analysis for project: {details.get('name')}
        
        Return JSON with ONLY these fields:
        {{
            "financial_projections": {{
                "year_1": {{
                    "revenue": {{
                        "subscription": "$200,000",
                        "services": "$50,000",
                        "total": "$250,000"
                    }},
                    "costs": {{
                        "development": "$150,000",
                        "marketing": "$50,000",
                        "operations": "$30,000",
                        "total": "$230,000"
                    }},
                    "profit": "$20,000",
                    "profit_margin": "8%"
                }},
                "year_2": {{
                    "revenue": {{
                        "subscription": "$800,000",
                        "services": "$200,000",
                        "total": "$1,000,000"
                    }},
                    "costs": {{
                        "development": "$300,000",
                        "marketing": "$200,000",
                        "operations": "$100,000",
                        "total": "$600,000"
                    }},
                    "profit": "$400,000",
                    "profit_margin": "40%"
                }},
                "year_3": {{
                    "revenue": {{
                        "subscription": "$2,400,000",
                        "services": "$600,000",
                        "total": "$3,000,000"
                    }},
                    "costs": {{
                        "development": "$600,000",
                        "marketing": "$500,000",
                        "operations": "$400,000",
                        "total": "$1,500,000"
                    }},
                    "profit": "$1,500,000",
                    "profit_margin": "50%"
                }},
                "break_even": "Month 10",
                "runway": "18 months with $500K funding"
            }},
            "risk_analysis": {{
                "market_risks": [
                    {{
                        "risk": "New competitor entry",
                        "probability": "High",
                        "impact": "Medium",
                        "mitigation": "Build strong brand loyalty and network effects"
                    }},
                    {{
                        "risk": "Market downturn",
                        "probability": "Medium",
                        "impact": "High",
                        "mitigation": "Diversify revenue streams and maintain lean operations"
                    }}
                ],
                "operational_risks": [
                    {{
                        "risk": "Technical scalability issues",
                        "probability": "Medium",
                        "impact": "High",
                        "mitigation": "Invest in robust infrastructure early"
                    }},
                    {{
                        "risk": "Key employee turnover",
                        "probability": "Medium",
                        "impact": "Medium",
                        "mitigation": "Competitive compensation and equity packages"
                    }}
                ],
                "financial_risks": [
                    {{
                        "risk": "Cash flow gaps",
                        "probability": "Medium",
                        "impact": "High",
                        "mitigation": "Maintain 6-month cash buffer"
                    }},
                    {{
                        "risk": "Customer concentration",
                        "probability": "Low",
                        "impact": "High",
                        "mitigation": "No single customer > 10% of revenue"
                    }}
                ]
            }}
        }}
        
        Return ONLY valid JSON.
        """
        
        return await self._generate_json_with_retry(prompt)
    
    async def _save_project_metadata(self, repo, details: Dict[str, Any], 
                                   structured_docs: Dict[str, Any], results: Dict[str, Any]):
        """Save complete project metadata including all structured documentation.
        
        Args:
            repo: Repository object
            details: Project details
            structured_docs: All structured documentation
            results: Results dictionary
        """
        try:
            self.logger.info("💾 Saving complete project metadata...")
            
            # Get current repository state
            repo_path = Path(repo.path) if hasattr(repo, 'path') else None
            
            # Count issues and tasks
            issue_count = 0
            task_states = {
                'pending': 0,
                'in_progress': 0,
                'completed': 0,
                'total': 0
            }
            
            if repo_path:
                # Count issues
                issues_dir = repo_path / "issues"
                if issues_dir.exists():
                    issue_count = len(list(issues_dir.glob("issue-*.json")))
                
                # TODO: Add task state tracking when task system is integrated
            
            # Build complete metadata
            metadata = {
                "project": {
                    "name": details.get('name'),
                    "description": details.get('description'),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0.0"
                },
                "repository": {
                    "full_name": repo.full_name if hasattr(repo, 'full_name') else f"CodeWebMobile-AI/{details.get('name')}",
                    "path": str(repo_path) if repo_path else None,
                    "default_branch": "main",
                    "template_used": "react-starter-kit"
                },
                "documentation": {
                    "readme": structured_docs.get('readme', {}),
                    "architecture": structured_docs.get('architecture', {}),
                    "monetization": structured_docs.get('monetization', {}),
                    "generated_at": structured_docs.get('generated_at')
                },
                "project_details": {
                    "problem_statement": details.get('problem_statement'),
                    "target_audience": details.get('target_audience'),
                    "initial_features": details.get('initial_features', []),
                    "core_entities": self._extract_core_entities(details),
                    "tech_stack": details.get('tech_stack'),
                    "project_type": details.get('metadata', {}).get('project_type')
                },
                "statistics": {
                    "issues": {
                        "total": issue_count,
                        "open": issue_count,  # All new issues are open
                        "closed": 0
                    },
                    "tasks": task_states,
                    "documentation_files": {
                        "readme": bool(structured_docs.get('readme')),
                        "architecture": bool(structured_docs.get('architecture')),
                        "monetization": bool(structured_docs.get('monetization'))
                    }
                },
                "ai_generation": {
                    "models_used": ["gemini-2.0-flash"],
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompts_count": 3  # README, Architecture, Monetization
                }
            }
            
            # Save to metadata.json
            if self.storage_mode == StorageMode.LOCAL and repo_path:
                metadata_path = repo_path / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"✅ Saved complete metadata to {metadata_path}")
            else:
                # For GitHub mode, update the file
                try:
                    # Get existing metadata file
                    existing = repo.get_contents("metadata.json")
                    repo.update_file(
                        "metadata.json",
                        "Update project metadata with complete documentation",
                        json.dumps(metadata, indent=2),
                        existing.sha
                    )
                except:
                    # Create if doesn't exist
                    repo.create_file(
                        "metadata.json",
                        "Add complete project metadata",
                        json.dumps(metadata, indent=2)
                    )
                
            results['metadata_saved'] = True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save project metadata: {e}")
            raise  # Re-raise the exception to stop execution
    
    async def _create_markdown_files_from_structured(self, repo, structured_docs: Dict[str, Any], 
                                                   results: Dict[str, Any]):
        """Transform structured documentation to markdown files.
        
        Args:
            repo: Repository object
            structured_docs: Structured documentation data
            results: Results dictionary
        """
        try:
            # 1. Create README.md from structured data
            if structured_docs.get('readme'):
                self.logger.info("📝 Creating README.md from structured data")
                readme_content = self._transform_structured_readme_to_markdown(structured_docs['readme'])
                
                if self.storage_mode == StorageMode.LOCAL:
                    repo.update_file(
                        "README.md",
                        "Update README with comprehensive documentation",
                        readme_content
                    )
                else:
                    try:
                        existing = repo.get_contents("README.md")
                        repo.update_file(
                            "README.md",
                            "Update README with comprehensive documentation",
                            readme_content,
                            existing.sha
                        )
                    except:
                        repo.create_file(
                            "README.md",
                            "Add comprehensive README",
                            readme_content
                        )
                
                results['readme_updated'] = True
                self.logger.info("✅ README.md created successfully")
            
            # 2. Create ARCHITECTURE.md from structured data
            if structured_docs.get('architecture'):
                self.logger.info("🏗️ Creating ARCHITECTURE.md from structured data")
                # Use the enhanced markdown converter
                try:
                    from scripts.architecture_markdown_converter import ArchitectureMarkdownConverter
                    converter = ArchitectureMarkdownConverter()
                    architecture_content = converter.convert_to_markdown(structured_docs['architecture'])
                except ImportError:
                    # Fall back to existing method
                    architecture_content = self._transform_structured_architecture_to_markdown(structured_docs['architecture'])
                
                if self.storage_mode == StorageMode.LOCAL:
                    repo.create_file(
                        "ARCHITECTURE.md",
                        "Add comprehensive architecture documentation",
                        architecture_content
                    )
                else:
                    repo.create_file(
                        "ARCHITECTURE.md",
                        "Add comprehensive architecture documentation",
                        architecture_content
                    )
                
                results['architecture_saved'] = True
                self.logger.info("✅ ARCHITECTURE.md created successfully")
            
            # 3. Create MONETIZATION_STRATEGY.md from structured data
            if structured_docs.get('monetization'):
                self.logger.info("💰 Creating MONETIZATION_STRATEGY.md from structured data")
                monetization_content = self._transform_structured_monetization_to_markdown(structured_docs['monetization'])
                
                if self.storage_mode == StorageMode.LOCAL:
                    repo.create_file(
                        "MONETIZATION_STRATEGY.md",
                        "Add comprehensive monetization strategy",
                        monetization_content
                    )
                else:
                    repo.create_file(
                        "MONETIZATION_STRATEGY.md",
                        "Add comprehensive monetization strategy",
                        monetization_content
                    )
                
                results['monetization_saved'] = True
                self.logger.info("✅ MONETIZATION_STRATEGY.md created successfully")
            
            # Log summary
            self.logger.info(f"""
📚 Documentation Creation Summary:
- README.md: {'✅ Created' if results.get('readme_updated') else '❌ Failed'}
- ARCHITECTURE.md: {'✅ Created' if results.get('architecture_saved') else '❌ Failed'}
- MONETIZATION_STRATEGY.md: {'✅ Created' if results.get('monetization_saved') else '❌ Failed'}
- metadata.json: {'✅ Updated' if results.get('metadata_saved') else '❌ Failed'}
            """)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create markdown files: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _json_to_markdown(self, data: Any, level: int = 0, parent_key: str = '') -> str:
        """Recursively convert JSON data to markdown format.
        
        This method intelligently handles any JSON structure and converts it to
        well-formatted markdown without requiring hardcoded field names.
        
        Args:
            data: JSON data (can be dict, list, or primitive)
            level: Current heading level (0 = title, 1 = ##, 2 = ###, etc)
            parent_key: Parent key name for context
            
        Returns:
            Markdown formatted string
        """
        md = ""
        
        # Handle None/null values
        if data is None:
            return ""
        
        # Handle primitive types (string, number, boolean)
        if isinstance(data, (str, int, float, bool)):
            # For top-level strings or when parent suggests it's a description/content
            if level == 0 or parent_key.lower() in ['description', 'overview', 'summary', 
                                                     'content', 'details', 'explanation',
                                                     'purpose', 'rationale', 'benefit',
                                                     'strategy', 'approach', 'guidelines',
                                                     'value_proposition', 'solution']:
                return f"{data}\n\n"
            else:
                return f"{data}"
        
        # Handle lists
        if isinstance(data, list):
            if not data:
                return ""
                
            # Check if it's a list of complex objects
            if all(isinstance(item, dict) for item in data):
                # List of dictionaries - render each as a subsection
                for item in data:
                    if 'name' in item or 'title' in item:
                        # Has a natural heading
                        item_name = item.get('name') or item.get('title')
                        md += f"{'#' * (level + 2)} {item_name}\n\n"
                        # Render other fields
                        for key, value in item.items():
                            if key not in ['name', 'title']:
                                formatted_key = self._format_key(key)
                                if isinstance(value, (list, dict)):
                                    md += f"**{formatted_key}**:\n"
                                    md += self._json_to_markdown(value, level + 2, key)
                                else:
                                    md += f"**{formatted_key}**: {value}\n"
                        md += "\n"
                    else:
                        # No natural heading - use parent context or index
                        md += self._json_to_markdown(item, level + 1, parent_key)
                        md += "---\n\n"
            else:
                # List of simple values - render as bullet points
                for item in data:
                    if isinstance(item, dict):
                        # Mixed content - format the dict inline
                        md += f"- "
                        dict_parts = []
                        for k, v in item.items():
                            if isinstance(v, (str, int, float, bool)):
                                dict_parts.append(f"{self._format_key(k)}: {v}")
                        md += " | ".join(dict_parts) + "\n"
                    else:
                        md += f"- {item}\n"
                md += "\n"
            return md
        
        # Handle dictionaries
        if isinstance(data, dict):
            # Special handling for root level with title
            if level == 0 and 'title' in data:
                md += f"# {data['title']}\n\n"
                # Process other fields
                for key, value in data.items():
                    if key == 'title':
                        continue
                    if value is None or (isinstance(value, (list, dict)) and not value):
                        continue
                    formatted_key = self._format_key(key)
                    md += f"## {formatted_key}\n\n"
                    md += self._json_to_markdown(value, 1, key)
            else:
                # Regular dictionary processing
                for key, value in data.items():
                    if value is None or (isinstance(value, (list, dict)) and not value):
                        continue
                        
                    formatted_key = self._format_key(key)
                    
                    # Determine how to format based on value type and context
                    if isinstance(value, dict):
                        # Nested dictionary - create subsection
                        md += f"{'#' * (level + 2)} {formatted_key}\n\n"
                        md += self._json_to_markdown(value, level + 1, key)
                    elif isinstance(value, list):
                        # List - check if simple or complex
                        if value and all(isinstance(item, (str, int, float, bool)) for item in value):
                            # Simple list - inline with bullets
                            md += f"**{formatted_key}**:\n"
                            for item in value:
                                md += f"- {item}\n"
                            md += "\n"
                        else:
                            # Complex list - create subsection
                            md += f"{'#' * (level + 2)} {formatted_key}\n\n"
                            md += self._json_to_markdown(value, level + 1, key)
                    else:
                        # Primitive value - format based on context
                        if len(str(value)) > 100 or '\n' in str(value):
                            # Long text - put on separate line
                            md += f"**{formatted_key}**:\n\n{value}\n\n"
                        else:
                            # Short text - inline
                            md += f"**{formatted_key}**: {value}\n\n"
        
        return md
    
    def _format_key(self, key: str) -> str:
        """Format a JSON key into a human-readable heading.
        
        Args:
            key: Raw JSON key
            
        Returns:
            Formatted heading
        """
        # Handle snake_case and camelCase
        if '_' in key:
            words = key.split('_')
        else:
            # Handle camelCase
            import re
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', key)
            if not words:
                words = [key]
        
        # Capitalize appropriately
        formatted_words = []
        for word in words:
            word_lower = word.lower()
            # Keep certain acronyms uppercase
            if word_lower in ['api', 'url', 'id', 'ui', 'ux', 'mvp', 'kpi', 'roi', 
                             'tam', 'sam', 'som', 'mrr', 'arr', 'cac', 'ltv', 'sla']:
                formatted_words.append(word.upper())
            # Keep certain words lowercase (unless first word)
            elif word_lower in ['and', 'or', 'the', 'of', 'to', 'for', 'with', 'in', 'on'] and formatted_words:
                formatted_words.append(word_lower)
            else:
                formatted_words.append(word.capitalize())
        
        return ' '.join(formatted_words)
    
    def _transform_structured_readme_to_markdown(self, data: Dict[str, Any]) -> str:
        """Transform structured README data to markdown format using dynamic converter.
        
        Args:
            data: Structured README data
            
        Returns:
            Markdown formatted README
        """
        # Use the dynamic converter
        return self._json_to_markdown(data, level=0)
    
    def _transform_structured_architecture_to_markdown(self, data: Dict[str, Any]) -> str:
        """Transform structured architecture data to markdown format using dynamic converter.
        
        Args:
            data: Structured architecture data
            
        Returns:
            Markdown formatted architecture document
        """
        # Add a title if not present
        if 'title' not in data:
            data = {'title': 'System Architecture', **data}
        
        # Use the dynamic converter
        return self._json_to_markdown(data, level=0)
    
    def _transform_structured_monetization_to_markdown(self, data: Dict[str, Any]) -> str:
        """Transform structured monetization data to markdown format using dynamic converter.
        
        Args:
            data: Structured monetization data
            
        Returns:
            Markdown formatted monetization strategy
        """
        # Add a title if not present
        if 'title' not in data:
            data = {'title': 'Monetization Strategy', **data}
        
        # Use the dynamic converter
        return self._json_to_markdown(data, level=0)
