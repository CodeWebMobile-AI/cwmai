"""
Architecture Generator for Existing Projects

Analyzes existing projects and generates comprehensive architecture documentation.
Can be used to create ARCHITECTURE.md for projects that don't have one.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from github import Github
import base64


class ArchitectureGenerator:
    """Generate architecture documentation for existing projects."""
    
    def __init__(self, github_token: str, ai_brain=None):
        """Initialize architecture generator.
        
        Args:
            github_token: GitHub personal access token
            ai_brain: AI brain for intelligent analysis
        """
        self.github = Github(github_token)
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
    async def generate_architecture_for_project(self, 
                                               repo_full_name: str,
                                               repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture document for an existing project.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            repo_analysis: Repository analysis from RepositoryAnalyzer
            
        Returns:
            Generated architecture data
        """
        self.logger.info(f"Generating architecture for {repo_full_name}")
        
        # Extract key information from analysis
        basic_info = repo_analysis.get('basic_info', {})
        tech_stack = repo_analysis.get('technical_stack', {})
        code_structure = repo_analysis.get('code_analysis', {})
        
        # Analyze codebase to extract architecture
        extracted_architecture = await self._extract_architecture_from_code(
            repo_full_name, 
            repo_analysis
        )
        
        # Generate comprehensive architecture using AI
        if self.ai_brain:
            architecture = await self._generate_architecture_with_ai(
                repo_full_name,
                basic_info,
                tech_stack,
                code_structure,
                extracted_architecture
            )
        else:
            # Fallback to basic extraction
            architecture = self._create_basic_architecture(
                basic_info,
                tech_stack,
                extracted_architecture
            )
            
        return architecture
    
    async def _extract_architecture_from_code(self, 
                                            repo_full_name: str,
                                            repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract architectural patterns from existing codebase.
        
        Args:
            repo_full_name: Repository full name
            repo_analysis: Repository analysis
            
        Returns:
            Extracted architecture patterns
        """
        try:
            repo = self.github.get_repo(repo_full_name)
            
            extracted = {
                'models': [],
                'controllers': [],
                'components': [],
                'api_endpoints': [],
                'database_entities': [],
                'frontend_structure': {},
                'backend_structure': {},
                'configuration': {}
            }
            
            # Analyze Laravel models
            try:
                models_content = repo.get_contents("app/Models")
                if isinstance(models_content, list):
                    for file in models_content:
                        if file.name.endswith('.php'):
                            model_name = file.name.replace('.php', '')
                            extracted['models'].append(model_name)
                            extracted['database_entities'].append(model_name)
            except:
                self.logger.debug("No Laravel models directory found")
                
            # Analyze controllers
            try:
                controllers = repo.get_contents("app/Http/Controllers")
                if isinstance(controllers, list):
                    for file in controllers:
                        if file.name.endswith('Controller.php'):
                            controller_name = file.name.replace('.php', '')
                            extracted['controllers'].append(controller_name)
                            
                            # Try to extract API routes
                            if 'Api' in file.path:
                                resource = controller_name.replace('Controller', '').lower()
                                extracted['api_endpoints'].extend([
                                    f"/api/{resource}",
                                    f"/api/{resource}/{{id}}"
                                ])
            except:
                self.logger.debug("No controllers directory found")
                
            # Analyze React components
            try:
                components = repo.get_contents("resources/js/components")
                if isinstance(components, list):
                    for file in components:
                        if file.name.endswith('.tsx') or file.name.endswith('.jsx'):
                            component_name = file.name.split('.')[0]
                            extracted['components'].append(component_name)
            except:
                self.logger.debug("No React components directory found")
                
            # Check for configuration files
            config_files = ['package.json', 'composer.json', '.env.example', 'tsconfig.json']
            for config_file in config_files:
                try:
                    file_content = repo.get_contents(config_file)
                    extracted['configuration'][config_file] = True
                    
                    # Extract dependencies if JSON
                    if config_file.endswith('.json'):
                        content = base64.b64decode(file_content.content).decode('utf-8')
                        data = json.loads(content)
                        
                        if config_file == 'package.json':
                            extracted['frontend_structure']['dependencies'] = list(data.get('dependencies', {}).keys())[:10]
                        elif config_file == 'composer.json':
                            extracted['backend_structure']['dependencies'] = list(data.get('require', {}).keys())[:10]
                except:
                    pass
                    
            # Analyze database migrations
            try:
                migrations = repo.get_contents("database/migrations")
                if isinstance(migrations, list):
                    migration_tables = []
                    for file in migrations:
                        if 'create_' in file.name and '_table' in file.name:
                            # Extract table name from migration file
                            table_name = file.name.split('create_')[1].split('_table')[0]
                            migration_tables.append(table_name)
                    extracted['database_entities'].extend(migration_tables)
            except:
                self.logger.debug("No migrations found")
                
            return extracted
            
        except Exception as e:
            self.logger.error(f"Error extracting architecture: {e}")
            return {}
    
    async def _generate_architecture_with_ai(self, 
                                           repo_name: str,
                                           basic_info: Dict[str, Any],
                                           tech_stack: Dict[str, Any],
                                           code_structure: Dict[str, Any],
                                           extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive architecture using AI.
        
        Args:
            repo_name: Repository name (e.g., 'project-analytics-dashboard')
            basic_info: Basic repository information
            tech_stack: Detected technology stack
            code_structure: Code structure analysis
            extracted: Extracted architecture patterns
            
        Returns:
            Complete architecture specification
        """
        # Parse meaningful information from repo name
        name_parts = repo_name.lower().replace('-', ' ').replace('_', ' ').split()
        prefixes_to_remove = ['project', 'app', 'application', 'system', 'platform']
        filtered_parts = [part for part in name_parts if part not in prefixes_to_remove]
        project_type = ' '.join(filtered_parts).title()
        
        # Prepare context for AI
        context = {
            'repository': repo_name,
            'project_type': project_type,
            'description': basic_info.get('description', ''),
            'language': basic_info.get('language', ''),
            'tech_stack': tech_stack,
            'detected_models': extracted.get('models', []),
            'detected_controllers': extracted.get('controllers', []),
            'detected_components': extracted.get('components', []),
            'api_endpoints': extracted.get('api_endpoints', []),
            'database_entities': list(set(extracted.get('database_entities', []))),
            'file_structure': code_structure
        }
        
        prompt = f"""
        Analyze this existing project and generate a comprehensive architecture document.
        
        IMPORTANT: The repository name is HIGHLY DESCRIPTIVE and should guide your understanding of the project's purpose.
        
        Repository Name: {repo_name}
        Interpreted Project Type: {project_type}
        
        Repository Name Examples and Their Meanings:
        - "project-analytics-dashboard" → A dashboard for analyzing business metrics and data visualization
        - "inventory-management-system" → System for managing inventory, stock levels, and supply chain
        - "customer-support-portal" → Portal for managing customer support tickets and interactions
        - "employee-timesheet-tracker" → Application for tracking employee work hours and timesheets
        
        Current Description: {basic_info.get('description', 'Generic or missing - USE THE REPO NAME INSTEAD')}
        
        Detected Technical Stack:
        {json.dumps(tech_stack, indent=2)}
        
        Detected Architecture Elements:
        - Models/Entities: {json.dumps(extracted.get('models', []), indent=2)}
        - Controllers: {json.dumps(extracted.get('controllers', []), indent=2)}
        - Components: {json.dumps(extracted.get('components', []), indent=2)}
        - API Endpoints: {json.dumps(extracted.get('api_endpoints', []), indent=2)}
        - Database Tables: {json.dumps(list(set(extracted.get('database_entities', []))), indent=2)}
        
        Based on this analysis, generate a complete architecture specification that includes:
        
        1. Core entities and their relationships
        2. System architecture overview
        3. API design patterns
        4. Frontend component structure
        5. Database schema design
        6. Feature roadmap (infer from existing features)
        7. Design system (if detectable)
        
        Return as JSON matching this schema:
        {{
          "title": "Architecture for {repo_name}",
          "description": "Comprehensive architecture extracted from existing codebase",
          "core_entities": ["List of core business entities"],
          "design_system": {{
            "detected": true/false,
            "suggested_font": {{
              "font_name": "Suggested font",
              "rationale": "Why this font"
            }},
            "color_palette": {{
              "primary": {{"hex": "#HEX", "usage": "Usage description"}},
              "secondary": {{"hex": "#HEX", "usage": "Usage description"}}
            }}
          }},
          "foundational_architecture": {{
            "core_components": {{
              "section_title": "1. Core Components & Rationale",
              "content": "Description of core components and their purposes"
            }},
            "database_schema": {{
              "section_title": "2. Database Schema Design",
              "content": "Description of database structure and relationships"
            }},
            "api_design": {{
              "section_title": "3. API Design & Key Endpoints",
              "content": "API patterns and main endpoints"
            }},
            "frontend_structure": {{
              "section_title": "4. Frontend Structure",
              "content": "Frontend architecture and component hierarchy"
            }}
          }},
          "feature_implementation_roadmap": [
            {{
              "feature_name": "Inferred feature name",
              "description": "What this feature does",
              "status": "existing|planned",
              "components": ["Related components"]
            }}
          ]
        }}
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt)
            
            # Parse AI response
            try:
                import re
                # Handle dict response
                if isinstance(response, dict):
                    response_text = response.get('content', str(response))
                else:
                    response_text = str(response)
                    
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    architecture = json.loads(json_match.group())
                    
                    # Add metadata
                    architecture['generated_from'] = 'existing_codebase'
                    architecture['generation_date'] = datetime.now(timezone.utc).isoformat()
                    architecture['extracted_elements'] = {
                        'models_found': len(extracted.get('models', [])),
                        'controllers_found': len(extracted.get('controllers', [])),
                        'components_found': len(extracted.get('components', []))
                    }
                    
                    return architecture
            except Exception as parse_error:
                self.logger.error(f"Failed to parse AI response: {parse_error}")
                
        except Exception as e:
            self.logger.error(f"AI generation failed for architecture: {e}")
            
        # Fallback to basic architecture
        self.logger.info("Using fallback architecture generation based on code analysis")
        return self._create_basic_architecture(basic_info, tech_stack, extracted)
    
    def _create_basic_architecture(self, 
                                  basic_info: Dict[str, Any],
                                  tech_stack: Dict[str, Any],
                                  extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic architecture from extracted information.
        
        Args:
            basic_info: Basic repository info
            tech_stack: Technology stack
            extracted: Extracted patterns
            
        Returns:
            Basic architecture document
        """
        # Parse repo name for project understanding
        repo_name = basic_info.get('name', 'project')
        name_parts = repo_name.lower().replace('-', ' ').replace('_', ' ').split()
        prefixes_to_remove = ['project', 'app', 'application', 'system', 'platform']
        filtered_parts = [part for part in name_parts if part not in prefixes_to_remove]
        project_type = ' '.join(filtered_parts).title()
        
        # Deduplicate and clean entities
        core_entities = list(set(extracted.get('models', []) + 
                               [e.title() for e in extracted.get('database_entities', [])]))
        
        # Generate description based on repo name
        if project_type:
            generated_description = f"A {project_type} system built with Laravel and React"
        else:
            generated_description = basic_info.get('description', 'Architecture extracted from existing codebase')
        
        return {
            'title': f"Architecture for {project_type or basic_info.get('name', 'Project')}",
            'description': generated_description,
            'core_entities': core_entities[:10],  # Top 10 entities
            'design_system': {
                'detected': False,
                'suggested_font': {
                    'font_name': 'Inter',
                    'rationale': 'Default suggestion for readability'
                },
                'color_palette': {
                    'primary': {'hex': '#3B82F6', 'usage': 'Primary brand color'},
                    'secondary': {'hex': '#10B981', 'usage': 'Secondary accent color'}
                }
            },
            'foundational_architecture': {
                'core_components': {
                    'section_title': '1. Core Components',
                    'content': f"Detected {len(extracted.get('controllers', []))} controllers and {len(extracted.get('components', []))} frontend components"
                },
                'database_schema': {
                    'section_title': '2. Database Schema',
                    'content': f"Database includes tables for: {', '.join(core_entities[:5])}"
                },
                'api_design': {
                    'section_title': '3. API Design',
                    'content': f"RESTful API with endpoints: {', '.join(extracted.get('api_endpoints', [])[:5])}"
                },
                'frontend_structure': {
                    'section_title': '4. Frontend Structure',
                    'content': f"{tech_stack.get('primary_language', 'JavaScript')} frontend with {len(extracted.get('components', []))} components"
                }
            },
            'feature_implementation_roadmap': [
                {
                    'feature_name': f"{entity} Management",
                    'description': f"CRUD operations for {entity}",
                    'status': 'existing',
                    'components': [f"{entity}Controller", f"{entity}Component"]
                }
                for entity in core_entities[:3]
            ],
            'generated_from': 'basic_extraction',
            'generation_date': datetime.now(timezone.utc).isoformat()
        }
    
    async def update_repository_description(self,
                                          repo_full_name: str,
                                          architecture: Dict[str, Any]) -> bool:
        """Update repository description if it's generic.
        
        Args:
            repo_full_name: Repository full name
            architecture: Architecture data with project details
            
        Returns:
            True if updated successfully
        """
        try:
            repo = self.github.get_repo(repo_full_name)
            current_desc = repo.description or ""
            
            # Check if description is generic
            generic_phrases = [
                "Project created from Laravel React starter kit",
                "Forked from",
                "starter kit",
                "template",
                "boilerplate"
            ]
            
            is_generic = any(phrase.lower() in current_desc.lower() for phrase in generic_phrases)
            
            if is_generic or not current_desc:
                # Generate proper description
                new_description = await self._generate_repository_description(
                    repo.name,
                    architecture
                )
                
                if new_description and new_description != current_desc:
                    repo.edit(description=new_description[:350])  # GitHub limit
                    self.logger.info(f"Updated repository description for {repo_full_name}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update repository description: {e}")
            return False
    
    async def _generate_repository_description(self,
                                             repo_name: str,
                                             architecture: Dict[str, Any]) -> str:
        """Generate a proper repository description.
        
        Args:
            repo_name: Repository name (e.g., 'project-analytics-dashboard')
            architecture: Architecture data
            
        Returns:
            Generated description
        """
        # Parse meaningful information from repo name
        name_parts = repo_name.lower().replace('-', ' ').replace('_', ' ').split()
        
        # Remove common prefixes
        prefixes_to_remove = ['project', 'app', 'application', 'system', 'platform']
        filtered_parts = [part for part in name_parts if part not in prefixes_to_remove]
        
        # Reconstruct the meaningful name
        meaningful_name = ' '.join(filtered_parts).title()
        
        if not self.ai_brain:
            # Enhanced fallback using repo name
            self.logger.warning("AI Brain not available, using fallback description generation")
            if filtered_parts:
                return f"A {meaningful_name} built with Laravel and React for modern web applications"
            core_entities = architecture.get('core_entities', [])
            return f"A Laravel React application managing {', '.join(core_entities[:3])}"
            
        prompt = f"""
        Generate a concise, professional repository description for a GitHub project.
        
        IMPORTANT: The repository name is highly descriptive and should be the PRIMARY source for understanding the project.
        
        Repository Name: {repo_name}
        Parsed Project Type: {meaningful_name}
        
        Examples of how to interpret repository names:
        - "project-analytics-dashboard" → "Analytics dashboard for tracking and visualizing business metrics and KPIs"
        - "inventory-management-system" → "Inventory management system for tracking stock levels, orders, and supplier relationships"
        - "task-automation-platform" → "Task automation platform for streamlining workflows and reducing manual processes"
        
        Additional Context (if available):
        Architecture Description: {architecture.get('description', 'Not available')}
        Core Entities: {', '.join(architecture.get('core_entities', [])[:5]) if architecture.get('core_entities') else 'Not detected'}
        
        Requirements:
        - Maximum 350 characters (GitHub limit)
        - USE THE REPOSITORY NAME as the primary guide for what the project does
        - Expand on the name to explain the business value
        - Be specific about capabilities implied by the name
        - Don't mention technical stack unless it's the main feature
        
        Return just the description text, no JSON or formatting.
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt)
            
            # Clean up response
            if isinstance(response, dict):
                description = response.get('content', str(response))
            else:
                description = str(response)
            description = description.strip().strip('"').strip("'")
            
            # Check if it's an error message
            if description.lower().startswith('error') or len(description) < 10:
                raise ValueError(f"Invalid description generated: {description}")
            
            # Ensure it's not too long
            if len(description) > 350:
                description = description[:347] + "..."
                
            return description
            
        except Exception as e:
            self.logger.error(f"AI generation failed for repository description: {e}")
            # Fallback to name-based description
            self.logger.info("Using fallback description based on repository name")
            if filtered_parts:
                return f"A {meaningful_name} built with Laravel and React for modern web applications"
            return "A Laravel React web application"
    
    async def save_architecture_to_repo(self, 
                                       repo_full_name: str,
                                       architecture: Dict[str, Any]) -> bool:
        """Save generated architecture to repository as ARCHITECTURE.md.
        
        Args:
            repo_full_name: Repository full name
            architecture: Architecture data
            
        Returns:
            True if saved successfully
        """
        try:
            repo = self.github.get_repo(repo_full_name)
            
            # Format architecture as markdown
            from scripts.project_creator import ProjectCreator
            creator = ProjectCreator(self.github._Github__requester._Requester__auth.token)
            
            # Create a details dict that matches what the formatter expects
            details = {
                'name': architecture.get('title', 'Project'),
                'description': architecture.get('description', ''),
                'problem_statement': 'Extracted from existing codebase',
                'target_audience': 'Development team',
                'core_entities': architecture.get('core_entities', []),
                'architecture': architecture
            }
            
            content = creator._format_architecture_document(details)
            
            # Create or update ARCHITECTURE.md
            try:
                # Try to get existing file
                existing_file = repo.get_contents("ARCHITECTURE.md")
                repo.update_file(
                    "ARCHITECTURE.md",
                    "Update architecture documentation",
                    content,
                    existing_file.sha
                )
                self.logger.info(f"Updated ARCHITECTURE.md in {repo_full_name}")
            except:
                # File doesn't exist, create it
                repo.create_file(
                    "ARCHITECTURE.md",
                    "Add architecture documentation",
                    content
                )
                self.logger.info(f"Created ARCHITECTURE.md in {repo_full_name}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save architecture: {e}")
            return False