#!/usr/bin/env python3
"""
Documentation Generator

Comprehensive documentation tool for generating API docs, user guides,
technical documentation, architecture diagrams, and more.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
import subprocess

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import Logger


class DocumentationGenerator:
    """AI-enhanced documentation generation tool."""
    
    def __init__(self):
        self.logger = Logger()
        self.ai_client = HTTPAIClient()
        self.doc_templates = {
            'api': self._get_api_template(),
            'readme': self._get_readme_template(),
            'guide': self._get_guide_template(),
            'architecture': self._get_architecture_template()
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Main entry point with operation routing."""
        operations = {
            'generate_api_docs': self.generate_api_documentation,
            'create_readme': self.create_readme,
            'user_guide': self.create_user_guide,
            'technical_docs': self.create_technical_documentation,
            'architecture_diagram': self.create_architecture_diagram,
            'wiki': self.manage_wiki,
            'changelog': self.generate_changelog,
            'docstring': self.generate_docstrings
        }
        
        if operation not in operations:
            return {
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': list(operations.keys())
            }
            
        try:
            return await operations[operation](**kwargs)
        except Exception as e:
            self.logger.error(f"Error in {operation}: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    async def generate_api_documentation(self, source_path: str = '.',
                                       output_format: str = 'markdown',
                                       include_examples: bool = True) -> Dict[str, Any]:
        """Generate comprehensive API documentation."""
        try:
            # Analyze code to find API endpoints and functions
            api_info = await self._analyze_api_code(source_path)
            
            if not api_info['endpoints'] and not api_info['functions']:
                return {'success': False, 'error': 'No API endpoints or functions found'}
                
            # Generate documentation using AI
            prompt = f"""Generate comprehensive API documentation for this codebase:

API Endpoints:
{json.dumps(api_info['endpoints'], indent=2)}

Functions/Methods:
{json.dumps(api_info['functions'], indent=2)}

Classes:
{json.dumps(api_info['classes'], indent=2)}

Requirements:
1. Clear descriptions for each endpoint/function
2. Parameter documentation with types
3. Return value documentation
4. Error handling documentation
5. Authentication requirements
6. Rate limiting information
7. Example requests/responses: {include_examples}
8. Format: {output_format}

Use professional technical writing style."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            documentation = response.get('content', '')
            
            # Post-process documentation
            documentation = await self._enhance_documentation(documentation, api_info)
            
            # Save documentation
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if output_format == 'markdown':
                doc_file = f'api_documentation_{timestamp}.md'
            elif output_format == 'html':
                doc_file = f'api_documentation_{timestamp}.html'
                documentation = self._convert_to_html(documentation)
            else:
                doc_file = f'api_documentation_{timestamp}.txt'
                
            with open(doc_file, 'w') as f:
                f.write(documentation)
                
            # Generate additional formats if requested
            additional_files = {}
            if kwargs.get('generate_openapi', False):
                openapi_file = await self._generate_openapi_spec(api_info)
                additional_files['openapi'] = openapi_file
                
            if kwargs.get('generate_postman', False):
                postman_file = await self._generate_postman_collection(api_info)
                additional_files['postman'] = postman_file
                
            return {
                'success': True,
                'documentation_file': doc_file,
                'format': output_format,
                'api_summary': {
                    'endpoints': len(api_info['endpoints']),
                    'functions': len(api_info['functions']),
                    'classes': len(api_info['classes'])
                },
                'additional_files': additional_files
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def create_readme(self, project_path: str = '.',
                          template: str = 'standard',
                          sections: List[str] = None) -> Dict[str, Any]:
        """Create or update README file."""
        try:
            # Analyze project
            project_info = await self._analyze_project(project_path)
            
            # Default sections if not specified
            if not sections:
                sections = [
                    'title', 'description', 'features', 'installation',
                    'usage', 'api', 'configuration', 'contributing',
                    'license', 'contact'
                ]
                
            # Generate README using AI
            prompt = f"""Create a professional README.md for this project:

Project Info:
{json.dumps(project_info, indent=2)}

Sections to include:
{json.dumps(sections, indent=2)}

Template style: {template}

Requirements:
1. Clear and concise writing
2. Proper markdown formatting
3. Include badges where appropriate
4. Code examples with syntax highlighting
5. Professional tone
6. SEO-friendly description
7. Clear installation instructions
8. Usage examples
9. Link to documentation

Make it engaging and informative."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            readme_content = response.get('content', '')
            
            # Add badges
            if kwargs.get('add_badges', True):
                badges = self._generate_badges(project_info)
                readme_content = badges + '\n\n' + readme_content
                
            # Save README
            readme_path = os.path.join(project_path, 'README.md')
            
            # Backup existing README
            if os.path.exists(readme_path):
                backup_path = readme_path + '.backup'
                with open(readme_path, 'r') as f:
                    backup_content = f.read()
                with open(backup_path, 'w') as f:
                    f.write(backup_content)
                    
            with open(readme_path, 'w') as f:
                f.write(readme_content)
                
            return {
                'success': True,
                'readme_file': readme_path,
                'backup_file': backup_path if os.path.exists(readme_path + '.backup') else None,
                'sections_included': sections,
                'project_info': project_info,
                'word_count': len(readme_content.split())
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def create_user_guide(self, product_name: str,
                              features: List[str] = None,
                              target_audience: str = 'general',
                              guide_type: str = 'comprehensive') -> Dict[str, Any]:
        """Create user guide documentation."""
        try:
            # Gather information about the product
            if not features:
                # Try to auto-detect features
                features = await self._detect_product_features('.')
                
            # Generate user guide
            prompt = f"""Create a {guide_type} user guide for {product_name}:

Target Audience: {target_audience}

Features to document:
{json.dumps(features, indent=2)}

Requirements:
1. Clear step-by-step instructions
2. Screenshots placeholders [Screenshot: description]
3. Common use cases
4. Troubleshooting section
5. FAQ section
6. Tips and best practices
7. Glossary of terms
8. {target_audience} appropriate language
9. Visual aids descriptions
10. Quick start section

Structure the guide logically with clear headings."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            guide_content = response.get('content', '')
            
            # Create guide structure
            guide_dir = f'{product_name.lower().replace(" ", "_")}_guide'
            os.makedirs(guide_dir, exist_ok=True)
            
            # Save main guide
            guide_file = os.path.join(guide_dir, 'user_guide.md')
            with open(guide_file, 'w') as f:
                f.write(guide_content)
                
            # Create additional sections
            sections_created = []
            
            # Quick start guide
            if kwargs.get('create_quickstart', True):
                quickstart = await self._create_quickstart(product_name, features[:5])
                quickstart_file = os.path.join(guide_dir, 'quickstart.md')
                with open(quickstart_file, 'w') as f:
                    f.write(quickstart)
                sections_created.append('quickstart.md')
                
            # FAQ
            if kwargs.get('create_faq', True):
                faq = await self._create_faq(product_name, features)
                faq_file = os.path.join(guide_dir, 'faq.md')
                with open(faq_file, 'w') as f:
                    f.write(faq)
                sections_created.append('faq.md')
                
            return {
                'success': True,
                'guide_directory': guide_dir,
                'main_guide': guide_file,
                'sections_created': sections_created,
                'features_documented': len(features),
                'target_audience': target_audience,
                'guide_type': guide_type
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def create_technical_documentation(self, source_path: str = '.',
                                           doc_type: str = 'developer',
                                           depth: str = 'detailed') -> Dict[str, Any]:
        """Create technical documentation."""
        try:
            # Analyze codebase
            code_analysis = await self._analyze_codebase(source_path)
            
            # Generate technical documentation
            prompt = f"""Create {depth} {doc_type} documentation for this codebase:

Code Analysis:
{json.dumps(code_analysis, indent=2)[:3000]}

Documentation type: {doc_type}
Depth: {depth}

Requirements:
1. Architecture overview
2. Component descriptions
3. Data flow diagrams (as text descriptions)
4. API references
5. Database schema
6. Configuration options
7. Deployment instructions
8. Development setup
9. Testing guidelines
10. Performance considerations
11. Security considerations
12. Troubleshooting guide

Use technical language appropriate for {doc_type}s."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            tech_docs = response.get('content', '')
            
            # Create documentation structure
            docs_dir = 'technical_docs'
            os.makedirs(docs_dir, exist_ok=True)
            
            # Save main documentation
            main_doc = os.path.join(docs_dir, 'index.md')
            with open(main_doc, 'w') as f:
                f.write(tech_docs)
                
            # Create section files
            sections = {
                'architecture.md': await self._create_architecture_doc(code_analysis),
                'api_reference.md': await self._create_api_reference(code_analysis),
                'database.md': await self._create_database_doc(code_analysis),
                'deployment.md': await self._create_deployment_doc(code_analysis),
                'development.md': await self._create_development_doc(code_analysis)
            }
            
            for filename, content in sections.items():
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
                    
            # Generate table of contents
            toc = self._generate_toc(sections)
            toc_file = os.path.join(docs_dir, 'README.md')
            with open(toc_file, 'w') as f:
                f.write(toc)
                
            return {
                'success': True,
                'documentation_directory': docs_dir,
                'main_document': main_doc,
                'sections': list(sections.keys()),
                'table_of_contents': toc_file,
                'analysis_summary': {
                    'files_analyzed': code_analysis.get('total_files', 0),
                    'components': len(code_analysis.get('components', [])),
                    'complexity': code_analysis.get('complexity', 'medium')
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def create_architecture_diagram(self, source_path: str = '.',
                                        diagram_type: str = 'system',
                                        format: str = 'mermaid') -> Dict[str, Any]:
        """Create architecture diagrams."""
        try:
            # Analyze architecture
            architecture = await self._analyze_architecture(source_path)
            
            # Generate diagram based on type
            if diagram_type == 'system':
                diagram = await self._create_system_diagram(architecture, format)
            elif diagram_type == 'component':
                diagram = await self._create_component_diagram(architecture, format)
            elif diagram_type == 'sequence':
                diagram = await self._create_sequence_diagram(architecture, format)
            elif diagram_type == 'deployment':
                diagram = await self._create_deployment_diagram(architecture, format)
            else:
                return {'success': False, 'error': f'Unknown diagram type: {diagram_type}'}
                
            # Save diagram
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if format == 'mermaid':
                diagram_file = f'architecture_{diagram_type}_{timestamp}.mmd'
            elif format == 'plantuml':
                diagram_file = f'architecture_{diagram_type}_{timestamp}.puml'
            else:
                diagram_file = f'architecture_{diagram_type}_{timestamp}.txt'
                
            with open(diagram_file, 'w') as f:
                f.write(diagram)
                
            # Generate description
            description = await self._generate_diagram_description(
                architecture, diagram_type
            )
            
            desc_file = f'architecture_{diagram_type}_{timestamp}_description.md'
            with open(desc_file, 'w') as f:
                f.write(description)
                
            # Try to render diagram if tools available
            rendered_file = None
            if format == 'mermaid' and kwargs.get('render', False):
                rendered_file = await self._render_mermaid(diagram_file)
                
            return {
                'success': True,
                'diagram_file': diagram_file,
                'description_file': desc_file,
                'diagram_type': diagram_type,
                'format': format,
                'rendered_file': rendered_file,
                'components_included': len(architecture.get('components', [])),
                'relationships': len(architecture.get('relationships', []))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def manage_wiki(self, action: str = 'create',
                        wiki_path: str = 'wiki',
                        content: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage project wiki."""
        try:
            if action == 'create':
                # Create wiki structure
                return await self._create_wiki_structure(wiki_path, content)
                
            elif action == 'update':
                # Update existing wiki
                return await self._update_wiki(wiki_path, content)
                
            elif action == 'generate_page':
                # Generate specific wiki page
                page_title = content.get('title', 'New Page')
                page_content = await self._generate_wiki_page(page_title, content)
                
                page_file = os.path.join(
                    wiki_path,
                    page_title.lower().replace(' ', '_') + '.md'
                )
                
                with open(page_file, 'w') as f:
                    f.write(page_content)
                    
                return {
                    'success': True,
                    'page_file': page_file,
                    'page_title': page_title
                }
                
            elif action == 'index':
                # Generate wiki index
                index = await self._generate_wiki_index(wiki_path)
                index_file = os.path.join(wiki_path, 'Home.md')
                
                with open(index_file, 'w') as f:
                    f.write(index)
                    
                return {
                    'success': True,
                    'index_file': index_file,
                    'pages_indexed': len(list(Path(wiki_path).glob('*.md')))
                }
                
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def generate_changelog(self, repo_path: str = '.',
                               version: str = None,
                               since: str = None) -> Dict[str, Any]:
        """Generate changelog from git history."""
        try:
            # Get git history
            git_log = await self._get_git_history(repo_path, since)
            
            if not git_log:
                return {'success': False, 'error': 'No git history found'}
                
            # Generate changelog using AI
            prompt = f"""Generate a professional CHANGELOG.md from this git history:

Git commits:
{git_log[:5000]}

Version: {version or 'Latest'}

Requirements:
1. Group by version if tags exist
2. Categorize changes (Added, Changed, Fixed, Removed)
3. Clean up commit messages
4. Highlight breaking changes
5. Include contributor acknowledgments
6. Follow Keep a Changelog format
7. Make it user-friendly
8. Include release dates

Format the changelog professionally."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            changelog = response.get('content', '')
            
            # Save changelog
            changelog_file = os.path.join(repo_path, 'CHANGELOG.md')
            
            # Backup existing changelog
            if os.path.exists(changelog_file):
                with open(changelog_file, 'r') as f:
                    existing = f.read()
                with open(changelog_file + '.backup', 'w') as f:
                    f.write(existing)
                    
            with open(changelog_file, 'w') as f:
                f.write(changelog)
                
            # Parse changelog for summary
            summary = self._parse_changelog_summary(changelog)
            
            return {
                'success': True,
                'changelog_file': changelog_file,
                'version': version or 'Latest',
                'changes_summary': summary,
                'commits_processed': len(git_log.split('\n')),
                'backup_created': os.path.exists(changelog_file + '.backup')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def generate_docstrings(self, file_path: str = None,
                                directory: str = None,
                                style: str = 'google') -> Dict[str, Any]:
        """Generate or improve docstrings for Python code."""
        try:
            files_to_process = []
            
            if file_path:
                files_to_process = [file_path]
            elif directory:
                files_to_process = list(Path(directory).rglob('*.py'))
            else:
                return {'success': False, 'error': 'Specify file_path or directory'}
                
            processed_files = []
            total_functions = 0
            total_classes = 0
            
            for file in files_to_process:
                # Skip test files
                if 'test' in str(file).lower():
                    continue
                    
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                        
                    # Parse Python code
                    tree = ast.parse(content)
                    
                    # Extract functions and classes without docstrings
                    items_needing_docs = self._find_undocumented_items(tree)
                    
                    if not items_needing_docs:
                        continue
                        
                    # Generate docstrings using AI
                    updated_content = await self._add_docstrings(
                        content, items_needing_docs, style
                    )
                    
                    # Backup original file
                    backup_file = str(file) + '.backup'
                    with open(backup_file, 'w') as f:
                        f.write(content)
                        
                    # Save updated file
                    with open(file, 'w') as f:
                        f.write(updated_content)
                        
                    processed_files.append(str(file))
                    total_functions += len([i for i in items_needing_docs if i['type'] == 'function'])
                    total_classes += len([i for i in items_needing_docs if i['type'] == 'class'])
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file}: {str(e)}")
                    continue
                    
            return {
                'success': True,
                'files_processed': len(processed_files),
                'functions_documented': total_functions,
                'classes_documented': total_classes,
                'style': style,
                'processed_files': processed_files[:10]  # First 10
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    # Helper methods
    async def _analyze_api_code(self, source_path: str) -> Dict[str, Any]:
        """Analyze code to find API endpoints and functions."""
        api_info = {
            'endpoints': [],
            'functions': [],
            'classes': []
        }
        
        # Analyze Python files
        for py_file in Path(source_path).rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Find Flask/FastAPI endpoints
                endpoint_patterns = [
                    r'@app\.route\([\'"](.+?)[\'"]\)',
                    r'@app\.(get|post|put|delete|patch)\([\'"](.+?)[\'"]',
                    r'@router\.(get|post|put|delete|patch)\([\'"](.+?)[\'"]'
                ]
                
                for pattern in endpoint_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if isinstance(match, tuple):
                            method = match[0]
                            path = match[1]
                        else:
                            method = 'GET'
                            path = match
                            
                        api_info['endpoints'].append({
                            'path': path,
                            'method': method,
                            'file': str(py_file),
                            'description': self._extract_function_description(content, path)
                        })
                        
                # Parse AST for functions and classes
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            api_info['functions'].append({
                                'name': node.name,
                                'file': str(py_file),
                                'args': [arg.arg for arg in node.args.args],
                                'docstring': ast.get_docstring(node),
                                'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')]
                            })
                        elif isinstance(node, ast.ClassDef):
                            api_info['classes'].append({
                                'name': node.name,
                                'file': str(py_file),
                                'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                                'docstring': ast.get_docstring(node)
                            })
                except:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {py_file}: {str(e)}")
                
        # Also check JavaScript files
        for js_file in Path(source_path).rglob('*.js'):
            try:
                with open(js_file, 'r') as f:
                    content = f.read()
                    
                # Find Express routes
                route_pattern = r'(app|router)\.(get|post|put|delete|patch)\([\'"](.+?)[\'"]'
                matches = re.findall(route_pattern, content)
                
                for _, method, path in matches:
                    api_info['endpoints'].append({
                        'path': path,
                        'method': method.upper(),
                        'file': str(js_file),
                        'description': ''
                    })
            except:
                continue
                
        return api_info
        
    def _extract_function_description(self, content: str, path: str) -> str:
        """Extract function description from comments."""
        # Look for comments above the route
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if path in line:
                # Check previous lines for comments
                for j in range(max(0, i-5), i):
                    if '#' in lines[j] or '//' in lines[j]:
                        return lines[j].strip('# /').strip()
        return ''
        
    async def _enhance_documentation(self, documentation: str, 
                                   api_info: Dict[str, Any]) -> str:
        """Enhance documentation with additional information."""
        # Add table of contents
        toc = self._generate_api_toc(api_info)
        
        # Add overview section
        overview = f"""## API Overview

- **Total Endpoints**: {len(api_info['endpoints'])}
- **Total Functions**: {len(api_info['functions'])}
- **Total Classes**: {len(api_info['classes'])}

"""
        
        return toc + '\n\n' + overview + documentation
        
    def _generate_api_toc(self, api_info: Dict[str, Any]) -> str:
        """Generate table of contents for API documentation."""
        toc = "# Table of Contents\n\n"
        
        if api_info['endpoints']:
            toc += "## Endpoints\n"
            for endpoint in api_info['endpoints']:
                toc += f"- [{endpoint['method']} {endpoint['path']}](#{endpoint['path'].replace('/', '')})\n"
                
        if api_info['functions']:
            toc += "\n## Functions\n"
            for func in api_info['functions'][:10]:  # First 10
                toc += f"- [{func['name']}](#{func['name']})\n"
                
        return toc
        
    async def _analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze project for README generation."""
        project_info = {
            'name': os.path.basename(os.path.abspath(project_path)),
            'description': '',
            'language': 'Unknown',
            'dependencies': [],
            'features': [],
            'structure': {}
        }
        
        # Detect project type and language
        if os.path.exists(os.path.join(project_path, 'package.json')):
            project_info['language'] = 'JavaScript/Node.js'
            with open(os.path.join(project_path, 'package.json'), 'r') as f:
                pkg = json.load(f)
                project_info['name'] = pkg.get('name', project_info['name'])
                project_info['description'] = pkg.get('description', '')
                project_info['dependencies'] = list(pkg.get('dependencies', {}).keys())
                
        elif os.path.exists(os.path.join(project_path, 'requirements.txt')):
            project_info['language'] = 'Python'
            with open(os.path.join(project_path, 'requirements.txt'), 'r') as f:
                project_info['dependencies'] = [line.strip() for line in f if line.strip()]
                
        elif os.path.exists(os.path.join(project_path, 'composer.json')):
            project_info['language'] = 'PHP'
            
        # Analyze project structure
        project_info['structure'] = self._analyze_project_structure(project_path)
        
        # Detect features
        project_info['features'] = await self._detect_project_features(project_path)
        
        return project_info
        
    def _analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze project directory structure."""
        structure = {
            'directories': [],
            'key_files': [],
            'test_directory': None,
            'docs_directory': None
        }
        
        for item in os.listdir(project_path):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(project_path, item)
            
            if os.path.isdir(item_path):
                structure['directories'].append(item)
                if 'test' in item.lower():
                    structure['test_directory'] = item
                elif 'doc' in item.lower():
                    structure['docs_directory'] = item
            else:
                if item in ['README.md', 'LICENSE', 'setup.py', 'package.json', 
                           'requirements.txt', 'Makefile', 'Dockerfile']:
                    structure['key_files'].append(item)
                    
        return structure
        
    async def _detect_product_features(self, path: str) -> List[str]:
        """Auto-detect product features from code."""
        features = []
        
        # Look for feature files or modules
        for file_path in Path(path).rglob('*.py'):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Look for class definitions
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Extract feature from class name and docstring
                        feature = node.name
                        if ast.get_docstring(node):
                            feature += f": {ast.get_docstring(node)[:100]}"
                        features.append(feature)
            except:
                continue
                
        return features[:20]  # Return top 20 features
        
    def _generate_badges(self, project_info: Dict[str, Any]) -> str:
        """Generate README badges."""
        badges = []
        
        # Language badge
        lang = project_info['language']
        if 'Python' in lang:
            badges.append('![Python](https://img.shields.io/badge/python-3.7+-blue.svg)')
        elif 'JavaScript' in lang:
            badges.append('![Node.js](https://img.shields.io/badge/node-%3E%3D%2014.0.0-green.svg)')
            
        # License badge
        if 'LICENSE' in project_info['structure']['key_files']:
            badges.append('![License](https://img.shields.io/badge/license-MIT-blue.svg)')
            
        # Version badge
        badges.append('![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)')
        
        return ' '.join(badges)
        
    async def _create_quickstart(self, product_name: str, features: List[str]) -> str:
        """Create quickstart guide."""
        prompt = f"""Create a concise quickstart guide for {product_name}.

Key features to highlight:
{json.dumps(features, indent=2)}

Requirements:
1. 5-minute setup
2. Most basic usage example
3. Clear numbered steps
4. Minimal configuration
5. Link to full documentation

Keep it very brief and actionable."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _create_faq(self, product_name: str, features: List[str]) -> str:
        """Create FAQ section."""
        prompt = f"""Create an FAQ for {product_name}.

Features:
{json.dumps(features, indent=2)}

Generate 10-15 common questions and answers that users might have.
Include technical and non-technical questions."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _analyze_codebase(self, source_path: str) -> Dict[str, Any]:
        """Deep analysis of codebase for technical documentation."""
        analysis = {
            'total_files': 0,
            'languages': {},
            'components': [],
            'dependencies': {},
            'complexity': 'medium',
            'patterns': [],
            'architecture_style': ''
        }
        
        # Count files by language
        for ext in ['*.py', '*.js', '*.java', '*.php', '*.go']:
            files = list(Path(source_path).rglob(ext))
            if files:
                lang = ext[2:]
                analysis['languages'][lang] = len(files)
                analysis['total_files'] += len(files)
                
        # Detect components
        component_indicators = ['controller', 'service', 'model', 'view', 'api', 
                              'database', 'auth', 'utils', 'helpers']
        
        for indicator in component_indicators:
            for file_path in Path(source_path).rglob(f'*{indicator}*'):
                if file_path.is_file():
                    analysis['components'].append({
                        'type': indicator,
                        'file': str(file_path)
                    })
                    
        # Detect architecture patterns
        if any('controller' in str(f) for f in Path(source_path).rglob('*')):
            analysis['architecture_style'] = 'MVC'
        elif any('handler' in str(f) for f in Path(source_path).rglob('*')):
            analysis['architecture_style'] = 'Event-driven'
        else:
            analysis['architecture_style'] = 'Modular'
            
        return analysis
        
    async def _create_architecture_doc(self, analysis: Dict[str, Any]) -> str:
        """Create architecture documentation."""
        prompt = f"""Create architecture documentation based on this analysis:

{json.dumps(analysis, indent=2)[:2000]}

Include:
1. Architecture overview
2. Component descriptions
3. Data flow
4. Technology stack
5. Design patterns used
6. Scalability considerations

Format as technical documentation."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _create_api_reference(self, analysis: Dict[str, Any]) -> str:
        """Create API reference documentation."""
        # Use the analyze_api_code method
        api_info = await self._analyze_api_code('.')
        
        prompt = f"""Create comprehensive API reference documentation:

API Information:
{json.dumps(api_info, indent=2)[:2000]}

Include:
1. Endpoint documentation
2. Request/response formats
3. Authentication
4. Error codes
5. Rate limiting
6. Examples

Format as API reference."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _create_database_doc(self, analysis: Dict[str, Any]) -> str:
        """Create database documentation."""
        # Look for models or schema files
        schema_info = self._find_database_schema('.')
        
        prompt = f"""Create database documentation:

Schema Information:
{json.dumps(schema_info, indent=2)[:1000]}

Include:
1. Database schema
2. Table descriptions
3. Relationships
4. Indexes
5. Data types
6. Constraints

Format as database documentation."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    def _find_database_schema(self, path: str) -> Dict[str, Any]:
        """Find database schema information."""
        schema_info = {
            'tables': [],
            'models': []
        }
        
        # Look for model files
        for model_file in Path(path).rglob('*model*.py'):
            try:
                with open(model_file, 'r') as f:
                    content = f.read()
                    
                # Extract model classes
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if any(base.id in ['Model', 'Base'] for base in node.bases if hasattr(base, 'id')):
                            schema_info['models'].append({
                                'name': node.name,
                                'file': str(model_file)
                            })
            except:
                continue
                
        # Look for migration files
        for migration in Path(path).rglob('*migration*'):
            schema_info['tables'].append(str(migration))
            
        return schema_info
        
    async def _create_deployment_doc(self, analysis: Dict[str, Any]) -> str:
        """Create deployment documentation."""
        # Check for deployment files
        deployment_files = {
            'docker': Path('.').glob('*Dockerfile*'),
            'kubernetes': Path('.').glob('*.yaml'),
            'ci': Path('.').glob('.github/workflows/*')
        }
        
        prompt = """Create deployment documentation including:

1. Prerequisites
2. Environment setup
3. Configuration
4. Deployment steps
5. Docker deployment
6. Cloud deployment options
7. Monitoring setup
8. Troubleshooting

Make it comprehensive and actionable."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _create_development_doc(self, analysis: Dict[str, Any]) -> str:
        """Create development setup documentation."""
        prompt = """Create development setup documentation including:

1. Development environment requirements
2. Getting started steps
3. Running locally
4. Testing procedures
5. Code style guide
6. Git workflow
7. Contributing guidelines
8. Debugging tips

Format for developers."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    def _generate_toc(self, sections: Dict[str, str]) -> str:
        """Generate table of contents."""
        toc = "# Technical Documentation\n\n## Table of Contents\n\n"
        
        for filename, content in sections.items():
            # Extract first heading from content
            lines = content.split('\n')
            title = filename.replace('.md', '').replace('_', ' ').title()
            
            for line in lines:
                if line.startswith('#'):
                    title = line.strip('# ')
                    break
                    
            toc += f"- [{title}]({filename})\n"
            
        return toc
        
    async def _analyze_architecture(self, source_path: str) -> Dict[str, Any]:
        """Analyze system architecture."""
        architecture = {
            'components': [],
            'relationships': [],
            'layers': [],
            'external_services': []
        }
        
        # Find main components
        component_patterns = ['controller', 'service', 'repository', 'model', 
                            'handler', 'manager', 'factory', 'provider']
        
        for pattern in component_patterns:
            for file in Path(source_path).rglob(f'*{pattern}*.py'):
                component_name = file.stem
                architecture['components'].append({
                    'name': component_name,
                    'type': pattern,
                    'file': str(file)
                })
                
        # Detect layers
        if any('controller' in c['type'] for c in architecture['components']):
            architecture['layers'].append('presentation')
        if any('service' in c['type'] for c in architecture['components']):
            architecture['layers'].append('business')
        if any('repository' in c['type'] for c in architecture['components']):
            architecture['layers'].append('data')
            
        # Find relationships by analyzing imports
        for component in architecture['components'][:10]:  # Limit to prevent too much analysis
            try:
                with open(component['file'], 'r') as f:
                    content = f.read()
                    
                # Find imports
                import_lines = [line for line in content.split('\n') if line.startswith('from') or line.startswith('import')]
                
                for line in import_lines:
                    for other in architecture['components']:
                        if other['name'] in line and other['name'] != component['name']:
                            architecture['relationships'].append({
                                'from': component['name'],
                                'to': other['name'],
                                'type': 'depends_on'
                            })
            except:
                continue
                
        return architecture
        
    async def _create_system_diagram(self, architecture: Dict[str, Any], format: str) -> str:
        """Create system architecture diagram."""
        if format == 'mermaid':
            diagram = "graph TB\n"
            
            # Add components
            for i, component in enumerate(architecture['components'][:20]):
                node_id = f"C{i}"
                diagram += f"    {node_id}[{component['name']}]\n"
                
            # Add relationships
            for rel in architecture['relationships'][:20]:
                from_idx = next((i for i, c in enumerate(architecture['components']) if c['name'] == rel['from']), None)
                to_idx = next((i for i, c in enumerate(architecture['components']) if c['name'] == rel['to']), None)
                
                if from_idx is not None and to_idx is not None:
                    diagram += f"    C{from_idx} --> C{to_idx}\n"
                    
            # Add styling
            diagram += "\n    classDef controller fill:#f9f,stroke:#333,stroke-width:2px;\n"
            diagram += "    classDef service fill:#bbf,stroke:#333,stroke-width:2px;\n"
            diagram += "    classDef repository fill:#bfb,stroke:#333,stroke-width:2px;\n"
            
            return diagram
            
        elif format == 'plantuml':
            diagram = "@startuml\n"
            
            # Add components
            for component in architecture['components'][:20]:
                diagram += f"component [{component['name']}] as {component['name'].replace(' ', '_')}\n"
                
            # Add relationships
            for rel in architecture['relationships'][:20]:
                from_name = rel['from'].replace(' ', '_')
                to_name = rel['to'].replace(' ', '_')
                diagram += f"{from_name} --> {to_name}\n"
                
            diagram += "@enduml"
            return diagram
            
        else:
            return "Unsupported format"
            
    async def _create_component_diagram(self, architecture: Dict[str, Any], format: str) -> str:
        """Create component diagram."""
        prompt = f"""Create a {format} component diagram for this architecture:

Components:
{json.dumps(architecture['components'][:20], indent=2)}

Relationships:
{json.dumps(architecture['relationships'][:20], indent=2)}

Create a clear component diagram showing the main components and their relationships."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _create_sequence_diagram(self, architecture: Dict[str, Any], format: str) -> str:
        """Create sequence diagram."""
        if format == 'mermaid':
            diagram = """sequenceDiagram
    participant Client
    participant API
    participant Service
    participant Database
    
    Client->>API: Request
    API->>Service: Process
    Service->>Database: Query
    Database-->>Service: Data
    Service-->>API: Response
    API-->>Client: Result
"""
            return diagram
        else:
            return "Basic sequence flow"
            
    async def _create_deployment_diagram(self, architecture: Dict[str, Any], format: str) -> str:
        """Create deployment diagram."""
        if format == 'mermaid':
            diagram = """graph TB
    subgraph "Production Environment"
        LB[Load Balancer]
        subgraph "Application Servers"
            AS1[App Server 1]
            AS2[App Server 2]
        end
        subgraph "Data Layer"
            DB[(Database)]
            CACHE[(Cache)]
        end
    end
    
    Client[Client] --> LB
    LB --> AS1
    LB --> AS2
    AS1 --> DB
    AS2 --> DB
    AS1 --> CACHE
    AS2 --> CACHE
"""
            return diagram
        else:
            return "Deployment architecture"
            
    async def _generate_diagram_description(self, architecture: Dict[str, Any],
                                          diagram_type: str) -> str:
        """Generate description for architecture diagram."""
        prompt = f"""Generate a clear description for a {diagram_type} architecture diagram.

Architecture info:
- Components: {len(architecture['components'])}
- Relationships: {len(architecture['relationships'])}
- Layers: {architecture['layers']}

Explain what the diagram shows and key architectural decisions."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '')
        
    async def _render_mermaid(self, mermaid_file: str) -> Optional[str]:
        """Try to render Mermaid diagram to image."""
        # This would require mermaid-cli to be installed
        # For now, return None
        return None
        
    async def _create_wiki_structure(self, wiki_path: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial wiki structure."""
        os.makedirs(wiki_path, exist_ok=True)
        
        # Create standard wiki pages
        pages = {
            'Home.md': 'Welcome to the project wiki!',
            'Getting-Started.md': 'Getting started guide',
            'Installation.md': 'Installation instructions',
            'Configuration.md': 'Configuration guide',
            'API-Reference.md': 'API documentation',
            'FAQ.md': 'Frequently asked questions',
            'Troubleshooting.md': 'Troubleshooting guide',
            'Contributing.md': 'Contributing guidelines'
        }
        
        created_pages = []
        
        for page_name, default_content in pages.items():
            page_path = os.path.join(wiki_path, page_name)
            
            # Generate content for each page
            page_content = await self._generate_wiki_page(
                page_name.replace('.md', '').replace('-', ' '),
                content or {'default': default_content}
            )
            
            with open(page_path, 'w') as f:
                f.write(page_content)
                
            created_pages.append(page_name)
            
        # Create sidebar
        sidebar = self._generate_wiki_sidebar(created_pages)
        with open(os.path.join(wiki_path, '_Sidebar.md'), 'w') as f:
            f.write(sidebar)
            
        return {
            'success': True,
            'wiki_path': wiki_path,
            'pages_created': created_pages,
            'sidebar_created': True
        }
        
    async def _update_wiki(self, wiki_path: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing wiki."""
        if not os.path.exists(wiki_path):
            return {'success': False, 'error': 'Wiki path does not exist'}
            
        updated_pages = []
        
        for page_name, page_content in content.items():
            page_file = os.path.join(wiki_path, f"{page_name}.md")
            
            # Backup existing page
            if os.path.exists(page_file):
                with open(page_file, 'r') as f:
                    backup_content = f.read()
                with open(page_file + '.backup', 'w') as f:
                    f.write(backup_content)
                    
            # Write new content
            with open(page_file, 'w') as f:
                f.write(page_content)
                
            updated_pages.append(page_name)
            
        return {
            'success': True,
            'updated_pages': updated_pages,
            'backups_created': True
        }
        
    async def _generate_wiki_page(self, title: str, content: Dict[str, Any]) -> str:
        """Generate content for a wiki page."""
        prompt = f"""Create wiki page content for: {title}

Context:
{json.dumps(content, indent=2)[:1000]}

Create comprehensive wiki page with:
1. Clear structure
2. Helpful examples
3. Links to related pages
4. Practical information

Format as GitHub wiki markdown."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', f'# {title}\n\nContent coming soon...')
        
    async def _generate_wiki_index(self, wiki_path: str) -> str:
        """Generate wiki index/home page."""
        pages = []
        
        for page_file in Path(wiki_path).glob('*.md'):
            if page_file.name not in ['Home.md', '_Sidebar.md']:
                pages.append(page_file.stem.replace('-', ' '))
                
        index = "# Wiki Home\n\nWelcome to the project wiki!\n\n## Pages\n\n"
        
        for page in sorted(pages):
            index += f"- [[{page}]]\n"
            
        index += "\n## Quick Links\n\n"
        index += "- [[Getting Started]]\n"
        index += "- [[API Reference]]\n"
        index += "- [[FAQ]]\n"
        index += "- [[Troubleshooting]]\n"
        
        return index
        
    def _generate_wiki_sidebar(self, pages: List[str]) -> str:
        """Generate wiki sidebar."""
        sidebar = "## Navigation\n\n"
        sidebar += "- [[Home]]\n\n"
        
        sidebar += "### Getting Started\n"
        starter_pages = ['Getting-Started', 'Installation', 'Configuration']
        for page in pages:
            if any(s in page for s in starter_pages):
                page_title = page.replace('.md', '').replace('-', ' ')
                sidebar += f"- [[{page_title}]]\n"
                
        sidebar += "\n### Documentation\n"
        doc_pages = ['API', 'Reference', 'Guide']
        for page in pages:
            if any(d in page for d in doc_pages):
                page_title = page.replace('.md', '').replace('-', ' ')
                sidebar += f"- [[{page_title}]]\n"
                
        sidebar += "\n### Resources\n"
        sidebar += "- [[FAQ]]\n"
        sidebar += "- [[Troubleshooting]]\n"
        sidebar += "- [[Contributing]]\n"
        
        return sidebar
        
    async def _get_git_history(self, repo_path: str, since: str = None) -> str:
        """Get git commit history."""
        try:
            cmd = ['git', 'log', '--pretty=format:%h|%an|%ad|%s', '--date=short']
            
            if since:
                cmd.extend(['--since', since])
            else:
                cmd.extend(['-n', '100'])  # Last 100 commits
                
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return ""
        except:
            return ""
            
    def _parse_changelog_summary(self, changelog: str) -> Dict[str, int]:
        """Parse changelog for summary statistics."""
        summary = {
            'added': 0,
            'changed': 0,
            'fixed': 0,
            'removed': 0
        }
        
        lines = changelog.lower().split('\n')
        for line in lines:
            if line.strip().startswith('- '):
                if 'add' in line:
                    summary['added'] += 1
                elif 'change' in line or 'update' in line:
                    summary['changed'] += 1
                elif 'fix' in line:
                    summary['fixed'] += 1
                elif 'remove' in line or 'delete' in line:
                    summary['removed'] += 1
                    
        return summary
        
    def _find_undocumented_items(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find functions and classes without docstrings."""
        items = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not ast.get_docstring(node):
                    items.append({
                        'type': 'function',
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    items.append({
                        'type': 'class',
                        'name': node.name,
                        'line': node.lineno
                    })
                    
        return items
        
    async def _add_docstrings(self, content: str, items: List[Dict[str, Any]], 
                            style: str) -> str:
        """Add docstrings to undocumented items."""
        lines = content.split('\n')
        
        # Sort items by line number in reverse order to avoid offset issues
        items.sort(key=lambda x: x['line'], reverse=True)
        
        for item in items:
            # Generate docstring
            if item['type'] == 'function':
                docstring = await self._generate_function_docstring(
                    item['name'], item.get('args', []), style
                )
            else:
                docstring = await self._generate_class_docstring(
                    item['name'], style
                )
                
            # Find the line and indentation
            line_idx = item['line'] - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                indent = len(line) - len(line.lstrip())
                
                # Insert docstring after the def/class line
                docstring_lines = docstring.split('\n')
                indented_docstring = []
                
                for doc_line in docstring_lines:
                    if doc_line:
                        indented_docstring.append(' ' * (indent + 4) + doc_line)
                    else:
                        indented_docstring.append('')
                        
                # Insert the docstring
                lines.insert(line_idx + 1, '\n'.join(indented_docstring))
                
        return '\n'.join(lines)
        
    async def _generate_function_docstring(self, name: str, args: List[str], 
                                         style: str) -> str:
        """Generate function docstring."""
        if style == 'google':
            docstring = f'''"""Brief description of {name}.

Args:'''
            for arg in args:
                if arg != 'self':
                    docstring += f'\n    {arg}: Description of {arg}.'
                    
            docstring += '''

Returns:
    Description of return value.
"""'''
            
        elif style == 'numpy':
            docstring = f'''"""
Brief description of {name}.

Parameters
----------'''
            for arg in args:
                if arg != 'self':
                    docstring += f'\n{arg} : type\n    Description of {arg}.'
                    
            docstring += '''

Returns
-------
type
    Description of return value.
"""'''
            
        else:  # sphinx
            docstring = f'''"""Brief description of {name}.'''
            for arg in args:
                if arg != 'self':
                    docstring += f'\n:param {arg}: Description of {arg}.'
            docstring += '\n:return: Description of return value.\n"""'
            
        return docstring
        
    async def _generate_class_docstring(self, name: str, style: str) -> str:
        """Generate class docstring."""
        return f'''"""Brief description of {name} class.

More detailed description if needed.
"""'''
        
    def _convert_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML."""
        # Basic conversion - in production use a proper markdown parser
        # Replace newlines with HTML line breaks
        newline = '\n'
        br_newline = '<br>\n'
        markdown_with_br = markdown.replace(newline, br_newline)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Documentation</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        code {{ 
            background: #f4f4f4; 
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        pre {{ 
            background: #f4f4f4; 
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            margin: 0;
            padding-left: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
{markdown_with_br}
</body>
</html>"""
        return html
        
    async def _generate_openapi_spec(self, api_info: Dict[str, Any]) -> str:
        """Generate OpenAPI specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "API Documentation",
                "version": "1.0.0"
            },
            "paths": {}
        }
        
        for endpoint in api_info['endpoints']:
            path = endpoint['path']
            method = endpoint['method'].lower()
            
            if path not in spec['paths']:
                spec['paths'][path] = {}
                
            spec['paths'][path][method] = {
                "summary": endpoint.get('description', f"{method.upper()} {path}"),
                "responses": {
                    "200": {"description": "Successful response"},
                    "400": {"description": "Bad request"},
                    "500": {"description": "Internal server error"}
                }
            }
            
        spec_file = 'openapi_spec.json'
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)
            
        return spec_file
        
    async def _generate_postman_collection(self, api_info: Dict[str, Any]) -> str:
        """Generate Postman collection."""
        collection = {
            "info": {
                "name": "API Collection",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": []
        }
        
        for endpoint in api_info['endpoints']:
            collection["item"].append({
                "name": f"{endpoint['method']} {endpoint['path']}",
                "request": {
                    "method": endpoint['method'],
                    "url": {
                        "raw": f"{{{{base_url}}}}{endpoint['path']}",
                        "host": ["{{base_url}}"],
                        "path": endpoint['path'].strip('/').split('/')
                    }
                }
            })
            
        collection_file = 'postman_collection.json'
        with open(collection_file, 'w') as f:
            json.dump(collection, f, indent=2)
            
        return collection_file
        
    def _get_api_template(self) -> str:
        """Get API documentation template."""
        return """# API Documentation

## Overview
{overview}

## Authentication
{authentication}

## Endpoints

{endpoints}

## Error Codes
{error_codes}

## Rate Limiting
{rate_limiting}
"""
        
    def _get_readme_template(self) -> str:
        """Get README template."""
        return """# {project_name}

{description}

## Features
{features}

## Installation
{installation}

## Usage
{usage}

## Documentation
{documentation}

## Contributing
{contributing}

## License
{license}
"""
        
    def _get_guide_template(self) -> str:
        """Get user guide template."""
        return """# User Guide

## Getting Started
{getting_started}

## Features
{features}

## How To
{how_to}

## FAQ
{faq}

## Troubleshooting
{troubleshooting}
"""
        
    def _get_architecture_template(self) -> str:
        """Get architecture documentation template."""
        return """# Architecture Documentation

## Overview
{overview}

## Components
{components}

## Data Flow
{data_flow}

## Technology Stack
{tech_stack}

## Design Decisions
{design_decisions}
"""
        
    async def _detect_project_features(self, path: str) -> List[str]:
        """Detect project features from code analysis."""
        features = []
        
        # Look for common feature indicators
        feature_indicators = {
            'auth': 'User authentication and authorization',
            'api': 'RESTful API endpoints',
            'database': 'Database integration',
            'cache': 'Caching functionality',
            'email': 'Email notifications',
            'upload': 'File upload capabilities',
            'search': 'Search functionality',
            'admin': 'Admin dashboard',
            'payment': 'Payment processing',
            'webhook': 'Webhook support'
        }
        
        for indicator, description in feature_indicators.items():
            for file in Path(path).rglob('*'):
                if file.is_file() and indicator in str(file).lower():
                    features.append(description)
                    break
                    
        return list(set(features))  # Remove duplicates


async def main(operation: str = None, **kwargs):
    """Tool entry point."""
    generator = DocumentationGenerator()
    
    if not operation:
        return {
            'success': True,
            'available_operations': [
                'generate_api_docs', 'create_readme', 'user_guide', 'technical_docs',
                'architecture_diagram', 'wiki', 'changelog', 'docstring'
            ],
            'description': 'Comprehensive documentation generation tool'
        }
        
    return await generator.execute(operation, **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        # Show available operations
        result = await main()
        print("Documentation Generator Operations:", result)
        
        # Generate API documentation
        result = await main('generate_api_docs', output_format='markdown')
        print(f"\nGenerated API docs: {result.get('documentation_file')}")
        
        # Create README
        result = await main('create_readme', template='standard')
        print(f"\nCreated README: {result.get('readme_file')}")
        
    asyncio.run(demo())