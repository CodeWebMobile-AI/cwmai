#!/usr/bin/env python3
"""
Generate All Documentation Tool

Unified tool to generate README.md, ARCHITECTURE.md, and MONETIZATION_STRATEGY.md
for local repositories. Ensures all documentation is aligned and consistent.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import Logger


class UnifiedDocumentationGenerator:
    """Generate all documentation files for local repositories."""
    
    def __init__(self):
        self.logger = Logger()
        self.ai_client = HTTPAIClient()
        self.local_repos_path = Path("local_repos/repositories/CodeWebMobile-AI")
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Main entry point with operation routing."""
        operations = {
            'generate': self.generate_all_docs,
            'batch': self.batch_generate_docs,
            'check': self.check_documentation_status,
            'update': self.update_existing_docs,
            'list': self.list_repositories
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
            
    async def generate_all_docs(self, project_path: str = None,
                              project_name: str = None,
                              force: bool = False) -> Dict[str, Any]:
        """Generate all documentation files for a single repository."""
        try:
            # Resolve project path
            if project_path:
                project_path = Path(project_path)
            elif project_name:
                project_path = self.local_repos_path / project_name
            else:
                return {'success': False, 'error': 'Specify project_path or project_name'}
                
            if not project_path.exists():
                return {'success': False, 'error': f'Project not found: {project_path}'}
                
            project_name = project_path.name
            self.logger.info(f"Generating documentation for: {project_name}")
            
            # Check existing documentation
            existing_docs = self._check_existing_docs(project_path)
            
            # Collect project information
            project_info = await self._analyze_project_comprehensive(project_path)
            
            results = {
                'project': project_name,
                'path': str(project_path),
                'generated': [],
                'skipped': [],
                'errors': []
            }
            
            # Generate README.md
            if force or not existing_docs['readme']:
                readme_result = await self._generate_readme(project_path, project_info)
                if readme_result['success']:
                    results['generated'].append('README.md')
                else:
                    results['errors'].append(f"README: {readme_result['error']}")
            else:
                results['skipped'].append('README.md (already exists)')
                # Read existing README for context
                project_info['readme_content'] = (project_path / 'README.md').read_text()
                
            # Generate ARCHITECTURE.md
            if force or not existing_docs['architecture']:
                arch_result = await self._generate_architecture(project_path, project_info)
                if arch_result['success']:
                    results['generated'].append('ARCHITECTURE.md')
                else:
                    results['errors'].append(f"ARCHITECTURE: {arch_result['error']}")
            else:
                results['skipped'].append('ARCHITECTURE.md (already exists)')
                # Read existing architecture for context
                project_info['architecture_content'] = (project_path / 'ARCHITECTURE.md').read_text()
                
            # Generate MONETIZATION_STRATEGY.md
            if force or not existing_docs['monetization']:
                # Ensure we have the latest context from generated/existing docs
                if 'readme_content' not in project_info and (project_path / 'README.md').exists():
                    project_info['readme_content'] = (project_path / 'README.md').read_text()
                if 'architecture_content' not in project_info and (project_path / 'ARCHITECTURE.md').exists():
                    project_info['architecture_content'] = (project_path / 'ARCHITECTURE.md').read_text()
                    
                monetization_result = await self._generate_monetization(project_path, project_info)
                if monetization_result['success']:
                    results['generated'].append('MONETIZATION_STRATEGY.md')
                else:
                    results['errors'].append(f"MONETIZATION: {monetization_result['error']}")
            else:
                results['skipped'].append('MONETIZATION_STRATEGY.md (already exists)')
                
            results['success'] = len(results['errors']) == 0
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def batch_generate_docs(self, skip_existing: bool = True,
                                max_repos: int = None) -> Dict[str, Any]:
        """Generate documentation for all repositories in batch."""
        try:
            repos = self._find_local_repositories()
            
            if max_repos:
                repos = repos[:max_repos]
                
            results = {
                'total_repos': len(repos),
                'processed': 0,
                'generated': 0,
                'skipped': 0,
                'errors': [],
                'details': []
            }
            
            for repo_path in repos:
                try:
                    # Check if we should skip
                    if skip_existing:
                        existing = self._check_existing_docs(repo_path)
                        if all(existing.values()):
                            results['skipped'] += 1
                            results['details'].append({
                                'repo': repo_path.name,
                                'status': 'skipped',
                                'reason': 'All docs exist'
                            })
                            continue
                            
                    # Generate docs
                    repo_result = await self.generate_all_docs(
                        project_path=str(repo_path),
                        force=False
                    )
                    
                    results['processed'] += 1
                    if repo_result['success']:
                        results['generated'] += 1
                    else:
                        results['errors'].append(f"{repo_path.name}: {repo_result.get('error')}")
                        
                    results['details'].append({
                        'repo': repo_path.name,
                        'status': 'success' if repo_result['success'] else 'error',
                        'generated': repo_result.get('generated', []),
                        'errors': repo_result.get('errors', [])
                    })
                    
                except Exception as e:
                    results['errors'].append(f"{repo_path.name}: {str(e)}")
                    results['details'].append({
                        'repo': repo_path.name,
                        'status': 'error',
                        'error': str(e)
                    })
                    
            results['success'] = len(results['errors']) == 0
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def check_documentation_status(self) -> Dict[str, Any]:
        """Check documentation status for all repositories."""
        try:
            repos = self._find_local_repositories()
            
            status = {
                'total_repos': len(repos),
                'fully_documented': 0,
                'partially_documented': 0,
                'undocumented': 0,
                'details': []
            }
            
            for repo_path in repos:
                existing = self._check_existing_docs(repo_path)
                doc_count = sum(1 for v in existing.values() if v)
                
                repo_status = {
                    'repo': repo_path.name,
                    'readme': existing['readme'],
                    'architecture': existing['architecture'],
                    'monetization': existing['monetization'],
                    'doc_count': doc_count
                }
                
                if doc_count == 3:
                    status['fully_documented'] += 1
                    repo_status['status'] = 'fully_documented'
                elif doc_count > 0:
                    status['partially_documented'] += 1
                    repo_status['status'] = 'partially_documented'
                else:
                    status['undocumented'] += 1
                    repo_status['status'] = 'undocumented'
                    
                status['details'].append(repo_status)
                
            return {
                'success': True,
                'status': status
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def update_existing_docs(self, project_path: str = None,
                                 project_name: str = None,
                                 doc_type: str = 'all') -> Dict[str, Any]:
        """Update existing documentation with latest project information."""
        try:
            # Resolve project path
            if project_path:
                project_path = Path(project_path)
            elif project_name:
                project_path = self.local_repos_path / project_name
            else:
                return {'success': False, 'error': 'Specify project_path or project_name'}
                
            if not project_path.exists():
                return {'success': False, 'error': f'Project not found: {project_path}'}
                
            # Analyze current project state
            project_info = await self._analyze_project_comprehensive(project_path)
            
            results = {
                'project': project_path.name,
                'updated': [],
                'errors': []
            }
            
            # Update specific or all docs
            if doc_type in ['readme', 'all']:
                if (project_path / 'README.md').exists():
                    readme_result = await self._update_readme(project_path, project_info)
                    if readme_result['success']:
                        results['updated'].append('README.md')
                    else:
                        results['errors'].append(f"README: {readme_result['error']}")
                        
            if doc_type in ['architecture', 'all']:
                if (project_path / 'ARCHITECTURE.md').exists():
                    arch_result = await self._update_architecture(project_path, project_info)
                    if arch_result['success']:
                        results['updated'].append('ARCHITECTURE.md')
                    else:
                        results['errors'].append(f"ARCHITECTURE: {arch_result['error']}")
                        
            if doc_type in ['monetization', 'all']:
                if (project_path / 'MONETIZATION_STRATEGY.md').exists():
                    monetization_result = await self._update_monetization(project_path, project_info)
                    if monetization_result['success']:
                        results['updated'].append('MONETIZATION_STRATEGY.md')
                    else:
                        results['errors'].append(f"MONETIZATION: {monetization_result['error']}")
                        
            results['success'] = len(results['errors']) == 0
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def list_repositories(self) -> Dict[str, Any]:
        """List all local repositories."""
        try:
            repos = self._find_local_repositories()
            
            repo_list = []
            for repo_path in repos:
                existing = self._check_existing_docs(repo_path)
                repo_list.append({
                    'name': repo_path.name,
                    'path': str(repo_path),
                    'has_readme': existing['readme'],
                    'has_architecture': existing['architecture'],
                    'has_monetization': existing['monetization']
                })
                
            return {
                'success': True,
                'repositories': repo_list,
                'total': len(repo_list)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    # Helper methods
    def _find_local_repositories(self) -> List[Path]:
        """Find all local repositories."""
        repos = []
        
        if self.local_repos_path.exists():
            for item in self.local_repos_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    repos.append(item)
                    
        return sorted(repos)
        
    def _check_existing_docs(self, project_path: Path) -> Dict[str, bool]:
        """Check which documentation files exist."""
        return {
            'readme': (project_path / 'README.md').exists(),
            'architecture': (project_path / 'ARCHITECTURE.md').exists(),
            'monetization': (project_path / 'MONETIZATION_STRATEGY.md').exists()
        }
        
    async def _analyze_project_comprehensive(self, project_path: Path) -> Dict[str, Any]:
        """Comprehensive analysis of project for documentation generation."""
        info = {
            'name': project_path.name,
            'path': str(project_path),
            'description': '',
            'tech_stack': [],
            'features': [],
            'structure': {},
            'dependencies': {},
            'metadata': {}
        }
        
        # Read metadata.json if exists
        metadata_path = project_path / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    info['metadata'] = json.load(f)
                    info['description'] = info['metadata'].get('description', '')
            except Exception as e:
                self.logger.error(f"Error reading metadata: {e}")
                
        # Detect tech stack
        if (project_path / 'package.json').exists():
            info['tech_stack'].append('Node.js')
            try:
                with open(project_path / 'package.json', 'r') as f:
                    pkg = json.load(f)
                    info['dependencies']['npm'] = list(pkg.get('dependencies', {}).keys())
            except:
                pass
                
        if (project_path / 'composer.json').exists():
            info['tech_stack'].append('PHP')
            try:
                with open(project_path / 'composer.json', 'r') as f:
                    composer = json.load(f)
                    info['dependencies']['composer'] = list(composer.get('require', {}).keys())
            except:
                pass
                
        if (project_path / 'requirements.txt').exists():
            info['tech_stack'].append('Python')
            try:
                with open(project_path / 'requirements.txt', 'r') as f:
                    info['dependencies']['pip'] = [line.strip() for line in f if line.strip()]
            except:
                pass
                
        # Analyze structure
        info['structure'] = self._analyze_directory_structure(project_path)
        
        # Detect features from file/folder names
        feature_indicators = {
            'auth': 'Authentication system',
            'api': 'API endpoints',
            'database': 'Database integration',
            'admin': 'Admin panel',
            'payment': 'Payment processing',
            'email': 'Email functionality',
            'upload': 'File upload',
            'search': 'Search functionality'
        }
        
        for indicator, description in feature_indicators.items():
            for item in project_path.rglob('*'):
                if indicator in str(item).lower():
                    info['features'].append(description)
                    break
                    
        info['features'] = list(set(info['features']))  # Remove duplicates
        
        return info
        
    def _analyze_directory_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project directory structure."""
        structure = {
            'directories': [],
            'key_files': [],
            'total_files': 0,
            'file_types': {}
        }
        
        for item in project_path.iterdir():
            if item.name.startswith('.'):
                continue
                
            if item.is_dir():
                structure['directories'].append(item.name)
            else:
                structure['total_files'] += 1
                ext = item.suffix.lower()
                if ext:
                    structure['file_types'][ext] = structure['file_types'].get(ext, 0) + 1
                    
                if item.name in ['package.json', 'composer.json', 'requirements.txt',
                               'Dockerfile', 'docker-compose.yml', '.env.example']:
                    structure['key_files'].append(item.name)
                    
        return structure
        
    async def _generate_readme(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate README.md file."""
        try:
            prompt = f"""Create a professional README.md for this project:

Project Name: {project_info['name']}
Description: {project_info.get('description', 'No description provided')}
Tech Stack: {', '.join(project_info['tech_stack']) if project_info['tech_stack'] else 'Not detected'}
Features: {', '.join(project_info['features']) if project_info['features'] else 'To be determined'}

Project Structure:
- Directories: {', '.join(project_info['structure']['directories'][:10])}
- Key Files: {', '.join(project_info['structure']['key_files'])}
- Total Files: {project_info['structure']['total_files']}

Requirements:
1. Professional title and description
2. Clear features list
3. Prerequisites and installation steps
4. Usage instructions with examples
5. Configuration guide
6. API documentation (if applicable)
7. Contributing guidelines
8. License information
9. Contact/support information

Make it comprehensive, well-structured, and developer-friendly."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            readme_content = response.get('content', '')
            
            # Add metadata comment
            readme_content = f"<!-- Generated by Unified Documentation Generator on {datetime.now().isoformat()} -->\n\n{readme_content}"
            
            # Save README
            readme_path = project_path / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(readme_content)
                
            # Store in project info for subsequent generation
            project_info['readme_content'] = readme_content
                
            return {'success': True, 'file': str(readme_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _generate_architecture(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ARCHITECTURE.md file."""
        try:
            # Include README context if available
            readme_context = ""
            if 'readme_content' in project_info:
                readme_context = f"\n\nREADME Context:\n{project_info['readme_content'][:1500]}"
                
            prompt = f"""Create a comprehensive ARCHITECTURE.md for this project:

Project Name: {project_info['name']}
Tech Stack: {', '.join(project_info['tech_stack']) if project_info['tech_stack'] else 'Not detected'}
{readme_context}

Project Structure:
{json.dumps(project_info['structure'], indent=2)}

Dependencies:
{json.dumps(project_info['dependencies'], indent=2) if project_info['dependencies'] else 'None detected'}

Create a technical architecture document that includes:
1. System Overview - high-level architecture
2. Core Components - main modules/services
3. Data Flow - how data moves through the system
4. Database Design - schema and relationships
5. API Design - endpoints and contracts
6. Frontend Architecture - UI components and state management
7. Security Architecture - authentication and authorization
8. Deployment Architecture - infrastructure and scaling
9. Development Workflow - setup and practices
10. Future Considerations - scalability and extensions

Make it technical, detailed, and aligned with the project's actual structure."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            architecture_content = response.get('content', '')
            
            # Add metadata comment
            architecture_content = f"<!-- Generated by Unified Documentation Generator on {datetime.now().isoformat()} -->\n\n{architecture_content}"
            
            # Save Architecture
            arch_path = project_path / 'ARCHITECTURE.md'
            with open(arch_path, 'w') as f:
                f.write(architecture_content)
                
            # Store in project info for subsequent generation
            project_info['architecture_content'] = architecture_content
                
            return {'success': True, 'file': str(arch_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _generate_monetization(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MONETIZATION_STRATEGY.md file with full context."""
        try:
            # Build comprehensive context from README and ARCHITECTURE
            readme_context = ""
            if 'readme_content' in project_info:
                # README provides overview and description
                readme_context = f"""
README Overview:
{project_info['readme_content'][:1000]}
"""
                
            architecture_context = ""
            if 'architecture_content' in project_info:
                # Extract features and components from ARCHITECTURE
                arch_lines = project_info['architecture_content'].split('\n')
                features_section = []
                core_components = []
                api_design = []
                key_features = []
                
                in_features = False
                in_components = False
                in_api = False
                in_key_features = False
                
                for line in arch_lines:
                    # Look for various feature sections
                    if any(marker in line for marker in ['## Features', '## Key Features', '## Feature', '### Key Features']):
                        in_features = True
                        in_components = False
                        in_api = False
                        in_key_features = False
                    elif '## Core Components' in line or '## Key Components' in line:
                        in_components = True
                        in_features = False
                        in_api = False
                        in_key_features = False
                    elif '## API' in line:
                        in_api = True
                        in_features = False
                        in_components = False
                        in_key_features = False
                    elif line.startswith('##'):
                        in_features = False
                        in_components = False
                        in_api = False
                        in_key_features = False
                    elif in_features:
                        features_section.append(line)
                    elif in_components:
                        core_components.append(line)
                    elif in_api:
                        api_design.append(line)
                        
                architecture_context = f"""
Architecture Overview:
Key Features from ARCHITECTURE:
{''.join(features_section[:800])}

Core Components:
{''.join(core_components[:500])}

API Design:
{''.join(api_design[:500])}
"""
                
            prompt = f"""Create a comprehensive MONETIZATION_STRATEGY.md for this project:

Project Name: {project_info['name']}
Description: {project_info.get('description', 'No description provided')}
{readme_context}
{architecture_context}

Based on the documentation above, create a monetization strategy that:

1. **Aligns pricing tiers with ACTUAL features** found in the ARCHITECTURE.md
2. **Leverages the core components** described in ARCHITECTURE
3. **Prices API endpoints and services** that exist in the system
4. **Targets the audience** mentioned in project documentation
5. **IMPORTANT**: Features come from ARCHITECTURE.md, not README.md

Include:
1. Revenue Models - aligned with project capabilities
2. Pricing Tiers - based on actual features from ARCHITECTURE
3. Market Analysis - for the specific solution
4. Growth Projections - realistic for this type of product
5. Implementation Roadmap - leveraging existing architecture
6. Competitive Advantages - based on unique features
7. Risk Mitigation - specific to this business model

Ensure the monetization strategy is realistic and directly tied to the project's actual capabilities."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            monetization_content = response.get('content', '')
            
            # Add metadata comment
            monetization_content = f"<!-- Generated by Unified Documentation Generator on {datetime.now().isoformat()} -->\n<!-- Context: README.md and ARCHITECTURE.md were analyzed for alignment -->\n\n{monetization_content}"
            
            # Save Monetization Strategy
            monetization_path = project_path / 'MONETIZATION_STRATEGY.md'
            with open(monetization_path, 'w') as f:
                f.write(monetization_content)
                
            return {'success': True, 'file': str(monetization_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _update_readme(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing README with new information."""
        try:
            existing_readme = (project_path / 'README.md').read_text()
            
            prompt = f"""Update this existing README.md with new project information:

Current README:
{existing_readme[:2000]}

New Project Information:
- Features detected: {', '.join(project_info['features'])}
- Tech stack: {', '.join(project_info['tech_stack'])}
- New files: {', '.join(project_info['structure']['key_files'])}

Update the README to:
1. Add any new features not mentioned
2. Update installation instructions if needed
3. Add new configuration options
4. Keep existing content that's still relevant
5. Maintain the current style and tone

Return the complete updated README."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            updated_content = response.get('content', '')
            
            # Backup existing
            backup_path = project_path / 'README.md.backup'
            with open(backup_path, 'w') as f:
                f.write(existing_readme)
                
            # Save updated
            with open(project_path / 'README.md', 'w') as f:
                f.write(updated_content)
                
            return {'success': True, 'backup': str(backup_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _update_architecture(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing ARCHITECTURE.md."""
        try:
            existing_arch = (project_path / 'ARCHITECTURE.md').read_text()
            
            prompt = f"""Update this existing ARCHITECTURE.md with new project information:

Current Architecture:
{existing_arch[:2000]}

New Project Structure:
{json.dumps(project_info['structure'], indent=2)}

New Dependencies:
{json.dumps(project_info['dependencies'], indent=2)}

Update the architecture document to:
1. Reflect new components or modules
2. Update technology choices
3. Add new architectural decisions
4. Keep existing valuable content
5. Maintain technical accuracy

Return the complete updated ARCHITECTURE.md."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            updated_content = response.get('content', '')
            
            # Backup and save
            backup_path = project_path / 'ARCHITECTURE.md.backup'
            with open(backup_path, 'w') as f:
                f.write(existing_arch)
                
            with open(project_path / 'ARCHITECTURE.md', 'w') as f:
                f.write(updated_content)
                
            return {'success': True, 'backup': str(backup_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    async def _update_monetization(self, project_path: Path, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing MONETIZATION_STRATEGY.md."""
        try:
            existing_monetization = (project_path / 'MONETIZATION_STRATEGY.md').read_text()
            
            # Read current README and ARCHITECTURE for context
            readme_content = ""
            if (project_path / 'README.md').exists():
                readme_content = (project_path / 'README.md').read_text()[:1500]
                
            arch_content = ""
            if (project_path / 'ARCHITECTURE.md').exists():
                arch_content = (project_path / 'ARCHITECTURE.md').read_text()[:1500]
                
            prompt = f"""Update this existing MONETIZATION_STRATEGY.md based on current project state:

Current Monetization Strategy:
{existing_monetization[:1500]}

Current README features:
{readme_content}

Current Architecture:
{arch_content}

Update the monetization strategy to:
1. Align pricing with current features in README
2. Adjust tiers based on actual architecture
3. Update market analysis if needed
4. Revise projections based on new capabilities
5. Keep successful strategies that still apply

Ensure pricing and features are perfectly aligned with what the project actually offers.

Return the complete updated MONETIZATION_STRATEGY.md."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            updated_content = response.get('content', '')
            
            # Backup and save
            backup_path = project_path / 'MONETIZATION_STRATEGY.md.backup'
            with open(backup_path, 'w') as f:
                f.write(existing_monetization)
                
            with open(project_path / 'MONETIZATION_STRATEGY.md', 'w') as f:
                f.write(updated_content)
                
            return {'success': True, 'backup': str(backup_path)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


async def main(operation: str = None, **kwargs):
    """Tool entry point."""
    generator = UnifiedDocumentationGenerator()
    
    if not operation:
        return {
            'success': True,
            'tool': 'generate_all_docs',
            'description': 'Unified documentation generator for README, ARCHITECTURE, and MONETIZATION files',
            'available_operations': {
                'generate': 'Generate all docs for a single repository',
                'batch': 'Generate docs for multiple repositories',
                'check': 'Check documentation status',
                'update': 'Update existing documentation',
                'list': 'List all repositories'
            },
            'usage_examples': [
                "generate_all_docs(operation='generate', project_name='my-project')",
                "generate_all_docs(operation='batch', skip_existing=True, max_repos=5)",
                "generate_all_docs(operation='check')",
                "generate_all_docs(operation='update', project_name='my-project', doc_type='all')"
            ]
        }
        
    return await generator.execute(operation, **kwargs)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        # Show available operations
        result = await main()
        print("Unified Documentation Generator:", json.dumps(result, indent=2))
        
        # Check documentation status
        result = await main('check')
        if result['success']:
            print(f"\nDocumentation Status:")
            print(f"Total repos: {result['status']['total_repos']}")
            print(f"Fully documented: {result['status']['fully_documented']}")
            print(f"Partially documented: {result['status']['partially_documented']}")
            print(f"Undocumented: {result['status']['undocumented']}")
            
        # Generate docs for a specific project
        # result = await main('generate', project_name='example-project')
        # print(f"\nGenerated docs: {result}")
        
    asyncio.run(demo())