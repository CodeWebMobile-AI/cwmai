#!/usr/bin/env python3
"""
Update All Metadata Script

This script updates all existing metadata.json files in local repositories
with essential metrics only. Uses dynamic parsing that adapts to content structure.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import argparse


class DynamicMetadataUpdater:
    """Updates metadata.json files with dynamically parsed content."""
    
    def __init__(self, base_path: str = "local_repos/repositories"):
        self.base_path = Path(base_path)
        self.file_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript', '.jsx': 'JavaScript',
            '.ts': 'TypeScript', '.tsx': 'TypeScript',
            '.php': 'PHP',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++', '.cc': 'C++', '.cxx': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
    def update_all_repositories(self):
        """Update metadata for all repositories."""
        if not self.base_path.exists():
            print(f"❌ Directory {self.base_path} does not exist")
            return
            
        organizations = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        total_updated = 0
        for org in organizations:
            repos = [r for r in org.iterdir() if r.is_dir() and not r.name.startswith('.')]
            
            for repo in repos:
                print(f"\n📦 Processing {org.name}/{repo.name}...")
                if self.update_repository_metadata(repo):
                    total_updated += 1
                    
        print(f"\n✅ Successfully updated {total_updated} repositories")
        
    def update_repository_metadata(self, repo_path: Path) -> bool:
        """Update metadata for a single repository."""
        metadata_file = repo_path / "metadata.json"
        
        # Load existing metadata or create new
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                print(f"  📄 Loaded existing metadata.json")
            except Exception as e:
                print(f"  ⚠️ Error loading metadata.json: {e}")
                existing_metadata = {}
        else:
            existing_metadata = {}
            print(f"  📝 Creating new metadata.json")
            
        # Start with all existing metadata
        metadata = existing_metadata.copy()
                
        # Update with timestamp
        metadata['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Add the new analyzed fields
        # 1. Check README status
        readme_data = self.check_readme(repo_path)
        metadata['readme'] = readme_data
        
        # 2. Check MONETIZATION_STRATEGY status
        monetization_data = self.check_monetization(repo_path)
        metadata['monetization_strategy'] = monetization_data
        
        # 3. Calculate code metrics
        code_metrics = self.calculate_code_metrics(repo_path)
        metadata['code_metrics'] = code_metrics
        
        # 4. Analyze test coverage
        test_info = self.analyze_tests(repo_path)
        metadata['test_info'] = test_info
        
        # 5. Check documentation status
        doc_status = self.check_documentation(repo_path)
        metadata['documentation'] = doc_status
        
        # Save updated metadata - preserving all existing fields
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Successfully updated metadata.json")
            return True
        except Exception as e:
            print(f"  ❌ Error saving metadata.json: {e}")
            return False
            
    def check_readme(self, repo_path: Path) -> Dict[str, Any]:
        """Check README.md status with dynamic parsing."""
        readme_path = repo_path / "README.md"
        
        result = {
            'exists': readme_path.exists()
        }
        
        if readme_path.exists():
            try:
                result['last_modified'] = datetime.fromtimestamp(
                    readme_path.stat().st_mtime, timezone.utc
                ).isoformat()
                
                content = readme_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                # Basic content metrics
                result['content_summary'] = {
                    'total_lines': len(lines),
                    'total_words': len(content.split()),
                    'has_installation': self._content_has_section(content, ['install', 'setup', 'getting started']),
                    'has_usage': self._content_has_section(content, ['usage', 'example', 'how to']),
                    'has_contributing': self._content_has_section(content, ['contribut']),
                    'todo_count': self._count_pattern(content, r'\btodo\b', case_sensitive=False),
                    'known_issues_count': self._count_pattern(content, r'known\s+issue', case_sensitive=False),
                    'features_count': self._count_pattern(content, r'\bfeatures?\b', case_sensitive=False)
                }
                
                # Dynamic extraction
                result['extracted_data'] = {
                    'project_title': self._extract_title(lines),
                    'description': self._extract_description(lines),
                    'badges': self._extract_badges(content),
                    'target_audience': self._extract_target_audience(content),
                    'tech_stack': self._extract_tech_stack(content),
                    'architecture_overview': self._extract_architecture_overview(content),
                    'prerequisites': self._extract_prerequisites(content),
                    'installation_steps': self._extract_installation_steps(content),
                    'environment_setup': self._extract_environment_setup(content),
                    'usage_examples': self._extract_usage_examples(content),
                    'testing': self._extract_testing_info(content),
                    'deployment': self._extract_deployment_info(content),
                    'features': self._extract_features(content),
                    'how_it_works': self._extract_how_it_works(content),
                    'data_sources': self._extract_data_sources(content),
                    'customization': self._extract_customization(content),
                    'todo_items': self._extract_todos(content),
                    'known_issues': self._extract_known_issues(content),
                    'contributing': {
                        'has_section': self._content_has_section(content, ['contribut']),
                        'guidelines': self._extract_contributing_guidelines(content)
                    },
                    'license': {
                        'mentioned': self._content_has_section(content, ['license']),
                        'type': self._extract_license_type(content)
                    },
                    'disclaimer': self._extract_disclaimer(content),
                    'privacy_policy': self._extract_privacy_policy(content),
                    'links': self._extract_links(content)
                }
                
            except Exception as e:
                print(f"    ⚠️ Error reading README.md: {e}")
                
        return result
        
    def check_monetization(self, repo_path: Path) -> Dict[str, Any]:
        """Check MONETIZATION_STRATEGY.md status with dynamic parsing."""
        monetization_path = repo_path / "MONETIZATION_STRATEGY.md"
        
        result = {
            'exists': monetization_path.exists()
        }
        
        if monetization_path.exists():
            try:
                result['last_modified'] = datetime.fromtimestamp(
                    monetization_path.stat().st_mtime, timezone.utc
                ).isoformat()
                
                content = monetization_path.read_text(encoding='utf-8')
                
                # Dynamic business model detection
                business_type = None
                business_types = ['SUBSCRIPTION', 'SAAS', 'B2B', 'B2C', 'MARKETPLACE', 'FREEMIUM']
                for btype in business_types:
                    if btype.lower() in content.lower():
                        business_type = btype
                        break
                
                result['extracted_data'] = {
                    'revenue_streams': self._extract_revenue_streams(content),
                    'pricing_models': self._extract_pricing_models(content),
                    'target_markets': self._extract_target_markets(content),
                    'business_model': {
                        'type': business_type,
                        'description': self._extract_business_description(content)
                    },
                    'key_metrics': self._extract_key_metrics(content),
                    'monetization_timeline': self._extract_timeline(content)
                }
                
            except Exception as e:
                print(f"    ⚠️ Error reading MONETIZATION_STRATEGY.md: {e}")
                
        return result
        
    def calculate_code_metrics(self, repo_path: Path) -> Dict[str, Any]:
        """Calculate code metrics."""
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'total_size_kb': 0,
            'languages': {},
            'file_types': {},
            'largest_files': [],
            'directory_structure': {
                'max_depth': 0,
                'total_directories': 0
            },
            'code_complexity': {}
        }
        
        # Directories to ignore
        ignore_dirs = {'.git', 'node_modules', 'vendor', '.idea', '__pycache__', 
                      'dist', 'build', '.next', 'coverage', '.pytest_cache'}
        
        max_depth = 0
        dir_count = 0
        
        for file_path in repo_path.rglob('*'):
            # Skip ignored directories
            if any(ignored in file_path.parts for ignored in ignore_dirs):
                continue
                
            if file_path.is_dir():
                dir_count += 1
                depth = len(file_path.relative_to(repo_path).parts)
                max_depth = max(max_depth, depth)
            elif file_path.is_file():
                metrics['total_files'] += 1
                
                try:
                    # File size
                    size_kb = file_path.stat().st_size / 1024
                    metrics['total_size_kb'] += size_kb
                    
                    # Track largest files
                    metrics['largest_files'].append({
                        'path': str(file_path.relative_to(repo_path)),
                        'size_kb': round(size_kb, 2)
                    })
                    
                    # Language detection
                    ext = file_path.suffix.lower()
                    if ext in self.file_extensions:
                        lang = self.file_extensions[ext]
                        metrics['languages'][lang] = metrics['languages'].get(lang, 0) + 1
                    
                    # File type tracking
                    metrics['file_types'][ext] = metrics['file_types'].get(ext, 0) + 1
                    
                    # Count lines for code files
                    if ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.php', '.java', 
                              '.go', '.rs', '.cpp', '.c', '.cs', '.rb', '.swift', '.kt']:
                        try:
                            lines = file_path.read_text(encoding='utf-8').count('\n')
                            metrics['total_lines'] += lines
                        except:
                            pass
                            
                except Exception:
                    pass
        
        # Sort and limit largest files
        metrics['largest_files'] = sorted(
            metrics['largest_files'], 
            key=lambda x: x['size_kb'], 
            reverse=True
        )[:10]
        
        metrics['total_size_kb'] = round(metrics['total_size_kb'], 2)
        metrics['directory_structure']['max_depth'] = max_depth
        metrics['directory_structure']['total_directories'] = dir_count
        
        return metrics
        
    def analyze_tests(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze test information."""
        test_info = {
            'has_tests': False,
            'test_files': [],
            'test_frameworks': [],
            'test_commands': [],
            'coverage_config_exists': False,
            'test_count': 0
        }
        
        # Dynamic test pattern detection
        test_patterns = ['test', 'tests', 'spec', 'specs', '__tests__']
        test_file_patterns = ['test.', 'spec.', '_test.', '.test.', '.spec.']
        
        # Find test files
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                file_str = str(file_path).lower()
                file_name = file_path.name.lower()
                
                # Check if it's a test file
                is_test_file = (
                    any(pattern in file_str for pattern in test_patterns) or
                    any(pattern in file_name for pattern in test_file_patterns)
                )
                
                if is_test_file and not any(ignored in file_str for ignored in ['node_modules', 'vendor']):
                    test_info['test_files'].append(str(file_path.relative_to(repo_path)))
                    test_info['has_tests'] = True
        
        test_info['test_count'] = len(test_info['test_files'])
        
        # Detect test frameworks dynamically
        framework_files = {
            ('phpunit.xml', 'phpunit.xml.dist'): ('phpunit', 'vendor/bin/phpunit'),
            ('jest.config.js', 'jest.config.ts'): ('jest', 'npm test'),
            ('vitest.config.js', 'vitest.config.ts'): ('vitest', 'npm run test'),
            ('pytest.ini', 'setup.cfg', 'tox.ini'): ('pytest', 'pytest'),
            ('.rspec',): ('rspec', 'rspec'),
            ('karma.conf.js',): ('karma', 'npm test')
        }
        
        for file_patterns, (framework, command) in framework_files.items():
            if any((repo_path / pattern).exists() for pattern in file_patterns):
                test_info['test_frameworks'].append(framework)
                test_info['test_commands'].append(command)
                if framework in ['phpunit', 'jest', 'vitest', 'pytest']:
                    test_info['coverage_config_exists'] = True
            
        return test_info
        
    def check_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """Check documentation status."""
        doc_files = {
            'README.md': 'readme',
            'LICENSE': 'license',
            'LICENSE.md': 'license',
            'CONTRIBUTING.md': 'contributing',
            'CODE_OF_CONDUCT.md': 'code_of_conduct',
            'CHANGELOG.md': 'changelog',
            'ARCHITECTURE.md': 'architecture',
            'MONETIZATION_STRATEGY.md': 'monetization',
            '.env.example': 'env_example',
            'docker-compose.yml': 'docker',
            'Dockerfile': 'docker'
        }
        
        documentation = {
            'has_readme': False,
            'has_license': False,
            'has_contributing': False,
            'has_code_of_conduct': False,
            'has_changelog': False,
            'has_architecture': False,
            'has_monetization': False,
            'has_env_example': False,
            'has_docker': False,
            'documentation_files': []
        }
        
        for file_name, doc_type in doc_files.items():
            if (repo_path / file_name).exists():
                documentation[f'has_{doc_type}'] = True
                documentation['documentation_files'].append(file_name)
                
        # Check for docs directory
        docs_dir = repo_path / 'docs'
        if docs_dir.exists() and docs_dir.is_dir():
            documentation['has_docs_directory'] = True
            documentation['docs_file_count'] = len(list(docs_dir.rglob('*.md')))
        else:
            documentation['has_docs_directory'] = False
            documentation['docs_file_count'] = 0
            
        return documentation
    
    # Helper methods for dynamic parsing
    def _content_has_section(self, content: str, keywords: List[str]) -> bool:
        """Check if content has a section containing any of the keywords."""
        content_lower = content.lower()
        # Check for headers with keywords
        for keyword in keywords:
            # Check for markdown headers
            if re.search(rf'^#+\s*.*{keyword}', content_lower, re.MULTILINE | re.IGNORECASE):
                return True
            # Check for bold/emphasis
            if re.search(rf'\*\*.*{keyword}.*\*\*', content_lower, re.IGNORECASE):
                return True
            # Check for general presence in a substantial way
            if content_lower.count(keyword) >= 2:
                return True
        return False
    
    def _count_pattern(self, content: str, pattern: str, case_sensitive: bool = True) -> int:
        """Count occurrences of a pattern in content."""
        flags = 0 if case_sensitive else re.IGNORECASE
        return len(re.findall(pattern, content, flags))
    
    def _extract_title(self, lines: List[str]) -> str:
        """Extract title from first markdown header."""
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('#'):
                return line.strip('#').strip()
        return ''
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract all links from content."""
        links = []
        # Match URLs
        url_pattern = re.compile(r'https?://[^\s\)]+')
        for match in url_pattern.finditer(content):
            url = match.group(0).rstrip('.,;:')  # Clean trailing punctuation
            links.append({'text': url, 'url': url})
        return links[:20]  # Limit to 20 links
    
    def _extract_description(self, lines: List[str]) -> str:
        """Extract description from first few lines after title."""
        description = []
        in_description = False
        
        for i, line in enumerate(lines):
            if i == 0 and line.startswith('#'):
                continue
            elif line.strip() and not line.startswith('#') and not line.startswith('!'):
                description.append(line.strip())
                if len(description) >= 3:
                    break
                    
        return ' '.join(description)
    
    def _extract_badges(self, content: str) -> List[Dict[str, str]]:
        """Extract badges from README."""
        badges = []
        pattern = re.compile(r'!\[([^\]]+)\]\(([^)]+)\)')
        
        for match in pattern.finditer(content):
            badges.append({
                'alt': match.group(1),
                'url': match.group(2)
            })
            
        return badges[:10]
    
    def _extract_installation_steps(self, content: str) -> List[str]:
        """Extract installation steps dynamically."""
        steps = []
        
        # Find installation section using dynamic patterns
        install_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Installation|Setup|Getting Started|Quick Start)',
            r'##?\s*(?:\d+\.\s*)?(?:Install|Set up|Get Started)',
            r'##?\s*(?:\d+\.\s*)?Installation Steps'
        ]
        
        for pattern in install_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract content after the header until next header
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract numbered steps
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                steps.extend(numbered)
                
                # Extract code blocks
                code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', section, re.DOTALL)
                for block in code_blocks:
                    commands = [cmd.strip() for cmd in block.strip().split('\n') 
                              if cmd.strip() and not cmd.strip().startswith('#')]
                    steps.extend(commands)
                    
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                steps.extend(bullets)
                
                break
                
        return steps[:20]
    
    def _extract_usage_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract usage examples."""
        examples = []
        
        usage_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Usage|Examples?|How to Use)',
            r'##?\s*(?:\d+\.\s*)?(?:Quick Example|Sample|Demo)',
            r'##?\s*(?:\d+\.\s*)?(?:Development Workflow)'
        ]
        
        for pattern in usage_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract code blocks as examples
                code_blocks = re.findall(r'```(\w*)\n(.*?)```', section, re.DOTALL)
                for lang, code in code_blocks:
                    examples.append({
                        'language': lang or 'plain',
                        'code': code.strip()
                    })
                    
                break
                
        return examples[:10]
    
    def _extract_features(self, content: str) -> List[str]:
        """Extract features from content."""
        features = []
        
        # Look for features section
        feature_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Features|Key Features|Main Features)',
            r'##?\s*(?:\d+\.\s*)?(?:What.s Included|Capabilities)'
        ]
        
        for pattern in feature_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                features.extend(bullets)
                
                break
                
        return features[:50]
    
    def _extract_todos(self, content: str) -> List[str]:
        """Extract TODO items."""
        todos = []
        
        # Look for TODO comments
        todo_patterns = [
            r'TODO:\s*(.+)',
            r'\\[ \\]\s+(.+)',  # Markdown checkboxes
            r'- \\[ \\]\s+(.+)'
        ]
        
        for pattern in todo_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            todos.extend(matches)
            
        return todos[:20]
    
    def _extract_known_issues(self, content: str) -> List[str]:
        """Extract known issues."""
        issues = []
        
        # Look for issues section
        issue_patterns = [
            r'##?\s*(?:Known Issues|Issues|Problems|Limitations)',
            r'##?\s*(?:Bugs|Current Issues)'
        ]
        
        for pattern in issue_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                issues.extend(bullets)
                
                break
                
        return issues[:20]
    
    def _extract_contributing_guidelines(self, content: str) -> List[str]:
        """Extract contributing guidelines."""
        guidelines = []
        
        contrib_match = re.search(r'##?\s*Contributing', content, re.IGNORECASE)
        if contrib_match:
            start = contrib_match.end()
            next_header = re.search(r'\n##?\s', content[start:])
            end = start + next_header.start() if next_header else len(content)
            
            section = content[start:end]
            bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
            guidelines.extend(bullets)
            
        return guidelines[:10]
    
    def _extract_license_type(self, content: str) -> Optional[str]:
        """Extract license type."""
        license_patterns = [
            r'MIT License',
            r'Apache License',
            r'GPL',
            r'BSD',
            r'ISC License',
            r'Creative Commons'
        ]
        
        for pattern in license_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return pattern
                
        return None
    
    def _extract_target_audience(self, content: str) -> List[str]:
        """Extract target audience information."""
        audience = []
        
        # Look for target audience section
        audience_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Target Audience|Who is this for|For Who|Target Users?)',
            r'##?\s*(?:\d+\.\s*)?(?:Target Audience and Value Proposition)',
            r'##?\s*(?:Why Use|Benefits|Value Proposition)'
        ]
        
        for pattern in audience_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                audience.extend(bullets)
                
                # Extract numbered items
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                audience.extend(numbered)
                
                break
                
        return audience[:10]
    
    def _extract_tech_stack(self, content: str) -> Dict[str, str]:
        """Extract technology stack information."""
        tech_stack = {}
        
        # Look for tech stack section
        tech_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Tech(?:nology)?\s*Stack|Technologies|Built With|Stack)',
            r'##?\s*(?:\d+\.\s*)?(?:Technical Requirements|Technology Used)'
        ]
        
        for pattern in tech_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract key-value pairs (e.g., "Backend: Laravel")
                kv_pattern = r'^\s*[-*+]?\s*\*?\*?([^:]+):\*?\*?\s*(.+)$'
                for match in re.finditer(kv_pattern, section, re.MULTILINE):
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    tech_stack[key] = value
                
                # If no key-value pairs, extract bullet points
                if not tech_stack:
                    bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                    for i, bullet in enumerate(bullets):
                        tech_stack[f'tech_{i+1}'] = bullet
                
                break
                
        return tech_stack
    
    def _extract_architecture_overview(self, content: str) -> str:
        """Extract architecture overview."""
        overview = ""
        
        # Look for architecture section
        arch_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Architecture|System Architecture|Architecture Overview)',
            r'##?\s*(?:\d+\.\s*)?(?:How it Works|System Design|Overview)'
        ]
        
        for pattern in arch_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end].strip()
                
                # Take the first few paragraphs
                paragraphs = []
                for para in section.split('\n\n'):
                    if para.strip() and not para.strip().startswith('#'):
                        paragraphs.append(para.strip())
                        if len(paragraphs) >= 3:
                            break
                
                overview = '\n\n'.join(paragraphs)
                break
                
        return overview[:1500]  # Limit to 1500 chars
    
    def _extract_prerequisites(self, content: str) -> List[str]:
        """Extract prerequisites."""
        prerequisites = []
        
        # Look for prerequisites section
        prereq_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Prerequisites?|Requirements?|Before You Begin)',
            r'##?\s*(?:\d+\.\s*)?(?:System Requirements?|Dependencies)',
            r'##?\s*(?:\d+\.\s*)?(?:Installation)',
            r'##?\s*(?:\d+\.\s*)?(?:Getting Started)'
        ]
        
        for pattern in prereq_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Look for a prerequisites subsection
                prereq_subsection = re.search(r'###?\s*Prerequisites?', section, re.IGNORECASE)
                if prereq_subsection:
                    sub_start = prereq_subsection.end()
                    sub_next = re.search(r'\n###?\s', section[sub_start:])
                    sub_end = sub_start + sub_next.start() if sub_next else len(section)
                    section = section[sub_start:sub_end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                prerequisites.extend(bullets)
                
                # Extract numbered items
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                prerequisites.extend(numbered)
                
                if prerequisites:  # If we found some, don't look further
                    break
                
        return prerequisites[:20]
    
    def _extract_environment_setup(self, content: str) -> List[str]:
        """Extract environment setup information."""
        setup_steps = []
        
        # Look for environment setup section
        env_patterns = [
            r'##?\s*(?:Environment Setup|Environment Variables|Configuration)',
            r'##?\s*(?:Config(?:uration)?|Settings|Setup Environment)'
        ]
        
        for pattern in env_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract code blocks
                code_blocks = re.findall(r'```(?:\w*)\n(.*?)```', section, re.DOTALL)
                for block in code_blocks:
                    setup_steps.append(block.strip())
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                setup_steps.extend(bullets)
                
                break
                
        return setup_steps[:15]
    
    def _extract_testing_info(self, content: str) -> Dict[str, Any]:
        """Extract testing information."""
        testing = {
            'has_section': False,
            'backend_tests': [],
            'frontend_tests': [],
            'commands': []
        }
        
        # Look for testing section
        test_patterns = [
            r'##?\s*(?:Testing|Tests|Running Tests)',
            r'##?\s*(?:Test Suite|Unit Tests|Test Coverage)'
        ]
        
        for pattern in test_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                testing['has_section'] = True
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract test commands
                code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', section, re.DOTALL)
                for block in code_blocks:
                    commands = [cmd.strip() for cmd in block.strip().split('\n') 
                              if cmd.strip() and not cmd.strip().startswith('#')]
                    testing['commands'].extend(commands)
                
                # Look for backend/frontend specific tests
                if 'backend' in section.lower() or 'php' in section.lower():
                    bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                    testing['backend_tests'] = [b for b in bullets if 'backend' in b.lower() or 'php' in b.lower()]
                
                if 'frontend' in section.lower() or 'react' in section.lower() or 'javascript' in section.lower():
                    bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                    testing['frontend_tests'] = [b for b in bullets if 'frontend' in b.lower() or 'react' in b.lower()]
                
                break
                
        return testing
    
    def _extract_deployment_info(self, content: str) -> Dict[str, Any]:
        """Extract deployment information."""
        deployment = {
            'has_section': False,
            'steps': [],
            'platforms': [],
            'containerization': False
        }
        
        # Look for deployment section
        deploy_patterns = [
            r'##?\s*(?:Deployment|Deploy|Production)',
            r'##?\s*(?:Hosting|Going Live|Release)'
        ]
        
        for pattern in deploy_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                deployment['has_section'] = True
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract deployment steps
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                deployment['steps'].extend(bullets)
                
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                deployment['steps'].extend(numbered)
                
                # Check for containerization
                if any(word in section.lower() for word in ['docker', 'container', 'kubernetes', 'k8s']):
                    deployment['containerization'] = True
                
                # Extract platform mentions
                platforms = ['AWS', 'Google Cloud', 'Azure', 'Heroku', 'DigitalOcean', 'Vercel', 'Netlify']
                for platform in platforms:
                    if platform.lower() in section.lower():
                        deployment['platforms'].append(platform)
                
                break
                
        return deployment
    
    def _extract_how_it_works(self, content: str) -> str:
        """Extract how it works information."""
        how_it_works = ""
        
        # Look for how it works section
        how_patterns = [
            r'##?\s*(?:How (?:it|It) Works?|How Does (?:it|It) Work)',
            r'##?\s*(?:Workflow|Process Flow|Architecture)'
        ]
        
        for pattern in how_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end].strip()
                
                # Take the content up to 1000 chars
                how_it_works = section[:1000]
                break
                
        return how_it_works
    
    def _extract_data_sources(self, content: str) -> List[str]:
        """Extract data sources information."""
        sources = []
        
        # Look for data sources section
        source_patterns = [
            r'##?\s*(?:Data Sources?|Data Sets?|APIs)',
            r'##?\s*(?:External (?:Data|Sources)|Integrations)'
        ]
        
        for pattern in source_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                sources.extend(bullets)
                
                break
                
        return sources[:15]
    
    def _extract_customization(self, content: str) -> List[str]:
        """Extract customization options."""
        customizations = []
        
        # Look for customization section
        custom_patterns = [
            r'##?\s*(?:Customization|Configuration|Options)',
            r'##?\s*(?:Settings|Preferences|Personalization)'
        ]
        
        for pattern in custom_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                customizations.extend(bullets)
                
                break
                
        return customizations[:20]
    
    def _extract_disclaimer(self, content: str) -> str:
        """Extract disclaimer information."""
        disclaimer = ""
        
        # Look for disclaimer section
        disclaimer_patterns = [
            r'##?\s*(?:Disclaimer|Legal Notice|Warning)',
            r'##?\s*(?:Important Notice|Caution)'
        ]
        
        for pattern in disclaimer_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end].strip()
                disclaimer = section[:500]  # Limit to 500 chars
                break
                
        return disclaimer
    
    def _extract_privacy_policy(self, content: str) -> str:
        """Extract privacy policy information."""
        privacy = ""
        
        # Look for privacy section
        privacy_patterns = [
            r'##?\s*(?:Privacy Policy|Privacy|Data Privacy)',
            r'##?\s*(?:Data Protection|User Privacy)'
        ]
        
        for pattern in privacy_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end].strip()
                privacy = section[:500]  # Limit to 500 chars
                break
                
        return privacy
    
    # Monetization extraction methods
    def _extract_revenue_streams(self, content: str) -> List[str]:
        """Extract revenue streams."""
        streams = []
        
        # Look for revenue section with flexible numbering
        revenue_patterns = [
            r'\*\*\d+\.\s*Revenue (?:Models?|Streams?):\*\*',
            r'##?\s*(?:\d+\.\s*)?Revenue (?:Models?|Streams?)',
            r'##?\s*(?:\d+\.\s*)?Income Sources?'
        ]
        
        for pattern in revenue_patterns:
            revenue_match = re.search(pattern, content, re.IGNORECASE)
            if revenue_match:
                start = revenue_match.end()
                # Find next numbered section (bold or heading)
                next_section = re.search(r'\n(?:\*\*\d+\.|##?\s*\d+\.)', content[start:])
                end = start + next_section.start() if next_section else len(content)
                
                section = content[start:end]
                
                # Extract bullet points with bold titles
                bold_bullets = re.findall(r'^\s*[-*+]\s+\*\*([^:*]+)(?:\*\*)?:', section, re.MULTILINE)
                for bullet in bold_bullets:
                    clean_stream = bullet.strip()
                    if clean_stream and len(clean_stream) > 3:
                        streams.append(clean_stream)
                
                break
            
        return streams[:10]
    
    def _extract_pricing_models(self, content: str) -> List[str]:
        """Extract pricing models."""
        models = []
        
        # Look for pricing section with flexible numbering
        pricing_patterns = [
            r'\*\*\d+\.\s*Pricing (?:Tiers?|Models?|Plans?):\*\*',
            r'##?\s*(?:\d+\.\s*)?Pricing (?:Tiers?|Models?|Plans?)',
            r'##?\s*(?:\d+\.\s*)?Subscription Plans?'
        ]
        
        for pattern in pricing_patterns:
            pricing_match = re.search(pattern, content, re.IGNORECASE)
            if pricing_match:
                start = pricing_match.end()
                next_section = re.search(r'\n(?:\*\*\d+\.|##?\s*\d+\.)', content[start:])
                end = start + next_section.start() if next_section else len(content)
                
                section = content[start:end]
                
                # Look for table format first (pipe-separated)
                if '|' in section:
                    # Extract tier names from table header
                    lines = section.split('\n')
                    for i, line in enumerate(lines):
                        if '|' in line and i < len(lines) - 1:
                            # Check if next line has pricing info
                            cols = [col.strip() for col in line.split('|')]
                            if len(cols) >= 4:  # At least 4 columns
                                # Skip first column (usually feature name)
                                for col in cols[1:]:
                                    if col and any(word in col.lower() for word in ['free', 'starter', 'basic', 'pro', 'growth', 'enterprise', 'scale']):
                                        models.append(col)
                
                # Extract tier descriptions with bold formatting
                # Match patterns like: *   **Free Tier:** (with multiple spaces)
                tier_patterns = [
                    re.compile(r'^\*\s+\*\*([^:]+)\*\*:', re.MULTILINE),      # Match * **Tier:**
                    re.compile(r'^\s*\*\s+\*\*([^:]+)\*\*:', re.MULTILINE)   # Match with leading spaces
                ]
                
                for tier_pattern in tier_patterns:
                    for match in tier_pattern.finditer(section):
                        tier_name = match.group(1).strip()
                        # Remove "Tier" suffix if present
                        tier_name = re.sub(r'\s+Tier$', '', tier_name, flags=re.IGNORECASE)
                        
                        if any(word in tier_name.lower() for word in ['free', 'starter', 'basic', 'pro', 'professional', 'enterprise', 'premium', 'growth', 'scale']):
                            # Look for price in the next few lines
                            tier_start = match.end()
                            tier_section = section[tier_start:tier_start + 500]  # Look ahead 500 chars
                            price_match = re.search(r'\*\*Price:\*\*\s*([^\n]+)', tier_section)
                            if price_match and price_match.group(1).strip() not in ['Custom pricing based on needs and usage.', 'Custom pricing']:
                                models.append(f"{tier_name}: {price_match.group(1).strip()}")
                            else:
                                models.append(tier_name)
                
                break
                
        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)
                
        return unique_models[:10]
    
    def _extract_target_markets(self, content: str) -> List[str]:
        """Extract target markets."""
        markets = []
        
        # Look for market analysis section
        market_patterns = [
            r'##?\s*(?:\d+\.\s*)?(?:Target Markets?|Target Audience|Market Analysis)',
            r'##?\s*(?:\d+\.\s*)?Customer Segments?'
        ]
        
        for pattern in market_patterns:
            market_match = re.search(pattern, content, re.IGNORECASE)
            if market_match:
                start = market_match.end()
                next_section = re.search(r'\n##?\s*\d+\.', content[start:])
                end = start + next_section.start() if next_section else len(content)
                
                section = content[start:end]
                
                # Extract bullet points that describe markets
                bullets = re.findall(r'^\s*[-*+]\s+([^:\n]+(?:businesses|entrepreneurs|professionals|individuals|companies|organizations|users|customers|agencies|brands)[^:\n]*)', section, re.MULTILINE | re.IGNORECASE)
                for bullet in bullets:
                    clean_market = bullet.strip()
                    if clean_market and len(clean_market) > 10:
                        markets.append(clean_market)
                
                break
                
        # Remove duplicates while preserving order
        seen = set()
        unique_markets = []
        for market in markets:
            if market not in seen:
                seen.add(market)
                unique_markets.append(market)
                
        return unique_markets[:20]
    
    def _extract_business_description(self, content: str) -> str:
        """Extract business model description."""
        model_match = re.search(r'##?\s*(?:Business Model|Business Strategy)', content, re.IGNORECASE)
        if model_match:
            start = model_match.end()
            # Get first paragraph
            lines = content[start:].split('\n\n')
            if lines and lines[0].strip():
                return lines[0].strip()
                
        return ''
    
    def _extract_key_metrics(self, content: str) -> List[Dict[str, str]]:
        """Extract key metrics."""
        metrics = []
        
        # Look for metrics/KPI section or growth projections table
        metrics_patterns = [
            r'\*\*\d+\.\s*(?:Key Metrics|Success Metrics|KPIs?|Performance Indicators|Growth Projections?):\*\*',
            r'##?\s*(?:\d+\.\s*)?(?:Key Metrics|Success Metrics|KPIs?|Performance Indicators)',
            r'##?\s*(?:\d+\.\s*)?Growth Projections?'
        ]
        
        for pattern in metrics_patterns:
            metrics_match = re.search(pattern, content, re.IGNORECASE)
            if metrics_match:
                start = metrics_match.end()
                next_section = re.search(r'\n(?:\*\*\d+\.|##?\s*\d+\.)', content[start:])
                end = start + next_section.start() if next_section else len(content)
                
                section = content[start:end]
                
                # Extract year-based bullet points first
                year_bullets = re.findall(r'^\s*[-*+]\s+\*\*(Year \d+):\*\*\s*(.+)$', section, re.MULTILINE)
                for year, value in year_bullets:
                    metrics.append({
                        'metric': year,
                        'value': value.strip()
                    })
                
                # Try to extract from table format
                if '|' in section and not metrics:
                    # Look for metric rows in table
                    lines = section.split('\n')
                    for line in lines:
                        if '|' in line and not line.strip().startswith('|--'):
                            cols = [col.strip() for col in line.split('|')]
                            if len(cols) >= 4 and cols[1]:  # At least 4 columns with non-empty first col
                                metric = cols[1].strip()
                                if metric and not metric.startswith('-') and metric.lower() not in ['metric', 'year']:
                                    # Try to get year columns
                                    if len(cols) >= 5:
                                        metrics.append({
                                            'metric': metric,
                                            'year1': cols[2].strip(),
                                            'year2': cols[3].strip(),
                                            'year3': cols[4].strip() if len(cols) > 4 else ''
                                        })
                
                # If no metrics found yet, try simple bullet extraction
                if not metrics:
                    bullets = re.findall(r'^\s*[-*+]\s+([^:\n]+):\s*(.+)$', section, re.MULTILINE)
                    for metric, value in bullets:
                        metrics.append({
                            'metric': metric.strip(),
                            'value': value.strip()
                        })
                
                break
                
        return metrics[:15]
    
    def _extract_timeline(self, content: str) -> List[Dict[str, Any]]:
        """Extract timeline/milestones."""
        timeline = []
        
        # Look for timeline/roadmap sections with flexible numbering
        timeline_patterns = [
            r'\*\*\d+\.\s*(?:Implementation )?(?:Timeline|Roadmap|Milestones?):\*\*',
            r'##?\s*(?:\d+\.\s*)?(?:Implementation )?(?:Timeline|Roadmap|Milestones?)',
            r'##?\s*(?:\d+\.\s*)?Development Phases?'
        ]
        
        for pattern in timeline_patterns:
            timeline_match = re.search(pattern, content, re.IGNORECASE)
            if timeline_match:
                start = timeline_match.end()
                next_section = re.search(r'\n(?:\*\*\d+\.|##?\s*\d+\.)', content[start:])
                end = start + next_section.start() if next_section else len(content)
                
                section = content[start:end]
                
                # Look for phase patterns
                phase_pattern = re.compile(r'\*\*Phase\s+(\d+)\s*\(([^)]+)\)\s*:\*\*', re.MULTILINE)
                matches = list(phase_pattern.finditer(section))
                
                if matches:
                    for i, match in enumerate(matches):
                        phase = match.group(1)
                        timeframe = match.group(2)
                        
                        # Get content after this phase header until next phase
                        phase_start = match.end()
                        phase_end = matches[i+1].start() if i+1 < len(matches) else len(section)
                        
                        phase_content = section[phase_start:phase_end]
                        items = re.findall(r'^\s*[-*+]\s+(.+)$', phase_content, re.MULTILINE)
                        
                        timeline.append({
                            'phase': f"Phase {phase}",
                            'timeframe': timeframe,
                            'items': items[:5]  # Limit items per phase
                        })
                else:
                    # Try alternative format with bullet points
                    phase_bullets = re.findall(r'^\s*[-*+]\s+\*\*Phase\s+(\d+)[^:]*:\*\*\s*(.+)', section, re.MULTILINE)
                    for phase, desc in phase_bullets:
                        timeline.append({
                            'phase': f"Phase {phase}",
                            'timeframe': 'TBD',
                            'items': [desc.strip()]
                        })
                
                break
                
        return timeline[:10]
    
    # New extraction methods for missing sections
    def _extract_target_audience(self, content: str) -> List[str]:
        """Extract target audience information."""
        audiences = []
        
        patterns = [
            r'##?\s*(?:Target Audience|Who.?s it for|For Who)',
            r'##?\s*Target Audience and Value Proposition'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+\*\*([^:*]+):\*\*\s*(.+)$', section, re.MULTILINE)
                for audience, desc in bullets:
                    audiences.append(f"{audience.strip()}: {desc.strip()}")
                
                # Also extract regular bullets
                if not audiences:
                    bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                    audiences.extend([b.strip() for b in bullets if len(b.strip()) > 10])
                
                break
                
        return audiences[:10]
    
    def _extract_tech_stack(self, content: str) -> Dict[str, str]:
        """Extract tech stack information."""
        tech_stack = {}
        
        patterns = [
            r'##?\s*(?:Tech Stack|Technology Stack|Technologies)',
            r'##?\s*Stack'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points with technology names
                bullets = re.findall(r'^\s*[-*+]\s+\*\*([^:*]+):\*\*\s*(.+)$', section, re.MULTILINE)
                for tech, desc in bullets:
                    tech_stack[tech.strip()] = desc.strip()
                
                break
                
        return tech_stack
    
    def _extract_architecture_overview(self, content: str) -> str:
        """Extract architecture overview."""
        patterns = [
            r'##?\s*(?:Architecture Overview|System Architecture|Architecture)',
            r'##?\s*Overview'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else start + 2000
                
                section = content[start:end].strip()
                # Return first paragraph or up to 500 chars
                paragraphs = section.split('\n\n')
                return '\n\n'.join(paragraphs[:3])[:1000]
                
        return ''
    
    def _extract_prerequisites(self, content: str) -> List[str]:
        """Extract prerequisites."""
        prereqs = []
        
        patterns = [
            r'##?\s*(?:Prerequisites|Requirements|System Requirements)',
            r'##?\s*Before you begin'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                prereqs.extend([b.strip() for b in bullets])
                
                break
                
        return prereqs[:20]
    
    def _extract_environment_setup(self, content: str) -> List[str]:
        """Extract environment setup steps."""
        setup_steps = []
        
        patterns = [
            r'##?\s*(?:Environment Setup|Configuration|Setup)',
            r'##?\s*(?:Configure|Setting up)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract numbered steps
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                setup_steps.extend(numbered)
                
                # Extract code blocks
                code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', section, re.DOTALL)
                for block in code_blocks:
                    setup_steps.append(f"```\n{block.strip()}\n```")
                
                break
                
        return setup_steps[:30]
    
    def _extract_testing_info(self, content: str) -> Dict[str, Any]:
        """Extract testing information."""
        testing = {
            'has_section': False,
            'backend_tests': [],
            'frontend_tests': [],
            'commands': []
        }
        
        patterns = [
            r'##?\s*(?:Testing|Tests|Running Tests)',
            r'##?\s*Test'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                testing['has_section'] = True
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract test commands
                code_blocks = re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', section, re.DOTALL)
                for block in code_blocks:
                    commands = [cmd.strip() for cmd in block.strip().split('\n') if cmd.strip()]
                    testing['commands'].extend(commands)
                
                # Check for backend/frontend specific tests
                if 'backend' in section.lower():
                    backend_match = re.search(r'backend.*?(?=frontend|\Z)', section, re.IGNORECASE | re.DOTALL)
                    if backend_match:
                        testing['backend_tests'] = re.findall(r'^\s*[-*+]\s+(.+)$', backend_match.group(), re.MULTILINE)
                
                if 'frontend' in section.lower():
                    frontend_match = re.search(r'frontend.*', section, re.IGNORECASE | re.DOTALL)
                    if frontend_match:
                        testing['frontend_tests'] = re.findall(r'^\s*[-*+]\s+(.+)$', frontend_match.group(), re.MULTILINE)
                
                break
                
        return testing
    
    def _extract_deployment_info(self, content: str) -> Dict[str, Any]:
        """Extract deployment information."""
        deployment = {
            'has_section': False,
            'steps': [],
            'platforms': [],
            'containerization': False
        }
        
        patterns = [
            r'##?\s*(?:Deployment|Deploy|Production)',
            r'##?\s*(?:Deploying|How to Deploy)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                deployment['has_section'] = True
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract deployment steps
                numbered = re.findall(r'^\s*\d+\.\s+(.+)$', section, re.MULTILINE)
                deployment['steps'].extend(numbered)
                
                # Check for platforms
                if 'aws' in section.lower():
                    deployment['platforms'].append('AWS')
                if 'google cloud' in section.lower() or 'gcp' in section.lower():
                    deployment['platforms'].append('Google Cloud')
                if 'azure' in section.lower():
                    deployment['platforms'].append('Azure')
                if 'docker' in section.lower():
                    deployment['containerization'] = True
                
                break
                
        return deployment
    
    def _extract_how_it_works(self, content: str) -> str:
        """Extract how it works section."""
        patterns = [
            r'##?\s*(?:How it Works|How Does it Work|Process)',
            r'##?\s*(?:Workflow|Architecture|System Design)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else start + 1500
                
                section = content[start:end].strip()
                return section[:1000]
                
        return ''
    
    def _extract_data_sources(self, content: str) -> List[str]:
        """Extract data sources information."""
        sources = []
        
        patterns = [
            r'##?\s*(?:Data Sources|Sources|APIs)',
            r'##?\s*(?:External Services|Third.?party)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                sources.extend([b.strip() for b in bullets if len(b.strip()) > 10])
                
                break
                
        return sources[:20]
    
    def _extract_customization(self, content: str) -> List[str]:
        """Extract customization details."""
        customizations = []
        
        patterns = [
            r'##?\s*(?:Customization|Customizations|Custom Features)',
            r'##?\s*(?:Key Customizations|Project Customizations)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else len(content)
                
                section = content[start:end]
                
                # Extract bullet points
                bullets = re.findall(r'^\s*[-*+]\s+(.+)$', section, re.MULTILINE)
                customizations.extend([b.strip() for b in bullets if len(b.strip()) > 10])
                
                break
                
        return customizations[:20]
    
    def _extract_disclaimer(self, content: str) -> str:
        """Extract disclaimer section."""
        patterns = [
            r'##?\s*(?:Disclaimer|Legal Notice|Warning)',
            r'##?\s*(?:Important Notice)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else start + 1000
                
                section = content[start:end].strip()
                # Return first paragraph
                paragraphs = section.split('\n\n')
                return paragraphs[0] if paragraphs else section[:500]
                
        return ''
    
    def _extract_privacy_policy(self, content: str) -> str:
        """Extract privacy policy section."""
        patterns = [
            r'##?\s*(?:Privacy Policy|Privacy|Data Privacy)',
            r'##?\s*(?:Privacy Statement)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()
                next_header = re.search(r'\n##?\s', content[start:])
                end = start + next_header.start() if next_header else start + 1000
                
                section = content[start:end].strip()
                # Return first paragraph
                paragraphs = section.split('\n\n')
                return paragraphs[0] if paragraphs else section[:500]
                
        return ''


def main():
    """Main function to run the metadata updater."""
    parser = argparse.ArgumentParser(
        description='Update metadata.json files for all local repositories'
    )
    parser.add_argument(
        '--path', 
        default='local_repos/repositories',
        help='Base path to repositories (default: local_repos/repositories)'
    )
    parser.add_argument(
        '--repo',
        help='Update only a specific repository (e.g., CodeWebMobile-AI/ai-text-summarizer)'
    )
    
    args = parser.parse_args()
    
    updater = DynamicMetadataUpdater(args.path)
    
    if args.repo:
        # Update specific repository
        org, name = args.repo.split('/')
        repo_path = Path(args.path) / org / name
        
        if repo_path.exists():
            print(f"Updating metadata for {args.repo}...")
            updater.update_repository_metadata(repo_path)
        else:
            print(f"❌ Repository {args.repo} not found at {repo_path}")
    else:
        # Update all repositories
        updater.update_all_repositories()


if __name__ == '__main__':
    main()