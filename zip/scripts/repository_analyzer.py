"""
Repository Analyzer

Deeply analyzes a specific repository to understand its current state,
needs, and opportunities for improvement.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from github import Github
import os

# Import lifecycle analyzer
try:
    from project_lifecycle_analyzer import ProjectLifecycleAnalyzer
    LIFECYCLE_ANALYZER_AVAILABLE = True
except ImportError:
    LIFECYCLE_ANALYZER_AVAILABLE = False


class RepositoryAnalyzer:
    """Analyzes repositories to understand their specific needs and context."""
    
    def __init__(self, github_token: str = None, ai_brain=None):
        """Initialize repository analyzer.
        
        Args:
            github_token: GitHub personal access token
            ai_brain: AI brain for intelligent analysis
        """
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github = Github(self.github_token) if self.github_token else None
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain
        
        # Initialize lifecycle analyzer if available
        self.lifecycle_analyzer = None
        if LIFECYCLE_ANALYZER_AVAILABLE:
            self.lifecycle_analyzer = ProjectLifecycleAnalyzer(ai_brain)
        
    async def analyze_repository(self, repo_full_name: str) -> Dict[str, Any]:
        """Deeply analyze a specific repository.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            
        Returns:
            Comprehensive analysis of the repository
        """
        self.logger.info(f"Analyzing repository: {repo_full_name}")
        
        try:
            repo = self.github.get_repo(repo_full_name)
            
            analysis = {
                'repository': repo_full_name,
                'analyzed_at': datetime.now(timezone.utc).isoformat(),
                'basic_info': self._get_basic_info(repo),
                'health_metrics': self._analyze_health(repo),
                'code_analysis': await self._analyze_code_structure(repo),
                'issues_analysis': self._analyze_issues(repo),
                'recent_activity': self._analyze_recent_activity(repo),
                'technical_stack': self._identify_tech_stack(repo),
                'architecture': self._get_architecture_document(repo),
                'improvement_opportunities': [],
                'specific_needs': []
            }
            
            # Identify specific needs based on analysis
            analysis['specific_needs'] = self._identify_specific_needs(analysis)
            
            # Find improvement opportunities
            analysis['improvement_opportunities'] = self._find_opportunities(analysis)
            
            # Add lifecycle analysis if available
            if self.lifecycle_analyzer:
                analysis['lifecycle_analysis'] = await self.lifecycle_analyzer.analyze_project_stage(analysis)
            
            # Enhanced analysis with AI if available
            if self.ai_brain:
                analysis['ai_insights'] = await self._generate_ai_insights(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository {repo_full_name}: {e}")
            return {
                'repository': repo_full_name,
                'error': str(e),
                'analyzed_at': datetime.now(timezone.utc).isoformat()
            }
    
    def _get_basic_info(self, repo) -> Dict[str, Any]:
        """Get basic repository information."""
        return {
            'name': repo.name,
            'description': repo.description,
            'language': repo.language,
            'topics': repo.get_topics(),
            'created_at': repo.created_at.isoformat(),
            'updated_at': repo.updated_at.isoformat(),
            'size': repo.size,
            'default_branch': repo.default_branch,
            'has_issues': repo.has_issues,
            'has_wiki': repo.has_wiki,
            'has_pages': repo.has_pages,
            'fork': repo.fork,
            'stargazers_count': repo.stargazers_count,
            'watchers_count': repo.watchers_count,
            'forks_count': repo.forks_count,
            'open_issues_count': repo.open_issues_count
        }
    
    def _analyze_health(self, repo) -> Dict[str, Any]:
        """Analyze repository health metrics."""
        now = datetime.now(timezone.utc)
        
        # Calculate days since last update
        last_update = repo.updated_at
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)
        days_inactive = (now - last_update).days
        
        # Get recent commits
        try:
            commits = list(repo.get_commits(since=now - timedelta(days=30)))
            recent_commit_count = len(commits)
        except:
            recent_commit_count = 0
            
        # Calculate health score
        health_score = 100
        if days_inactive > 30:
            health_score -= 20
        if days_inactive > 90:
            health_score -= 30
        if recent_commit_count < 5:
            health_score -= 10
        if repo.open_issues_count > 20:
            health_score -= 15
            
        return {
            'health_score': max(0, health_score),
            'days_since_update': days_inactive,
            'recent_commits': recent_commit_count,
            'open_issues': repo.open_issues_count,
            'is_active': days_inactive < 30,
            'needs_attention': health_score < 70
        }
    
    async def _analyze_code_structure(self, repo) -> Dict[str, Any]:
        """Analyze code structure and identify patterns."""
        structure = {
            'file_types': {},
            'key_directories': [],
            'config_files': [],
            'documentation': [],
            'test_coverage': 'unknown'
        }
        
        try:
            # Get repository contents
            contents = repo.get_contents("")
            
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    structure['key_directories'].append(file_content.path)
                    # Don't recurse too deep
                    if file_content.path.count('/') < 2:
                        contents.extend(repo.get_contents(file_content.path))
                else:
                    # Track file types
                    ext = file_content.name.split('.')[-1] if '.' in file_content.name else 'none'
                    structure['file_types'][ext] = structure['file_types'].get(ext, 0) + 1
                    
                    # Identify config files
                    config_files = ['package.json', 'composer.json', 'requirements.txt', 
                                  'Gemfile', 'Cargo.toml', '.env.example', 'docker-compose.yml']
                    if file_content.name in config_files:
                        structure['config_files'].append(file_content.path)
                    
                    # Identify documentation
                    if file_content.name.lower() in ['readme.md', 'readme.txt', 'readme']:
                        structure['documentation'].append(file_content.path)
                        
                    # Check for test directories
                    if 'test' in file_content.path.lower() or 'spec' in file_content.path.lower():
                        structure['test_coverage'] = 'has_tests'
                        
        except Exception as e:
            self.logger.warning(f"Error analyzing code structure: {e}")
            
        return structure
    
    def _analyze_issues(self, repo) -> Dict[str, Any]:
        """Analyze repository issues to understand needs."""
        issues_data = {
            'total_open': repo.open_issues_count,
            'bug_count': 0,
            'feature_requests': 0,
            'high_priority': 0,
            'recent_issues': [],
            'common_themes': []
        }
        
        try:
            # Get open issues
            issues = repo.get_issues(state='open')
            
            for issue in issues[:20]:  # Analyze first 20 issues
                issue_info = {
                    'number': issue.number,
                    'title': issue.title,
                    'labels': [label.name for label in issue.labels],
                    'created_at': issue.created_at.isoformat(),
                    'comments': issue.comments
                }
                
                # Categorize issues
                labels_lower = [l.lower() for l in issue_info['labels']]
                if 'bug' in labels_lower:
                    issues_data['bug_count'] += 1
                if any(label in labels_lower for label in ['enhancement', 'feature']):
                    issues_data['feature_requests'] += 1
                if any(label in labels_lower for label in ['urgent', 'critical', 'high']):
                    issues_data['high_priority'] += 1
                    
                issues_data['recent_issues'].append(issue_info)
                
        except Exception as e:
            self.logger.warning(f"Error analyzing issues: {e}")
            
        return issues_data
    
    def _analyze_recent_activity(self, repo) -> Dict[str, Any]:
        """Analyze recent repository activity."""
        try:
            commits = list(repo.get_commits()[:10])
            
            activity = {
                'last_commit': commits[0].commit.author.date.isoformat() if commits else None,
                'recent_commits': len(commits),
                'commit_messages': [c.commit.message.split('\n')[0] for c in commits[:5]],
                'active_contributors': len(set(c.commit.author.name for c in commits if c.commit.author))
            }
            
            return activity
        except:
            return {
                'last_commit': None,
                'recent_commits': 0,
                'commit_messages': [],
                'active_contributors': 0
            }
    
    def _get_architecture_document(self, repo) -> Optional[Dict[str, Any]]:
        """Retrieve and parse the ARCHITECTURE.md document if it exists.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            Parsed architecture data or None
        """
        try:
            # Try to get ARCHITECTURE.md file
            arch_file = repo.get_contents("ARCHITECTURE.md")
            content = arch_file.decoded_content.decode('utf-8')
            
            # Parse the markdown to extract key sections
            architecture = {
                'document_exists': True,
                'raw_content': content,
                'core_entities': [],
                'design_system': {},
                'system_architecture': {},
                'feature_roadmap': [],
                'metadata': {}
            }
            
            # Extract core entities
            if "### Core Entities" in content:
                entities_section = content.split("### Core Entities")[1].split("\n")[1]
                architecture['core_entities'] = [e.strip() for e in entities_section.split(',')]
            
            # Extract design system colors
            if "### Color Palette" in content:
                color_section = content.split("### Color Palette")[1].split("## ")[0]
                colors = {}
                for line in color_section.split('\n'):
                    if line.startswith('- **') and '`#' in line:
                        color_name = line.split('**')[1].lower().replace(' ', '_')
                        hex_value = line.split('`')[1]
                        colors[color_name] = hex_value
                architecture['design_system']['colors'] = colors
            
            # Extract feature roadmap
            if "## Feature Implementation Roadmap" in content:
                roadmap_section = content.split("## Feature Implementation Roadmap")[1]
                features = []
                
                # Simple parsing for feature names
                for line in roadmap_section.split('\n'):
                    if line.startswith('### ') and '. ' in line:
                        feature_name = line.split('. ', 1)[1]
                        features.append(feature_name)
                
                architecture['feature_roadmap'] = features
            
            # Extract metadata
            if "## Metadata" in content:
                metadata_section = content.split("## Metadata")[1].split("---")[0]
                for line in metadata_section.split('\n'):
                    if '**Generated**:' in line:
                        architecture['metadata']['generated'] = line.split(':', 1)[1].strip()
                    elif '**Architecture Version**:' in line:
                        architecture['metadata']['version'] = line.split(':', 1)[1].strip()
            
            self.logger.info(f"Architecture document found and parsed for {repo.full_name}")
            return architecture
            
        except Exception as e:
            self.logger.info(f"No architecture document found for {repo.full_name}: {e}")
            
            # Generate architecture if AI brain is available
            if self.ai_brain:
                self.logger.info(f"Generating architecture for {repo.full_name}")
                try:
                    from scripts.architecture_generator import ArchitectureGenerator
                    generator = ArchitectureGenerator(self.github._Github__requester._Requester__auth.token, self.ai_brain)
                    
                    # Need full analysis for generation
                    # Return None here to avoid circular dependency
                    # Architecture generation should be triggered separately
                    return {
                        'document_exists': False,
                        'generation_available': True,
                        'message': 'Architecture can be generated for this project'
                    }
                except Exception as gen_error:
                    self.logger.error(f"Failed to prepare architecture generation: {gen_error}")
                    
            return None
    
    def _identify_tech_stack(self, repo) -> Dict[str, Any]:
        """Identify technology stack from repository."""
        tech_stack = {
            'primary_language': repo.language,
            'frameworks': [],
            'dependencies': [],
            'infrastructure': []
        }
        
        try:
            # Check for specific files that indicate frameworks
            framework_indicators = {
                'package.json': ['JavaScript', 'Node.js'],
                'composer.json': ['PHP', 'Laravel'],
                'requirements.txt': ['Python'],
                'Gemfile': ['Ruby', 'Rails'],
                'pom.xml': ['Java', 'Maven'],
                'build.gradle': ['Java', 'Gradle'],
                'Cargo.toml': ['Rust'],
                'go.mod': ['Go'],
                'mix.exs': ['Elixir'],
                '.csproj': ['C#', '.NET']
            }
            
            for file_name, techs in framework_indicators.items():
                try:
                    repo.get_contents(file_name)
                    tech_stack['frameworks'].extend(techs)
                except:
                    pass
                    
            # Check for infrastructure files
            infra_files = ['Dockerfile', 'docker-compose.yml', '.github/workflows', 
                          'kubernetes', '.gitlab-ci.yml', 'Jenkinsfile']
            for file_name in infra_files:
                try:
                    repo.get_contents(file_name)
                    tech_stack['infrastructure'].append(file_name)
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Error identifying tech stack: {e}")
            
        return tech_stack
    
    def _identify_specific_needs(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific needs based on analysis."""
        needs = []
        
        # Check for missing architecture documentation
        architecture = analysis.get('architecture', {})
        if not architecture or not architecture.get('document_exists', False):
            needs.append({
                'type': 'architecture_documentation',
                'priority': 'high',
                'description': 'Repository lacks architecture documentation',
                'suggested_action': 'Generate and save ARCHITECTURE.md to document system design',
                'can_generate': architecture.get('generation_available', False)
            })
        
        # Check for generic repository description
        description = analysis.get('basic_info', {}).get('description', '')
        generic_phrases = [
            "Project created from Laravel React starter kit",
            "Forked from",
            "starter kit",
            "template",
            "boilerplate"
        ]
        
        if any(phrase.lower() in description.lower() for phrase in generic_phrases) or not description:
            needs.append({
                'type': 'repository_description',
                'priority': 'medium',
                'description': 'Repository has generic or missing description',
                'suggested_action': 'Update repository description to reflect actual purpose',
                'current_description': description
            })
        
        # Check for missing documentation
        if not analysis['code_analysis']['documentation']:
            needs.append({
                'type': 'documentation',
                'priority': 'medium',
                'description': 'Repository lacks README documentation',
                'suggested_action': 'Create comprehensive README.md with setup instructions'
            })
        
        # Check for missing tests
        if analysis['code_analysis']['test_coverage'] == 'unknown':
            needs.append({
                'type': 'testing',
                'priority': 'high',
                'description': 'No test suite detected',
                'suggested_action': 'Implement unit tests for core functionality'
            })
        
        # Check for high bug count
        if analysis['issues_analysis']['bug_count'] > 5:
            needs.append({
                'type': 'bug_fixes',
                'priority': 'high',
                'description': f"{analysis['issues_analysis']['bug_count']} open bugs need attention",
                'suggested_action': 'Address critical bugs starting with oldest/highest impact'
            })
        
        # Check for stale repository
        if analysis['health_metrics']['days_since_update'] > 60:
            needs.append({
                'type': 'maintenance',
                'priority': 'medium',
                'description': 'Repository has been inactive for over 60 days',
                'suggested_action': 'Update dependencies and ensure project still builds'
            })
        
        # Check for security needs
        if 'package.json' in analysis['code_analysis']['config_files']:
            needs.append({
                'type': 'security',
                'priority': 'high',
                'description': 'JavaScript project may have outdated dependencies',
                'suggested_action': 'Run npm audit and update vulnerable packages'
            })
            
        return needs
    
    def _find_opportunities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find improvement opportunities based on analysis."""
        opportunities = []
        
        # Feature opportunities from issues
        if analysis['issues_analysis']['feature_requests'] > 0:
            opportunities.append({
                'type': 'feature',
                'description': f"{analysis['issues_analysis']['feature_requests']} feature requests from users",
                'impact': 'high',
                'effort': 'variable'
            })
        
        # Performance opportunities
        if analysis['basic_info']['size'] > 100000:  # Large repo
            opportunities.append({
                'type': 'performance',
                'description': 'Large repository may benefit from optimization',
                'impact': 'medium',
                'effort': 'medium'
            })
        
        # CI/CD opportunities
        if '.github/workflows' not in analysis['technical_stack']['infrastructure']:
            opportunities.append({
                'type': 'automation',
                'description': 'Add GitHub Actions for CI/CD',
                'impact': 'high',
                'effort': 'low'
            })
            
        return opportunities
    
    async def _generate_ai_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights about the repository."""
        lifecycle = analysis.get('lifecycle_analysis', {})
        
        prompt = f"""
        Analyze this repository and provide strategic insights:
        
        Repository: {analysis.get('repository')}
        Language: {analysis.get('basic_info', {}).get('language')}
        Age: {lifecycle.get('stage_indicators', {}).get('repository_age_days', 'Unknown')} days
        Lifecycle Stage: {lifecycle.get('current_stage', 'Unknown')}
        Health Score: {analysis.get('health_metrics', {}).get('health_score', 0)}
        
        Key Metrics:
        - Open Issues: {analysis.get('issues_analysis', {}).get('total_open', 0)}
        - Bug Count: {analysis.get('issues_analysis', {}).get('bug_count', 0)}
        - Feature Requests: {analysis.get('issues_analysis', {}).get('feature_requests', 0)}
        - Recent Commits: {analysis.get('health_metrics', {}).get('recent_commits', 0)}
        
        Specific Needs:
        {json.dumps(analysis.get('specific_needs', []), indent=2)}
        
        Provide insights on:
        1. What is the most critical need for this repository right now?
        2. What type of tasks would provide the most value?
        3. Are there any hidden opportunities not captured in the basic analysis?
        4. What should be the immediate priority?
        5. What long-term strategy would benefit this project?
        
        Format as JSON with: critical_need, high_value_tasks, hidden_opportunities, 
        immediate_priority, long_term_strategy
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        return {}