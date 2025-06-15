#!/usr/bin/env python3
"""
Fix Repository Customizations

This script finds and fixes repositories that have:
1. Generic or missing descriptions
2. Missing ARCHITECTURE.md files
3. Generic README.md files
4. Missing customization from the starter kit

It can run in check-only mode or actually apply fixes.
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from github import Github, GithubException
import base64
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryCustomizationFixer:
    """Finds and fixes repositories with missing customizations."""
    
    GENERIC_DESCRIPTIONS = [
        "Project created from Laravel React starter kit",
        "Forked from",
        "starter kit",
        "template",
        "boilerplate",
        "Created by AI",
        "New project"
    ]
    
    EXCLUDED_REPOS = ['cwmai', '.github', 'cwmai.git']
    ORGANIZATION = "CodeWebMobile-AI"
    
    def __init__(self, github_token: str, organization: str = None, ai_brain=None):
        """Initialize with GitHub client and optional AI brain.
        
        Args:
            github_token: GitHub personal access token
            organization: GitHub organization name (optional, defaults to ORGANIZATION)
            ai_brain: AI brain for generating content (optional)
        """
        self.github = Github(github_token)
        self.github_token = github_token
        self.organization = organization or self.ORGANIZATION
        self.ai_brain = ai_brain
        self.logger = logger
        self.issues_found = []
        self.fixes_applied = []
        
    async def scan_repositories(self, fix_mode: bool = False) -> Dict[str, Any]:
        """Scan all repositories for customization issues.
        
        Args:
            fix_mode: If True, apply fixes. If False, only report issues.
            
        Returns:
            Summary of issues found and fixes applied
        """
        self.logger.info(f"🔍 Starting repository scan (fix_mode={fix_mode})")
        
        # Get organization
        try:
            org = self.github.get_organization(self.organization)
            self.logger.info(f"✓ Connected to organization: {self.organization}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to organization: {e}")
            return {'error': str(e)}
        
        # Scan all repositories
        total_repos = 0
        repos_with_issues = 0
        
        for repo in org.get_repos():
            if repo.name in self.EXCLUDED_REPOS:
                self.logger.info(f"⏭️  Skipping excluded repository: {repo.name}")
                continue
                
            total_repos += 1
            self.logger.info(f"\n📦 Checking repository: {repo.name}")
            
            # Check for issues
            issues = await self._check_repository_issues(repo)
            
            if issues:
                repos_with_issues += 1
                self.issues_found.append({
                    'repository': repo.full_name,
                    'issues': issues,
                    'checked_at': datetime.now(timezone.utc).isoformat()
                })
                
                self.logger.warning(f"  ⚠️  Found {len(issues)} issues:")
                for issue in issues:
                    self.logger.warning(f"    - {issue['type']}: {issue['description']}")
                
                # Apply fixes if in fix mode
                if fix_mode:
                    self.logger.info(f"  🔧 Applying fixes...")
                    fixes = await self._fix_repository_issues(repo, issues)
                    
                    if fixes:
                        self.fixes_applied.append({
                            'repository': repo.full_name,
                            'fixes': fixes,
                            'fixed_at': datetime.now(timezone.utc).isoformat()
                        })
                        
                        self.logger.info(f"  ✅ Applied {len(fixes)} fixes:")
                        for fix in fixes:
                            self.logger.info(f"    - {fix['type']}: {fix['status']}")
            else:
                self.logger.info(f"  ✅ No issues found - repository is properly customized")
        
        # Generate summary
        summary = {
            'scan_completed': datetime.now(timezone.utc).isoformat(),
            'total_repositories': total_repos,
            'repositories_with_issues': repos_with_issues,
            'total_issues_found': sum(len(r['issues']) for r in self.issues_found),
            'total_fixes_applied': sum(len(r['fixes']) for r in self.fixes_applied) if fix_mode else 0,
            'fix_mode': fix_mode,
            'issues_by_repository': self.issues_found,
            'fixes_by_repository': self.fixes_applied if fix_mode else []
        }
        
        # Save detailed report
        report_filename = f"repository_customization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\n📊 Scan complete. Report saved to: {report_filename}")
        
        return summary
    
    async def _check_repository_issues(self, repo) -> List[Dict[str, Any]]:
        """Check a repository for customization issues.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Check 1: Generic or missing description
        if not repo.description or any(phrase.lower() in repo.description.lower() for phrase in self.GENERIC_DESCRIPTIONS):
            issues.append({
                'type': 'generic_description',
                'description': f'Repository has generic description: "{repo.description}"',
                'current_value': repo.description,
                'severity': 'medium'
            })
        
        # Check 2: Missing ARCHITECTURE.md
        try:
            repo.get_contents("ARCHITECTURE.md")
        except GithubException:
            issues.append({
                'type': 'missing_architecture',
                'description': 'ARCHITECTURE.md file is missing',
                'severity': 'high'
            })
        
        # Check 3: Generic README
        try:
            readme = repo.get_contents("README.md")
            readme_content = readme.decoded_content.decode('utf-8')
            
            # Check for generic content indicators
            generic_indicators = [
                "Laravel React Starter Kit",
                "Laravel + React Starter Kit",
                "React starter kit",
                "This is a starter kit",
                "TODO: Add project description",
                "Your application description",
                "Our React starter kit provides",
                "Documentation for all Laravel starter kits"
            ]
            
            if any(indicator.lower() in readme_content.lower() for indicator in generic_indicators):
                issues.append({
                    'type': 'generic_readme',
                    'description': 'README.md contains generic starter kit content',
                    'severity': 'medium'
                })
                
            # Check if README is too short (less than 500 characters)
            if len(readme_content) < 500:
                issues.append({
                    'type': 'incomplete_readme',
                    'description': f'README.md is too short ({len(readme_content)} characters)',
                    'severity': 'medium'
                })
                
        except GithubException:
            issues.append({
                'type': 'missing_readme',
                'description': 'README.md file is missing',
                'severity': 'high'
            })
        
        # Check 4: Missing customization in package.json
        try:
            package_file = repo.get_contents("package.json")
            package_data = json.loads(package_file.decoded_content)
            
            if package_data.get('name') == 'laravel-react-starter' or \
               'starter' in package_data.get('name', '').lower():
                issues.append({
                    'type': 'generic_package_name',
                    'description': f'package.json has generic name: "{package_data.get("name")}"',
                    'current_value': package_data.get('name'),
                    'severity': 'low'
                })
                
        except GithubException:
            pass  # package.json might not exist in all repos
        except json.JSONDecodeError:
            issues.append({
                'type': 'invalid_package_json',
                'description': 'package.json contains invalid JSON',
                'severity': 'medium'
            })
        
        # Check 5: Missing topics/tags
        topics = repo.get_topics()
        if not topics:
            issues.append({
                'type': 'missing_topics',
                'description': 'Repository has no topics/tags',
                'severity': 'low'
            })
        
        return issues
    
    async def _fix_repository_issues(self, repo, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix identified issues in a repository.
        
        Args:
            repo: GitHub repository object
            issues: List of issues to fix
            
        Returns:
            List of fixes applied
        """
        fixes = []
        
        # Group issues by type for efficient fixing
        issues_by_type = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Fix generic description
        if 'generic_description' in issues_by_type:
            fix_result = await self._fix_repository_description(repo)
            fixes.append({
                'type': 'description_updated',
                'status': 'success' if fix_result else 'failed',
                'details': fix_result
            })
        
        # Fix missing architecture
        if 'missing_architecture' in issues_by_type:
            fix_result = await self._generate_and_save_architecture(repo)
            fixes.append({
                'type': 'architecture_generated',
                'status': 'success' if fix_result else 'failed',
                'details': fix_result
            })
        
        # Fix generic or incomplete README
        if any(issue_type in issues_by_type for issue_type in ['generic_readme', 'incomplete_readme', 'missing_readme']):
            fix_result = await self._fix_readme(repo)
            fixes.append({
                'type': 'readme_updated',
                'status': 'success' if fix_result else 'failed',
                'details': fix_result
            })
        
        # Fix package.json
        if 'generic_package_name' in issues_by_type:
            fix_result = await self._fix_package_json(repo)
            fixes.append({
                'type': 'package_json_updated',
                'status': 'success' if fix_result else 'failed',
                'details': fix_result
            })
        
        # Fix missing topics
        if 'missing_topics' in issues_by_type:
            fix_result = await self._add_repository_topics(repo)
            fixes.append({
                'type': 'topics_added',
                'status': 'success' if fix_result else 'failed',
                'details': fix_result
            })
        
        return fixes
    
    async def _fix_repository_description(self, repo) -> Optional[str]:
        """Generate and update repository description.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            New description if successful, None otherwise
        """
        try:
            # If AI brain is available, generate intelligent description
            if self.ai_brain:
                # Analyze repository to understand its purpose
                from scripts.repository_analyzer import RepositoryAnalyzer
                analyzer = RepositoryAnalyzer(self.github_token, self.ai_brain)
                analysis = await analyzer.analyze_repository(repo.full_name)
                
                # Generate description based on analysis
                prompt = f"""
                Generate a clear, specific repository description (max 200 characters) based on this analysis:
                
                Repository: {repo.name}
                Language: {repo.language or 'Mixed'}
                Features: {json.dumps(analysis.get('specific_needs', [])[:3])}
                Tech Stack: {json.dumps(analysis.get('technical_stack', {}))}
                
                The description should:
                1. Clearly state what the project does
                2. Mention the key technology (Laravel + React)
                3. Highlight the main value proposition
                4. Be specific to this project, not generic
                
                Format: Just the description text, no quotes or explanation.
                """
                
                response = await self.ai_brain.generate_enhanced_response(prompt)
                new_description = response.get('content', '').strip()
                
                # Ensure it's not empty and within GitHub's limit
                if new_description and len(new_description) <= 200:
                    repo.edit(description=new_description)
                    return new_description
            
            # Fallback: Generate basic description from repo name
            repo_name_words = repo.name.replace('-', ' ').replace('_', ' ').title()
            new_description = f"{repo_name_words} - A Laravel React application with TypeScript"
            
            repo.edit(description=new_description)
            return new_description
            
        except Exception as e:
            self.logger.error(f"Failed to update description: {e}")
            return None
    
    async def _generate_and_save_architecture(self, repo) -> bool:
        """Generate and save ARCHITECTURE.md for the repository.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.ai_brain:
                self.logger.warning("AI brain required for architecture generation")
                return False
            
            # Use the architecture generator
            from scripts.architecture_generator import ArchitectureGenerator
            generator = ArchitectureGenerator(self.github_token, self.ai_brain)
            
            # Generate architecture
            # First get repository analysis
            from scripts.repository_analyzer import RepositoryAnalyzer
            analyzer = RepositoryAnalyzer(self.github_token, self.ai_brain)
            repo_analysis = await analyzer.analyze_repository(repo.full_name)
            
            # Then generate architecture document
            architecture_data = await generator.generate_architecture_for_project(repo.full_name, repo_analysis)
            
            # Format architecture as markdown
            if architecture_data:
                from scripts.project_creator import ProjectCreator
                creator = ProjectCreator(self.github_token)
                
                # Create a details dict that matches what the formatter expects
                details = {
                    'name': repo.name,
                    'description': repo.description or architecture_data.get('description', ''),
                    'problem_statement': 'Extracted from existing codebase',
                    'target_audience': 'Development team',
                    'core_entities': architecture_data.get('core_entities', []),
                    'architecture': architecture_data
                }
                
                architecture_content = creator._format_architecture_document(details)
            else:
                architecture_content = None
            
            if architecture_content:
                # Save to repository
                repo.create_file(
                    "ARCHITECTURE.md",
                    "Add comprehensive architecture documentation",
                    architecture_content,
                    branch='main'
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to generate architecture: {e}")
            return False
    
    async def _fix_readme(self, repo) -> bool:
        """Update README.md with proper project-specific content.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.ai_brain:
                self.logger.warning("AI brain required for README generation")
                return False
            
            # Analyze repository first
            from scripts.repository_analyzer import RepositoryAnalyzer
            analyzer = RepositoryAnalyzer(self.github_token, self.ai_brain)
            analysis = await analyzer.analyze_repository(repo.full_name)
            
            # Check if architecture exists
            architecture = None
            try:
                arch_file = repo.get_contents("ARCHITECTURE.md")
                architecture = arch_file.decoded_content.decode('utf-8')
            except:
                pass
            
            # Generate comprehensive README
            prompt = f"""
            Generate a comprehensive, project-specific README.md for this repository:
            
            Repository: {repo.name}
            Description: {repo.description}
            Language: {repo.language}
            Analysis: {json.dumps(analysis, indent=2)}
            
            {f"Architecture Document: {architecture[:1000]}..." if architecture else ""}
            
            Create a professional README that includes:
            1. Clear project title and description
            2. What problem this solves
            3. Key features (be specific)
            4. Tech stack details (Laravel 11+, React 19, TypeScript, etc.)
            5. Prerequisites
            6. Installation instructions
            7. Environment setup
            8. Development workflow
            9. Testing instructions
            10. Deployment guide
            11. Contributing guidelines
            12. License
            
            Make it specific to THIS project, not generic.
            Use proper Markdown formatting.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            readme_content = response.get('content', '')
            
            if readme_content:
                try:
                    # Update existing README
                    readme_file = repo.get_contents("README.md")
                    repo.update_file(
                        "README.md",
                        "Update README with project-specific content",
                        readme_content,
                        readme_file.sha,
                        branch='main'
                    )
                except GithubException:
                    # Create new README
                    repo.create_file(
                        "README.md",
                        "Add comprehensive project README",
                        readme_content,
                        branch='main'
                    )
                
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update README: {e}")
            return False
    
    async def _fix_package_json(self, repo) -> bool:
        """Update package.json with proper project name.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            package_file = repo.get_contents("package.json")
            package_data = json.loads(package_file.decoded_content)
            
            # Update name to match repository
            package_data['name'] = repo.name
            package_data['description'] = repo.description or f"{repo.name} application"
            
            # Update the file
            repo.update_file(
                "package.json",
                f"Update package.json with project-specific details",
                json.dumps(package_data, indent=2),
                package_file.sha,
                branch=repo.default_branch
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update package.json: {e}")
            return False
    
    async def _add_repository_topics(self, repo) -> List[str]:
        """Add appropriate topics/tags to repository.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            List of topics added
        """
        try:
            # Default topics for Laravel React projects
            topics = ['laravel', 'react', 'typescript', 'fullstack']
            
            # Add language-specific topics
            if repo.language:
                lang_lower = repo.language.lower()
                if lang_lower == 'php':
                    topics.extend(['php', 'backend'])
                elif lang_lower == 'javascript':
                    topics.extend(['javascript', 'frontend'])
            
            # Add feature-based topics by analyzing repo
            try:
                # Check for API routes
                api_file = repo.get_contents("routes/api.php")
                if api_file:
                    topics.append('rest-api')
            except:
                pass
            
            try:
                # Check for authentication
                auth_controller = repo.get_contents("app/Http/Controllers/Auth")
                if auth_controller:
                    topics.append('authentication')
            except:
                pass
            
            # Remove duplicates and apply
            topics = list(set(topics))
            repo.replace_topics(topics)
            
            return topics
            
        except Exception as e:
            self.logger.error(f"Failed to add topics: {e}")
            return []
    
    async def check_repositories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Check all repositories for issues.
        
        Returns:
            Dictionary mapping repository names to their issues
        """
        org = self.github.get_organization(self.organization)
        repos = org.get_repos()
        
        issues_by_repo = {}
        
        for repo in repos:
            # Skip excluded repositories
            if repo.name in self.EXCLUDED_REPOS:
                continue
                
            issues = await self._check_repository_issues(repo)
            if issues:
                issues_by_repo[repo.name] = issues
                
        return issues_by_repo
    
    async def fix_repository(self, repo_name: str) -> Dict[str, Any]:
        """Alias for fix_specific_repository for compatibility."""
        return await self.fix_specific_repository(repo_name)
    
    async def fix_specific_repository(self, repo_name: str) -> Dict[str, Any]:
        """Fix issues in a specific repository.
        
        Args:
            repo_name: Repository name (can be full name or just name)
            
        Returns:
            Summary of fixes applied
        """
        try:
            # Get repository
            if '/' in repo_name:
                repo = self.github.get_repo(repo_name)
            else:
                org = self.github.get_organization(self.organization)
                repo = org.get_repo(repo_name)
            
            self.logger.info(f"🔧 Fixing repository: {repo.full_name}")
            
            # Check for issues
            issues = await self._check_repository_issues(repo)
            
            if not issues:
                self.logger.info("✅ No issues found - repository is already properly customized")
                return {
                    'repository': repo.full_name,
                    'issues_found': 0,
                    'fixes_applied': 0,
                    'status': 'already_customized'
                }
            
            # Apply fixes
            fixes = await self._fix_repository_issues(repo, issues)
            
            return {
                'repository': repo.full_name,
                'issues_found': len(issues),
                'issues': issues,
                'fixes_applied': len(fixes),
                'fixes': fixes,
                'status': 'fixed'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fix repository: {e}")
            return {
                'repository': repo_name,
                'error': str(e),
                'status': 'failed'
            }


async def main():
    """Main function to run the customization fixer."""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv('.env.local')
    
    parser = argparse.ArgumentParser(description='Fix repository customizations')
    parser.add_argument('--check', action='store_true', help='Check for issues without fixing')
    parser.add_argument('--fix', action='store_true', help='Fix found issues')
    parser.add_argument('--repo', type=str, help='Fix specific repository')
    parser.add_argument('--with-ai', action='store_true', help='Use AI for intelligent fixes')
    
    args = parser.parse_args()
    
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN not found in environment")
        return
    
    # Initialize AI brain if requested
    ai_brain = None
    if args.with_ai:
        try:
            from scripts.ai_brain import IntelligentAIBrain
            ai_brain = IntelligentAIBrain()
            print("🤖 AI brain initialized for intelligent fixes")
        except ImportError:
            from scripts.ai_brain import AIBrain
            ai_brain = AIBrain()
            print("🤖 AI brain initialized for intelligent fixes")
    
    # Create fixer
    fixer = RepositoryCustomizationFixer(github_token, ai_brain)
    
    # Run appropriate action
    if args.repo:
        # Fix specific repository
        result = await fixer.fix_specific_repository(args.repo)
        print(f"\n📊 Fix Summary for {args.repo}:")
        print(f"  Issues found: {result.get('issues_found', 0)}")
        print(f"  Fixes applied: {result.get('fixes_applied', 0)}")
        print(f"  Status: {result.get('status', 'unknown')}")
        
    elif args.fix:
        # Fix all repositories
        summary = await fixer.scan_repositories(fix_mode=True)
        print(f"\n📊 Fix Summary:")
        print(f"  Total repositories: {summary['total_repositories']}")
        print(f"  Repositories with issues: {summary['repositories_with_issues']}")
        print(f"  Total issues found: {summary['total_issues_found']}")
        print(f"  Total fixes applied: {summary['total_fixes_applied']}")
        
    else:
        # Check only (default)
        summary = await fixer.scan_repositories(fix_mode=False)
        print(f"\n📊 Check Summary:")
        print(f"  Total repositories: {summary['total_repositories']}")
        print(f"  Repositories with issues: {summary['repositories_with_issues']}")
        print(f"  Total issues found: {summary['total_issues_found']}")
        
        if summary['repositories_with_issues'] > 0:
            print("\n⚠️  Issues found in repositories:")
            for repo_issues in summary['issues_by_repository']:
                print(f"\n  {repo_issues['repository']}:")
                for issue in repo_issues['issues']:
                    print(f"    - {issue['type']}: {issue['description']}")
            
            print("\n💡 Run with --fix to automatically fix these issues")
            print("   Or use --repo <name> to fix a specific repository")


if __name__ == "__main__":
    asyncio.run(main())