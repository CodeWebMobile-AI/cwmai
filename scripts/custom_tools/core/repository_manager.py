"""
Repository Manager

Unified tool for all repository operations including management, analysis, 
issue handling, and reporting. Consolidates 14 repository tools into one.

Capabilities:
- Get, list, search repositories
- Analyze repository health and metrics
- Manage issues and pull requests
- Generate reports and summaries
- Fix repository problems
- Track repository metadata
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

from scripts.state_manager import StateManager
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.http_ai_client import HTTPAIClient
from scripts.storage_factory import StorageFactory


class RepositoryManager:
    """Comprehensive repository management system."""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.repo_analyzer = RepositoryAnalyzer()
        self.issue_creator = GitHubIssueCreator()
        self.ai_client = HTTPAIClient()
        self.storage_factory = StorageFactory()
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute any repository operation.
        
        Operations:
        - get: Get repository information
        - list: List all repositories
        - search: Search repositories
        - analyze: Analyze repository health
        - create_issue: Create GitHub issue
        - get_issues: Get repository issues
        - fix: Fix repository problems
        - report: Generate repository report
        """
        operations = {
            'get': self.get_repository,
            'list': self.list_repositories,
            'search': self.search_repositories,
            'analyze': self.analyze_repository,
            'create_issue': self.create_issue,
            'get_issues': self.get_issues,
            'fix': self.fix_repository,
            'report': self.generate_report,
            'ensure_workflow': self.ensure_workflow,
            'get_metadata': self.get_metadata,
            'evaluate': self.evaluate_opportunities,
            'weekly_report': self.generate_weekly_report
        }
        
        handler = operations.get(operation)
        if not handler:
            # Try fuzzy matching
            for op, func in operations.items():
                if op in operation or operation in op:
                    handler = func
                    break
                    
        if not handler:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": list(operations.keys())
            }
            
        return await handler(**kwargs)
        
    async def get_repository(self, name: str = None, **kwargs) -> Dict[str, Any]:
        """Get detailed repository information."""
        if not name:
            # Try to extract from kwargs
            name = kwargs.get('repo') or kwargs.get('repository')
            
        if not name:
            return {"error": "Repository name required"}
            
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        # Check different name formats
        repo_data = None
        if name in projects:
            repo_data = projects[name]
        else:
            # Try with organization prefix
            for repo_key in projects:
                if repo_key.endswith(f'/{name}') or repo_key == name:
                    repo_data = projects[repo_key]
                    break
                    
        if not repo_data:
            # Try to fetch from GitHub/local
            try:
                analysis = await self.repo_analyzer.analyze_repository(name)
                if isinstance(analysis, dict) and not analysis.get('error'):
                    return {
                        "success": True,
                        "repository": name,
                        "data": analysis
                    }
            except:
                pass
                
            return {"error": f"Repository '{name}' not found"}
            
        # Enrich with current analysis if possible
        try:
            current_analysis = await self.repo_analyzer.analyze_repository(name)
            if isinstance(current_analysis, dict) and not current_analysis.get('error'):
                repo_data['current_analysis'] = current_analysis
        except:
            pass
            
        return {
            "success": True,
            "repository": name,
            "data": repo_data,
            "health_score": repo_data.get('health_score', 0),
            "metrics": repo_data.get('metrics', {}),
            "last_updated": repo_data.get('last_checked', 'unknown')
        }
        
    async def list_repositories(self, filter_by: str = None, sort_by: str = 'name', **kwargs) -> Dict[str, Any]:
        """List all repositories with optional filtering."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        repositories = []
        for name, data in projects.items():
            if isinstance(data, dict):
                repo_info = {
                    'name': name,
                    'health_score': data.get('health_score', 0),
                    'stars': data.get('metrics', {}).get('stars', 0),
                    'issues': data.get('metrics', {}).get('issues_open', 0),
                    'last_updated': data.get('last_checked', 'unknown')
                }
                
                # Apply filter
                if filter_by:
                    if filter_by == 'healthy' and repo_info['health_score'] < 80:
                        continue
                    elif filter_by == 'issues' and repo_info['issues'] == 0:
                        continue
                    elif filter_by == 'popular' and repo_info['stars'] < 10:
                        continue
                        
                repositories.append(repo_info)
                
        # Sort
        if sort_by == 'health':
            repositories.sort(key=lambda x: x['health_score'], reverse=True)
        elif sort_by == 'stars':
            repositories.sort(key=lambda x: x['stars'], reverse=True)
        elif sort_by == 'issues':
            repositories.sort(key=lambda x: x['issues'], reverse=True)
        else:
            repositories.sort(key=lambda x: x['name'])
            
        return {
            "success": True,
            "total": len(repositories),
            "repositories": repositories,
            "filters_available": ['healthy', 'issues', 'popular'],
            "sort_options": ['name', 'health', 'stars', 'issues']
        }
        
    async def search_repositories(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search repositories by name or content."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        query_lower = query.lower()
        matches = []
        
        for name, data in projects.items():
            score = 0
            
            # Name match
            if query_lower in name.lower():
                score += 10
                
            # Description match (if available)
            if isinstance(data, dict):
                desc = data.get('description', '').lower()
                if query_lower in desc:
                    score += 5
                    
                # Technology match
                techs = data.get('technologies', [])
                if any(query_lower in tech.lower() for tech in techs):
                    score += 3
                    
            if score > 0:
                matches.append({
                    'name': name,
                    'score': score,
                    'health': data.get('health_score', 0) if isinstance(data, dict) else 0
                })
                
        # Sort by relevance
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "success": True,
            "query": query,
            "results": len(matches),
            "repositories": matches[:20]  # Top 20 results
        }
        
    async def analyze_repository(self, name: str, deep: bool = False, **kwargs) -> Dict[str, Any]:
        """Analyze repository health and metrics."""
        try:
            # Get fresh analysis
            analysis = await self.repo_analyzer.analyze_repository(name)
            
            if isinstance(analysis, dict) and analysis.get('error'):
                return {"error": analysis['error']}
                
            # Calculate health score if not present
            if 'health_metrics' not in analysis:
                health_score = self._calculate_health_score(analysis)
                analysis['health_metrics'] = {'health_score': health_score}
                
            # Deep analysis with AI
            if deep:
                ai_insights = await self._get_ai_insights(name, analysis)
                analysis['ai_insights'] = ai_insights
                
            # Update state
            state = self.state_manager.get_state()
            if name in state.get('projects', {}):
                state['projects'][name].update({
                    'health_score': analysis['health_metrics']['health_score'],
                    'last_analysis': datetime.now(timezone.utc).isoformat(),
                    'analysis_summary': analysis.get('summary', {})
                })
                self.state_manager.save_state()
                
            return {
                "success": True,
                "repository": name,
                "analysis": analysis,
                "health_score": analysis['health_metrics']['health_score'],
                "recommendations": self._generate_recommendations(analysis)
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
            
    async def create_issue(self, repo: str, title: str, body: str, labels: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Create a GitHub issue."""
        try:
            # Use issue creator
            result = await self.issue_creator.create_issue(
                repo=repo,
                title=title,
                body=body,
                labels=labels or []
            )
            
            return {
                "success": result.get('success', False),
                "issue_url": result.get('html_url', ''),
                "issue_number": result.get('number', 0),
                "repository": repo
            }
            
        except Exception as e:
            return {"error": f"Failed to create issue: {str(e)}"}
            
    async def get_issues(self, repo: str, state: str = 'open', **kwargs) -> Dict[str, Any]:
        """Get repository issues."""
        try:
            # Check if we're in local mode
            if self.storage_factory.storage_mode == 'local':
                # Get from local storage
                from scripts.local_github_simulator import LocalGitHubSimulator
                simulator = LocalGitHubSimulator()
                issues = simulator.get_issues(repo, state)
            else:
                # Get from GitHub
                # This would need GitHub API implementation
                issues = []
                
            return {
                "success": True,
                "repository": repo,
                "total_issues": len(issues),
                "issues": issues,
                "state_filter": state
            }
            
        except Exception as e:
            return {"error": f"Failed to get issues: {str(e)}"}
            
    async def fix_repository(self, name: str, problems: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Fix common repository problems."""
        fixes_applied = []
        
        # Analyze first
        analysis = await self.analyze_repository(name)
        if analysis.get('error'):
            return analysis
            
        # Identify problems if not specified
        if not problems:
            problems = self._identify_problems(analysis['analysis'])
            
        # Apply fixes
        for problem in problems:
            if problem == 'missing_readme':
                # Create README
                fixes_applied.append({
                    'problem': 'missing_readme',
                    'action': 'Would create README.md',
                    'status': 'suggested'
                })
                
            elif problem == 'no_tests':
                # Suggest test creation
                fixes_applied.append({
                    'problem': 'no_tests',
                    'action': 'Would create test structure',
                    'status': 'suggested'
                })
                
            elif problem == 'outdated_dependencies':
                # Suggest updates
                fixes_applied.append({
                    'problem': 'outdated_dependencies',
                    'action': 'Would update dependencies',
                    'status': 'suggested'
                })
                
        return {
            "success": True,
            "repository": name,
            "problems_found": len(problems),
            "fixes_applied": len(fixes_applied),
            "fixes": fixes_applied
        }
        
    async def generate_report(self, repos: List[str] = None, report_type: str = 'summary', **kwargs) -> Dict[str, Any]:
        """Generate repository report."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        # Select repositories
        if not repos:
            repos = list(projects.keys())
            
        report_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'repositories_analyzed': len(repos),
            'report_type': report_type
        }
        
        if report_type == 'summary':
            # Summary statistics
            total_health = 0
            total_issues = 0
            total_stars = 0
            
            for repo in repos:
                if repo in projects and isinstance(projects[repo], dict):
                    total_health += projects[repo].get('health_score', 0)
                    metrics = projects[repo].get('metrics', {})
                    total_issues += metrics.get('issues_open', 0)
                    total_stars += metrics.get('stars', 0)
                    
            report_data.update({
                'average_health': round(total_health / max(len(repos), 1), 1),
                'total_open_issues': total_issues,
                'total_stars': total_stars,
                'repositories': repos
            })
            
        elif report_type == 'detailed':
            # Detailed analysis for each repo
            detailed_reports = []
            for repo in repos[:10]:  # Limit to 10 for detailed
                analysis = await self.analyze_repository(repo)
                if not analysis.get('error'):
                    detailed_reports.append({
                        'repository': repo,
                        'health_score': analysis['health_score'],
                        'recommendations': analysis['recommendations']
                    })
                    
            report_data['detailed_analysis'] = detailed_reports
            
        return {
            "success": True,
            "report": report_data
        }
        
    async def ensure_workflow(self, repo: str, workflow: str = 'ci', **kwargs) -> Dict[str, Any]:
        """Ensure repository has required workflows."""
        workflows = {
            'ci': {
                'name': 'CI/CD Pipeline',
                'file': '.github/workflows/ci.yml',
                'purpose': 'Continuous Integration'
            },
            'security': {
                'name': 'Security Scan',
                'file': '.github/workflows/security.yml',
                'purpose': 'Security vulnerability scanning'
            },
            'docs': {
                'name': 'Documentation',
                'file': '.github/workflows/docs.yml',
                'purpose': 'Documentation generation'
            }
        }
        
        workflow_info = workflows.get(workflow, workflows['ci'])
        
        return {
            "success": True,
            "repository": repo,
            "workflow": workflow_info['name'],
            "status": "Would ensure workflow exists",
            "file_path": workflow_info['file']
        }
        
    async def get_metadata(self, repo: str, **kwargs) -> Dict[str, Any]:
        """Get repository metadata including creation date, contributors, etc."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        if repo not in projects:
            return {"error": f"Repository '{repo}' not found"}
            
        metadata = {
            'name': repo,
            'created_at': projects[repo].get('created_at', 'unknown'),
            'last_updated': projects[repo].get('last_checked', 'unknown'),
            'health_score': projects[repo].get('health_score', 0),
            'metrics': projects[repo].get('metrics', {}),
            'technologies': projects[repo].get('technologies', []),
            'description': projects[repo].get('description', '')
        }
        
        return {
            "success": True,
            "repository": repo,
            "metadata": metadata
        }
        
    async def evaluate_opportunities(self, repo: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate improvement opportunities for repositories."""
        opportunities = []
        
        if repo:
            repos = [repo]
        else:
            state = self.state_manager.get_state()
            repos = list(state.get('projects', {}).keys())
            
        for repo_name in repos:
            analysis = await self.analyze_repository(repo_name)
            if not analysis.get('error'):
                repo_opportunities = self._identify_opportunities(analysis['analysis'])
                if repo_opportunities:
                    opportunities.append({
                        'repository': repo_name,
                        'opportunities': repo_opportunities,
                        'potential_impact': len(repo_opportunities) * 10  # Simple scoring
                    })
                    
        # Sort by impact
        opportunities.sort(key=lambda x: x['potential_impact'], reverse=True)
        
        return {
            "success": True,
            "total_opportunities": sum(len(o['opportunities']) for o in opportunities),
            "repositories_analyzed": len(repos),
            "top_opportunities": opportunities[:10]
        }
        
    async def generate_weekly_report(self, **kwargs) -> Dict[str, Any]:
        """Generate weekly repository report."""
        state = self.state_manager.get_state()
        projects = state.get('projects', {})
        
        # Calculate weekly metrics
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        weekly_stats = {
            'total_repositories': len(projects),
            'healthy_repos': 0,
            'repos_needing_attention': 0,
            'total_issues': 0,
            'repos_updated': 0
        }
        
        repo_details = []
        
        for repo, data in projects.items():
            if isinstance(data, dict):
                health = data.get('health_score', 0)
                
                if health >= 80:
                    weekly_stats['healthy_repos'] += 1
                elif health < 60:
                    weekly_stats['repos_needing_attention'] += 1
                    
                weekly_stats['total_issues'] += data.get('metrics', {}).get('issues_open', 0)
                
                # Check if updated this week
                last_checked = data.get('last_checked', '')
                if last_checked:
                    try:
                        checked_date = datetime.fromisoformat(last_checked.replace('Z', '+00:00'))
                        if checked_date > week_ago:
                            weekly_stats['repos_updated'] += 1
                    except:
                        pass
                        
                repo_details.append({
                    'name': repo,
                    'health': health,
                    'issues': data.get('metrics', {}).get('issues_open', 0)
                })
                
        # Sort by health (worst first)
        repo_details.sort(key=lambda x: x['health'])
        
        return {
            "success": True,
            "report_type": "weekly",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": weekly_stats,
            "repos_needing_attention": repo_details[:5],
            "recommendations": self._generate_weekly_recommendations(weekly_stats)
        }
        
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate repository health score."""
        score = 100
        
        # Check various factors
        if not analysis.get('has_readme', True):
            score -= 20
        if not analysis.get('has_tests', True):
            score -= 15
        if analysis.get('code_quality', {}).get('issues', 0) > 10:
            score -= 10
        if analysis.get('metrics', {}).get('issues_open', 0) > 20:
            score -= 10
            
        return max(0, score)
        
    async def _get_ai_insights(self, repo: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered insights about the repository."""
        prompt = f"""Analyze this repository data and provide insights:
Repository: {repo}
Health Score: {analysis.get('health_metrics', {}).get('health_score', 0)}
Metrics: {json.dumps(analysis.get('metrics', {}), indent=2)}

Provide insights on:
1. Main strengths
2. Key weaknesses  
3. Improvement priorities
4. Growth potential"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            return {
                "insights": response.get('content', ''),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        return {}
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        health = analysis.get('health_metrics', {}).get('health_score', 100)
        
        if health < 60:
            recommendations.append("Urgent attention needed - health score below 60")
        if not analysis.get('has_readme', True):
            recommendations.append("Add comprehensive README documentation")
        if not analysis.get('has_tests', True):
            recommendations.append("Implement unit tests for better reliability")
        if analysis.get('metrics', {}).get('issues_open', 0) > 10:
            recommendations.append("Address open issues to improve maintainability")
            
        return recommendations
        
    def _identify_problems(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify problems in repository."""
        problems = []
        
        if not analysis.get('has_readme', True):
            problems.append('missing_readme')
        if not analysis.get('has_tests', True):
            problems.append('no_tests')
        if analysis.get('dependencies', {}).get('outdated', 0) > 5:
            problems.append('outdated_dependencies')
            
        return problems
        
    def _identify_opportunities(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify improvement opportunities."""
        opportunities = []
        
        if not analysis.get('has_ci', True):
            opportunities.append({
                'type': 'automation',
                'description': 'Add CI/CD pipeline',
                'impact': 'high'
            })
            
        if analysis.get('code_quality', {}).get('complexity', 0) > 10:
            opportunities.append({
                'type': 'refactoring',
                'description': 'Reduce code complexity',
                'impact': 'medium'
            })
            
        return opportunities
        
    def _generate_weekly_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate weekly recommendations."""
        recommendations = []
        
        if stats['repos_needing_attention'] > stats['total_repositories'] * 0.3:
            recommendations.append("Focus on improving repository health - 30%+ need attention")
            
        if stats['total_issues'] > stats['total_repositories'] * 10:
            recommendations.append("Issue backlog is growing - organize issue triage sessions")
            
        if stats['repos_updated'] < stats['total_repositories'] * 0.5:
            recommendations.append("Many repositories haven't been checked recently - schedule reviews")
            
        return recommendations


async def main(operation: str = None, **kwargs):
    """Entry point for repository manager."""
    manager = RepositoryManager()
    
    if operation:
        return await manager.execute(operation, **kwargs)
    else:
        # Default to list
        return await manager.list_repositories(**kwargs)
        

if __name__ == "__main__":
    # Test the manager
    import asyncio
    
    async def test():
        manager = RepositoryManager()
        
        # Test operations
        print("List repos:", await manager.list_repositories())
        print("\nSearch:", await manager.search_repositories("ai"))
        print("\nWeekly report:", await manager.generate_weekly_report())
        
    asyncio.run(test())