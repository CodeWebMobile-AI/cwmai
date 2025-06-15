"""AI-Powered Task Content Generator

Generates unique, contextually relevant task content using AI instead of templates.
No hardcoded strings, no random choices - pure AI-driven content generation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import random

from scripts.ai_brain import IntelligentAIBrain


class AITaskContentGenerator:
    """Generates task content using AI based on context and repository needs."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, logger=None):
        """Initialize with AI brain.
        
        Args:
            ai_brain: AI brain instance for content generation
            logger: Optional logger instance
        """
        self.ai_brain = ai_brain
        self.logger = logger or logging.getLogger(__name__)
        
    async def generate_new_project_content(self, context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered new project task content.
        
        Args:
            context: System context including trends, portfolio gaps, etc.
            
        Returns:
            Tuple of (title, description)
        """
        # Extract relevant context
        trends = context.get('github_trending', [])[:3]
        portfolio_gaps = context.get('portfolio_gaps', [])
        existing_projects = context.get('projects', {})
        
        prompt = f"""Generate a new project idea that fills a gap in the current portfolio.

Current Portfolio:
{json.dumps([p.get('name', '') for p in existing_projects.values()], indent=2)}

Portfolio Gaps Identified:
{json.dumps(portfolio_gaps, indent=2)}

Current Technology Trends:
{json.dumps([t.get('title', '') for t in trends], indent=2)}

Create a NEW PROJECT that:
1. Fills an identified gap or explores a new area
2. Uses Laravel React starter kit as foundation
3. Has clear business value and use case
4. Can showcase AI capabilities
5. Is different from existing projects

Return as JSON with:
- title: Specific, actionable project name (50-80 chars)
- description: Detailed description including:
  * What the application does
  * Key features (5-7 specific features)
  * Technology stack details
  * How to customize Laravel React starter kit
  * Business value proposition
  * Target users
  * Competitive advantages

Make it specific, innovative, and different from typical examples.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback (should rarely happen)
        return "AI-Powered Innovation Platform", "Create an innovative platform using Laravel React"
    
    async def generate_feature_content(self, 
                                     repository: str,
                                     repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered feature task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        # Extract repository information
        repo_info = repo_context.get('basic_info', {})
        recent_issues = repo_context.get('issues_analysis', {}).get('recent_issues', [])
        tech_stack = repo_context.get('technical_stack', {})
        
        # Extract architecture information if available
        architecture = repo_context.get('architecture', {})
        architecture_context = ""
        if architecture and architecture.get('document_exists'):
            core_entities = architecture.get('core_entities', [])
            feature_roadmap = architecture.get('feature_roadmap', [])
            architecture_context = f"""

Architecture Context (from ARCHITECTURE.md):
- Core Entities: {', '.join(core_entities) if core_entities else 'Not specified'}
- Planned Features in Roadmap: {json.dumps(feature_roadmap[:5], indent=2) if feature_roadmap else 'None'}

IMPORTANT: The feature should align with the documented architecture and complement the planned feature roadmap.
"""
        
        prompt = f"""Generate a valuable feature for the {repository} repository.

Repository Information:
- Description: {repo_info.get('description', 'No description')}
- Language: {repo_info.get('language', 'Unknown')}
- Open Issues: {repo_info.get('open_issues_count', 0)}
- Tech Stack: {json.dumps(tech_stack, indent=2)}
{architecture_context}

Recent Issues (for context):
{json.dumps([issue.get('title', '') for issue in recent_issues[:3]], indent=2)}

Create a FEATURE task that:
1. Adds significant value to the project
2. Is technically appropriate for the stack
3. Addresses user needs or enhances functionality
4. Is specific and implementable
5. Hasn't been recently implemented

Return as JSON with:
- title: Clear, specific feature name (50-80 chars)
- description: Detailed description including:
  * What the feature does
  * User stories (2-3)
  * Technical implementation approach
  * Required components/changes
  * Integration points
  * Success metrics

Make it specific to THIS repository, not generic.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Enhance {repository} functionality", f"Add new capabilities to {repository}"
    
    async def generate_bug_fix_content(self,
                                     repository: str,
                                     repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered bug fix task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        recent_commits = repo_context.get('recent_activity', {}).get('recent_commits', [])
        open_issues = repo_context.get('issues_analysis', {}).get('bug_issues', [])
        
        prompt = f"""Generate a bug fix task for the {repository} repository.

Repository Context:
- Recent Commits: {len(recent_commits)}
- Open Bug Issues: {len(open_issues)}
- Last Activity: {repo_context.get('recent_activity', {}).get('last_commit_date', 'Unknown')}

Known Issues:
{json.dumps([issue.get('title', '') for issue in open_issues[:3]], indent=2)}

Create a BUG_FIX task that:
1. Addresses a realistic bug scenario
2. Is specific to this repository's functionality
3. Includes investigation and fix steps
4. Has clear reproduction steps
5. Considers edge cases

Return as JSON with:
- title: Specific bug description (50-80 chars)
- description: Detailed description including:
  * Bug symptoms
  * Steps to reproduce (if applicable)
  * Expected vs actual behavior
  * Investigation approach
  * Fix requirements
  * Testing requirements

Make it realistic and repository-specific.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Fix issue in {repository}", f"Investigate and fix reported issue in {repository}"
    
    async def generate_documentation_content(self,
                                           repository: str,
                                           repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered documentation task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        existing_docs = repo_context.get('documentation_status', {})
        tech_stack = repo_context.get('technical_stack', {})
        
        prompt = f"""Generate a documentation task for the {repository} repository.

Repository Information:
- Description: {repo_context.get('basic_info', {}).get('description', '')}
- Tech Stack: {json.dumps(tech_stack, indent=2)}
- Existing Docs: {json.dumps(existing_docs, indent=2)}

Create a DOCUMENTATION task that:
1. Fills a specific documentation gap
2. Adds value for developers or users
3. Is appropriate for the project type
4. Uses clear structure and examples
5. Goes beyond basic README updates

Return as JSON with:
- title: Specific documentation task (50-80 chars)
- description: Detailed description including:
  * What documentation is needed
  * Target audience
  * Content structure
  * Key sections to include
  * Examples to provide
  * Format and location

Make it specific and valuable for THIS project.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Document {repository} components", f"Create documentation for {repository}"
    
    async def generate_testing_content(self,
                                     repository: str,
                                     repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered testing task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        tech_stack = repo_context.get('technical_stack', {})
        test_coverage = repo_context.get('test_coverage', {})
        
        prompt = f"""Generate a testing task for the {repository} repository.

Repository Information:
- Language: {repo_context.get('basic_info', {}).get('language', 'Unknown')}
- Tech Stack: {json.dumps(tech_stack, indent=2)}
- Current Test Coverage: {json.dumps(test_coverage, indent=2)}

Create a TESTING task that:
1. Improves test coverage meaningfully
2. Uses appropriate testing frameworks
3. Covers critical functionality
4. Includes different test types
5. Is specific to this project's needs

Return as JSON with:
- title: Specific testing task (50-80 chars)
- description: Detailed description including:
  * What to test
  * Test scenarios (3-5)
  * Testing approach
  * Frameworks to use
  * Coverage goals
  * Edge cases to consider

Make it specific to THIS repository's testing needs.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Add tests for {repository}", f"Improve test coverage in {repository}"
    
    async def generate_security_content(self,
                                      repository: str,
                                      repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered security task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        tech_stack = repo_context.get('technical_stack', {})
        security_issues = repo_context.get('security_analysis', {})
        
        prompt = f"""Generate a security task for the {repository} repository.

Repository Information:
- Tech Stack: {json.dumps(tech_stack, indent=2)}
- Known Security Concerns: {json.dumps(security_issues, indent=2)}
- Last Security Review: {repo_context.get('last_security_review', 'Unknown')}

Create a SECURITY task that:
1. Addresses realistic security concerns
2. Is appropriate for the technology stack
3. Follows security best practices
4. Includes specific actions
5. Considers OWASP guidelines

Return as JSON with:
- title: Specific security task (50-80 chars)
- description: Detailed description including:
  * Security concern addressed
  * Audit scope
  * Vulnerabilities to check
  * Implementation requirements
  * Testing approach
  * Compliance considerations

Make it specific to THIS repository's security needs.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Security audit for {repository}", f"Conduct security review of {repository}"
    
    async def generate_optimization_content(self,
                                          repository: str,
                                          repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered optimization task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        performance_metrics = repo_context.get('performance_metrics', {})
        tech_stack = repo_context.get('technical_stack', {})
        
        prompt = f"""Generate an optimization task for the {repository} repository.

Repository Information:
- Tech Stack: {json.dumps(tech_stack, indent=2)}
- Performance Metrics: {json.dumps(performance_metrics, indent=2)}
- Known Bottlenecks: {repo_context.get('known_bottlenecks', [])}

Create an OPTIMIZATION task that:
1. Targets specific performance improvements
2. Is measurable and impactful
3. Uses appropriate optimization techniques
4. Considers the technology stack
5. Has clear success metrics

Return as JSON with:
- title: Specific optimization task (50-80 chars)
- description: Detailed description including:
  * Performance issue addressed
  * Current vs target metrics
  * Optimization approach
  * Implementation steps
  * Measurement methodology
  * Expected improvements

Make it specific to THIS repository's performance needs.
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return f"Optimize {repository} performance", f"Improve performance of {repository}"
    
    async def generate_architecture_documentation_content(self,
                                                        repository: str,
                                                        repo_context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate AI-powered architecture documentation task content.
        
        Args:
            repository: Target repository name
            repo_context: Repository-specific context
            
        Returns:
            Tuple of (title, description)
        """
        # Extract key information
        tech_stack = repo_context.get('technical_stack', {})
        basic_info = repo_context.get('basic_info', {})
        architecture = repo_context.get('architecture', {})
        
        prompt = f"""Generate an architecture documentation task for the {repository} repository.

Repository Information:
- Description: {basic_info.get('description', 'No description')}
- Language: {basic_info.get('language', 'Unknown')}
- Tech Stack: {json.dumps(tech_stack, indent=2)}
- Has Architecture Doc: {architecture.get('document_exists', False)}

Create an ARCHITECTURE_DOCUMENTATION task that:
1. Analyzes the existing codebase structure
2. Extracts architectural patterns and design decisions
3. Documents core entities and their relationships
4. Creates a comprehensive ARCHITECTURE.md file
5. Includes system design, API patterns, and database schema

Return as JSON with:
- title: Clear task title (e.g., "Generate architecture documentation for {repository}")
- description: Detailed description including:
  * Purpose of architecture documentation
  * What will be analyzed (models, controllers, components, etc.)
  * What will be documented (design patterns, entity relationships, API structure)
  * Expected deliverable (ARCHITECTURE.md file)
  * Benefits of having architecture documentation
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        return (f"Generate architecture documentation for {repository}", 
                f"Analyze the {repository} codebase and create comprehensive architecture documentation")
    
    async def generate_review_content(self,
                                    target: Optional[Dict[str, Any]] = None,
                                    repository: str = None) -> Tuple[str, str]:
        """Generate AI-powered code review task content.
        
        Args:
            target: Optional target issue/PR to review
            repository: Repository name
            
        Returns:
            Tuple of (title, description)
        """
        if target:
            prompt = f"""Generate a review task for a stale issue.

Issue Details:
- Number: #{target.get('number', 'Unknown')}
- Title: {target.get('title', 'Unknown')}
- Days Stale: {target.get('days_stale', 0)}
- Repository: {repository}

Create a REVIEW task that:
1. Addresses the stale issue specifically
2. Determines current relevance
3. Suggests next actions
4. Considers project priorities

Return as JSON with:
- title: Review task title mentioning issue number
- description: Review requirements and approach
"""
        else:
            prompt = f"""Generate a code review task for recent contributions.

Repository: {repository or 'various repositories'}

Create a REVIEW task that:
1. Reviews recent pull requests
2. Ensures code quality
3. Checks best practices
4. Provides constructive feedback

Return as JSON with:
- title: Specific review task
- description: Review focus areas and approach
"""

        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        if result:
            return result.get('title', ''), result.get('description', '')
        
        # Fallback
        if target:
            return f"Review issue #{target.get('number', '?')}", "Review and update stale issue"
        return "Review recent contributions", "Conduct code review of recent changes"
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response.
        
        Args:
            response: AI response dict with 'content' field
            
        Returns:
            Parsed JSON dict or None
        """
        content = response.get('content', '')
        
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse AI response: {e}")
            
        return None