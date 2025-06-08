"""
Context Gatherer Module

Performs external context gathering and market intelligence for the autonomous AI system.
Searches for trending technologies, security alerts, and market trends to inform decision-making.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


class ContextGatherer:
    """Gathers external context and market intelligence with AI-enhanced analysis."""
    
    def __init__(self, output_path: str = "context.json", ai_brain=None):
        """Initialize the ContextGatherer.
        
        Args:
            output_path: Path to save gathered context
            ai_brain: AI brain instance for enhanced analysis (optional)
        """
        self.output_path = output_path
        self.ai_brain = ai_brain
    
    def gather_context(self, charter: Dict[str, Any]) -> Dict[str, Any]:
        """Gather external context based on system charter.
        
        Args:
            charter: System charter defining goals and constraints
            
        Returns:
            Dictionary containing gathered context
        """
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "charter_goals": [charter.get("primary_goal", ""), charter.get("secondary_goal", "")],
            "market_trends": [],
            "security_alerts": [],
            "technology_updates": [],
            "github_trending": [],
            "programming_news": []
        }
        
        try:
            # Gather different types of context based on charter goals
            if charter.get("primary_goal") == "innovation":
                context["market_trends"] = self._get_innovation_trends()
                context["technology_updates"] = self._get_technology_updates()
                context["github_trending"] = self._get_github_trending()
            
            elif charter.get("primary_goal") == "security":
                context["security_alerts"] = self._get_security_alerts()
            
            elif charter.get("primary_goal") == "community_engagement":
                context["programming_news"] = self._get_programming_news()
                context["github_trending"] = self._get_github_trending()
            
            # Always gather general programming trends
            context["programming_news"].extend(self._get_general_programming_trends())
            
        except Exception as e:
            print(f"Error gathering context: {e}")
            context["error"] = str(e)
        
        # Enhance context with AI analysis if available
        if self.ai_brain:
            context = self._enhance_context_with_ai(context)
        
        # Save context to file
        self._save_context(context)
        return context
    
    def _get_innovation_trends(self) -> List[Dict[str, str]]:
        """Get innovation and technology trends using AI research.
        
        Returns:
            List of trend dictionaries
        """
        trends = []
        
        if not self.ai_brain:
            return trends
            
        try:
            # Use AI to research current trends
            research_prompt = """
            Research and provide the top 5 current technology and innovation trends in 2025.
            For each trend, provide:
            1. Title: A concise title for the trend
            2. Description: 2-3 sentences describing the trend
            3. Relevance: Why this trend matters for developers
            
            Focus on: AI breakthroughs, new programming languages, cloud innovations, and developer tools.
            Format as a list of dictionaries with keys: title, snippet, url (set url to 'ai-research')
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            # Parse AI response into trend format
            if response and 'content' in response:
                # The AI should return structured data we can parse
                import ast
                try:
                    trend_data = ast.literal_eval(response['content'])
                    if isinstance(trend_data, list):
                        trends = trend_data[:10]
                except:
                    # Fallback: create trends from text response
                    trends.append({
                        'title': 'AI Technology Trends 2025',
                        'snippet': response['content'][:200],
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error in _get_innovation_trends: {e}")
        
        return trends
    
    def _get_technology_updates(self) -> List[Dict[str, str]]:
        """Get technology updates and releases using AI research.
        
        Returns:
            List of technology update dictionaries
        """
        updates = []
        
        if not self.ai_brain:
            return updates
            
        try:
            research_prompt = """
            Research recent technology updates and releases in 2025. Include:
            - Python language updates and new features
            - JavaScript/TypeScript framework updates
            - Container technology (Docker, Kubernetes) updates
            - Major programming language releases
            
            For each update, provide: title, description, and impact.
            Format as a list of dictionaries with keys: title, snippet, url (set url to 'ai-research')
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            if response and 'content' in response:
                import ast
                try:
                    update_data = ast.literal_eval(response['content'])
                    if isinstance(update_data, list):
                        updates = update_data
                except:
                    updates.append({
                        'title': 'Technology Updates 2025',
                        'snippet': response['content'][:200],
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error in _get_technology_updates: {e}")
        
        return updates
    
    def _get_security_alerts(self) -> List[Dict[str, str]]:
        """Get security alerts and vulnerabilities using AI research.
        
        Returns:
            List of security alert dictionaries
        """
        alerts = []
        
        if not self.ai_brain:
            return alerts
            
        try:
            research_prompt = """
            Research current critical security vulnerabilities and alerts in 2025. Include:
            - Recent CVEs with high severity
            - npm/pip package vulnerabilities
            - Framework security advisories
            - Cloud platform security alerts
            
            For each alert, provide: title, severity, affected systems, and mitigation.
            Format as a list of dictionaries with keys: title, snippet, url (set url to 'ai-research')
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            if response and 'content' in response:
                import ast
                try:
                    alert_data = ast.literal_eval(response['content'])
                    if isinstance(alert_data, list):
                        alerts = alert_data[:8]
                except:
                    alerts.append({
                        'title': 'Security Alerts 2025',
                        'snippet': response['content'][:200],
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error in _get_security_alerts: {e}")
        
        return alerts
    
    def _get_github_trending(self) -> List[Dict[str, str]]:
        """Get trending GitHub repositories using AI research.
        
        Returns:
            List of trending repository information
        """
        trending = []
        
        if not self.ai_brain:
            return trending
            
        try:
            research_prompt = """
            Research the current trending GitHub repositories and open source projects in 2025.
            Focus on:
            - Popular new programming languages and frameworks
            - AI/ML projects gaining traction
            - Developer tools and productivity enhancers
            - Innovative web technologies
            
            For each project, provide: name, description, primary language, and why it's trending.
            Format as a list of dictionaries with keys: title (format as "GitHub Trending: {name}"), 
            description, source (set to "github_trending"), url (format as "https://github.com/{owner}/{repo}")
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            if response and 'content' in response:
                import ast
                try:
                    trending_data = ast.literal_eval(response['content'])
                    if isinstance(trending_data, list):
                        trending = trending_data[:5]
                except:
                    trending.append({
                        'title': 'GitHub Trending Projects',
                        'description': response['content'][:200],
                        'source': 'github_trending',
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error getting GitHub trending: {e}")
        
        return trending
    
    def _get_programming_news(self) -> List[Dict[str, str]]:
        """Get programming and development news using AI research.
        
        Returns:
            List of programming news dictionaries
        """
        news = []
        
        if not self.ai_brain:
            return news
            
        try:
            research_prompt = """
            Research current programming and software development news in June 2025.
            Include:
            - Major programming language updates or announcements
            - New framework releases or significant updates
            - Open source project milestones
            - Developer community news
            - Industry shifts or trends
            
            Format as a list of dictionaries with keys: title, snippet, url (set url to 'ai-research')
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            if response and 'content' in response:
                import ast
                try:
                    news_data = ast.literal_eval(response['content'])
                    if isinstance(news_data, list):
                        news = news_data
                except:
                    news.append({
                        'title': 'Programming News June 2025',
                        'snippet': response['content'][:200],
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error in _get_programming_news: {e}")
        
        return news
    
    def _get_general_programming_trends(self) -> List[Dict[str, str]]:
        """Get general programming trends using AI research.
        
        Returns:
            List of general trend dictionaries
        """
        trends = []
        
        if not self.ai_brain:
            return trends
            
        try:
            research_prompt = """
            Research current best programming practices and software architecture trends in 2025.
            Include:
            - Emerging design patterns and architectural styles
            - DevOps and deployment best practices
            - Code quality and testing trends
            - Performance optimization techniques
            - Team collaboration patterns
            
            Format as a list of dictionaries with keys: title, snippet, url (set url to 'ai-research')
            """
            
            response = self.ai_brain.generate_enhanced_response(research_prompt)
            
            if response and 'content' in response:
                import ast
                try:
                    trend_data = ast.literal_eval(response['content'])
                    if isinstance(trend_data, list):
                        trends = trend_data
                except:
                    trends.append({
                        'title': 'Programming Best Practices 2025',
                        'snippet': response['content'][:200],
                        'url': 'ai-research'
                    })
                    
        except Exception as e:
            print(f"Error in _get_general_programming_trends: {e}")
        
        return trends
    
    
    def _save_context(self, context: Dict[str, Any]) -> None:
        """Save gathered context to file.
        
        Args:
            context: Context dictionary to save
        """
        try:
            with open(self.output_path, 'w') as f:
                json.dump(context, f, indent=2, sort_keys=True)
            print(f"Context saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving context: {e}")
    
    def _enhance_context_with_ai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance gathered context with AI analysis.
        
        Args:
            context: Raw context data
            
        Returns:
            Enhanced context with AI insights
        """
        try:
            enhanced_context = context.copy()
            enhanced_context["ai_analysis"] = {}
            
            # Analyze market trends
            if context.get("market_trends"):
                trends_text = " ".join([item.get("snippet", "") for item in context["market_trends"][:5]])
                if trends_text:
                    analysis = self.ai_brain.analyze_with_research_ai(trends_text, "market trends")
                    if analysis:
                        enhanced_context["ai_analysis"]["market_trends_insights"] = analysis
            
            # Analyze security alerts
            if context.get("security_alerts"):
                security_text = " ".join([item.get("snippet", "") for item in context["security_alerts"][:3]])
                if security_text:
                    analysis = self.ai_brain.analyze_with_research_ai(security_text, "security")
                    if analysis:
                        enhanced_context["ai_analysis"]["security_insights"] = analysis
            
            # Analyze technology updates
            if context.get("technology_updates"):
                tech_text = " ".join([item.get("snippet", "") for item in context["technology_updates"][:3]])
                if tech_text:
                    analysis = self.ai_brain.analyze_with_research_ai(tech_text, "technical")
                    if analysis:
                        enhanced_context["ai_analysis"]["technology_insights"] = analysis
            
            # Overall strategic analysis
            all_content = []
            for key in ["market_trends", "security_alerts", "technology_updates", "github_trending"]:
                if context.get(key):
                    for item in context[key][:2]:
                        if item.get("snippet"):
                            all_content.append(item["snippet"][:200])
                        elif item.get("description"):
                            all_content.append(item["description"][:200])
            
            if all_content:
                combined_content = " | ".join(all_content)
                strategic_analysis = self.ai_brain.analyze_with_research_ai(
                    combined_content, 
                    "strategic software development"
                )
                if strategic_analysis:
                    enhanced_context["ai_analysis"]["strategic_recommendations"] = strategic_analysis
            
            # Add AI provider status
            if hasattr(self.ai_brain, 'get_research_ai_status'):
                enhanced_context["ai_analysis"]["research_ai_status"] = self.ai_brain.get_research_ai_status()
            
            print(f"Enhanced context with {len(enhanced_context.get('ai_analysis', {}))} AI insights")
            
        except Exception as e:
            print(f"Error enhancing context with AI: {e}")
            enhanced_context = context.copy()
        
        return enhanced_context
    
    def gather_workflow_context(self) -> Dict[str, Any]:
        """Gather minimal context optimized for GitHub Actions workflow.
        
        Collects only essential context data for fast CI execution.
        
        Returns:
            Minimal context dictionary for workflow
        """
        workflow_context = {
            'environment': 'github_actions',
            'capabilities': [
                'GitHub API integration',
                'AI Model access',
                'Task generation',
                'Repository management'
            ],
            'market_trends': [],  # Skip expensive API calls in CI
            'security_alerts': [],
            'technology_updates': [],
            'github_trending': [],
            'programming_news': [],
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'optimized_for': 'ci_performance',
            'limited_scope': True
        }
        
        # Add GitHub Actions specific information
        if os.getenv('GITHUB_ACTIONS'):
            workflow_context.update({
                'github_workspace': os.getenv('GITHUB_WORKSPACE'),
                'github_event_name': os.getenv('GITHUB_EVENT_NAME'),
                'github_sha': os.getenv('GITHUB_SHA'),
                'runner_os': os.getenv('RUNNER_OS'),
                'runner_arch': os.getenv('RUNNER_ARCH')
            })
        
        print("Gathered workflow-optimized context")
        return workflow_context
    
    def gather_production_context(self, charter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Gather complete context for production environment.
        
        Performs full context gathering with all data sources.
        
        Args:
            charter: System charter for context filtering
            
        Returns:
            Complete context dictionary for production
        """
        try:
            # Get full context with all data sources
            full_context = self.gather_context(charter or {})
            
            # Add production-specific metadata
            full_context.update({
                'environment': 'production',
                'full_data_gathering': True,
                'production_load_time': datetime.now(timezone.utc).isoformat()
            })
            
            print("Gathered complete production context")
            return full_context
            
        except Exception as e:
            print(f"Failed to gather production context: {e}")
            raise


def main():
    """Main function for standalone execution."""
    from state_manager import StateManager
    
    # Load system state to get charter
    state_manager = StateManager()
    state = state_manager.load_state()
    charter = state.get("charter", {})
    
    # Gather context
    gatherer = ContextGatherer()
    context = gatherer.gather_context(charter)
    
    print(f"Gathered {len(context.get('market_trends', []))} market trends")
    print(f"Gathered {len(context.get('security_alerts', []))} security alerts")
    print(f"Gathered {len(context.get('technology_updates', []))} technology updates")
    print(f"Gathered {len(context.get('github_trending', []))} GitHub trending items")
    print(f"Gathered {len(context.get('programming_news', []))} programming news items")


if __name__ == "__main__":
    main()