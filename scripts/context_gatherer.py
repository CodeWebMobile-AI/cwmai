"""
Context Gatherer Module

Performs external context gathering and market intelligence for the autonomous AI system.
Searches for trending technologies, security alerts, and market trends to inform decision-making.
"""

import json
import os
import requests
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import urllib.parse


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
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AI-Bot/1.0; +https://github.com/CodeWebMobile-AI/cwmai)'
        })
    
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
        """Get innovation and technology trends.
        
        Returns:
            List of trend dictionaries
        """
        trends = []
        
        try:
            # Search for AI and technology trends
            search_terms = [
                "AI breakthrough 2025",
                "new programming languages trending",
                "cloud technology innovations",
                "developer tools 2025"
            ]
            
            for term in search_terms:
                try:
                    results = self._search_duckduckgo(term)
                    if results:
                        trends.extend(results[:2])  # Limit results per term
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"Error searching for {term}: {e}")
                    
        except Exception as e:
            print(f"Error in _get_innovation_trends: {e}")
        
        return trends[:10]  # Limit total results
    
    def _get_technology_updates(self) -> List[Dict[str, str]]:
        """Get technology updates and releases.
        
        Returns:
            List of technology update dictionaries
        """
        updates = []
        
        try:
            # Search for recent technology updates
            search_terms = [
                "Python 3.13 new features",
                "JavaScript framework updates 2025",
                "Docker updates security",
                "Kubernetes new release"
            ]
            
            for term in search_terms:
                try:
                    results = self._search_duckduckgo(term)
                    if results:
                        updates.extend(results[:1])  # One result per term
                    time.sleep(1)
                except Exception as e:
                    print(f"Error searching for {term}: {e}")
                    
        except Exception as e:
            print(f"Error in _get_technology_updates: {e}")
        
        return updates
    
    def _get_security_alerts(self) -> List[Dict[str, str]]:
        """Get security alerts and vulnerabilities.
        
        Returns:
            List of security alert dictionaries
        """
        alerts = []
        
        try:
            # Search for recent security vulnerabilities
            search_terms = [
                "CVE 2025 critical vulnerabilities",
                "npm security advisory",
                "Python security vulnerabilities",
                "GitHub security alert"
            ]
            
            for term in search_terms:
                try:
                    results = self._search_duckduckgo(term)
                    if results:
                        alerts.extend(results[:2])
                    time.sleep(1)
                except Exception as e:
                    print(f"Error searching for {term}: {e}")
                    
        except Exception as e:
            print(f"Error in _get_security_alerts: {e}")
        
        return alerts[:8]
    
    def _get_github_trending(self) -> List[Dict[str, str]]:
        """Get trending GitHub repositories.
        
        Returns:
            List of trending repository information
        """
        trending = []
        
        try:
            # Try to get trending repos from GitHub's trending page
            url = "https://github.com/trending"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                repo_elements = soup.find_all('article', class_='Box-row')
                
                for repo in repo_elements[:5]:  # Limit to top 5
                    try:
                        title_elem = repo.find('h2', class_='h3')
                        if title_elem:
                            repo_name = title_elem.get_text(strip=True)
                            description_elem = repo.find('p')
                            description = description_elem.get_text(strip=True) if description_elem else ""
                            
                            trending.append({
                                "title": f"GitHub Trending: {repo_name}",
                                "description": description[:200],
                                "source": "github_trending",
                                "url": f"https://github.com/{repo_name}"
                            })
                    except Exception as e:
                        print(f"Error parsing trending repo: {e}")
                        
        except Exception as e:
            print(f"Error getting GitHub trending: {e}")
        
        return trending
    
    def _get_programming_news(self) -> List[Dict[str, str]]:
        """Get programming and development news.
        
        Returns:
            List of programming news dictionaries
        """
        news = []
        
        try:
            # Search for programming news
            search_terms = [
                "programming news June 2025",
                "software development trends",
                "open source projects news"
            ]
            
            for term in search_terms:
                try:
                    results = self._search_duckduckgo(term)
                    if results:
                        news.extend(results[:2])
                    time.sleep(1)
                except Exception as e:
                    print(f"Error searching for {term}: {e}")
                    
        except Exception as e:
            print(f"Error in _get_programming_news: {e}")
        
        return news
    
    def _get_general_programming_trends(self) -> List[Dict[str, str]]:
        """Get general programming trends.
        
        Returns:
            List of general trend dictionaries
        """
        trends = []
        
        try:
            search_terms = [
                "best programming practices 2025",
                "software architecture trends"
            ]
            
            for term in search_terms:
                try:
                    results = self._search_duckduckgo(term)
                    if results:
                        trends.extend(results[:1])
                    time.sleep(1)
                except Exception as e:
                    print(f"Error searching for {term}: {e}")
                    
        except Exception as e:
            print(f"Error in _get_general_programming_trends: {e}")
        
        return trends
    
    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search DuckDuckGo for the given query.
        
        Args:
            query: Search query string
            
        Returns:
            List of search result dictionaries
        """
        results = []
        
        try:
            # Use DuckDuckGo instant answers API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get abstract if available
                if data.get('Abstract'):
                    results.append({
                        "title": data.get('Heading', query),
                        "description": data['Abstract'][:300],
                        "source": "duckduckgo_abstract",
                        "url": data.get('AbstractURL', '')
                    })
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            "title": topic.get('Text', '')[:100],
                            "description": topic.get('Text', '')[:300],
                            "source": "duckduckgo_related",
                            "url": topic.get('FirstURL', '')
                        })
                        
        except Exception as e:
            print(f"Error searching DuckDuckGo for '{query}': {e}")
        
        return results
    
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
                trends_text = " ".join([item.get("description", "") for item in context["market_trends"][:5]])
                if trends_text:
                    analysis = self.ai_brain.analyze_with_research_ai(trends_text, "market trends")
                    if analysis:
                        enhanced_context["ai_analysis"]["market_trends_insights"] = analysis
            
            # Analyze security alerts
            if context.get("security_alerts"):
                security_text = " ".join([item.get("description", "") for item in context["security_alerts"][:3]])
                if security_text:
                    analysis = self.ai_brain.analyze_with_research_ai(security_text, "security")
                    if analysis:
                        enhanced_context["ai_analysis"]["security_insights"] = analysis
            
            # Analyze technology updates
            if context.get("technology_updates"):
                tech_text = " ".join([item.get("description", "") for item in context["technology_updates"][:3]])
                if tech_text:
                    analysis = self.ai_brain.analyze_with_research_ai(tech_text, "technical")
                    if analysis:
                        enhanced_context["ai_analysis"]["technology_insights"] = analysis
            
            # Overall strategic analysis
            all_content = []
            for key in ["market_trends", "security_alerts", "technology_updates", "github_trending"]:
                if context.get(key):
                    for item in context[key][:2]:
                        if item.get("description"):
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