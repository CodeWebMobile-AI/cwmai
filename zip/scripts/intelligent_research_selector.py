"""
Intelligent Research Selector - Autonomously selects research topics based on system needs.

This module maps performance gaps to specific research queries, prioritizes by expected
value, and ensures research efforts focus on CWMAI's core needs while avoiding redundancy.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib


class IntelligentResearchSelector:
    """Autonomously select research topics based on system needs."""
    
    def __init__(self, knowledge_store=None, need_analyzer=None):
        self.knowledge_store = knowledge_store
        self.need_analyzer = need_analyzer
        self.research_history = []
        self.topic_effectiveness = {}
        
        # Core research domains aligned with CWMAI needs
        self.research_domains = {
            "task_generation": {
                "topics": [
                    "Task decomposition strategies",
                    "Complexity scoring models", 
                    "Success criteria definition",
                    "Natural language patterns for implementable tasks",
                    "Task validation techniques"
                ],
                "keywords": ["task", "decomposition", "complexity", "criteria", "validation"]
            },
            "claude_interaction": {
                "topics": [
                    "GitHub issue formatting for AI implementation",
                    "Prompt engineering for code generation",
                    "Acceptance criteria patterns",
                    "Context optimization for AI assistants",
                    "Task clarity and specificity"
                ],
                "keywords": ["claude", "prompt", "github", "issue", "formatting", "ai"]
            },
            "multi_agent_coordination": {
                "topics": [
                    "Consensus building mechanisms",
                    "Agent specialization strategies",
                    "Weighted voting systems",
                    "Swarm decision optimization",
                    "Communication protocols"
                ],
                "keywords": ["agent", "consensus", "swarm", "coordination", "voting"]
            },
            "outcome_learning": {
                "topics": [
                    "Pattern recognition in task outcomes",
                    "Success/failure correlation analysis",
                    "Feedback loop optimization",
                    "Predictive modeling for task value",
                    "Learning rate optimization"
                ],
                "keywords": ["outcome", "pattern", "learning", "feedback", "prediction"]
            },
            "portfolio_management": {
                "topics": [
                    "Project selection criteria",
                    "Cross-project synergy identification",
                    "Resource allocation models",
                    "Portfolio diversification strategies",
                    "Project health metrics"
                ],
                "keywords": ["portfolio", "project", "resource", "allocation", "synergy"]
            }
        }
        
        # Research query templates
        self.query_templates = {
            "best_practices": "What are the best practices for {} in {}?",
            "implementation": "How to implement {} for {}?",
            "optimization": "How to optimize {} to improve {}?",
            "patterns": "What patterns lead to successful {} in {}?",
            "strategies": "What strategies can improve {} when {}?",
            "techniques": "What techniques are effective for {} in the context of {}?"
        }
        
    def select_research_topics(self, context: Dict) -> List[Dict]:
        """
        Intelligently select research topics based on system needs.
        
        Args:
            context: Current system state and performance metrics
            
        Returns:
            List of prioritized research queries
        """
        research_queries = []
        
        # Get performance gaps from need analyzer
        if self.need_analyzer:
            gaps = self.need_analyzer.analyze_performance_gaps()
        else:
            gaps = self._analyze_context_for_gaps(context)
        
        # Map gaps to specific research queries
        for priority_level in ["critical", "high", "medium"]:
            for gap in gaps.get(priority_level, []):
                queries = self._generate_queries_for_gap(gap, context)
                research_queries.extend(queries)
        
        # Remove redundant research (already done recently)
        filtered_queries = self._filter_redundant_research(research_queries)
        
        # Prioritize by expected impact
        prioritized_queries = self._prioritize_by_impact(filtered_queries, context)
        
        # Track selection for learning
        self._record_selection(prioritized_queries)
        
        return prioritized_queries[:10]  # Return top 10 queries
    
    def _analyze_context_for_gaps(self, context: Dict) -> Dict:
        """Analyze context directly if need analyzer not available."""
        gaps = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Check Claude success rate
        claude_success = context.get("metrics", {}).get("claude_success_rate", 1.0)
        if claude_success == 0:
            gaps["critical"].append({
                "area": "claude_interaction",
                "severity": "critical",
                "issues": ["Zero Claude interaction success rate"],
                "research_needs": [
                    "GitHub issue formatting for AI",
                    "Prompt engineering patterns",
                    "Task clarity optimization"
                ],
                "expected_impact": "Improve Claude success from 0% to 50%+"
            })
        
        # Check task completion rate
        task_completion = context.get("metrics", {}).get("task_completion_rate", 1.0)
        if task_completion < 0.2:
            gaps["high"].append({
                "area": "task_generation",
                "severity": "high",
                "issues": ["Very low task completion rate"],
                "research_needs": [
                    "Task decomposition strategies",
                    "Complexity management",
                    "Success criteria definition"
                ],
                "expected_impact": "Improve task completion from {}% to 40%+".format(
                    int(task_completion * 100)
                )
            })
        
        # Check agent coordination
        consensus_rate = context.get("swarm_metrics", {}).get("average_consensus", 1.0)
        if consensus_rate < 0.7:
            gaps["medium"].append({
                "area": "multi_agent_coordination",
                "severity": "medium",
                "issues": ["Low agent consensus levels"],
                "research_needs": [
                    "Consensus mechanisms",
                    "Agent specialization",
                    "Voting optimization"
                ],
                "expected_impact": "Boost consensus from {}% to 85%+".format(
                    int(consensus_rate * 100)
                )
            })
        
        return gaps
    
    def _generate_queries_for_gap(self, gap: Dict, context: Dict) -> List[Dict]:
        """Generate specific research queries for a performance gap."""
        queries = []
        
        area = gap["area"]
        severity = gap["severity"]
        issues = gap["issues"]
        research_needs = gap["research_needs"]
        
        # Generate queries for each research need
        for need in research_needs:
            # Create multiple query variations
            query_variations = self._create_query_variations(need, area, context)
            
            for query_text in query_variations:
                query = {
                    "topic": need,
                    "query": query_text,
                    "area": area,
                    "severity": severity,
                    "priority": self._calculate_query_priority(severity, area, context),
                    "expected_impact": gap["expected_impact"],
                    "context": {
                        "issues": issues,
                        "current_metrics": self._extract_relevant_metrics(area, context),
                        "failure_patterns": self._extract_failure_patterns(area, context)
                    },
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "gap_id": self._generate_gap_id(gap),
                        "estimated_research_time": self._estimate_research_time(need)
                    }
                }
                queries.append(query)
        
        return queries
    
    def _create_query_variations(self, need: str, area: str, context: Dict) -> List[str]:
        """Create multiple query variations for better research coverage."""
        variations = []
        
        # Base query
        base_query = need
        
        # Add context-specific variations
        if area == "claude_interaction" and context.get("metrics", {}).get("claude_success_rate", 1) == 0:
            variations.extend([
                f"Why do GitHub issues with Claude mentions fail: {need}",
                f"Successful Claude AI task patterns for {need}",
                f"How to format tasks for Claude to implement: {need}"
            ])
        
        elif area == "task_generation":
            variations.extend([
                f"Best practices for {need} in software development tasks",
                f"How to improve {need} for AI-implementable tasks",
                f"Common patterns in successful {need}"
            ])
        
        elif area == "multi_agent_coordination":
            variations.extend([
                f"Implementing {need} in multi-agent systems",
                f"Optimizing {need} for distributed decision making",
                f"Real-world examples of effective {need}"
            ])
        
        else:
            # Generic variations
            for template_name, template in self.query_templates.items():
                if template_name in ["best_practices", "implementation", "optimization"]:
                    variations.append(template.format(need, "autonomous development systems"))
        
        return variations[:3]  # Limit to 3 variations per need
    
    def _filter_redundant_research(self, queries: List[Dict]) -> List[Dict]:
        """Filter out research that was done recently."""
        filtered = []
        
        for query in queries:
            # Check if similar research was done recently
            if self._is_research_recent(query):
                continue
            
            # Check knowledge store for existing research
            if self.knowledge_store and self._has_valid_existing_research(query):
                continue
            
            filtered.append(query)
        
        return filtered
    
    def _is_research_recent(self, query: Dict) -> bool:
        """Check if similar research was done recently."""
        query_hash = self._generate_query_hash(query["query"])
        
        # Check research history
        for past_research in self.research_history[-50:]:  # Last 50 researches
            if past_research.get("query_hash") == query_hash:
                # Check if it was done in the last 24 hours
                research_time = datetime.fromisoformat(past_research["timestamp"])
                if datetime.now() - research_time < timedelta(hours=24):
                    return True
        
        return False
    
    def _has_valid_existing_research(self, query: Dict) -> bool:
        """Check if valid research already exists in knowledge store."""
        if not self.knowledge_store:
            return False
        
        # Search for similar research
        search_results = self.knowledge_store.search_research(
            query["topic"], 
            search_in=["content", "type"]
        )
        
        # Check if any recent, high-quality research exists
        for result in search_results:
            # Check age
            research_time = datetime.fromisoformat(result["timestamp"])
            if datetime.now() - research_time > timedelta(days=7):
                continue
            
            # Check quality
            if result.get("quality_score", 0) < 0.7:
                continue
            
            # Check relevance
            if self._calculate_relevance_score(result, query) > 0.8:
                return True
        
        return False
    
    def _prioritize_by_impact(self, queries: List[Dict], context: Dict) -> List[Dict]:
        """Prioritize research queries by expected impact."""
        # Calculate impact scores
        scored_queries = []
        
        for query in queries:
            impact_score = self._calculate_impact_score(query, context)
            query["impact_score"] = impact_score
            scored_queries.append((impact_score, query))
        
        # Sort by impact score (highest first)
        scored_queries.sort(key=lambda x: x[0], reverse=True)
        
        # Return prioritized queries
        return [query for _, query in scored_queries]
    
    def _calculate_impact_score(self, query: Dict, context: Dict) -> float:
        """Calculate expected impact score for a research query."""
        base_scores = {
            "claude_interaction": 0.9,  # Critical for execution
            "task_generation": 0.8,     # Core functionality
            "outcome_learning": 0.7,    # Improvement capability
            "multi_agent_coordination": 0.6,  # Quality enhancement
            "portfolio_management": 0.5      # Strategic value
        }
        
        severity_multipliers = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.0,
            "low": 0.5
        }
        
        # Base score from area
        area_score = base_scores.get(query["area"], 0.5)
        
        # Severity multiplier
        severity_mult = severity_multipliers.get(query["severity"], 1.0)
        
        # Current performance penalty (worse performance = higher priority)
        performance_penalty = 0
        if query["area"] == "claude_interaction":
            success_rate = context.get("metrics", {}).get("claude_success_rate", 1)
            performance_penalty = (1 - success_rate) * 0.5
        elif query["area"] == "task_generation":
            completion_rate = context.get("metrics", {}).get("task_completion_rate", 1)
            performance_penalty = (1 - completion_rate) * 0.4
        
        # Historical effectiveness bonus
        topic_effectiveness = self.topic_effectiveness.get(query["topic"], 0.5)
        effectiveness_bonus = topic_effectiveness * 0.2
        
        # Calculate final score
        impact_score = (area_score * severity_mult) + performance_penalty + effectiveness_bonus
        
        return min(1.0, impact_score)
    
    def _record_selection(self, queries: List[Dict]):
        """Record research selection for learning."""
        for query in queries:
            self.research_history.append({
                "query": query["query"],
                "query_hash": self._generate_query_hash(query["query"]),
                "topic": query["topic"],
                "area": query["area"],
                "timestamp": datetime.now().isoformat(),
                "impact_score": query.get("impact_score", 0)
            })
        
        # Keep history size manageable
        if len(self.research_history) > 1000:
            self.research_history = self.research_history[-1000:]
    
    def update_topic_effectiveness(self, topic: str, effectiveness: float):
        """Update effectiveness score for a research topic based on outcomes."""
        if topic in self.topic_effectiveness:
            # Weighted average with existing score
            old_score = self.topic_effectiveness[topic]
            self.topic_effectiveness[topic] = old_score * 0.7 + effectiveness * 0.3
        else:
            self.topic_effectiveness[topic] = effectiveness
    
    def _calculate_query_priority(self, severity: str, area: str, context: Dict) -> str:
        """Calculate priority level for a query."""
        if severity == "critical":
            return "CRITICAL"
        
        if severity == "high" and area in ["claude_interaction", "task_generation"]:
            return "HIGH"
        
        if area == "outcome_learning" and len(context.get("learned_patterns", [])) < 5:
            return "HIGH"
        
        return "MEDIUM"
    
    def _extract_relevant_metrics(self, area: str, context: Dict) -> Dict:
        """Extract metrics relevant to the research area."""
        metrics = {}
        
        if area == "claude_interaction":
            metrics = {
                "success_rate": context.get("metrics", {}).get("claude_success_rate", 0),
                "total_attempts": context.get("metrics", {}).get("claude_attempts", 0),
                "last_success": context.get("metrics", {}).get("last_claude_success")
            }
        elif area == "task_generation":
            metrics = {
                "completion_rate": context.get("metrics", {}).get("task_completion_rate", 0),
                "total_tasks": context.get("metrics", {}).get("total_tasks", 0),
                "average_task_age": context.get("metrics", {}).get("avg_task_age_hours", 0)
            }
        elif area == "multi_agent_coordination":
            metrics = {
                "consensus_rate": context.get("swarm_metrics", {}).get("average_consensus", 0),
                "decision_time": context.get("swarm_metrics", {}).get("avg_decision_time", 0),
                "agent_count": context.get("swarm_metrics", {}).get("agent_count", 0)
            }
        
        return metrics
    
    def _extract_failure_patterns(self, area: str, context: Dict) -> List[str]:
        """Extract failure patterns relevant to the research area."""
        patterns = []
        
        if area == "claude_interaction":
            patterns = context.get("failure_patterns", {}).get("claude_failures", [])
        elif area == "task_generation":
            patterns = context.get("failure_patterns", {}).get("task_failures", [])
        
        return patterns[:5]  # Limit to top 5 patterns
    
    def _generate_gap_id(self, gap: Dict) -> str:
        """Generate unique ID for a gap."""
        gap_string = f"{gap['area']}_{gap['severity']}_{','.join(gap['issues'])}"
        return hashlib.md5(gap_string.encode()).hexdigest()[:8]
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for a query for deduplication."""
        return hashlib.md5(query.lower().encode()).hexdigest()[:16]
    
    def _estimate_research_time(self, need: str) -> int:
        """Estimate time needed for research in minutes."""
        # Simple estimation based on complexity
        if "optimization" in need.lower() or "strategies" in need.lower():
            return 30
        elif "implementation" in need.lower() or "patterns" in need.lower():
            return 20
        else:
            return 15
    
    def _calculate_relevance_score(self, existing_research: Dict, query: Dict) -> float:
        """Calculate how relevant existing research is to a new query."""
        score = 0.0
        
        # Topic match
        if query["topic"].lower() in str(existing_research.get("content", "")).lower():
            score += 0.4
        
        # Area match
        if query["area"] == existing_research.get("type", ""):
            score += 0.3
        
        # Keyword overlap
        query_keywords = set(query["topic"].lower().split())
        content_keywords = set(str(existing_research.get("content", "")).lower().split())
        overlap = len(query_keywords & content_keywords) / max(1, len(query_keywords))
        score += overlap * 0.3
        
        return min(1.0, score)
    
    def get_research_summary(self) -> Dict:
        """Get summary of research selection activities."""
        return {
            "total_queries_generated": len(self.research_history),
            "topic_effectiveness": self.topic_effectiveness,
            "recent_selections": self.research_history[-10:],
            "most_effective_topics": sorted(
                self.topic_effectiveness.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }