"""
Research Query Generator - Creates specific, actionable research queries.

This module transforms high-level research needs into targeted queries that
include full context and are optimized for different AI research providers.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re


class ResearchQueryGenerator:
    """Transform high-level needs into specific research queries."""
    
    def __init__(self):
        # Query templates organized by intent
        self.query_templates = {
            "best_practices": [
                "What are the proven best practices for {topic} in {context}?",
                "What are the most effective strategies for {topic} when {situation}?",
                "What do successful implementations of {topic} have in common in {domain}?"
            ],
            "implementation": [
                "How to implement {topic} for {context} step by step?",
                "What is the most practical approach to {topic} in {environment}?",
                "How should {topic} be integrated into {system_type}?"
            ],
            "optimization": [
                "How to optimize {topic} to improve {metric}?",
                "What techniques can enhance {topic} performance in {context}?",
                "How to overcome common problems with {topic} in {scenario}?"
            ],
            "patterns": [
                "What patterns lead to successful {topic} in {domain}?",
                "What are the common failure patterns in {topic} and how to avoid them?",
                "What are the key indicators of effective {topic}?"
            ],
            "comparison": [
                "What are the different approaches to {topic} and their trade-offs?",
                "How do various {topic} strategies compare in terms of {criteria}?",
                "What are the pros and cons of different {topic} methods?"
            ],
            "troubleshooting": [
                "Why does {topic} fail in {context} and how to fix it?",
                "What are the most common issues with {topic} and their solutions?",
                "How to diagnose and resolve {topic} problems in {environment}?"
            ]
        }
        
        # Context-specific templates for CWMAI domains
        self.domain_templates = {
            "claude_interaction": {
                "failure_analysis": [
                    "Why do GitHub issues fail to get Claude AI responses: specific formatting and content requirements",
                    "What GitHub issue templates lead to successful Claude AI implementations",
                    "How to structure development tasks for AI assistant completion: acceptance criteria and clarity patterns"
                ],
                "optimization": [
                    "Best practices for prompt engineering in GitHub issues for code generation",
                    "How to write clear, implementable task descriptions for AI assistants",
                    "Successful patterns in AI-assisted software development task management"
                ]
            },
            "task_generation": {
                "quality": [
                    "How to decompose complex software development tasks into implementable subtasks",
                    "What task complexity scoring models work best for autonomous development systems",
                    "How to define success criteria that lead to high task completion rates"
                ],
                "patterns": [
                    "What patterns in task descriptions correlate with high completion rates",
                    "How to identify tasks that are too complex for single implementation cycles",
                    "What makes development tasks clear and actionable for AI implementation"
                ]
            },
            "multi_agent_coordination": {
                "consensus": [
                    "How to implement effective consensus mechanisms in multi-agent software development systems",
                    "What voting strategies work best for distributed AI decision making",
                    "How to optimize agent specialization for better collective decisions"
                ],
                "optimization": [
                    "How to reduce decision time while maintaining consensus quality in swarm intelligence",
                    "What agent communication patterns lead to better coordination outcomes",
                    "How to handle disagreement and conflict resolution in multi-agent systems"
                ]
            },
            "outcome_learning": {
                "patterns": [
                    "How to identify patterns in software development task success and failure",
                    "What metrics best predict task completion probability in autonomous development",
                    "How to build effective feedback loops for continuous system improvement"
                ],
                "optimization": [
                    "How to optimize learning algorithms for software development outcome prediction",
                    "What features are most important for predicting task success in AI-driven development",
                    "How to implement effective value assessment for completed development tasks"
                ]
            },
            "portfolio_management": {
                "selection": [
                    "How to select software projects for an autonomous development portfolio",
                    "What criteria identify high-synergy projects for parallel development",
                    "How to balance project diversity with resource efficiency in automated development"
                ],
                "optimization": [
                    "How to optimize resource allocation across multiple software development projects",
                    "What models work best for cross-project synergy identification",
                    "How to measure and improve portfolio health in autonomous development systems"
                ]
            }
        }
        
        # Provider-specific formatting
        self.provider_formats = {
            "anthropic": {
                "prefix": "Please provide a comprehensive analysis of:",
                "suffix": "Include specific examples, implementation details, and potential challenges.",
                "context_emphasis": "Given the context of autonomous software development systems"
            },
            "openai": {
                "prefix": "Analyze and explain:",
                "suffix": "Provide actionable insights with concrete examples.",
                "context_emphasis": "In the context of AI-driven development workflows"
            },
            "gemini": {
                "prefix": "Research and summarize:",
                "suffix": "Focus on practical applications and real-world outcomes.",
                "context_emphasis": "For autonomous development and task management systems"
            }
        }
        
    def generate_queries(self, research_need: Dict) -> List[Dict]:
        """
        Generate targeted research queries for a research need.
        
        Args:
            research_need: Dictionary containing research requirements
            
        Returns:
            List of formatted research queries
        """
        queries = []
        
        topic = research_need.get("topic", "")
        area = research_need.get("area", "")
        context = research_need.get("context", {})
        severity = research_need.get("severity", "medium")
        
        # Generate domain-specific queries first
        domain_queries = self._generate_domain_queries(area, topic, context)
        queries.extend(domain_queries)
        
        # Generate template-based queries
        template_queries = self._generate_template_queries(topic, area, context, severity)
        queries.extend(template_queries)
        
        # Add context-specific variations
        contextualized_queries = self._add_context_variations(queries, context)
        
        # Format for different providers
        formatted_queries = self._format_for_providers(contextualized_queries, research_need)
        
        return formatted_queries[:5]  # Limit to top 5 queries
    
    def _generate_domain_queries(self, area: str, topic: str, context: Dict) -> List[Dict]:
        """Generate queries specific to CWMAI domains."""
        queries = []
        
        if area not in self.domain_templates:
            return queries
        
        domain_templates = self.domain_templates[area]
        
        # Try to match topic to template category
        topic_lower = topic.lower()
        
        for category, templates in domain_templates.items():
            if self._topic_matches_category(topic_lower, category):
                for template in templates:
                    query = {
                        "query": template,
                        "category": category,
                        "confidence": 0.9,
                        "source": "domain_specific"
                    }
                    queries.append(query)
        
        return queries
    
    def _generate_template_queries(self, topic: str, area: str, context: Dict, severity: str) -> List[Dict]:
        """Generate queries using general templates."""
        queries = []
        
        # Determine query intent based on context and severity
        intents = self._determine_query_intents(topic, area, context, severity)
        
        for intent in intents:
            if intent in self.query_templates:
                templates = self.query_templates[intent]
                
                for template in templates[:2]:  # Limit to 2 per intent
                    formatted_query = self._format_template(template, topic, area, context)
                    
                    query = {
                        "query": formatted_query,
                        "category": intent,
                        "confidence": 0.7,
                        "source": "template"
                    }
                    queries.append(query)
        
        return queries
    
    def _determine_query_intents(self, topic: str, area: str, context: Dict, severity: str) -> List[str]:
        """Determine what types of queries to generate."""
        intents = []
        
        # High severity issues need troubleshooting
        if severity in ["critical", "high"]:
            intents.append("troubleshooting")
            intents.append("best_practices")
        
        # Optimization for performance issues
        if "rate" in topic.lower() or "performance" in topic.lower():
            intents.append("optimization")
            intents.append("patterns")
        
        # Implementation for new capabilities
        if "implementation" in topic.lower() or "strategies" in topic.lower():
            intents.append("implementation")
            intents.append("comparison")
        
        # Default intents
        if not intents:
            intents = ["best_practices", "implementation"]
        
        return intents[:3]  # Limit to 3 intents
    
    def _format_template(self, template: str, topic: str, area: str, context: Dict) -> str:
        """Format a template with specific values."""
        # Extract context information
        system_type = "autonomous development system"
        domain = "software development"
        environment = "AI-driven development environment"
        
        # Get specific context from the research need
        current_metrics = context.get("current_metrics", {})
        failure_patterns = context.get("failure_patterns", [])
        
        # Determine context values based on area
        context_values = self._get_context_values(area, current_metrics, failure_patterns)
        
        # Format the template
        formatted = template.format(
            topic=topic,
            context=context_values.get("context", system_type),
            situation=context_values.get("situation", "building autonomous systems"),
            domain=domain,
            environment=environment,
            metric=context_values.get("metric", "system performance"),
            system_type=system_type,
            scenario=context_values.get("scenario", "autonomous development"),
            criteria=context_values.get("criteria", "effectiveness and reliability")
        )
        
        return formatted
    
    def _get_context_values(self, area: str, metrics: Dict, patterns: List[str]) -> Dict:
        """Get area-specific context values."""
        context_maps = {
            "claude_interaction": {
                "context": "GitHub issue-based AI interaction",
                "situation": "Claude AI is not responding to development tasks",
                "metric": "Claude response rate and implementation success",
                "scenario": "AI assistant task delegation",
                "criteria": "response rate and code quality"
            },
            "task_generation": {
                "context": "autonomous task creation and management",
                "situation": "generating implementable development tasks",
                "metric": "task completion rate and quality",
                "scenario": "automated software development",
                "criteria": "completion rate and task clarity"
            },
            "multi_agent_coordination": {
                "context": "distributed AI agent collaboration",
                "situation": "coordinating multiple AI agents for decisions",
                "metric": "consensus quality and decision speed",
                "scenario": "swarm intelligence systems",
                "criteria": "consensus rate and decision accuracy"
            }
        }
        
        return context_maps.get(area, {
            "context": "autonomous systems",
            "situation": "optimizing system performance",
            "metric": "overall effectiveness",
            "scenario": "automated processes",
            "criteria": "performance and reliability"
        })
    
    def _add_context_variations(self, queries: List[Dict], context: Dict) -> List[Dict]:
        """Add context-specific variations to queries."""
        enhanced_queries = []
        
        for query in queries:
            base_query = query["query"]
            
            # Create variations with specific context
            variations = [base_query]
            
            # Add failure pattern context if available
            failure_patterns = context.get("failure_patterns", [])
            if failure_patterns:
                pattern_context = f" given these failure patterns: {', '.join(failure_patterns[:3])}"
                variations.append(base_query + pattern_context)
            
            # Add metric context if available
            current_metrics = context.get("current_metrics", {})
            if current_metrics:
                metric_info = []
                for key, value in current_metrics.items():
                    if isinstance(value, (int, float)):
                        if value < 0.3:  # Poor performance
                            metric_info.append(f"current {key}: {value:.1%} (poor)")
                        elif value > 0.8:  # Good performance
                            metric_info.append(f"current {key}: {value:.1%} (good)")
                
                if metric_info:
                    metric_context = f" (current performance: {', '.join(metric_info)})"
                    variations.append(base_query + metric_context)
            
            # Add the best variation
            for variation in variations:
                enhanced_query = query.copy()
                enhanced_query["query"] = variation
                enhanced_queries.append(enhanced_query)
        
        return enhanced_queries
    
    def _format_for_providers(self, queries: List[Dict], research_need: Dict) -> List[Dict]:
        """Format queries for different AI providers."""
        formatted_queries = []
        
        # Default to anthropic if no provider specified
        target_providers = research_need.get("providers", ["anthropic"])
        
        for query in queries:
            for provider in target_providers:
                if provider in self.provider_formats:
                    format_config = self.provider_formats[provider]
                    
                    formatted_query = {
                        "query": f"{format_config['prefix']} {query['query']}. {format_config['suffix']}",
                        "provider": provider,
                        "context_emphasis": format_config["context_emphasis"],
                        "category": query.get("category", "general"),
                        "confidence": query.get("confidence", 0.5),
                        "source": query.get("source", "template"),
                        "metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "area": research_need.get("area"),
                            "topic": research_need.get("topic"),
                            "priority": research_need.get("priority", "medium")
                        }
                    }
                    
                    formatted_queries.append(formatted_query)
        
        # Sort by confidence and uniqueness
        return self._rank_and_deduplicate(formatted_queries)
    
    def _rank_and_deduplicate(self, queries: List[Dict]) -> List[Dict]:
        """Rank queries by quality and remove duplicates."""
        # Remove near-duplicates
        unique_queries = []
        seen_queries = set()
        
        for query in queries:
            # Create a simplified version for comparison
            simplified = re.sub(r'[^\w\s]', '', query["query"].lower())
            simplified = ' '.join(simplified.split())
            
            if simplified not in seen_queries:
                seen_queries.add(simplified)
                unique_queries.append(query)
        
        # Sort by confidence, then by specificity
        unique_queries.sort(key=lambda q: (
            q.get("confidence", 0),
            len(q["query"]),  # Longer queries often more specific
            1 if q.get("source") == "domain_specific" else 0
        ), reverse=True)
        
        return unique_queries
    
    def _topic_matches_category(self, topic: str, category: str) -> bool:
        """Check if topic matches a template category."""
        category_keywords = {
            "failure_analysis": ["failure", "fail", "error", "problem", "issue"],
            "optimization": ["optimize", "improve", "enhance", "performance", "efficiency"],
            "quality": ["quality", "success", "completion", "effectiveness"],
            "patterns": ["pattern", "correlation", "success", "failure"],
            "consensus": ["consensus", "agreement", "coordination", "collaboration"],
            "selection": ["selection", "criteria", "choose", "portfolio"]
        }
        
        keywords = category_keywords.get(category, [])
        return any(keyword in topic for keyword in keywords)
    
    def create_contextual_query(self, topic: str, area: str, specific_context: str, 
                               severity: str = "medium") -> Dict:
        """
        Create a single, highly contextual query.
        
        Args:
            topic: Research topic
            area: CWMAI area
            specific_context: Specific situation context
            severity: Issue severity
            
        Returns:
            Single optimized query
        """
        # Build context-aware query
        if severity == "critical" and area == "claude_interaction":
            query_text = (
                f"Emergency analysis needed: {topic} - Claude AI has 0% response rate to GitHub issues. "
                f"Specific context: {specific_context}. "
                "What immediate changes to issue formatting, task description, or interaction patterns "
                "can restore Claude functionality?"
            )
        elif severity == "high" and area == "task_generation":
            query_text = (
                f"Urgent optimization needed for {topic}. "
                f"Current situation: {specific_context}. "
                "What proven strategies can immediately improve task completion rates and quality?"
            )
        else:
            query_text = (
                f"How to improve {topic} in the context of {specific_context}? "
                f"Focus on {area} optimization for autonomous development systems."
            )
        
        return {
            "query": query_text,
            "area": area,
            "topic": topic,
            "severity": severity,
            "confidence": 0.95,
            "source": "contextual",
            "provider": "anthropic",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "context": specific_context,
                "type": "emergency" if severity == "critical" else "optimization"
            }
        }