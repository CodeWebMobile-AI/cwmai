"""
Evolutionary Knowledge Synthesis

Bridges the evolutionary improvement system with the research knowledge system
to create breakthrough innovations through cross-domain insights.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
import numpy as np

from scripts.evolutionary_knowledge_store import EvolutionaryKnowledgeStore
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.simple_knowledge_synthesis_engine import SimpleKnowledgeSynthesisEngine
from scripts.http_ai_client import HTTPAIClient


class EvolutionaryKnowledgeSynthesis:
    """Synthesizes knowledge between improvement and research systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Knowledge stores
        self.improvement_store = EvolutionaryKnowledgeStore()
        self.research_store = ResearchKnowledgeStore()
        self.synthesis_engine = SimpleKnowledgeSynthesisEngine()
        self.ai_client = HTTPAIClient()
        
        # Cross-domain knowledge graph
        self.knowledge_graph = nx.DiGraph()
        self.cross_domain_insights = []
        
        # Synthesis parameters
        self.min_connection_strength = 0.3
        self.synthesis_interval = 10  # Synthesize every N improvements
        
        # Load existing synthesis
        self._load_synthesis_state()
        
    def _load_synthesis_state(self):
        """Load existing synthesis state."""
        state_file = Path("evolutionary_knowledge/synthesis_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.cross_domain_insights = state.get("insights", [])
                    # Rebuild graph from state
                    for edge in state.get("graph_edges", []):
                        self.knowledge_graph.add_edge(
                            edge["source"], 
                            edge["target"],
                            weight=edge["weight"],
                            relationship=edge["relationship"]
                        )
            except Exception as e:
                self.logger.warning(f"Could not load synthesis state: {e}")
                
    def _save_synthesis_state(self):
        """Save synthesis state."""
        state = {
            "insights": self.cross_domain_insights,
            "graph_edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                    "relationship": data.get("relationship", "related")
                }
                for u, v, data in self.knowledge_graph.edges(data=True)
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        state_file = Path("evolutionary_knowledge/synthesis_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    async def synthesize_cross_domain_insights(self) -> List[Dict[str, Any]]:
        """Find insights that bridge improvements and research."""
        insights = []
        
        # 1. Map improvements to research topics
        improvement_research_map = await self._map_improvements_to_research()
        
        # 2. Find research that could enhance improvements
        research_enhancement_map = await self._find_research_enhancements()
        
        # 3. Identify breakthrough combinations
        breakthrough_combinations = await self._identify_breakthrough_combinations()
        
        # 4. Generate synthesis insights
        for combo in breakthrough_combinations:
            insight = await self._generate_synthesis_insight(combo)
            if insight:
                insights.append(insight)
                self.cross_domain_insights.append(insight)
                
        # 5. Update knowledge graph
        self._update_knowledge_graph(insights)
        
        # Save state
        self._save_synthesis_state()
        
        return insights
        
    async def _map_improvements_to_research(self) -> Dict[str, List[str]]:
        """Map improvements to relevant research topics."""
        mapping = defaultdict(list)
        
        # Get all improvements
        improvements = []
        for imp_id, imp_data in self.improvement_store._iterate_all_improvements():
            improvements.append((imp_id, imp_data))
            
        # Get all research topics
        all_research = self.research_store.retrieve_research()
        research_topics = [r.get("research_type", "unknown") for r in all_research]
        research_topics = list(set(research_topics))  # Unique topics
        
        # Find connections
        for imp_id, imp_data in improvements:
            imp_text = self._create_improvement_text(imp_data)
            
            for topic in research_topics:
                relevance = await self._calculate_relevance(imp_text, topic)
                if relevance > self.min_connection_strength:
                    mapping[imp_id].append(topic)
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_edge(
                        f"imp:{imp_id}",
                        f"research:{topic}",
                        weight=relevance,
                        relationship="relates_to"
                    )
                    
        return mapping
        
    async def _find_research_enhancements(self) -> Dict[str, List[Dict]]:
        """Find research that could enhance existing improvements."""
        enhancements = defaultdict(list)
        
        # Get recent improvements
        recent_improvements = []
        for imp_id, imp_data in self.improvement_store._iterate_all_improvements():
            if "fitness" in imp_data and imp_data["fitness"] < 0.8:  # Room for improvement
                recent_improvements.append((imp_id, imp_data))
                
        # Find relevant research
        for imp_id, imp_data in recent_improvements[:20]:  # Top 20
            patterns = self.improvement_store._extract_patterns(imp_data)
            
            # Search research for enhancement ideas
            for pattern in patterns:
                research_results = self.research_store.search_research(pattern.split(':')[-1])
                
                for research in research_results[:3]:  # Top 3 matches
                    enhancement = {
                        "research_id": research.get("id", "unknown"),
                        "research_topic": research.get("topic", "unknown"),
                        "enhancement_type": self._classify_enhancement(research),
                        "potential_impact": self._estimate_impact(imp_data, research)
                    }
                    enhancements[imp_id].append(enhancement)
                    
        return enhancements
        
    async def _identify_breakthrough_combinations(self) -> List[Dict]:
        """Identify combinations that could lead to breakthroughs."""
        combinations = []
        
        # Use graph analysis to find interesting paths
        if len(self.knowledge_graph) > 10:
            # Find nodes with high betweenness centrality (bridges)
            centrality = nx.betweenness_centrality(self.knowledge_graph)
            bridge_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Find distant connections through bridges
            for bridge, _ in bridge_nodes:
                neighbors = list(self.knowledge_graph.neighbors(bridge))
                
                # Look for improvement-research pairs
                imp_neighbors = [n for n in neighbors if n.startswith("imp:")]
                research_neighbors = [n for n in neighbors if n.startswith("research:")]
                
                for imp in imp_neighbors:
                    for research in research_neighbors:
                        # Check if this is a non-obvious connection
                        if not self.knowledge_graph.has_edge(imp, research):
                            combinations.append({
                                "improvement": imp[4:],  # Remove prefix
                                "research": research[9:],  # Remove prefix
                                "bridge": bridge,
                                "type": "distant_connection"
                            })
                            
        # Find complementary patterns
        successful_patterns = self.improvement_store.get_successful_patterns()
        research_insights = self.synthesis_engine.synthesize_insights()
        
        for pattern, pattern_data in successful_patterns.items():
            for insight in research_insights:
                if self._are_complementary(pattern, insight):
                    combinations.append({
                        "pattern": pattern,
                        "insight": insight,
                        "type": "complementary_synthesis"
                    })
                    
        return combinations[:20]  # Top 20 combinations
        
    async def _generate_synthesis_insight(self, combination: Dict) -> Optional[Dict]:
        """Generate a synthesis insight from a combination."""
        try:
            if combination["type"] == "distant_connection":
                # Get improvement and research details
                imp_data = self.improvement_store._get_improvement_by_id(combination["improvement"])
                research_data = self.research_store.retrieve_research(research_id=combination["research"])
                
                if not imp_data or not research_data:
                    return None
                    
                prompt = f"""Synthesize a breakthrough insight by combining:

IMPROVEMENT: {imp_data.get('name', 'Unknown')}
- Type: {imp_data.get('type', 'Unknown')}
- Approach: {imp_data.get('approach', 'Unknown')}

RESEARCH: {research_data.get('topic', 'Unknown')}
- Content: {str(research_data.get('content', ''))[:500]}

BRIDGE CONCEPT: {combination.get('bridge', 'Unknown')}

Generate a novel insight that combines these in an unexpected way.
Focus on how the research could transform the improvement approach.

Return a JSON with:
- insight: The breakthrough insight
- implementation_approach: How to implement this
- expected_impact: What improvements to expect
- novel_aspects: What makes this unique
"""
                
            else:  # complementary_synthesis
                pattern = combination["pattern"]
                insight = combination["insight"]
                
                prompt = f"""Synthesize a breakthrough by combining:

SUCCESSFUL PATTERN: {pattern}
RESEARCH INSIGHT: {json.dumps(insight, indent=2)}

Create a novel approach that leverages both.

Return a JSON with:
- insight: The breakthrough insight
- implementation_approach: How to implement this
- expected_impact: What improvements to expect
- novel_aspects: What makes this unique
"""
            
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            # Parse response
            if isinstance(response, dict):
                synthesis = response
            else:
                try:
                    synthesis = json.loads(str(response))
                except:
                    return None
                    
            # Add metadata
            synthesis["combination"] = combination
            synthesis["synthesized_at"] = datetime.now(timezone.utc).isoformat()
            synthesis["synthesis_type"] = combination["type"]
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Synthesis generation failed: {e}")
            return None
            
    async def generate_breakthrough_improvements(self, count: int = 5) -> List[Dict]:
        """Generate breakthrough improvements from synthesis insights."""
        # First synthesize new insights
        new_insights = await self.synthesize_cross_domain_insights()
        
        # Select most promising insights
        all_insights = self.cross_domain_insights[-20:]  # Recent insights
        
        # Score insights by novelty and impact
        scored_insights = []
        for insight in all_insights:
            score = self._score_insight(insight)
            scored_insights.append((score, insight))
            
        scored_insights.sort(key=lambda x: x[0], reverse=True)
        
        # Generate improvements from top insights
        breakthroughs = []
        for score, insight in scored_insights[:count]:
            improvement = await self._generate_improvement_from_insight(insight)
            if improvement:
                improvement["breakthrough_score"] = score
                improvement["based_on_insight"] = insight
                breakthroughs.append(improvement)
                
        return breakthroughs
        
    async def _generate_improvement_from_insight(self, insight: Dict) -> Optional[Dict]:
        """Generate a concrete improvement from a synthesis insight."""
        prompt = f"""Generate a concrete improvement based on this breakthrough insight:

INSIGHT: {insight.get('insight', '')}
APPROACH: {insight.get('implementation_approach', '')}
EXPECTED IMPACT: {insight.get('expected_impact', '')}
NOVEL ASPECTS: {insight.get('novel_aspects', '')}

Create a specific, implementable improvement that embodies this insight.

Return ONLY a JSON with:
- name: A descriptive name
- type: The improvement type
- description: What it does
- approach: How it works (detailed)
- metrics: How to measure success
- integration_points: Where it fits in the system
- breakthrough_nature: What makes this a breakthrough
"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            # Parse response (reuse parsing logic from evolutionary engine)
            if isinstance(response, dict) and 'name' in response:
                improvement = response
            else:
                # Try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', str(response), re.DOTALL)
                if json_match:
                    improvement = json.loads(json_match.group())
                else:
                    return None
                    
            # Add synthesis metadata
            improvement["synthesis_generated"] = True
            improvement["insight_based"] = True
            
            return improvement
            
        except Exception as e:
            self.logger.error(f"Breakthrough generation failed: {e}")
            return None
            
    def analyze_knowledge_flows(self) -> Dict[str, Any]:
        """Analyze how knowledge flows between systems."""
        analysis = {
            "total_nodes": len(self.knowledge_graph),
            "total_edges": len(self.knowledge_graph.edges()),
            "improvement_nodes": len([n for n in self.knowledge_graph if n.startswith("imp:")]),
            "research_nodes": len([n for n in self.knowledge_graph if n.startswith("research:")]),
            "cross_connections": 0,
            "knowledge_bridges": [],
            "isolated_components": []
        }
        
        # Count cross-domain connections
        for u, v in self.knowledge_graph.edges():
            if (u.startswith("imp:") and v.startswith("research:")) or \
               (u.startswith("research:") and v.startswith("imp:")):
                analysis["cross_connections"] += 1
                
        # Find knowledge bridges
        if len(self.knowledge_graph) > 0:
            centrality = nx.betweenness_centrality(self.knowledge_graph)
            bridges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            analysis["knowledge_bridges"] = [
                {"node": node, "centrality": cent} for node, cent in bridges
            ]
            
        # Find isolated components
        components = list(nx.weakly_connected_components(self.knowledge_graph))
        analysis["isolated_components"] = [
            list(comp) for comp in components if len(comp) < 3
        ]
        
        return analysis
        
    def get_synthesis_recommendations(self) -> List[str]:
        """Get recommendations for improving knowledge synthesis."""
        recommendations = []
        
        # Analyze knowledge flows
        flow_analysis = self.analyze_knowledge_flows()
        
        # Check for low cross-connections
        if flow_analysis["cross_connections"] < flow_analysis["total_edges"] * 0.2:
            recommendations.append(
                "Increase cross-domain exploration - only "
                f"{flow_analysis['cross_connections']} connections between improvements and research"
            )
            
        # Check for isolated components
        if len(flow_analysis["isolated_components"]) > 5:
            recommendations.append(
                f"Connect {len(flow_analysis['isolated_components'])} isolated knowledge components"
            )
            
        # Check synthesis frequency
        if len(self.cross_domain_insights) < 10:
            recommendations.append(
                "Increase synthesis frequency - only "
                f"{len(self.cross_domain_insights)} insights generated"
            )
            
        # Check for stale insights
        if self.cross_domain_insights:
            latest = datetime.fromisoformat(
                self.cross_domain_insights[-1]["synthesized_at"].replace('Z', '+00:00')
            )
            age = (datetime.now(timezone.utc) - latest).days
            if age > 7:
                recommendations.append(
                    f"Synthesis is {age} days old - run new synthesis cycle"
                )
                
        return recommendations
        
    def _create_improvement_text(self, imp_data: Dict) -> str:
        """Create text representation of improvement for analysis."""
        components = []
        for field in ["name", "type", "description", "approach"]:
            if field in imp_data:
                components.append(str(imp_data[field]))
        return " ".join(components)
        
    async def _calculate_relevance(self, imp_text: str, research_topic: str) -> float:
        """Calculate relevance between improvement and research."""
        # Simple keyword overlap for now
        imp_words = set(imp_text.lower().split())
        topic_words = set(research_topic.lower().split())
        
        if not imp_words or not topic_words:
            return 0.0
            
        overlap = len(imp_words.intersection(topic_words))
        union = len(imp_words.union(topic_words))
        
        return overlap / union if union > 0 else 0.0
        
    def _classify_enhancement(self, research: Dict) -> str:
        """Classify type of enhancement research could provide."""
        content = str(research.get("content", "")).lower()
        
        if any(word in content for word in ["optimize", "performance", "speed"]):
            return "performance"
        elif any(word in content for word in ["secure", "safety", "protect"]):
            return "security"
        elif any(word in content for word in ["scale", "distributed", "parallel"]):
            return "scalability"
        elif any(word in content for word in ["ai", "ml", "predict", "learn"]):
            return "intelligence"
        else:
            return "general"
            
    def _estimate_impact(self, improvement: Dict, research: Dict) -> float:
        """Estimate potential impact of applying research to improvement."""
        # Simple heuristic based on current fitness and research quality
        current_fitness = improvement.get("fitness", 0.5)
        research_quality = research.get("metadata", {}).get("quality_score", 0.5)
        
        # Potential for improvement is higher when current fitness is lower
        improvement_potential = 1.0 - current_fitness
        
        return improvement_potential * research_quality
        
    def _are_complementary(self, pattern: str, insight: Dict) -> bool:
        """Check if pattern and insight are complementary."""
        # Extract key concepts
        pattern_type = pattern.split(':')[0] if ':' in pattern else ""
        insight_text = str(insight.get("insight", "")).lower()
        
        # Complementary pairs
        complements = {
            "approach": ["implementation", "method", "technique"],
            "integration": ["api", "interface", "connection"],
            "type": ["category", "classification", "domain"]
        }
        
        for key, values in complements.items():
            if key in pattern_type:
                if any(val in insight_text for val in values):
                    return True
                    
        return False
        
    def _score_insight(self, insight: Dict) -> float:
        """Score an insight based on novelty and potential impact."""
        score = 0.0
        
        # Novelty from synthesis type
        if insight.get("synthesis_type") == "distant_connection":
            score += 0.3
        else:
            score += 0.1
            
        # Expected impact
        impact = insight.get("expected_impact", "")
        if "breakthrough" in impact.lower() or "transform" in impact.lower():
            score += 0.3
        elif "improve" in impact.lower() or "enhance" in impact.lower():
            score += 0.2
            
        # Novel aspects
        novel_aspects = insight.get("novel_aspects", "")
        if len(novel_aspects) > 100:  # Detailed novelty description
            score += 0.2
            
        # Age penalty (favor recent insights)
        if "synthesized_at" in insight:
            age_days = (datetime.now(timezone.utc) - 
                       datetime.fromisoformat(insight["synthesized_at"].replace('Z', '+00:00'))).days
            score -= age_days * 0.01
            
        return max(0, score)
        
    def _update_knowledge_graph(self, insights: List[Dict]):
        """Update knowledge graph with new insights."""
        for insight in insights:
            # Add insight node
            insight_id = f"insight:{len(self.cross_domain_insights)}"
            
            # Connect to source nodes
            if "combination" in insight:
                combo = insight["combination"]
                
                if combo["type"] == "distant_connection":
                    self.knowledge_graph.add_edge(
                        f"imp:{combo['improvement']}", 
                        insight_id,
                        relationship="synthesized_to"
                    )
                    self.knowledge_graph.add_edge(
                        f"research:{combo['research']}", 
                        insight_id,
                        relationship="synthesized_to"
                    )
                    if "bridge" in combo:
                        self.knowledge_graph.add_edge(
                            combo["bridge"],
                            insight_id,
                            relationship="bridged_to"
                        )
                        
            # Add insight properties
            self.knowledge_graph.nodes[insight_id]["data"] = insight