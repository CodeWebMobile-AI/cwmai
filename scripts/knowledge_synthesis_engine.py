import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import asyncio
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from scripts.ai_brain import AIBrain
from scripts.redis_integration import get_redis_client
from scripts.pattern_learning_engine import PatternLearningEngine

logger = logging.getLogger(__name__)


class KnowledgeSynthesisEngine:
    """Synthesize knowledge from multiple sources to generate novel insights"""
    
    def __init__(self):
        self.ai_brain = AIBrain()
        self.redis_client = get_redis_client()
        self.pattern_engine = PatternLearningEngine()
        self.knowledge_graph = nx.DiGraph()
        self.synthesis_state_file = "knowledge_synthesis_state.json"
        self.load_synthesis_state()
        
    def load_synthesis_state(self):
        """Load previously synthesized knowledge"""
        if Path(self.synthesis_state_file).exists():
            with open(self.synthesis_state_file, 'r') as f:
                self.synthesis_state = json.load(f)
                # Rebuild knowledge graph
                for node in self.synthesis_state.get("nodes", []):
                    self.knowledge_graph.add_node(node["id"], **node["attributes"])
                for edge in self.synthesis_state.get("edges", []):
                    self.knowledge_graph.add_edge(edge["from"], edge["to"], **edge["attributes"])
        else:
            self.synthesis_state = {
                "concepts": {},
                "relationships": {},
                "insights": [],
                "last_synthesis": datetime.now().isoformat()
            }
    
    def save_synthesis_state(self):
        """Persist synthesized knowledge"""
        # Save knowledge graph
        nodes = []
        for node_id, attrs in self.knowledge_graph.nodes(data=True):
            nodes.append({"id": node_id, "attributes": attrs})
            
        edges = []
        for from_node, to_node, attrs in self.knowledge_graph.edges(data=True):
            edges.append({"from": from_node, "to": to_node, "attributes": attrs})
            
        self.synthesis_state["nodes"] = nodes
        self.synthesis_state["edges"] = edges
        self.synthesis_state["last_synthesis"] = datetime.now().isoformat()
        
        with open(self.synthesis_state_file, 'w') as f:
            json.dump(self.synthesis_state, f, indent=2)
    
    async def synthesize_knowledge(self) -> Dict[str, Any]:
        """Combine insights from multiple sources to generate new knowledge"""
        logger.info("Synthesizing knowledge from multiple sources")
        
        # Gather knowledge from various sources
        sources = await self._gather_knowledge_sources()
        
        # Extract concepts and relationships
        concepts = await self._extract_concepts(sources)
        relationships = await self._find_relationships(concepts)
        
        # Build/update knowledge graph
        self._update_knowledge_graph(concepts, relationships)
        
        # Cross-reference patterns
        cross_references = await self._cross_reference_patterns()
        
        # Generate novel insights
        novel_insights = await self._generate_novel_insights(cross_references)
        
        # Create actionable recommendations
        recommendations = await self._create_recommendations(novel_insights)
        
        # Update synthesis state
        self.synthesis_state["concepts"].update(concepts)
        self.synthesis_state["relationships"].update(relationships)
        self.synthesis_state["insights"].extend(novel_insights)
        
        self.save_synthesis_state()
        
        return {
            "concepts_discovered": len(concepts),
            "relationships_found": len(relationships),
            "novel_insights": novel_insights,
            "recommendations": recommendations,
            "knowledge_graph_size": len(self.knowledge_graph.nodes())
        }
    
    async def _gather_knowledge_sources(self) -> Dict[str, Any]:
        """Gather knowledge from all available sources"""
        sources = {
            "code_analysis": {},
            "documentation": {},
            "patterns": {},
            "metrics": {},
            "user_feedback": {},
            "external_research": {}
        }
        
        # Analyze codebase
        code_files = list(Path("scripts").rglob("*.py"))
        code_insights = []
        for file in code_files[:20]:  # Sample for efficiency
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    insight = await self._analyze_code_file(file.name, content)
                    if insight:
                        code_insights.append(insight)
            except:
                pass
        sources["code_analysis"] = {"insights": code_insights}
        
        # Read documentation
        doc_files = list(Path(".").glob("*.md"))
        doc_insights = []
        for file in doc_files:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    insight = await self._analyze_documentation(file.name, content)
                    if insight:
                        doc_insights.append(insight)
            except:
                pass
        sources["documentation"] = {"insights": doc_insights}
        
        # Get learned patterns
        if hasattr(self.pattern_engine, 'learning_state'):
            sources["patterns"] = self.pattern_engine.learning_state
        
        # Get system metrics
        if Path("system_state.json").exists():
            with open("system_state.json", 'r') as f:
                sources["metrics"] = json.load(f)
                
        # Get research knowledge
        research_dir = Path("research_knowledge")
        if research_dir.exists():
            research_data = []
            for file in research_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        research_data.append(json.load(f))
                except:
                    pass
            sources["external_research"] = {"data": research_data}
            
        return sources
    
    async def _extract_concepts(self, sources: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract key concepts from knowledge sources"""
        concepts = {}
        
        # Extract from code analysis
        for insight in sources.get("code_analysis", {}).get("insights", []):
            for concept in insight.get("concepts", []):
                if concept not in concepts:
                    concepts[concept] = {
                        "type": "technical",
                        "sources": [insight["source"]],
                        "frequency": 1,
                        "context": insight.get("context", "")
                    }
                else:
                    concepts[concept]["frequency"] += 1
                    concepts[concept]["sources"].append(insight["source"])
        
        # Extract from documentation
        for insight in sources.get("documentation", {}).get("insights", []):
            for concept in insight.get("concepts", []):
                if concept not in concepts:
                    concepts[concept] = {
                        "type": "documented",
                        "sources": [insight["source"]],
                        "frequency": 1,
                        "context": insight.get("context", "")
                    }
                else:
                    concepts[concept]["frequency"] += 1
                    
        # Extract from patterns
        pattern_concepts = self._extract_pattern_concepts(sources.get("patterns", {}))
        concepts.update(pattern_concepts)
        
        return concepts
    
    async def _find_relationships(self, concepts: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find relationships between concepts"""
        relationships = defaultdict(list)
        
        # Use TF-IDF to find similar concepts
        if len(concepts) > 1:
            concept_names = list(concepts.keys())
            contexts = [concepts[c].get("context", c) for c in concept_names]
            
            if contexts:
                vectorizer = TfidfVectorizer(max_features=100)
                try:
                    tfidf_matrix = vectorizer.fit_transform(contexts)
                    similarities = cosine_similarity(tfidf_matrix)
                    
                    # Find strong relationships
                    for i in range(len(concept_names)):
                        for j in range(i + 1, len(concept_names)):
                            similarity = similarities[i][j]
                            if similarity > 0.3:  # Threshold for relationship
                                rel = {
                                    "from": concept_names[i],
                                    "to": concept_names[j],
                                    "strength": float(similarity),
                                    "type": "semantic_similarity"
                                }
                                relationships[concept_names[i]].append(rel)
                except:
                    pass
        
        # Find causal relationships
        for concept, data in concepts.items():
            # Look for cause-effect patterns in context
            context = data.get("context", "")
            if "causes" in context or "leads to" in context or "results in" in context:
                # Extract potential causal relationships
                prompt = f"""
                Extract causal relationships from this context:
                {context}
                
                Return as JSON list: [{{"cause": "...", "effect": "...", "confidence": 0.0-1.0}}]
                """
                
                result = await self.ai_brain.process(prompt)
                try:
                    causal_rels = json.loads(result)
                    for rel in causal_rels:
                        relationships[concept].append({
                            "from": rel["cause"],
                            "to": rel["effect"],
                            "strength": rel["confidence"],
                            "type": "causal"
                        })
                except:
                    pass
                    
        return dict(relationships)
    
    def _update_knowledge_graph(self, concepts: Dict[str, Dict[str, Any]], 
                               relationships: Dict[str, List[Dict[str, Any]]]):
        """Update the knowledge graph with new concepts and relationships"""
        # Add concept nodes
        for concept, data in concepts.items():
            if not self.knowledge_graph.has_node(concept):
                self.knowledge_graph.add_node(concept, **data)
            else:
                # Update node attributes
                for key, value in data.items():
                    if key == "frequency":
                        self.knowledge_graph.nodes[concept][key] = \
                            self.knowledge_graph.nodes[concept].get(key, 0) + value
                    elif key == "sources":
                        existing = self.knowledge_graph.nodes[concept].get(key, [])
                        self.knowledge_graph.nodes[concept][key] = list(set(existing + value))
                        
        # Add relationship edges
        for from_concept, rels in relationships.items():
            for rel in rels:
                to_concept = rel["to"]
                if self.knowledge_graph.has_node(from_concept) and self.knowledge_graph.has_node(to_concept):
                    if not self.knowledge_graph.has_edge(from_concept, to_concept):
                        self.knowledge_graph.add_edge(
                            from_concept, to_concept,
                            strength=rel["strength"],
                            type=rel["type"]
                        )
                    else:
                        # Update edge strength
                        current_strength = self.knowledge_graph[from_concept][to_concept].get("strength", 0)
                        self.knowledge_graph[from_concept][to_concept]["strength"] = \
                            max(current_strength, rel["strength"])
    
    async def _cross_reference_patterns(self) -> List[Dict[str, Any]]:
        """Cross-reference patterns across different knowledge domains"""
        cross_references = []
        
        # Find concepts that appear in multiple domains
        domain_concepts = defaultdict(set)
        for node, attrs in self.knowledge_graph.nodes(data=True):
            domain = attrs.get("type", "unknown")
            domain_concepts[domain].add(node)
        
        # Find overlapping concepts
        domains = list(domain_concepts.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                overlap = domain_concepts[domains[i]] & domain_concepts[domains[j]]
                if overlap:
                    cross_references.append({
                        "domains": [domains[i], domains[j]],
                        "shared_concepts": list(overlap),
                        "strength": len(overlap) / min(
                            len(domain_concepts[domains[i]]),
                            len(domain_concepts[domains[j]])
                        )
                    })
        
        # Find bridge concepts (high betweenness centrality)
        if len(self.knowledge_graph) > 10:
            try:
                centrality = nx.betweenness_centrality(self.knowledge_graph)
                bridges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                for concept, score in bridges:
                    if score > 0.1:
                        cross_references.append({
                            "type": "bridge_concept",
                            "concept": concept,
                            "centrality": score,
                            "connects": list(self.knowledge_graph.neighbors(concept))
                        })
            except:
                pass
                
        return cross_references
    
    async def _generate_novel_insights(self, cross_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate novel insights by combining knowledge"""
        insights = []
        
        # Analyze cross-domain patterns
        for ref in cross_references:
            if ref.get("strength", 0) > 0.5 or ref.get("centrality", 0) > 0.2:
                prompt = f"""
                Generate a novel insight from this cross-domain pattern:
                {json.dumps(ref, indent=2)}
                
                Consider:
                1. What new capability could emerge from combining these domains?
                2. What optimization opportunity does this pattern suggest?
                3. What automation could be created?
                
                Return as JSON: {{
                    "insight": "description",
                    "type": "capability|optimization|automation",
                    "impact": "high|medium|low",
                    "implementation": "brief description"
                }}
                """
                
                result = await self.ai_brain.process(prompt)
                try:
                    insight = json.loads(result)
                    insight["source"] = ref
                    insight["generated_at"] = datetime.now().isoformat()
                    insights.append(insight)
                except:
                    pass
        
        # Find emergent patterns
        if len(self.knowledge_graph) > 20:
            # Look for communities/clusters
            try:
                communities = list(nx.community.greedy_modularity_communities(
                    self.knowledge_graph.to_undirected()
                ))
                
                for community in communities[:3]:  # Top 3 communities
                    if len(community) > 3:
                        prompt = f"""
                        This cluster of related concepts was discovered:
                        {list(community)}
                        
                        What emergent capability or pattern does this suggest?
                        
                        Return as JSON: {{
                            "insight": "description",
                            "pattern": "pattern name",
                            "applications": ["app1", "app2"],
                            "priority": "high|medium|low"
                        }}
                        """
                        
                        result = await self.ai_brain.process(prompt)
                        try:
                            insight = json.loads(result)
                            insight["type"] = "emergent_pattern"
                            insight["community"] = list(community)
                            insights.append(insight)
                        except:
                            pass
            except:
                pass
                
        return insights
    
    async def _create_recommendations(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create actionable recommendations from insights"""
        recommendations = []
        
        # Group insights by type
        by_type = defaultdict(list)
        for insight in insights:
            by_type[insight.get("type", "other")].append(insight)
        
        # Generate recommendations for each type
        for insight_type, type_insights in by_type.items():
            if type_insights:
                prompt = f"""
                Based on these {insight_type} insights:
                {json.dumps(type_insights, indent=2)}
                
                Create actionable recommendations that:
                1. Can be implemented automatically
                2. Will improve system capabilities
                3. Have measurable impact
                
                Return as JSON list: [{{
                    "title": "recommendation",
                    "description": "detailed description",
                    "implementation_steps": ["step1", "step2"],
                    "expected_impact": "description",
                    "priority": 1-10
                }}]
                """
                
                result = await self.ai_brain.process(prompt)
                try:
                    recs = json.loads(result)
                    for rec in recs:
                        rec["source_insights"] = [i.get("insight", "") for i in type_insights]
                        rec["created_at"] = datetime.now().isoformat()
                        recommendations.append(rec)
                except:
                    pass
                    
        # Prioritize recommendations
        recommendations.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def _analyze_code_file(self, filename: str, content: str) -> Optional[Dict[str, Any]]:
        """Analyze a code file for concepts and patterns"""
        if len(content) < 100:  # Skip small files
            return None
            
        prompt = f"""
        Analyze this code file and extract:
        1. Key concepts/capabilities
        2. Design patterns used
        3. Dependencies and relationships
        
        File: {filename}
        Code (first 500 chars): {content[:500]}
        
        Return as JSON: {{
            "source": "filename",
            "concepts": ["concept1", "concept2"],
            "patterns": ["pattern1"],
            "context": "brief description"
        }}
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            analysis = json.loads(result)
            analysis["source"] = filename
            return analysis
        except:
            return None
    
    async def _analyze_documentation(self, filename: str, content: str) -> Optional[Dict[str, Any]]:
        """Analyze documentation for concepts and insights"""
        if len(content) < 50:
            return None
            
        prompt = f"""
        Extract key concepts and insights from this documentation:
        
        File: {filename}
        Content (first 500 chars): {content[:500]}
        
        Return as JSON: {{
            "source": "filename",
            "concepts": ["concept1", "concept2"],
            "insights": ["insight1"],
            "context": "summary"
        }}
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            analysis = json.loads(result)
            analysis["source"] = filename
            return analysis
        except:
            return None
    
    def _extract_pattern_concepts(self, patterns: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract concepts from learned patterns"""
        concepts = {}
        
        # From workflow patterns
        for date, data in patterns.get("workflow_patterns", {}).items():
            for pattern in data.get("patterns", []):
                sequence = pattern.get("sequence", [])
                for task in sequence:
                    if task not in concepts:
                        concepts[task] = {
                            "type": "workflow_task",
                            "sources": ["workflow_patterns"],
                            "frequency": pattern.get("frequency", 1),
                            "context": f"Part of workflow pattern with confidence {pattern.get('confidence', 0)}"
                        }
                        
        # From error patterns
        for date, data in patterns.get("error_patterns", {}).items():
            for category in data.get("details", {}):
                if category not in concepts:
                    concepts[category] = {
                        "type": "error_category",
                        "sources": ["error_patterns"],
                        "frequency": 1,
                        "context": "Error pattern category"
                    }
                    
        return concepts
    
    async def apply_synthesis(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply synthesized knowledge recommendations"""
        results = {
            "applied": 0,
            "failed": 0,
            "implementations": []
        }
        
        for rec in recommendations[:5]:  # Apply top 5 recommendations
            try:
                # Create implementation based on recommendation
                implementation = await self._implement_recommendation(rec)
                if implementation:
                    results["implementations"].append(implementation)
                    results["applied"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to apply recommendation: {e}")
                results["failed"] += 1
                
        return results
    
    async def _implement_recommendation(self, recommendation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Implement a specific recommendation"""
        prompt = f"""
        Implement this recommendation:
        {json.dumps(recommendation, indent=2)}
        
        Generate:
        1. Specific code or configuration changes
        2. Validation steps
        3. Success metrics
        
        Return as JSON: {{
            "recommendation": "title",
            "changes": ["change1", "change2"],
            "validation": "how to validate",
            "metrics": "success metrics"
        }}
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            implementation = json.loads(result)
            implementation["implemented_at"] = datetime.now().isoformat()
            return implementation
        except:
            return None