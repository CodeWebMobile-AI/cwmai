"""
Simple Knowledge Synthesis Engine

A non-Redis version of the knowledge synthesis engine for the evolutionary system.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import networkx as nx


class SimpleKnowledgeSynthesisEngine:
    """Lightweight knowledge synthesis without external dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knowledge_graph = nx.DiGraph()
        self.synthesis_state_file = Path("evolutionary_knowledge/synthesis_state.json")
        self.synthesis_state = {
            "concepts": {},
            "relationships": {},
            "insights": [],
            "last_synthesis": datetime.now().isoformat()
        }
        self.load_synthesis_state()
        
    def load_synthesis_state(self):
        """Load previously synthesized knowledge."""
        if self.synthesis_state_file.exists():
            try:
                with open(self.synthesis_state_file, 'r') as f:
                    self.synthesis_state = json.load(f)
                    # Rebuild knowledge graph
                    for node in self.synthesis_state.get("nodes", []):
                        self.knowledge_graph.add_node(node["id"], **node.get("attributes", {}))
                    for edge in self.synthesis_state.get("edges", []):
                        self.knowledge_graph.add_edge(
                            edge["from"], 
                            edge["to"], 
                            **edge.get("attributes", {})
                        )
            except Exception as e:
                self.logger.warning(f"Could not load synthesis state: {e}")
                
    def save_synthesis_state(self):
        """Persist synthesized knowledge."""
        # Ensure directory exists
        self.synthesis_state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert graph to serializable format
        nodes = []
        for node in self.knowledge_graph.nodes(data=True):
            nodes.append({
                "id": node[0],
                "attributes": node[1]
            })
            
        edges = []
        for edge in self.knowledge_graph.edges(data=True):
            edges.append({
                "from": edge[0],
                "to": edge[1],
                "attributes": edge[2]
            })
            
        self.synthesis_state["nodes"] = nodes
        self.synthesis_state["edges"] = edges
        self.synthesis_state["last_synthesis"] = datetime.now().isoformat()
        
        with open(self.synthesis_state_file, 'w') as f:
            json.dump(self.synthesis_state, f, indent=2)
            
    def synthesize_insights(self) -> List[Dict[str, Any]]:
        """Synthesize insights from stored knowledge."""
        insights = self.synthesis_state.get("insights", [])
        
        # Add timestamp to new insights
        for insight in insights:
            if "timestamp" not in insight:
                insight["timestamp"] = datetime.now().isoformat()
                
        return insights
        
    def add_concept(self, concept_id: str, attributes: Dict[str, Any]):
        """Add a concept to the knowledge graph."""
        self.knowledge_graph.add_node(concept_id, **attributes)
        self.synthesis_state["concepts"][concept_id] = attributes
        
    def add_relationship(self, from_id: str, to_id: str, relationship_type: str, attributes: Optional[Dict] = None):
        """Add a relationship between concepts."""
        attrs = attributes or {}
        attrs["type"] = relationship_type
        self.knowledge_graph.add_edge(from_id, to_id, **attrs)
        
        rel_key = f"{from_id}->{to_id}"
        self.synthesis_state["relationships"][rel_key] = attrs
        
    def find_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns in the knowledge graph."""
        patterns = []
        
        # Find highly connected nodes (hubs)
        if len(self.knowledge_graph) > 0:
            degrees = dict(self.knowledge_graph.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            
            for node, degree in degrees.items():
                if degree > avg_degree * 2:  # Highly connected
                    patterns.append({
                        "type": "hub",
                        "node": node,
                        "connections": degree,
                        "significance": "Central concept with many connections"
                    })
                    
        # Find isolated clusters
        if len(self.knowledge_graph) > 0:
            components = list(nx.weakly_connected_components(self.knowledge_graph))
            for i, component in enumerate(components):
                if len(component) > 1 and len(component) < len(self.knowledge_graph) * 0.2:
                    patterns.append({
                        "type": "cluster",
                        "cluster_id": i,
                        "size": len(component),
                        "nodes": list(component),
                        "significance": "Isolated group of related concepts"
                    })
                    
        return patterns
        
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of synthesized insights."""
        return {
            "total_concepts": len(self.synthesis_state.get("concepts", {})),
            "total_relationships": len(self.synthesis_state.get("relationships", {})),
            "total_insights": len(self.synthesis_state.get("insights", [])),
            "graph_nodes": self.knowledge_graph.number_of_nodes(),
            "graph_edges": self.knowledge_graph.number_of_edges(),
            "patterns": self.find_patterns()
        }