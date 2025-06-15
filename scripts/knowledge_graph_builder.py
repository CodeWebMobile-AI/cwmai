"""
Knowledge Graph Builder - Constructs and manages knowledge graphs from research data.

This module extracts entities, relationships, and patterns from research results
to build a comprehensive knowledge graph that enables semantic search and insight discovery.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path
import networkx as nx
import hashlib
from collections import defaultdict, Counter
import re


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from research data."""
    
    def __init__(self, storage_path: str = "research_knowledge/knowledge_graph"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load graph
        self.graph = self._load_or_create_graph()
        
        # Entity and relationship extractors
        self.entity_patterns = {
            "component": r'\b(component|module|class|function|service|system)\s+(\w+)',
            "metric": r'\b(metric|measurement|score|rate|percentage)\s+(\w+)',
            "technique": r'\b(technique|method|approach|strategy|pattern)\s+(\w+)',
            "problem": r'\b(issue|problem|error|failure|bug)\s+(\w+)',
            "improvement": r'\b(improvement|optimization|enhancement|fix)\s+(\w+)',
            "capability": r'\b(capability|feature|functionality)\s+(\w+)',
            "technology": r'\b(framework|library|tool|api)\s+(\w+)'
        }
        
        self.relationship_patterns = {
            "improves": ["improves", "enhances", "optimizes", "fixes"],
            "causes": ["causes", "leads to", "results in", "triggers"],
            "uses": ["uses", "utilizes", "employs", "leverages"],
            "requires": ["requires", "needs", "depends on", "relies on"],
            "implements": ["implements", "realizes", "executes", "performs"],
            "relates_to": ["relates to", "connected to", "associated with", "linked to"]
        }
        
        # Metadata tracking
        self.entity_metadata = self._load_or_create_metadata("entity_metadata.json")
        self.relationship_metadata = self._load_or_create_metadata("relationship_metadata.json")
        self.pattern_metadata = self._load_or_create_metadata("pattern_metadata.json")
        
        # Statistics
        self.stats = {
            "total_entities": 0,
            "total_relationships": 0,
            "total_patterns": 0,
            "processing_history": []
        }
    
    def _load_or_create_graph(self) -> nx.MultiDiGraph:
        """Load existing graph or create new one."""
        graph_path = self.storage_path / "knowledge_graph.gexf"
        if graph_path.exists():
            try:
                return nx.read_gexf(str(graph_path))
            except:
                pass
        return nx.MultiDiGraph()
    
    def _load_or_create_metadata(self, filename: str) -> Dict:
        """Load or create metadata file."""
        metadata_path = self.storage_path / filename
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_graph(self):
        """Save graph and metadata to disk."""
        # Save graph in GEXF format (supports attributes)
        nx.write_gexf(self.graph, str(self.storage_path / "knowledge_graph.gexf"))
        
        # Save as JSON for easier access
        graph_data = nx.node_link_data(self.graph)
        with open(self.storage_path / "knowledge_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Save metadata
        metadata_files = {
            "entity_metadata.json": self.entity_metadata,
            "relationship_metadata.json": self.relationship_metadata,
            "pattern_metadata.json": self.pattern_metadata,
            "graph_stats.json": self.stats
        }
        
        for filename, data in metadata_files.items():
            with open(self.storage_path / filename, 'w') as f:
                json.dump(data, f, indent=2)
    
    def process_research(self, research_data: Dict) -> Dict[str, Any]:
        """
        Process research data to extract entities and relationships.
        
        Args:
            research_data: Research content to process
            
        Returns:
            Processing results including extracted entities and relationships
        """
        results = {
            "entities_extracted": [],
            "relationships_extracted": [],
            "patterns_identified": [],
            "graph_updates": 0
        }
        
        # Extract content
        content = self._extract_content(research_data)
        
        # Extract entities
        entities = self._extract_entities(content)
        results["entities_extracted"] = entities
        
        # Extract relationships
        relationships = self._extract_relationships(content, entities)
        results["relationships_extracted"] = relationships
        
        # Add to graph
        for entity in entities:
            self._add_entity_to_graph(entity, research_data)
            results["graph_updates"] += 1
        
        for relationship in relationships:
            self._add_relationship_to_graph(relationship, research_data)
            results["graph_updates"] += 1
        
        # Identify patterns
        patterns = self._identify_patterns()
        results["patterns_identified"] = patterns
        
        # Update statistics
        self.stats["total_entities"] = self.graph.number_of_nodes()
        self.stats["total_relationships"] = self.graph.number_of_edges()
        self.stats["total_patterns"] = len(patterns)
        self.stats["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "research_id": research_data.get("id", "unknown"),
            "entities_added": len(entities),
            "relationships_added": len(relationships)
        })
        
        # Save updates
        self._save_graph()
        
        return results
    
    def _extract_content(self, research_data: Dict) -> str:
        """Extract textual content from research data."""
        content_parts = []
        
        # Direct content
        if "content" in research_data:
            content_parts.append(str(research_data["content"]))
        
        # Query content
        if "query" in research_data:
            query = research_data["query"]
            if isinstance(query, dict):
                content_parts.append(query.get("query", ""))
            else:
                content_parts.append(str(query))
        
        # Topic content
        if "topic" in research_data:
            topic = research_data["topic"]
            if isinstance(topic, dict):
                content_parts.extend([
                    topic.get("topic", ""),
                    topic.get("area", ""),
                    str(topic.get("context", ""))
                ])
        
        return " ".join(content_parts)
    
    def _extract_entities(self, content: str) -> List[Dict]:
        """Extract entities from content using patterns."""
        entities = []
        seen = set()
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(2) if len(match.groups()) > 1 else match.group(1)
                entity_id = f"{entity_type}_{entity_name.lower()}"
                
                if entity_id not in seen:
                    seen.add(entity_id)
                    entities.append({
                        "id": entity_id,
                        "name": entity_name,
                        "type": entity_type,
                        "context": content[max(0, match.start()-50):match.end()+50],
                        "confidence": 0.8
                    })
        
        # Extract additional entities from structured content
        structured_entities = self._extract_structured_entities(content)
        entities.extend(structured_entities)
        
        return entities
    
    def _extract_structured_entities(self, content: str) -> List[Dict]:
        """Extract entities from structured content (lists, steps, etc.)."""
        entities = []
        
        # Look for numbered lists or bullet points
        list_patterns = [
            r'^\d+\.\s+(.+)$',
            r'^-\s+(.+)$',
            r'^\*\s+(.+)$',
            r'^•\s+(.+)$'
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in list_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    item_text = match.group(1)
                    # Try to identify entity type from content
                    if any(keyword in item_text.lower() for keyword in ["implement", "add", "create"]):
                        entity_type = "improvement"
                    elif any(keyword in item_text.lower() for keyword in ["fix", "resolve", "handle"]):
                        entity_type = "problem"
                    else:
                        entity_type = "concept"
                    
                    entity_id = f"{entity_type}_{hashlib.md5(item_text.encode()).hexdigest()[:8]}"
                    entities.append({
                        "id": entity_id,
                        "name": item_text[:50],
                        "type": entity_type,
                        "context": item_text,
                        "confidence": 0.7
                    })
        
        return entities
    
    def _extract_relationships(self, content: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities."""
        relationships = []
        entity_ids = {e["id"] for e in entities}
        
        # Look for relationship patterns
        for rel_type, keywords in self.relationship_patterns.items():
            for keyword in keywords:
                pattern = rf'(\w+)\s+{keyword}\s+(\w+)'
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)
                    
                    # Try to match with existing entities
                    source_entity = self._find_entity_match(source, entities)
                    target_entity = self._find_entity_match(target, entities)
                    
                    if source_entity and target_entity:
                        relationships.append({
                            "source": source_entity["id"],
                            "target": target_entity["id"],
                            "type": rel_type,
                            "context": content[max(0, match.start()-30):match.end()+30],
                            "confidence": 0.7
                        })
        
        # Infer relationships from proximity
        proximity_relationships = self._infer_proximity_relationships(content, entities)
        relationships.extend(proximity_relationships)
        
        return relationships
    
    def _find_entity_match(self, text: str, entities: List[Dict]) -> Optional[Dict]:
        """Find matching entity for given text."""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity["name"].lower() or entity["name"].lower() in text_lower:
                return entity
        return None
    
    def _infer_proximity_relationships(self, content: str, entities: List[Dict]) -> List[Dict]:
        """Infer relationships based on entity proximity in text."""
        relationships = []
        
        # Find entity positions in content
        entity_positions = []
        for entity in entities:
            pos = content.lower().find(entity["name"].lower())
            if pos != -1:
                entity_positions.append((pos, entity))
        
        # Sort by position
        entity_positions.sort(key=lambda x: x[0])
        
        # Create relationships between nearby entities
        for i in range(len(entity_positions) - 1):
            pos1, entity1 = entity_positions[i]
            pos2, entity2 = entity_positions[i + 1]
            
            # If entities are close (within 100 characters)
            if pos2 - pos1 < 100:
                relationships.append({
                    "source": entity1["id"],
                    "target": entity2["id"],
                    "type": "relates_to",
                    "context": content[pos1:pos2+len(entity2["name"])],
                    "confidence": 0.5
                })
        
        return relationships
    
    def _add_entity_to_graph(self, entity: Dict, research_data: Dict):
        """Add entity to knowledge graph."""
        entity_id = entity["id"]
        
        # Add or update node
        if entity_id not in self.graph:
            self.graph.add_node(entity_id)
        
        # Update node attributes
        self.graph.nodes[entity_id].update({
            "name": entity["name"],
            "type": entity["type"],
            "first_seen": research_data.get("timestamp", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "occurrence_count": self.graph.nodes[entity_id].get("occurrence_count", 0) + 1,
            "confidence": max(entity["confidence"], 
                            self.graph.nodes[entity_id].get("confidence", 0))
        })
        
        # Update entity metadata
        if entity_id not in self.entity_metadata:
            self.entity_metadata[entity_id] = {
                "contexts": [],
                "research_sources": [],
                "quality_scores": []
            }
        
        self.entity_metadata[entity_id]["contexts"].append(entity["context"])
        self.entity_metadata[entity_id]["research_sources"].append(research_data.get("id"))
        self.entity_metadata[entity_id]["quality_scores"].append(
            research_data.get("quality_score", 0.5)
        )
    
    def _add_relationship_to_graph(self, relationship: Dict, research_data: Dict):
        """Add relationship to knowledge graph."""
        source = relationship["source"]
        target = relationship["target"]
        rel_type = relationship["type"]
        
        # Ensure both nodes exist
        if source not in self.graph:
            self.graph.add_node(source)
        if target not in self.graph:
            self.graph.add_node(target)
        
        # Add edge with attributes
        edge_id = f"{source}_{rel_type}_{target}_{datetime.now().timestamp()}"
        self.graph.add_edge(source, target, key=edge_id,
            type=rel_type,
            confidence=relationship["confidence"],
            context=relationship["context"],
            timestamp=datetime.now().isoformat(),
            research_source=research_data.get("id")
        )
        
        # Update relationship metadata
        rel_key = f"{source}-{rel_type}-{target}"
        if rel_key not in self.relationship_metadata:
            self.relationship_metadata[rel_key] = {
                "occurrences": 0,
                "contexts": [],
                "research_sources": []
            }
        
        self.relationship_metadata[rel_key]["occurrences"] += 1
        self.relationship_metadata[rel_key]["contexts"].append(relationship["context"])
        self.relationship_metadata[rel_key]["research_sources"].append(research_data.get("id"))
    
    def _identify_patterns(self) -> List[Dict]:
        """Identify patterns in the knowledge graph."""
        patterns = []
        
        # Pattern 1: Frequently co-occurring entities
        co_occurrence_patterns = self._find_co_occurrence_patterns()
        patterns.extend(co_occurrence_patterns)
        
        # Pattern 2: Common relationship chains
        relationship_chains = self._find_relationship_chains()
        patterns.extend(relationship_chains)
        
        # Pattern 3: Hub entities (highly connected)
        hub_patterns = self._find_hub_patterns()
        patterns.extend(hub_patterns)
        
        # Pattern 4: Cyclic relationships
        cyclic_patterns = self._find_cyclic_patterns()
        patterns.extend(cyclic_patterns)
        
        # Store patterns
        for pattern in patterns:
            pattern_id = pattern["id"]
            self.pattern_metadata[pattern_id] = pattern
        
        return patterns
    
    def _find_co_occurrence_patterns(self) -> List[Dict]:
        """Find entities that frequently appear together."""
        patterns = []
        co_occurrences = defaultdict(int)
        
        # Count co-occurrences based on shared research sources
        for entity1, data1 in self.entity_metadata.items():
            for entity2, data2 in self.entity_metadata.items():
                if entity1 < entity2:  # Avoid duplicates
                    shared_sources = set(data1["research_sources"]) & set(data2["research_sources"])
                    if len(shared_sources) > 1:
                        co_occurrences[(entity1, entity2)] = len(shared_sources)
        
        # Create patterns for high co-occurrences
        for (entity1, entity2), count in co_occurrences.items():
            if count >= 3:  # Threshold for pattern
                patterns.append({
                    "id": f"cooccur_{entity1}_{entity2}",
                    "type": "co_occurrence",
                    "entities": [entity1, entity2],
                    "frequency": count,
                    "confidence": min(1.0, count / 10.0),
                    "description": f"{entity1} and {entity2} frequently appear together"
                })
        
        return patterns
    
    def _find_relationship_chains(self) -> List[Dict]:
        """Find common chains of relationships."""
        patterns = []
        
        # Look for paths of length 3
        for node in self.graph.nodes():
            paths = []
            for neighbor1 in self.graph.neighbors(node):
                for neighbor2 in self.graph.neighbors(neighbor1):
                    if neighbor2 != node:  # Avoid immediate cycles
                        path = [node, neighbor1, neighbor2]
                        edge1 = self.graph[node][neighbor1]
                        edge2 = self.graph[neighbor1][neighbor2]
                        
                        # Get relationship types
                        rel_types = []
                        for _, edge_data in edge1.items():
                            rel_types.append(edge_data.get("type", "unknown"))
                        for _, edge_data in edge2.items():
                            rel_types.append(edge_data.get("type", "unknown"))
                        
                        paths.append({
                            "path": path,
                            "relationships": rel_types
                        })
            
            # Group similar paths
            if len(paths) > 2:
                patterns.append({
                    "id": f"chain_{node}_{len(patterns)}",
                    "type": "relationship_chain",
                    "central_entity": node,
                    "paths": paths[:5],  # Top 5 paths
                    "frequency": len(paths),
                    "confidence": min(1.0, len(paths) / 10.0),
                    "description": f"Common relationship chains through {node}"
                })
        
        return patterns
    
    def _find_hub_patterns(self) -> List[Dict]:
        """Find hub entities with many connections."""
        patterns = []
        
        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Find high-centrality nodes
        threshold = 0.1  # Top 10% centrality
        hubs = [(node, cent) for node, cent in centrality.items() if cent > threshold]
        hubs.sort(key=lambda x: x[1], reverse=True)
        
        for node, centrality_score in hubs[:10]:  # Top 10 hubs
            # Analyze connection types
            connection_types = defaultdict(int)
            for _, neighbor, data in self.graph.edges(node, data=True):
                connection_types[data.get("type", "unknown")] += 1
            
            patterns.append({
                "id": f"hub_{node}",
                "type": "hub_entity",
                "entity": node,
                "centrality": centrality_score,
                "connection_count": self.graph.degree(node),
                "connection_types": dict(connection_types),
                "confidence": centrality_score,
                "description": f"{node} is a highly connected hub entity"
            })
        
        return patterns
    
    def _find_cyclic_patterns(self) -> List[Dict]:
        """Find cyclic relationships in the graph."""
        patterns = []
        
        # Find simple cycles (limit to small cycles for performance)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles[:20]:  # Limit to first 20 cycles
                if 3 <= len(cycle) <= 5:  # Only consider cycles of length 3-5
                    # Analyze cycle relationships
                    relationships = []
                    for i in range(len(cycle)):
                        source = cycle[i]
                        target = cycle[(i + 1) % len(cycle)]
                        if self.graph.has_edge(source, target):
                            edge_data = list(self.graph[source][target].values())[0]
                            relationships.append(edge_data.get("type", "unknown"))
                    
                    patterns.append({
                        "id": f"cycle_{'_'.join(cycle[:3])}",
                        "type": "cyclic_relationship",
                        "entities": cycle,
                        "relationships": relationships,
                        "length": len(cycle),
                        "confidence": 0.7,
                        "description": f"Cyclic relationship between {', '.join(cycle)}"
                    })
        except:
            # Graph might not support cycle detection
            pass
        
        return patterns
    
    def semantic_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform semantic search on the knowledge graph.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of relevant entities and relationships
        """
        results = []
        query_lower = query.lower()
        
        # Search entities
        entity_scores = {}
        for node, data in self.graph.nodes(data=True):
            score = 0
            
            # Name match
            if query_lower in data.get("name", "").lower():
                score += 5
            
            # Type match
            if query_lower in data.get("type", "").lower():
                score += 2
            
            # Context match (from metadata)
            if node in self.entity_metadata:
                contexts = self.entity_metadata[node]["contexts"]
                for context in contexts[-5:]:  # Check recent contexts
                    if query_lower in context.lower():
                        score += 1
            
            if score > 0:
                entity_scores[node] = score
        
        # Sort by score and get top results
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        
        for entity_id, score in sorted_entities[:max_results]:
            entity_data = self.graph.nodes[entity_id]
            
            # Get related entities
            related = []
            for neighbor in self.graph.neighbors(entity_id):
                related.append({
                    "id": neighbor,
                    "name": self.graph.nodes[neighbor].get("name", neighbor),
                    "relationship": self._get_relationship_type(entity_id, neighbor)
                })
            
            results.append({
                "entity_id": entity_id,
                "name": entity_data.get("name", entity_id),
                "type": entity_data.get("type", "unknown"),
                "relevance_score": score,
                "occurrence_count": entity_data.get("occurrence_count", 1),
                "related_entities": related[:5],
                "metadata": self.entity_metadata.get(entity_id, {})
            })
        
        return results
    
    def _get_relationship_type(self, source: str, target: str) -> str:
        """Get relationship type between two entities."""
        if self.graph.has_edge(source, target):
            edge_data = list(self.graph[source][target].values())[0]
            return edge_data.get("type", "unknown")
        return "unknown"
    
    def get_insights(self) -> Dict[str, Any]:
        """
        Get insights from the knowledge graph.
        
        Returns:
            Dictionary containing various insights
        """
        insights = {
            "graph_summary": {
                "total_entities": self.graph.number_of_nodes(),
                "total_relationships": self.graph.number_of_edges(),
                "entity_types": self._count_entity_types(),
                "relationship_types": self._count_relationship_types()
            },
            "top_entities": self._get_top_entities(),
            "key_patterns": self._get_key_patterns(),
            "recent_trends": self._get_recent_trends(),
            "actionable_insights": self._generate_actionable_insights()
        }
        
        return insights
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type."""
        type_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get("type", "unknown")
            type_counts[entity_type] += 1
        return dict(type_counts)
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count relationships by type."""
        type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "unknown")
            type_counts[rel_type] += 1
        return dict(type_counts)
    
    def _get_top_entities(self, limit: int = 10) -> List[Dict]:
        """Get most important entities."""
        # Calculate importance based on multiple factors
        entity_scores = {}
        
        for node, data in self.graph.nodes(data=True):
            score = 0
            
            # Occurrence frequency
            score += data.get("occurrence_count", 0) * 2
            
            # Degree centrality
            score += self.graph.degree(node) * 3
            
            # Average quality of research sources
            if node in self.entity_metadata:
                quality_scores = self.entity_metadata[node]["quality_scores"]
                if quality_scores:
                    score += sum(quality_scores) / len(quality_scores) * 10
            
            entity_scores[node] = score
        
        # Sort and return top entities
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_entities = []
        for entity_id, score in sorted_entities[:limit]:
            entity_data = self.graph.nodes[entity_id]
            top_entities.append({
                "id": entity_id,
                "name": entity_data.get("name", entity_id),
                "type": entity_data.get("type", "unknown"),
                "importance_score": round(score, 2),
                "occurrence_count": entity_data.get("occurrence_count", 1),
                "connection_count": self.graph.degree(entity_id)
            })
        
        return top_entities
    
    def _get_key_patterns(self) -> List[Dict]:
        """Get most significant patterns."""
        # Sort patterns by confidence and frequency
        all_patterns = list(self.pattern_metadata.values())
        
        # Score patterns
        for pattern in all_patterns:
            score = pattern.get("confidence", 0) * pattern.get("frequency", 1)
            pattern["significance_score"] = score
        
        # Sort by significance
        all_patterns.sort(key=lambda x: x["significance_score"], reverse=True)
        
        return all_patterns[:10]  # Top 10 patterns
    
    def _get_recent_trends(self) -> List[Dict]:
        """Analyze recent trends in the knowledge graph."""
        trends = []
        
        # Trend 1: Recently added entities
        recent_entities = []
        for node, data in self.graph.nodes(data=True):
            last_updated = data.get("last_updated")
            if last_updated:
                try:
                    update_time = datetime.fromisoformat(last_updated)
                    if (datetime.now() - update_time).days <= 1:
                        recent_entities.append({
                            "entity": node,
                            "name": data.get("name", node),
                            "type": data.get("type", "unknown")
                        })
                except:
                    pass
        
        if recent_entities:
            trends.append({
                "type": "new_entities",
                "description": "Recently discovered entities",
                "entities": recent_entities[:5]
            })
        
        # Trend 2: Emerging patterns
        if self.stats["processing_history"]:
            recent_processing = self.stats["processing_history"][-10:]
            total_recent_entities = sum(p["entities_added"] for p in recent_processing)
            total_recent_relationships = sum(p["relationships_added"] for p in recent_processing)
            
            trends.append({
                "type": "activity_trend",
                "description": "Recent knowledge graph activity",
                "metrics": {
                    "recent_entities_added": total_recent_entities,
                    "recent_relationships_added": total_recent_relationships,
                    "processing_count": len(recent_processing)
                }
            })
        
        return trends
    
    def _generate_actionable_insights(self) -> List[Dict]:
        """Generate actionable insights from the knowledge graph."""
        insights = []
        
        # Insight 1: Problematic components (high problem connections)
        problem_entities = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "component":
                problem_count = 0
                for neighbor in self.graph.neighbors(node):
                    neighbor_type = self.graph.nodes[neighbor].get("type")
                    if neighbor_type == "problem":
                        problem_count += 1
                
                if problem_count >= 2:
                    problem_entities.append({
                        "component": data.get("name", node),
                        "problem_count": problem_count,
                        "problems": [self.graph.nodes[n].get("name", n) 
                                   for n in self.graph.neighbors(node) 
                                   if self.graph.nodes[n].get("type") == "problem"][:3]
                    })
        
        if problem_entities:
            insights.append({
                "type": "problematic_components",
                "description": "Components with multiple associated problems",
                "action": "Prioritize refactoring or fixing these components",
                "data": problem_entities[:5]
            })
        
        # Insight 2: High-value improvements
        improvement_impacts = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "improvement":
                impact_score = 0
                affected_components = []
                
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get("type") in ["component", "metric", "capability"]:
                        impact_score += neighbor_data.get("occurrence_count", 1)
                        affected_components.append(neighbor_data.get("name", neighbor))
                
                if impact_score > 0:
                    improvement_impacts.append({
                        "improvement": data.get("name", node),
                        "impact_score": impact_score,
                        "affects": affected_components[:3]
                    })
        
        improvement_impacts.sort(key=lambda x: x["impact_score"], reverse=True)
        
        if improvement_impacts:
            insights.append({
                "type": "high_impact_improvements",
                "description": "Improvements that would affect multiple components",
                "action": "Implement these improvements for maximum impact",
                "data": improvement_impacts[:5]
            })
        
        # Insight 3: Technology recommendations
        tech_usage = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "technology":
                tech_usage[data.get("name", node)] = data.get("occurrence_count", 0)
        
        if tech_usage:
            top_tech = sorted(tech_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            insights.append({
                "type": "technology_recommendations",
                "description": "Most referenced technologies in research",
                "action": "Consider adopting or improving integration with these technologies",
                "data": [{"technology": tech, "references": count} for tech, count in top_tech]
            })
        
        return insights
    
    def export_graph_visualization(self) -> Dict[str, Any]:
        """
        Export graph data for visualization.
        
        Returns:
            Graph data in a format suitable for visualization tools
        """
        # Prepare nodes
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("name", node_id),
                "type": data.get("type", "unknown"),
                "size": min(50, 10 + data.get("occurrence_count", 1) * 2),
                "color": self._get_node_color(data.get("type", "unknown"))
            })
        
        # Prepare edges
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get("type", "unknown"),
                "weight": data.get("confidence", 0.5)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type."""
        color_map = {
            "component": "#4CAF50",
            "problem": "#F44336",
            "improvement": "#2196F3",
            "metric": "#FF9800",
            "capability": "#9C27B0",
            "technology": "#00BCD4",
            "technique": "#8BC34A"
        }
        return color_map.get(node_type, "#757575")