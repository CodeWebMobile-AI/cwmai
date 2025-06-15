"""
Research Insight Processor - Extracts, processes, and categorizes insights from research.

This module provides intelligent processing of raw research data to extract patterns,
recommendations, and actionable insights that can drive system improvements.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter
import hashlib


class ResearchInsightProcessor:
    """Processes raw research to extract and categorize insights."""
    
    def __init__(self, storage_path: str = "research_knowledge/processed_insights"):
        self.storage_path = Path(storage_path)
        
        # Ensure directories exist
        self.patterns_path = self.storage_path / "patterns"
        self.recommendations_path = self.storage_path / "recommendations"
        self.success_factors_path = self.storage_path / "success_factors"
        self.failure_analysis_path = self.storage_path / "failure_analysis"
        
        for path in [self.patterns_path, self.recommendations_path, 
                     self.success_factors_path, self.failure_analysis_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Insight extraction patterns
        self.insight_patterns = {
            "recommendation": [
                r"(?:should|recommend|suggest|advise)\s+(.+?)(?:\.|;|$)",
                r"(?:best practice|recommendation):\s*(.+?)(?:\.|;|$)",
                r"(?:consider|try|implement)\s+(.+?)(?:\.|;|$)"
            ],
            "pattern": [
                r"pattern:\s*(.+?)(?:\.|;|$)",
                r"(?:commonly|frequently|often)\s+(.+?)(?:\.|;|$)",
                r"(?:trend|tendency):\s*(.+?)(?:\.|;|$)"
            ],
            "success_factor": [
                r"(?:success factor|key to success):\s*(.+?)(?:\.|;|$)",
                r"(?:works well when|successful when)\s+(.+?)(?:\.|;|$)",
                r"(?:effective|efficient) (?:when|if)\s+(.+?)(?:\.|;|$)"
            ],
            "failure_cause": [
                r"(?:failure|issue|problem) (?:caused by|due to)\s+(.+?)(?:\.|;|$)",
                r"(?:fails when|breaks when)\s+(.+?)(?:\.|;|$)",
                r"(?:common mistake|pitfall):\s*(.+?)(?:\.|;|$)"
            ],
            "improvement": [
                r"(?:improve|enhance|optimize) by\s+(.+?)(?:\.|;|$)",
                r"(?:can be improved|enhancement):\s*(.+?)(?:\.|;|$)",
                r"(?:optimization|improvement) (?:opportunity|suggestion):\s*(.+?)(?:\.|;|$)"
            ]
        }
        
        # Category keywords for classification
        self.category_keywords = {
            "performance": ["performance", "speed", "efficiency", "latency", "throughput"],
            "reliability": ["reliability", "stability", "robust", "fault", "error"],
            "scalability": ["scale", "scaling", "growth", "capacity", "load"],
            "security": ["security", "secure", "vulnerability", "protection", "safety"],
            "usability": ["usability", "user experience", "interface", "ease of use"],
            "maintainability": ["maintain", "maintenance", "code quality", "technical debt"],
            "integration": ["integration", "integrate", "compatibility", "interoperability"],
            "architecture": ["architecture", "design", "structure", "pattern", "framework"]
        }
        
        # Insight metadata tracking
        self.insight_registry = self._load_insight_registry()
        self.pattern_registry = self._load_pattern_registry()
        
        # Statistics
        self.processing_stats = {
            "total_processed": 0,
            "insights_extracted": 0,
            "patterns_identified": 0,
            "recommendations_generated": 0
        }
    
    def _load_insight_registry(self) -> Dict:
        """Load or create insight registry."""
        registry_path = self.storage_path / "insight_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {
            "insights": {},
            "categories": defaultdict(list),
            "confidence_scores": {}
        }
    
    def _load_pattern_registry(self) -> Dict:
        """Load or create pattern registry."""
        registry_path = self.storage_path / "pattern_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {
            "patterns": {},
            "occurrences": defaultdict(int),
            "correlations": {}
        }
    
    def process_research_batch(self, research_items: List[Dict]) -> Dict[str, Any]:
        """
        Process a batch of research items to extract insights.
        
        Args:
            research_items: List of research results to process
            
        Returns:
            Processing results including extracted insights
        """
        batch_results = {
            "total_processed": len(research_items),
            "insights": [],
            "patterns": [],
            "recommendations": [],
            "success_factors": [],
            "failure_analyses": [],
            "aggregated_insights": {}
        }
        
        # Process each research item
        for research in research_items:
            item_results = self.process_single_research(research)
            
            batch_results["insights"].extend(item_results["insights"])
            batch_results["patterns"].extend(item_results["patterns"])
            batch_results["recommendations"].extend(item_results["recommendations"])
            batch_results["success_factors"].extend(item_results["success_factors"])
            batch_results["failure_analyses"].extend(item_results["failure_analyses"])
        
        # Aggregate insights across batch
        batch_results["aggregated_insights"] = self._aggregate_insights(batch_results)
        
        # Identify cross-research patterns
        cross_patterns = self._identify_cross_patterns(batch_results["insights"])
        batch_results["cross_patterns"] = cross_patterns
        
        # Update statistics
        self.processing_stats["total_processed"] += len(research_items)
        self.processing_stats["insights_extracted"] += len(batch_results["insights"])
        self.processing_stats["patterns_identified"] += len(batch_results["patterns"])
        self.processing_stats["recommendations_generated"] += len(batch_results["recommendations"])
        
        # Save results
        self._save_batch_results(batch_results)
        
        return batch_results
    
    def process_single_research(self, research: Dict) -> Dict[str, Any]:
        """
        Process a single research item to extract insights.
        
        Args:
            research: Research data to process
            
        Returns:
            Extracted insights categorized by type
        """
        results = {
            "research_id": research.get("id", "unknown"),
            "insights": [],
            "patterns": [],
            "recommendations": [],
            "success_factors": [],
            "failure_analyses": []
        }
        
        # Extract content
        content = self._extract_content(research)
        
        # Extract insights by type
        for insight_type, patterns in self.insight_patterns.items():
            extracted = self._extract_insights_by_pattern(content, patterns, insight_type)
            
            for insight in extracted:
                # Categorize insight
                categories = self._categorize_insight(insight["text"])
                
                # Calculate confidence
                confidence = self._calculate_confidence(insight, research)
                
                # Create insight object
                insight_obj = {
                    "id": self._generate_insight_id(insight["text"]),
                    "type": insight_type,
                    "text": insight["text"],
                    "context": insight["context"],
                    "categories": categories,
                    "confidence": confidence,
                    "source": research.get("id"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to appropriate list
                if insight_type == "recommendation":
                    results["recommendations"].append(insight_obj)
                elif insight_type == "pattern":
                    results["patterns"].append(insight_obj)
                elif insight_type == "success_factor":
                    results["success_factors"].append(insight_obj)
                elif insight_type == "failure_cause":
                    results["failure_analyses"].append(insight_obj)
                
                results["insights"].append(insight_obj)
        
        # Extract structured insights
        structured_insights = self._extract_structured_insights(content, research)
        results["insights"].extend(structured_insights)
        
        # Process and store insights
        self._store_insights(results)
        
        return results
    
    def _extract_content(self, research: Dict) -> str:
        """Extract textual content from research data."""
        content_parts = []
        
        def extract_text(obj):
            if isinstance(obj, str):
                return obj
            elif isinstance(obj, dict):
                return " ".join(extract_text(v) for v in obj.values() if v)
            elif isinstance(obj, list):
                return " ".join(extract_text(item) for item in obj)
            else:
                return str(obj)
        
        # Extract from various fields
        for field in ["content", "query", "topic", "results", "analysis"]:
            if field in research:
                content_parts.append(extract_text(research[field]))
        
        return " ".join(content_parts)
    
    def _extract_insights_by_pattern(self, content: str, patterns: List[str], 
                                   insight_type: str) -> List[Dict]:
        """Extract insights using regex patterns."""
        insights = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                insight_text = match.group(1).strip()
                
                # Get context (surrounding text)
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                insights.append({
                    "text": insight_text,
                    "context": context,
                    "pattern": pattern,
                    "type": insight_type
                })
        
        return insights
    
    def _extract_structured_insights(self, content: str, research: Dict) -> List[Dict]:
        """Extract insights from structured content."""
        insights = []
        
        # Look for numbered lists
        numbered_pattern = r'^\s*\d+\.\s+(.+)$'
        for match in re.finditer(numbered_pattern, content, re.MULTILINE):
            item_text = match.group(1).strip()
            
            # Determine insight type based on content
            insight_type = self._determine_insight_type(item_text)
            categories = self._categorize_insight(item_text)
            
            insights.append({
                "id": self._generate_insight_id(item_text),
                "type": insight_type,
                "text": item_text,
                "context": item_text,
                "categories": categories,
                "confidence": 0.7,
                "source": research.get("id"),
                "timestamp": datetime.now().isoformat(),
                "structured": True
            })
        
        # Look for bullet points
        bullet_patterns = [r'^\s*[-*•]\s+(.+)$']
        for pattern in bullet_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                item_text = match.group(1).strip()
                
                insight_type = self._determine_insight_type(item_text)
                categories = self._categorize_insight(item_text)
                
                insights.append({
                    "id": self._generate_insight_id(item_text),
                    "type": insight_type,
                    "text": item_text,
                    "context": item_text,
                    "categories": categories,
                    "confidence": 0.7,
                    "source": research.get("id"),
                    "timestamp": datetime.now().isoformat(),
                    "structured": True
                })
        
        return insights
    
    def _determine_insight_type(self, text: str) -> str:
        """Determine insight type based on content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["should", "recommend", "suggest", "consider"]):
            return "recommendation"
        elif any(word in text_lower for word in ["pattern", "trend", "commonly", "frequently"]):
            return "pattern"
        elif any(word in text_lower for word in ["success", "effective", "works well"]):
            return "success_factor"
        elif any(word in text_lower for word in ["fail", "issue", "problem", "error"]):
            return "failure_cause"
        elif any(word in text_lower for word in ["improve", "enhance", "optimize"]):
            return "improvement"
        else:
            return "general"
    
    def _categorize_insight(self, text: str) -> List[str]:
        """Categorize insight based on keywords."""
        categories = []
        text_lower = text.lower()
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        # Default category if none found
        if not categories:
            categories.append("general")
        
        return categories
    
    def _calculate_confidence(self, insight: Dict, research: Dict) -> float:
        """Calculate confidence score for an insight."""
        confidence = 0.5  # Base confidence
        
        # Boost for specific patterns
        if insight.get("pattern"):
            confidence += 0.1
        
        # Boost for longer insights (more context)
        if len(insight["text"]) > 50:
            confidence += 0.1
        if len(insight["text"]) > 100:
            confidence += 0.1
        
        # Boost based on research quality
        research_quality = research.get("quality_score", 0.5)
        confidence += research_quality * 0.2
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    def _generate_insight_id(self, text: str) -> str:
        """Generate unique ID for insight."""
        return f"insight_{hashlib.md5(text.encode()).hexdigest()[:12]}"
    
    def _aggregate_insights(self, batch_results: Dict) -> Dict[str, Any]:
        """Aggregate insights to find common themes."""
        aggregated = {
            "common_recommendations": [],
            "recurring_patterns": [],
            "top_categories": [],
            "confidence_distribution": {}
        }
        
        # Count recommendation occurrences
        recommendation_counts = Counter()
        for rec in batch_results["recommendations"]:
            # Normalize text for comparison
            normalized = rec["text"].lower().strip()
            recommendation_counts[normalized] += 1
        
        # Find common recommendations
        for text, count in recommendation_counts.most_common(10):
            if count > 1:
                aggregated["common_recommendations"].append({
                    "text": text,
                    "occurrences": count,
                    "sources": [r["source"] for r in batch_results["recommendations"] 
                              if r["text"].lower().strip() == text]
                })
        
        # Find recurring patterns
        pattern_texts = [p["text"] for p in batch_results["patterns"]]
        pattern_counts = Counter(pattern_texts)
        
        for text, count in pattern_counts.most_common(10):
            if count > 1:
                aggregated["recurring_patterns"].append({
                    "pattern": text,
                    "frequency": count
                })
        
        # Category distribution
        all_categories = []
        for insight in batch_results["insights"]:
            all_categories.extend(insight.get("categories", []))
        
        category_counts = Counter(all_categories)
        aggregated["top_categories"] = [
            {"category": cat, "count": count} 
            for cat, count in category_counts.most_common()
        ]
        
        # Confidence distribution
        confidence_buckets = defaultdict(int)
        for insight in batch_results["insights"]:
            confidence = insight.get("confidence", 0.5)
            bucket = f"{int(confidence * 10) / 10:.1f}"
            confidence_buckets[bucket] += 1
        
        aggregated["confidence_distribution"] = dict(confidence_buckets)
        
        return aggregated
    
    def _identify_cross_patterns(self, insights: List[Dict]) -> List[Dict]:
        """Identify patterns across multiple insights."""
        cross_patterns = []
        
        # Group insights by category
        category_groups = defaultdict(list)
        for insight in insights:
            for category in insight.get("categories", ["general"]):
                category_groups[category].append(insight)
        
        # Look for patterns within categories
        for category, category_insights in category_groups.items():
            if len(category_insights) >= 3:
                # Extract common words/phrases
                common_terms = self._extract_common_terms(category_insights)
                
                if common_terms:
                    cross_patterns.append({
                        "id": f"pattern_{category}_{len(cross_patterns)}",
                        "category": category,
                        "insight_count": len(category_insights),
                        "common_terms": common_terms[:5],
                        "confidence": min(1.0, len(category_insights) / 10.0),
                        "description": f"Pattern identified across {len(category_insights)} insights in {category}"
                    })
        
        # Look for insight chains (one insight leading to another)
        chains = self._find_insight_chains(insights)
        cross_patterns.extend(chains)
        
        return cross_patterns
    
    def _extract_common_terms(self, insights: List[Dict]) -> List[Tuple[str, int]]:
        """Extract common terms from insights."""
        # Simple word frequency analysis
        word_counts = Counter()
        
        for insight in insights:
            text = insight["text"].lower()
            # Extract meaningful words (simple approach)
            words = re.findall(r'\b\w{4,}\b', text)
            
            # Filter out common words
            common_words = {"that", "this", "with", "from", "have", "been", "will", "when"}
            words = [w for w in words if w not in common_words]
            
            word_counts.update(words)
        
        # Return words that appear in multiple insights
        min_occurrences = max(2, len(insights) // 3)
        return [(word, count) for word, count in word_counts.most_common() 
                if count >= min_occurrences]
    
    def _find_insight_chains(self, insights: List[Dict]) -> List[Dict]:
        """Find chains of related insights."""
        chains = []
        
        # Simple approach: look for insights that reference similar concepts
        for i, insight1 in enumerate(insights):
            chain = [insight1]
            
            for j, insight2 in enumerate(insights[i+1:], i+1):
                # Check if insights are related
                if self._are_insights_related(insight1, insight2):
                    chain.append(insight2)
            
            if len(chain) >= 3:
                chains.append({
                    "id": f"chain_{len(chains)}",
                    "type": "insight_chain",
                    "insights": [{"id": ins["id"], "text": ins["text"][:100]} for ins in chain],
                    "length": len(chain),
                    "confidence": 0.7,
                    "description": f"Chain of {len(chain)} related insights"
                })
        
        return chains[:5]  # Limit to top 5 chains
    
    def _are_insights_related(self, insight1: Dict, insight2: Dict) -> bool:
        """Check if two insights are related."""
        # Simple overlap check
        words1 = set(re.findall(r'\b\w{4,}\b', insight1["text"].lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', insight2["text"].lower()))
        
        overlap = len(words1 & words2)
        min_size = min(len(words1), len(words2))
        
        # Related if significant overlap
        return overlap / max(1, min_size) > 0.3
    
    def _store_insights(self, results: Dict):
        """Store processed insights to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store by type
        type_mapping = {
            "recommendations": self.recommendations_path,
            "patterns": self.patterns_path,
            "success_factors": self.success_factors_path,
            "failure_analyses": self.failure_analysis_path
        }
        
        for insight_type, path in type_mapping.items():
            if results[insight_type]:
                filename = f"{results['research_id']}_{timestamp}.json"
                with open(path / filename, 'w') as f:
                    json.dump({
                        "research_id": results["research_id"],
                        "timestamp": timestamp,
                        "insights": results[insight_type]
                    }, f, indent=2)
        
        # Update registry
        for insight in results["insights"]:
            insight_id = insight["id"]
            self.insight_registry["insights"][insight_id] = insight
            
            for category in insight.get("categories", ["general"]):
                self.insight_registry["categories"][category].append(insight_id)
            
            self.insight_registry["confidence_scores"][insight_id] = insight["confidence"]
    
    def _save_batch_results(self, batch_results: Dict):
        """Save batch processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated insights
        if batch_results["aggregated_insights"]:
            agg_path = self.storage_path / f"aggregated_insights_{timestamp}.json"
            with open(agg_path, 'w') as f:
                json.dump(batch_results["aggregated_insights"], f, indent=2)
        
        # Save cross patterns
        if batch_results["cross_patterns"]:
            patterns_path = self.storage_path / f"cross_patterns_{timestamp}.json"
            with open(patterns_path, 'w') as f:
                json.dump(batch_results["cross_patterns"], f, indent=2)
        
        # Update registries
        self._save_registries()
    
    def _save_registries(self):
        """Save insight and pattern registries."""
        registry_path = self.storage_path / "insight_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.insight_registry, f, indent=2)
        
        pattern_path = self.storage_path / "pattern_registry.json"
        with open(pattern_path, 'w') as f:
            json.dump(self.pattern_registry, f, indent=2)
    
    def get_insights_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get insights for a specific category."""
        insight_ids = self.insight_registry["categories"].get(category, [])[:limit]
        
        insights = []
        for insight_id in insight_ids:
            if insight_id in self.insight_registry["insights"]:
                insights.append(self.insight_registry["insights"][insight_id])
        
        # Sort by confidence
        insights.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return insights
    
    def get_top_recommendations(self, limit: int = 10) -> List[Dict]:
        """Get top recommendations by confidence and frequency."""
        recommendations = []
        
        for insight_id, insight in self.insight_registry["insights"].items():
            if insight.get("type") == "recommendation":
                recommendations.append(insight)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return recommendations[:limit]
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get analysis of identified patterns."""
        analysis = {
            "total_patterns": len([i for i in self.insight_registry["insights"].values() 
                                 if i.get("type") == "pattern"]),
            "pattern_categories": defaultdict(int),
            "pattern_confidence_avg": 0.0,
            "most_common_patterns": []
        }
        
        # Analyze patterns
        pattern_texts = []
        confidence_sum = 0
        pattern_count = 0
        
        for insight in self.insight_registry["insights"].values():
            if insight.get("type") == "pattern":
                pattern_texts.append(insight["text"])
                confidence_sum += insight.get("confidence", 0)
                pattern_count += 1
                
                for category in insight.get("categories", ["general"]):
                    analysis["pattern_categories"][category] += 1
        
        # Calculate average confidence
        if pattern_count > 0:
            analysis["pattern_confidence_avg"] = confidence_sum / pattern_count
        
        # Find most common patterns
        pattern_counts = Counter(pattern_texts)
        analysis["most_common_patterns"] = [
            {"pattern": pattern, "frequency": count}
            for pattern, count in pattern_counts.most_common(5)
        ]
        
        analysis["pattern_categories"] = dict(analysis["pattern_categories"])
        
        return analysis
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.processing_stats,
            "total_insights_stored": len(self.insight_registry["insights"]),
            "category_distribution": {
                cat: len(ids) for cat, ids in self.insight_registry["categories"].items()
            },
            "insight_types": self._count_insight_types()
        }
    
    def _count_insight_types(self) -> Dict[str, int]:
        """Count insights by type."""
        type_counts = defaultdict(int)
        
        for insight in self.insight_registry["insights"].values():
            insight_type = insight.get("type", "general")
            type_counts[insight_type] += 1
        
        return dict(type_counts)