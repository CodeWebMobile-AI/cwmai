"""
Cross-Research Analyzer - Discovers patterns and insights across multiple research results.

This module analyzes research data from different sources and time periods to identify
overarching patterns, contradictions, confirmations, and meta-insights.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from collections import defaultdict, Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging


class CrossResearchAnalyzer:
    """Analyzes patterns across multiple research results."""
    
    def __init__(self, knowledge_store=None, knowledge_graph=None):
        self.knowledge_store = knowledge_store
        self.knowledge_graph = knowledge_graph
        
        # Analysis configuration
        self.config = {
            "min_pattern_occurrences": 3,
            "similarity_threshold": 0.7,
            "time_window_days": 7,
            "max_research_items": 100,
            "pattern_confidence_threshold": 0.6
        }
        
        # Pattern detection
        self.pattern_types = {
            "convergent": "Multiple research items reaching similar conclusions",
            "divergent": "Research items with conflicting findings",
            "evolutionary": "Ideas that develop over time",
            "cyclical": "Patterns that repeat periodically",
            "emergent": "New patterns arising from combinations"
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.pattern_history = []
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "patterns_discovered": 0,
            "insights_generated": 0,
            "contradictions_found": 0
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def analyze_research_corpus(self, research_type: Optional[str] = None,
                              time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze entire research corpus or filtered subset.
        
        Args:
            research_type: Optional filter by research type
            time_window: Optional time window in days
            
        Returns:
            Comprehensive analysis results
        """
        analysis_id = f"analysis_{datetime.now().timestamp()}"
        self.logger.info(f"Starting cross-research analysis: {analysis_id}")
        
        # Retrieve research data
        research_items = self._retrieve_research_data(research_type, time_window)
        
        if not research_items:
            return {
                "analysis_id": analysis_id,
                "status": "no_data",
                "message": "No research data found for analysis"
            }
        
        # Perform various analyses
        results = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "research_count": len(research_items),
            "time_span": self._calculate_time_span(research_items),
            "pattern_analysis": self._analyze_patterns(research_items),
            "theme_analysis": self._analyze_themes(research_items),
            "evolution_analysis": self._analyze_evolution(research_items),
            "contradiction_analysis": self._find_contradictions(research_items),
            "convergence_analysis": self._find_convergences(research_items),
            "meta_insights": self._generate_meta_insights(research_items),
            "recommendations": self._generate_recommendations(research_items)
        }
        
        # Update statistics
        self.stats["total_analyses"] += 1
        self.stats["patterns_discovered"] += len(results["pattern_analysis"]["patterns"])
        self.stats["insights_generated"] += len(results["meta_insights"])
        self.stats["contradictions_found"] += len(results["contradiction_analysis"]["contradictions"])
        
        # Cache results
        self.analysis_cache[analysis_id] = results
        
        return results
    
    def _retrieve_research_data(self, research_type: Optional[str],
                               time_window: Optional[int]) -> List[Dict]:
        """Retrieve research data from knowledge store."""
        if not self.knowledge_store:
            return []
        
        # Determine time filter
        min_timestamp = None
        if time_window:
            min_timestamp = datetime.now() - timedelta(days=time_window)
        
        # Retrieve research
        if research_type:
            research_items = self.knowledge_store.retrieve_research(
                research_type=research_type,
                limit=self.config["max_research_items"]
            )
        else:
            # Get all research types
            research_items = []
            for rtype in ["innovation", "efficiency", "growth", "strategic_planning", 
                         "continuous_improvement", "task_performance", "claude_interactions"]:
                items = self.knowledge_store.retrieve_research(
                    research_type=rtype,
                    limit=20  # Limit per type
                )
                research_items.extend(items)
        
        # Filter by time if needed
        if min_timestamp:
            filtered_items = []
            for item in research_items:
                try:
                    item_time = datetime.fromisoformat(item.get("timestamp", ""))
                    if item_time > min_timestamp:
                        filtered_items.append(item)
                except:
                    pass
            research_items = filtered_items
        
        return research_items[:self.config["max_research_items"]]
    
    def _calculate_time_span(self, research_items: List[Dict]) -> Dict:
        """Calculate time span of research data."""
        if not research_items:
            return {"days": 0, "start": None, "end": None}
        
        timestamps = []
        for item in research_items:
            try:
                ts = datetime.fromisoformat(item.get("timestamp", ""))
                timestamps.append(ts)
            except:
                pass
        
        if not timestamps:
            return {"days": 0, "start": None, "end": None}
        
        start = min(timestamps)
        end = max(timestamps)
        days = (end - start).days
        
        return {
            "days": days,
            "start": start.isoformat(),
            "end": end.isoformat()
        }
    
    def _analyze_patterns(self, research_items: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across research items."""
        patterns = {
            "patterns": [],
            "pattern_types": defaultdict(int),
            "confidence_scores": []
        }
        
        # Extract text content from all items
        texts = []
        for item in research_items:
            content = self._extract_text_content(item)
            texts.append(content)
        
        if not texts:
            return patterns
        
        # Use TF-IDF to find important terms
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Find recurring important terms
            term_scores = tfidf_matrix.sum(axis=0).A1
            top_terms = [(feature_names[i], term_scores[i]) 
                        for i in term_scores.argsort()[-20:][::-1]]
            
            # Look for co-occurring terms (simple pattern detection)
            for i, (term1, score1) in enumerate(top_terms):
                for term2, score2 in top_terms[i+1:]:
                    co_occurrence = self._count_co_occurrences(texts, term1, term2)
                    
                    if co_occurrence >= self.config["min_pattern_occurrences"]:
                        pattern = {
                            "id": f"pattern_{term1}_{term2}",
                            "type": "co_occurrence",
                            "terms": [term1, term2],
                            "occurrences": co_occurrence,
                            "confidence": min(1.0, co_occurrence / len(texts)),
                            "description": f"'{term1}' and '{term2}' frequently appear together"
                        }
                        patterns["patterns"].append(pattern)
                        patterns["pattern_types"]["co_occurrence"] += 1
                        patterns["confidence_scores"].append(pattern["confidence"])
            
        except Exception as e:
            self.logger.error(f"Error in TF-IDF analysis: {e}")
        
        # Detect sequential patterns
        sequential_patterns = self._detect_sequential_patterns(research_items)
        patterns["patterns"].extend(sequential_patterns)
        
        # Detect cyclical patterns
        cyclical_patterns = self._detect_cyclical_patterns(research_items)
        patterns["patterns"].extend(cyclical_patterns)
        
        return patterns
    
    def _extract_text_content(self, item: Dict) -> str:
        """Extract text content from research item."""
        text_parts = []
        
        # Extract from various fields
        if isinstance(item.get("content"), dict):
            text_parts.append(json.dumps(item["content"]))
        elif isinstance(item.get("content"), str):
            text_parts.append(item["content"])
        
        if "query" in item:
            text_parts.append(str(item["query"]))
        
        if "topic" in item:
            text_parts.append(str(item["topic"]))
        
        return " ".join(text_parts)
    
    def _count_co_occurrences(self, texts: List[str], term1: str, term2: str) -> int:
        """Count co-occurrences of two terms."""
        count = 0
        for text in texts:
            if term1 in text.lower() and term2 in text.lower():
                count += 1
        return count
    
    def _detect_sequential_patterns(self, research_items: List[Dict]) -> List[Dict]:
        """Detect patterns that develop sequentially over time."""
        patterns = []
        
        # Sort by timestamp
        sorted_items = sorted(research_items, 
                            key=lambda x: x.get("timestamp", ""))
        
        # Look for evolving themes
        theme_evolution = defaultdict(list)
        
        for i, item in enumerate(sorted_items):
            content = self._extract_text_content(item)
            
            # Simple keyword extraction
            keywords = re.findall(r'\b\w{5,}\b', content.lower())
            common_keywords = Counter(keywords).most_common(5)
            
            for keyword, _ in common_keywords:
                theme_evolution[keyword].append({
                    "index": i,
                    "timestamp": item.get("timestamp"),
                    "context": content[:200]
                })
        
        # Find themes that evolve
        for theme, occurrences in theme_evolution.items():
            if len(occurrences) >= self.config["min_pattern_occurrences"]:
                # Check if occurrences are spread over time
                first_idx = occurrences[0]["index"]
                last_idx = occurrences[-1]["index"]
                
                if last_idx - first_idx > len(sorted_items) * 0.3:  # Spread over 30% of timeline
                    pattern = {
                        "id": f"sequential_{theme}",
                        "type": "evolutionary",
                        "theme": theme,
                        "occurrences": len(occurrences),
                        "time_span": f"{first_idx} to {last_idx}",
                        "confidence": 0.7,
                        "description": f"Theme '{theme}' evolves over time"
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_cyclical_patterns(self, research_items: List[Dict]) -> List[Dict]:
        """Detect patterns that repeat cyclically."""
        patterns = []
        
        # Group by research type and time
        type_timeline = defaultdict(list)
        
        for item in research_items:
            rtype = item.get("type", "unknown")
            timestamp = item.get("timestamp")
            
            if timestamp:
                try:
                    ts = datetime.fromisoformat(timestamp)
                    type_timeline[rtype].append(ts)
                except:
                    pass
        
        # Look for cyclical patterns in each type
        for rtype, timestamps in type_timeline.items():
            if len(timestamps) >= self.config["min_pattern_occurrences"]:
                # Calculate intervals
                timestamps.sort()
                intervals = []
                
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
                    intervals.append(interval)
                
                if intervals:
                    # Check for regular intervals (simple approach)
                    avg_interval = sum(intervals) / len(intervals)
                    std_dev = np.std(intervals) if len(intervals) > 1 else 0
                    
                    # Low standard deviation suggests regular pattern
                    if std_dev < avg_interval * 0.3:  # Less than 30% variation
                        pattern = {
                            "id": f"cyclical_{rtype}",
                            "type": "cyclical",
                            "research_type": rtype,
                            "average_interval_hours": round(avg_interval, 2),
                            "occurrences": len(timestamps),
                            "confidence": 0.6,
                            "description": f"{rtype} research occurs regularly every ~{round(avg_interval, 1)} hours"
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_themes(self, research_items: List[Dict]) -> Dict[str, Any]:
        """Analyze major themes across research."""
        themes = {
            "major_themes": [],
            "theme_distribution": {},
            "theme_evolution": []
        }
        
        # Collect all text
        all_text = []
        for item in research_items:
            text = self._extract_text_content(item)
            all_text.append(text)
        
        if not all_text:
            return themes
        
        # Use TF-IDF to identify themes
        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_text)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms as themes
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-20:][::-1]
            
            for idx in top_indices:
                theme_name = feature_names[idx]
                theme_score = scores[idx]
                
                # Count occurrences in documents
                doc_count = sum(1 for i in range(len(all_text)) if theme_name in all_text[i].lower())
                
                theme = {
                    "name": theme_name,
                    "score": float(theme_score),
                    "document_frequency": doc_count,
                    "percentage": (doc_count / len(all_text)) * 100
                }
                themes["major_themes"].append(theme)
            
            # Theme distribution by research type
            for item, text in zip(research_items, all_text):
                rtype = item.get("type", "unknown")
                if rtype not in themes["theme_distribution"]:
                    themes["theme_distribution"][rtype] = defaultdict(int)
                
                for theme in themes["major_themes"][:10]:  # Top 10 themes
                    if theme["name"] in text.lower():
                        themes["theme_distribution"][rtype][theme["name"]] += 1
            
        except Exception as e:
            self.logger.error(f"Error in theme analysis: {e}")
        
        return themes
    
    def _analyze_evolution(self, research_items: List[Dict]) -> Dict[str, Any]:
        """Analyze how research focus evolves over time."""
        evolution = {
            "focus_shifts": [],
            "emerging_topics": [],
            "declining_topics": [],
            "stability_score": 0.0
        }
        
        # Sort by timestamp
        sorted_items = sorted(research_items, 
                            key=lambda x: x.get("timestamp", ""))
        
        if len(sorted_items) < 10:
            return evolution
        
        # Divide into time periods
        period_size = max(1, len(sorted_items) // 3)  # 3 periods
        periods = [
            sorted_items[:period_size],
            sorted_items[period_size:period_size*2],
            sorted_items[period_size*2:]
        ]
        
        # Analyze themes in each period
        period_themes = []
        for i, period_items in enumerate(periods):
            themes = Counter()
            
            for item in period_items:
                text = self._extract_text_content(item).lower()
                # Simple keyword extraction
                keywords = re.findall(r'\b\w{5,}\b', text)
                themes.update(keywords)
            
            top_themes = themes.most_common(10)
            period_themes.append({
                "period": i,
                "themes": dict(top_themes)
            })
        
        # Find emerging topics (appear more in later periods)
        if len(period_themes) >= 2:
            early_themes = set(period_themes[0]["themes"].keys())
            late_themes = set(period_themes[-1]["themes"].keys())
            
            # Emerging topics
            emerging = late_themes - early_themes
            for topic in emerging:
                evolution["emerging_topics"].append({
                    "topic": topic,
                    "first_appearance": "late",
                    "frequency": period_themes[-1]["themes"].get(topic, 0)
                })
            
            # Declining topics
            declining = early_themes - late_themes
            for topic in declining:
                evolution["declining_topics"].append({
                    "topic": topic,
                    "last_appearance": "early",
                    "frequency": period_themes[0]["themes"].get(topic, 0)
                })
            
            # Calculate stability score
            common_themes = early_themes & late_themes
            if early_themes:
                evolution["stability_score"] = len(common_themes) / len(early_themes)
        
        return evolution
    
    def _find_contradictions(self, research_items: List[Dict]) -> Dict[str, Any]:
        """Find contradictory findings across research."""
        contradictions = {
            "contradictions": [],
            "total_found": 0,
            "by_type": defaultdict(int)
        }
        
        # Simple contradiction detection based on opposing keywords
        opposing_pairs = [
            ("increase", "decrease"),
            ("improve", "degrade"),
            ("success", "failure"),
            ("effective", "ineffective"),
            ("positive", "negative"),
            ("growth", "decline"),
            ("stable", "unstable")
        ]
        
        # Compare pairs of research items
        for i, item1 in enumerate(research_items):
            text1 = self._extract_text_content(item1).lower()
            
            for j, item2 in enumerate(research_items[i+1:], i+1):
                text2 = self._extract_text_content(item2).lower()
                
                # Check for opposing statements
                for pos_term, neg_term in opposing_pairs:
                    if ((pos_term in text1 and neg_term in text2) or 
                        (neg_term in text1 and pos_term in text2)):
                        
                        # Check if they're discussing similar topics
                        similarity = self._calculate_similarity(text1, text2)
                        
                        if similarity > 0.3:  # Some topical similarity
                            contradiction = {
                                "id": f"contradiction_{i}_{j}",
                                "items": [
                                    {"id": item1.get("id"), "stance": pos_term if pos_term in text1 else neg_term},
                                    {"id": item2.get("id"), "stance": pos_term if pos_term in text2 else neg_term}
                                ],
                                "topic_similarity": similarity,
                                "opposing_terms": [pos_term, neg_term],
                                "description": f"Research items show opposing views on similar topic"
                            }
                            contradictions["contradictions"].append(contradiction)
                            contradictions["by_type"][item1.get("type", "unknown")] += 1
        
        contradictions["total_found"] = len(contradictions["contradictions"])
        
        return contradictions
    
    def _find_convergences(self, research_items: List[Dict]) -> Dict[str, Any]:
        """Find convergent findings across research."""
        convergences = {
            "convergent_findings": [],
            "consensus_topics": [],
            "agreement_score": 0.0
        }
        
        # Group similar research items
        similarity_groups = self._group_by_similarity(research_items)
        
        # Find groups with high agreement
        for group in similarity_groups:
            if len(group) >= self.config["min_pattern_occurrences"]:
                # Extract common themes from group
                common_terms = self._extract_common_terms(group)
                
                if common_terms:
                    convergence = {
                        "id": f"convergence_{len(convergences['convergent_findings'])}",
                        "research_count": len(group),
                        "common_themes": common_terms[:5],
                        "research_types": list(set(item.get("type", "unknown") for item in group)),
                        "confidence": len(group) / len(research_items),
                        "description": f"{len(group)} research items converge on similar findings"
                    }
                    convergences["convergent_findings"].append(convergence)
        
        # Calculate overall agreement score
        if research_items:
            convergent_count = sum(len(g) for g in similarity_groups if len(g) > 1)
            convergences["agreement_score"] = convergent_count / len(research_items)
        
        return convergences
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _group_by_similarity(self, research_items: List[Dict]) -> List[List[Dict]]:
        """Group research items by similarity."""
        if not research_items:
            return []
        
        # Extract texts
        texts = [self._extract_text_content(item) for item in research_items]
        
        try:
            # Calculate similarity matrix
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Group by similarity threshold
            groups = []
            used_indices = set()
            
            for i in range(len(research_items)):
                if i in used_indices:
                    continue
                
                group = [research_items[i]]
                used_indices.add(i)
                
                for j in range(i + 1, len(research_items)):
                    if j not in used_indices and similarity_matrix[i][j] > self.config["similarity_threshold"]:
                        group.append(research_items[j])
                        used_indices.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error in similarity grouping: {e}")
            return [[item] for item in research_items]
    
    def _extract_common_terms(self, research_items: List[Dict]) -> List[str]:
        """Extract common terms from a group of research items."""
        all_terms = Counter()
        
        for item in research_items:
            text = self._extract_text_content(item).lower()
            # Extract meaningful terms
            terms = re.findall(r'\b\w{5,}\b', text)
            all_terms.update(terms)
        
        # Filter common English words
        common_words = {"would", "could", "should", "might", "these", "those", "there", "where"}
        filtered_terms = [(term, count) for term, count in all_terms.most_common(20) 
                         if term not in common_words and count >= len(research_items) * 0.5]
        
        return [term for term, _ in filtered_terms]
    
    def _generate_meta_insights(self, research_items: List[Dict]) -> List[Dict]:
        """Generate high-level insights from analysis."""
        insights = []
        
        # Insight 1: Research focus distribution
        type_counts = Counter(item.get("type", "unknown") for item in research_items)
        if type_counts:
            most_common = type_counts.most_common(1)[0]
            insights.append({
                "id": "meta_focus",
                "type": "research_focus",
                "insight": f"Research heavily focused on {most_common[0]} ({most_common[1]} items, {most_common[1]/len(research_items)*100:.1f}%)",
                "recommendation": "Consider diversifying research areas for comprehensive improvement",
                "confidence": 0.9
            })
        
        # Insight 2: Temporal patterns
        timestamps = []
        for item in research_items:
            try:
                ts = datetime.fromisoformat(item.get("timestamp", ""))
                timestamps.append(ts)
            except:
                pass
        
        if len(timestamps) > 10:
            # Check research frequency
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                        for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            if avg_interval < 1:
                insights.append({
                    "id": "meta_frequency",
                    "type": "research_pattern",
                    "insight": f"Very frequent research (avg {avg_interval:.1f} hours between items)",
                    "recommendation": "High research frequency may indicate system instability or over-optimization",
                    "confidence": 0.7
                })
            elif avg_interval > 24:
                insights.append({
                    "id": "meta_frequency",
                    "type": "research_pattern",
                    "insight": f"Infrequent research (avg {avg_interval:.1f} hours between items)",
                    "recommendation": "Consider more frequent research for continuous improvement",
                    "confidence": 0.7
                })
        
        # Insight 3: Quality distribution
        quality_scores = [item.get("quality_score", 0.5) for item in research_items]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            if avg_quality < 0.6:
                insights.append({
                    "id": "meta_quality",
                    "type": "quality_issue",
                    "insight": f"Low average research quality ({avg_quality:.2f})",
                    "recommendation": "Focus on improving research query generation and execution",
                    "confidence": 0.8
                })
            elif avg_quality > 0.8:
                insights.append({
                    "id": "meta_quality", 
                    "type": "quality_success",
                    "insight": f"High average research quality ({avg_quality:.2f})",
                    "recommendation": "Current research approach is effective, maintain strategy",
                    "confidence": 0.8
                })
        
        return insights
    
    def _generate_recommendations(self, research_items: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations from analysis."""
        recommendations = []
        
        # Analyze research distribution
        type_counts = Counter(item.get("type", "unknown") for item in research_items)
        total_items = len(research_items)
        
        # Recommendation 1: Balance research types
        if type_counts:
            max_percentage = max(count / total_items for count in type_counts.values())
            if max_percentage > 0.4:  # One type dominates > 40%
                recommendations.append({
                    "id": "rec_balance",
                    "priority": "high",
                    "recommendation": "Diversify research types for holistic system improvement",
                    "action": "Adjust research selection algorithm to ensure balanced coverage",
                    "expected_impact": "More comprehensive system optimization"
                })
        
        # Recommendation 2: Address contradictions
        contradictions = self._find_contradictions(research_items)
        if contradictions["total_found"] > 5:
            recommendations.append({
                "id": "rec_contradictions",
                "priority": "medium",
                "recommendation": "Resolve contradictory research findings",
                "action": "Implement research validation and A/B testing for conflicting recommendations",
                "expected_impact": "Improved research reliability and implementation success"
            })
        
        # Recommendation 3: Leverage convergences
        convergences = self._find_convergences(research_items)
        if convergences["convergent_findings"]:
            top_convergence = convergences["convergent_findings"][0]
            recommendations.append({
                "id": "rec_convergence",
                "priority": "high",
                "recommendation": f"Prioritize implementation of converged findings on {', '.join(top_convergence['common_themes'][:3])}",
                "action": "Create focused implementation tasks for highly-agreed research outcomes",
                "expected_impact": "High-confidence improvements with proven research backing"
            })
        
        # Recommendation 4: Research gaps
        research_gaps = self._identify_research_gaps(research_items)
        for gap in research_gaps[:2]:  # Top 2 gaps
            recommendations.append({
                "id": f"rec_gap_{gap['area']}",
                "priority": "medium",
                "recommendation": f"Address research gap in {gap['area']}",
                "action": f"Schedule targeted research for {gap['description']}",
                "expected_impact": gap['potential_impact']
            })
        
        return recommendations
    
    def _identify_research_gaps(self, research_items: List[Dict]) -> List[Dict]:
        """Identify gaps in research coverage."""
        gaps = []
        
        # Expected research areas
        expected_areas = {
            "error_handling": "Error recovery and resilience",
            "performance_optimization": "System performance and efficiency",
            "scalability": "Growth and scaling strategies",
            "user_experience": "User interaction and satisfaction",
            "security": "Security and safety measures",
            "integration": "External system integration"
        }
        
        # Check coverage
        covered_areas = set()
        for item in research_items:
            text = self._extract_text_content(item).lower()
            for area, description in expected_areas.items():
                if any(keyword in text for keyword in area.split("_")):
                    covered_areas.add(area)
        
        # Identify gaps
        for area, description in expected_areas.items():
            if area not in covered_areas:
                gaps.append({
                    "area": area,
                    "description": description,
                    "potential_impact": f"Improved {area.replace('_', ' ')}"
                })
        
        return gaps
    
    def get_latest_analysis(self) -> Optional[Dict]:
        """Get the most recent analysis results."""
        if not self.analysis_cache:
            return None
        
        latest_id = max(self.analysis_cache.keys())
        return self.analysis_cache[latest_id]
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics."""
        return {
            **self.stats,
            "cache_size": len(self.analysis_cache),
            "pattern_types_found": self._count_pattern_types()
        }
    
    def _count_pattern_types(self) -> Dict[str, int]:
        """Count different types of patterns found."""
        pattern_counts = defaultdict(int)
        
        for analysis in self.analysis_cache.values():
            if "pattern_analysis" in analysis:
                for pattern in analysis["pattern_analysis"].get("patterns", []):
                    pattern_counts[pattern.get("type", "unknown")] += 1
        
        return dict(pattern_counts)