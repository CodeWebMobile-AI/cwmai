"""
Evolutionary Knowledge Store

Manages knowledge about improvements, patterns, and evolution history
to prevent duplication and guide intelligent evolution.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class EvolutionaryKnowledgeStore:
    """Stores and manages knowledge about evolutionary improvements."""
    
    def __init__(self, base_path: str = "evolutionary_knowledge"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._init_directories()
        
        # Knowledge categories
        self.patterns_db = {}  # Successful patterns
        self.failures_db = {}  # Failed attempts
        self.combinations_db = {}  # Successful combinations
        self.deployment_results = {}  # Real-world performance
        self.quality_scores = {}  # Historical quality metrics
        
        # Pattern analysis
        self.pattern_frequency = Counter()
        self.pattern_success_rate = defaultdict(lambda: {"success": 0, "total": 0})
        
        # Similarity detection
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        self.improvement_vectors = {}
        
        # Load existing knowledge
        self._load_knowledge()
        
    def _init_directories(self):
        """Initialize directory structure."""
        directories = [
            "patterns/successful",
            "patterns/failed", 
            "combinations",
            "deployments",
            "quality_metrics",
            "lineage",
            "insights",
            "meta_learning"
        ]
        
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
            
    def _load_knowledge(self):
        """Load existing knowledge from disk."""
        # Load patterns
        patterns_file = self.base_path / "patterns_db.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    self.patterns_db = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load patterns: {e}")
                
        # Load deployment results
        deployments_file = self.base_path / "deployments/results.json"
        if deployments_file.exists():
            try:
                with open(deployments_file, 'r') as f:
                    self.deployment_results = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load deployments: {e}")
                
    def store_improvement(self, improvement: Dict[str, Any], 
                         verification_results: Optional[Dict] = None,
                         generation_context: Optional[Dict] = None) -> str:
        """Store improvement with deduplication and pattern extraction."""
        # Check for duplicates
        similar_improvements = self.find_similar_improvements(improvement)
        
        if similar_improvements and similar_improvements[0]["similarity"] > 0.95:
            # Near duplicate - just update metadata
            existing_id = similar_improvements[0]["id"]
            self._update_improvement_metadata(existing_id, improvement, generation_context)
            return existing_id
            
        # Generate unique ID
        imp_id = self._generate_improvement_id(improvement)
        
        # Extract patterns
        patterns = self._extract_patterns(improvement)
        
        # Store improvement
        imp_data = {
            "id": imp_id,
            "improvement": improvement,
            "patterns": patterns,
            "verification": verification_results,
            "context": generation_context,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "similar_to": [s["id"] for s in similar_improvements[:3]]  # Track relations
        }
        
        # Save to appropriate location
        if verification_results and verification_results.get("is_deployable"):
            self._store_successful_improvement(imp_id, imp_data)
        else:
            self._store_failed_improvement(imp_id, imp_data)
            
        # Update pattern statistics
        self._update_pattern_stats(patterns, verification_results)
        
        # Update vector store for similarity
        self._update_vector_store(imp_id, improvement)
        
        return imp_id
        
    def find_similar_improvements(self, candidate: Dict[str, Any], 
                                 threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find improvements similar to candidate."""
        # Create text representation
        candidate_text = self._create_text_representation(candidate)
        
        if not self.improvement_vectors:
            return []
            
        try:
            # Vectorize candidate
            candidate_vector = self.vectorizer.transform([candidate_text])
            
            # Calculate similarities
            similarities = []
            for imp_id, stored_vector in self.improvement_vectors.items():
                similarity = cosine_similarity(candidate_vector, stored_vector)[0][0]
                if similarity >= threshold:
                    similarities.append({
                        "id": imp_id,
                        "similarity": similarity,
                        "improvement": self._get_improvement_by_id(imp_id)
                    })
                    
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
            
    def get_successful_patterns(self, min_success_rate: float = 0.7) -> Dict[str, Any]:
        """Get patterns with high success rate."""
        successful_patterns = {}
        
        for pattern, stats in self.pattern_success_rate.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                if success_rate >= min_success_rate:
                    successful_patterns[pattern] = {
                        "success_rate": success_rate,
                        "occurrences": stats["total"],
                        "examples": self._get_pattern_examples(pattern)
                    }
                    
        return successful_patterns
        
    def get_failed_patterns(self, min_failures: int = 3) -> Dict[str, Any]:
        """Get patterns that frequently fail."""
        failed_patterns = {}
        
        for pattern, stats in self.pattern_success_rate.items():
            failures = stats["total"] - stats["success"]
            if failures >= min_failures:
                failed_patterns[pattern] = {
                    "failure_rate": failures / stats["total"] if stats["total"] > 0 else 0,
                    "failures": failures,
                    "reasons": self._analyze_failure_reasons(pattern)
                }
                
        return failed_patterns
        
    def predict_improvement_success(self, improvement: Dict[str, Any]) -> Dict[str, float]:
        """Predict likelihood of improvement success based on historical data."""
        patterns = self._extract_patterns(improvement)
        
        # Calculate success probability based on pattern history
        pattern_scores = []
        for pattern in patterns:
            if pattern in self.pattern_success_rate:
                stats = self.pattern_success_rate[pattern]
                if stats["total"] > 0:
                    pattern_scores.append(stats["success"] / stats["total"])
                    
        # Find similar improvements and their outcomes
        similar = self.find_similar_improvements(improvement, threshold=0.6)
        similarity_scores = []
        for sim in similar[:5]:  # Top 5 similar
            if sim["id"] in self.deployment_results:
                result = self.deployment_results[sim["id"]]
                if "success" in result:
                    similarity_scores.append(result["success"])
                    
        # Combine predictions
        predictions = {
            "pattern_based": np.mean(pattern_scores) if pattern_scores else 0.5,
            "similarity_based": np.mean(similarity_scores) if similarity_scores else 0.5,
            "overall": 0.0,
            "confidence": 0.0
        }
        
        # Weight predictions based on available data
        weights = []
        values = []
        
        if pattern_scores:
            weights.append(len(pattern_scores))
            values.append(predictions["pattern_based"])
            
        if similarity_scores:
            weights.append(len(similarity_scores) * 2)  # Weight similarity higher
            values.append(predictions["similarity_based"])
            
        if weights:
            predictions["overall"] = np.average(values, weights=weights)
            predictions["confidence"] = min(sum(weights) / 20.0, 1.0)  # Confidence based on data amount
        else:
            predictions["overall"] = 0.5
            predictions["confidence"] = 0.0
            
        return predictions
        
    def find_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify areas that haven't been explored."""
        # Analyze pattern combinations that haven't been tried
        all_patterns = set(self.pattern_frequency.keys())
        
        # Find rare patterns
        rare_patterns = [p for p, count in self.pattern_frequency.items() if count < 3]
        
        # Find unexplored combinations
        explored_combinations = set()
        for combo in self.combinations_db.values():
            if "patterns" in combo:
                explored_combinations.add(tuple(sorted(combo["patterns"])))
                
        # Generate potential combinations
        gaps = []
        
        # Rare pattern exploration
        for pattern in rare_patterns:
            gaps.append({
                "type": "underexplored_pattern",
                "pattern": pattern,
                "current_count": self.pattern_frequency[pattern],
                "priority": 1.0 / (1 + self.pattern_frequency[pattern])
            })
            
        # Novel combinations
        successful_patterns = list(self.get_successful_patterns().keys())
        for i, p1 in enumerate(successful_patterns):
            for p2 in successful_patterns[i+1:]:
                combo = tuple(sorted([p1, p2]))
                if combo not in explored_combinations:
                    gaps.append({
                        "type": "unexplored_combination",
                        "patterns": list(combo),
                        "priority": 0.8
                    })
                    
        # Sort by priority
        gaps.sort(key=lambda x: x["priority"], reverse=True)
        return gaps[:20]  # Top 20 gaps
        
    def get_improvement_lineage(self, improvement_id: str) -> Dict[str, Any]:
        """Get the evolutionary history of an improvement."""
        lineage = {
            "id": improvement_id,
            "ancestors": [],
            "descendants": [],
            "siblings": []
        }
        
        # Load improvement
        improvement = self._get_improvement_by_id(improvement_id)
        if not improvement:
            return lineage
            
        # Find ancestors (parents)
        if "parent" in improvement:
            lineage["ancestors"].append(improvement["parent"])
            # Recursive ancestry
            parent_lineage = self.get_improvement_lineage(improvement["parent"])
            lineage["ancestors"].extend(parent_lineage["ancestors"])
            
        if "parents" in improvement:  # Crossover
            lineage["ancestors"].extend(improvement["parents"])
            
        # Find descendants
        for imp_id, imp_data in self._iterate_all_improvements():
            if "parent" in imp_data and imp_data["parent"] == improvement_id:
                lineage["descendants"].append(imp_id)
            elif "parents" in imp_data and improvement_id in imp_data["parents"]:
                lineage["descendants"].append(imp_id)
                
        # Find siblings (same parent)
        if "parent" in improvement:
            for imp_id, imp_data in self._iterate_all_improvements():
                if (imp_id != improvement_id and 
                    "parent" in imp_data and 
                    imp_data["parent"] == improvement["parent"]):
                    lineage["siblings"].append(imp_id)
                    
        return lineage
        
    def record_deployment_result(self, improvement_id: str, 
                               metrics: Dict[str, Any],
                               success: bool):
        """Record real-world deployment results."""
        result = {
            "improvement_id": improvement_id,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "success": success
        }
        
        self.deployment_results[improvement_id] = result
        
        # Update pattern statistics based on deployment
        improvement = self._get_improvement_by_id(improvement_id)
        if improvement and "patterns" in improvement:
            for pattern in improvement["patterns"]:
                if success:
                    self.pattern_success_rate[pattern]["success"] += 1
                self.pattern_success_rate[pattern]["total"] += 1
                
        # Save deployment results
        self._save_deployment_results()
        
    def synthesize_meta_insights(self) -> Dict[str, Any]:
        """Extract high-level insights from accumulated knowledge."""
        insights = {
            "total_improvements": len(self.improvement_vectors),
            "success_patterns": len(self.get_successful_patterns()),
            "failure_patterns": len(self.get_failed_patterns()),
            "knowledge_gaps": len(self.find_knowledge_gaps()),
            "pattern_insights": [],
            "combination_insights": [],
            "evolution_trends": []
        }
        
        # Pattern insights
        successful = self.get_successful_patterns()
        for pattern, data in successful.items():
            if data["occurrences"] > 5:
                insights["pattern_insights"].append({
                    "pattern": pattern,
                    "success_rate": data["success_rate"],
                    "usage": data["occurrences"],
                    "insight": f"Pattern '{pattern}' shows {data['success_rate']:.0%} success rate"
                })
                
        # Combination insights
        for combo_id, combo_data in self.combinations_db.items():
            if "success_rate" in combo_data and combo_data["success_rate"] > 0.8:
                insights["combination_insights"].append({
                    "combination": combo_data.get("patterns", []),
                    "success_rate": combo_data["success_rate"],
                    "insight": "High-performing pattern combination"
                })
                
        # Evolution trends
        insights["evolution_trends"] = self._analyze_evolution_trends()
        
        return insights
        
    def _extract_patterns(self, improvement: Dict[str, Any]) -> List[str]:
        """Extract reusable patterns from improvement."""
        patterns = []
        
        # Type patterns
        if "type" in improvement:
            patterns.append(f"type:{improvement['type'].lower()}")
            
        # Approach patterns
        if "approach" in improvement:
            approach = improvement["approach"]
            # Handle dict or string
            if isinstance(approach, dict):
                approach_text = str(approach).lower()
            else:
                approach_text = str(approach).lower()
            # Extract key concepts
            keywords = ["cache", "async", "parallel", "event", "stream", "batch",
                       "real-time", "predictive", "adaptive", "distributed"]
            for keyword in keywords:
                if keyword in approach_text:
                    patterns.append(f"approach:{keyword}")
                    
        # Integration patterns
        if "integration_points" in improvement:
            for integration in improvement["integration_points"]:
                if isinstance(integration, str) and integration.strip():
                    # Split safely and handle empty strings
                    parts = integration.split()
                    if parts:
                        patterns.append(f"integration:{parts[0].lower()}")
                    
        # Code patterns
        if "code" in improvement:
            code_patterns = self._extract_code_patterns(improvement["code"])
            patterns.extend(code_patterns)
            
        return list(set(patterns))  # Unique patterns
        
    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract patterns from code."""
        patterns = []
        
        try:
            tree = ast.parse(code)
            
            # Class patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    patterns.append(f"class:{node.name.lower()}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    patterns.append("code:async")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        patterns.append(f"import:{alias.name.split('.')[0]}")
                        
        except:
            pass
            
        return patterns
        
    def _create_text_representation(self, improvement: Dict[str, Any]) -> str:
        """Create searchable text representation."""
        components = []
        
        for field in ["name", "type", "description", "approach"]:
            if field in improvement:
                components.append(str(improvement[field]))
                
        if "metrics" in improvement:
            components.extend([str(m) for m in improvement["metrics"]])
            
        if "code" in improvement:
            # Extract meaningful parts from code
            try:
                tree = ast.parse(improvement["code"])
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        components.append(node.name)
            except:
                pass
                
        return " ".join(components)
        
    def _generate_improvement_id(self, improvement: Dict[str, Any]) -> str:
        """Generate unique ID for improvement."""
        # Use content hash for deduplication
        content = json.dumps(improvement, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
        
    def _update_vector_store(self, imp_id: str, improvement: Dict[str, Any]):
        """Update vector store for similarity search."""
        text = self._create_text_representation(improvement)
        
        # Rebuild vectorizer if needed
        if not self.improvement_vectors:
            all_texts = [text]
            all_ids = [imp_id]
        else:
            all_texts = []
            all_ids = []
            for existing_id in self.improvement_vectors:
                existing_imp = self._get_improvement_by_id(existing_id)
                if existing_imp:
                    all_texts.append(self._create_text_representation(existing_imp))
                    all_ids.append(existing_id)
            all_texts.append(text)
            all_ids.append(imp_id)
            
        # Fit vectorizer on all texts
        vectors = self.vectorizer.fit_transform(all_texts)
        
        # Store vectors
        self.improvement_vectors = {}
        for i, vec_id in enumerate(all_ids):
            self.improvement_vectors[vec_id] = vectors[i]
            
    def _save_knowledge(self):
        """Persist knowledge to disk."""
        # Save patterns
        patterns_file = self.base_path / "patterns_db.json"
        with open(patterns_file, 'w') as f:
            json.dump(self.patterns_db, f, indent=2, cls=NumpyEncoder)
            
        # Save deployment results
        self._save_deployment_results()
        
    def _save_deployment_results(self):
        """Save deployment results."""
        deployments_file = self.base_path / "deployments/results.json"
        with open(deployments_file, 'w') as f:
            json.dump(self.deployment_results, f, indent=2, cls=NumpyEncoder)
            
    def _get_improvement_by_id(self, imp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve improvement by ID."""
        # Check successful improvements
        success_file = self.base_path / f"patterns/successful/{imp_id}.json"
        if success_file.exists():
            with open(success_file, 'r') as f:
                return json.load(f)
                
        # Check failed improvements
        failed_file = self.base_path / f"patterns/failed/{imp_id}.json"
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                return json.load(f)
                
        return None
        
    def _iterate_all_improvements(self):
        """Iterate through all stored improvements."""
        # Successful improvements
        for file_path in (self.base_path / "patterns/successful").glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    yield file_path.stem, data
            except json.JSONDecodeError as e:
                self.logger.warning(f"Skipping corrupted JSON file {file_path}: {e}")
                continue
                
        # Failed improvements
        for file_path in (self.base_path / "patterns/failed").glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    yield file_path.stem, data
            except json.JSONDecodeError as e:
                self.logger.warning(f"Skipping corrupted JSON file {file_path}: {e}")
                continue
                
    def _store_successful_improvement(self, imp_id: str, imp_data: Dict):
        """Store successful improvement."""
        file_path = self.base_path / f"patterns/successful/{imp_id}.json"
        with open(file_path, 'w') as f:
            json.dump(imp_data, f, indent=2, cls=NumpyEncoder)
            
    def _store_failed_improvement(self, imp_id: str, imp_data: Dict):
        """Store failed improvement."""
        file_path = self.base_path / f"patterns/failed/{imp_id}.json"
        with open(file_path, 'w') as f:
            json.dump(imp_data, f, indent=2, cls=NumpyEncoder)
            
    def _update_pattern_stats(self, patterns: List[str], verification_results: Optional[Dict]):
        """Update pattern statistics."""
        for pattern in patterns:
            self.pattern_frequency[pattern] += 1
            
            if verification_results and verification_results.get("is_deployable"):
                self.pattern_success_rate[pattern]["success"] += 1
            self.pattern_success_rate[pattern]["total"] += 1
            
    def _get_pattern_examples(self, pattern: str) -> List[str]:
        """Get example improvements using this pattern."""
        examples = []
        for imp_id, imp_data in self._iterate_all_improvements():
            if "patterns" in imp_data and pattern in imp_data["patterns"]:
                examples.append(imp_id)
                if len(examples) >= 3:
                    break
        return examples
        
    def _analyze_failure_reasons(self, pattern: str) -> List[str]:
        """Analyze why a pattern fails."""
        reasons = []
        for imp_id, imp_data in self._iterate_all_improvements():
            if ("patterns" in imp_data and pattern in imp_data["patterns"] and
                "verification" in imp_data and not imp_data["verification"].get("is_deployable")):
                if "issues" in imp_data["verification"]:
                    reasons.extend(imp_data["verification"]["issues"])
        return list(set(reasons))[:5]  # Top 5 unique reasons
        
    def _analyze_evolution_trends(self) -> List[Dict[str, Any]]:
        """Analyze trends in evolution over time."""
        trends = []
        
        # Pattern popularity over time
        time_series = defaultdict(lambda: defaultdict(int))
        for imp_id, imp_data in self._iterate_all_improvements():
            if "stored_at" in imp_data:
                date = imp_data["stored_at"][:10]  # YYYY-MM-DD
                if "patterns" in imp_data:
                    for pattern in imp_data["patterns"]:
                        time_series[date][pattern] += 1
                        
        # Find trending patterns
        dates = sorted(time_series.keys())
        if len(dates) > 1:
            recent_date = dates[-1]
            previous_date = dates[-2]
            
            for pattern in time_series[recent_date]:
                recent_count = time_series[recent_date][pattern]
                previous_count = time_series[previous_date].get(pattern, 0)
                
                if recent_count > previous_count:
                    trends.append({
                        "pattern": pattern,
                        "trend": "increasing",
                        "change": recent_count - previous_count
                    })
                    
        return trends
        
    def _update_improvement_metadata(self, existing_id: str, 
                                   new_improvement: Dict,
                                   context: Optional[Dict]):
        """Update metadata for existing improvement."""
        existing = self._get_improvement_by_id(existing_id)
        if existing:
            # Update generation count
            if "generation_count" not in existing:
                existing["generation_count"] = 1
            existing["generation_count"] += 1
            
            # Add regeneration context
            if "regeneration_contexts" not in existing:
                existing["regeneration_contexts"] = []
            existing["regeneration_contexts"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context
            })
            
            # Save updated improvement
            if existing.get("verification", {}).get("is_deployable"):
                self._store_successful_improvement(existing_id, existing)
            else:
                self._store_failed_improvement(existing_id, existing)