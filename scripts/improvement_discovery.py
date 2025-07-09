"""
Improvement Type Discovery System

Dynamically discovers and creates new improvement types based on
system analysis and pattern recognition.
"""

import ast
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import re
import json
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class ImprovementDiscoveryEngine:
    """Discovers new improvement types by analyzing code patterns and system needs."""
    
    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        
        # Pattern tracking
        self.code_patterns = defaultdict(int)
        self.improvement_patterns = defaultdict(list)
        self.success_patterns = defaultdict(float)
        
        # Discovery state
        self.discovered_types = set()
        self.type_embeddings = {}
        self.clustering_model = None
        
    async def discover_improvement_types(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover new improvement types based on system analysis.
        
        Returns:
            List of newly discovered improvement types with metadata
        """
        discoveries = []
        
        # 1. Analyze existing codebase patterns
        code_discoveries = await self._analyze_code_patterns()
        discoveries.extend(code_discoveries)
        
        # 2. Analyze successful improvements
        success_discoveries = await self._analyze_success_patterns()
        discoveries.extend(success_discoveries)
        
        # 3. Analyze system gaps
        gap_discoveries = await self._analyze_system_gaps(system_state)
        discoveries.extend(gap_discoveries)
        
        # 4. Generate novel combinations
        novel_discoveries = await self._generate_novel_types()
        discoveries.extend(novel_discoveries)
        
        # 5. Cluster and categorize
        categorized = self._categorize_discoveries(discoveries)
        
        # Update configuration with new types
        for discovery in categorized:
            if discovery["type"] not in self.discovered_types:
                self.discovered_types.add(discovery["type"])
                self.config.add_discovered_improvement_type(
                    discovery["type"], 
                    discovery.get("metadata", {})
                )
                
        return categorized
        
    async def _analyze_code_patterns(self) -> List[Dict[str, Any]]:
        """Analyze codebase to discover improvement patterns."""
        discoveries = []
        
        # Scan scripts directory
        scripts_dir = Path(__file__).parent
        pattern_counter = Counter()
        
        for py_file in scripts_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Look for patterns that suggest improvement opportunities
                patterns = self._extract_code_patterns(content)
                pattern_counter.update(patterns)
                
            except Exception as e:
                self.logger.debug(f"Error analyzing {py_file}: {e}")
                
        # Identify frequent patterns that could become improvement types
        for pattern, count in pattern_counter.most_common(10):
            if count > 3:  # Pattern appears in multiple files
                improvement_type = self._pattern_to_improvement_type(pattern)
                if improvement_type and improvement_type not in self.discovered_types:
                    discoveries.append({
                        "type": improvement_type,
                        "source": "code_analysis",
                        "confidence": min(0.9, count / 10),
                        "metadata": {
                            "pattern": pattern,
                            "frequency": count,
                            "discovered_at": datetime.now().isoformat()
                        }
                    })
                    
        return discoveries
        
    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract meaningful patterns from code."""
        patterns = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Class patterns
                if isinstance(node, ast.ClassDef):
                    # Look for specific base classes or patterns
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            if "Handler" in base.id:
                                patterns.append("handler_pattern")
                            elif "Manager" in base.id:
                                patterns.append("manager_pattern")
                            elif "Engine" in base.id:
                                patterns.append("engine_pattern")
                                
                    # Check for specific method patterns
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    if "process" in methods and "validate" in methods:
                        patterns.append("processor_pattern")
                    if "train" in methods or "predict" in methods:
                        patterns.append("ml_pattern")
                        
                # Function patterns
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith("analyze_"):
                        patterns.append("analyzer_pattern")
                    elif node.name.startswith("optimize_"):
                        patterns.append("optimizer_pattern")
                    elif node.name.startswith("generate_"):
                        patterns.append("generator_pattern")
                    elif "cache" in node.name:
                        patterns.append("caching_pattern")
                        
                # Import patterns
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if "sklearn" in alias.name:
                            patterns.append("ml_dependency")
                        elif "torch" in alias.name or "tensorflow" in alias.name:
                            patterns.append("deep_learning")
                        elif "redis" in alias.name:
                            patterns.append("distributed_pattern")
                            
        except:
            pass
            
        # Regex patterns for non-AST analysis
        if re.search(r'async\s+def.*websocket', code):
            patterns.append("realtime_pattern")
        if re.search(r'@retry|@backoff', code):
            patterns.append("resilience_pattern")
        if re.search(r'prometheus|grafana|metric', code, re.I):
            patterns.append("observability_pattern")
            
        return patterns
        
    def _pattern_to_improvement_type(self, pattern: str) -> Optional[str]:
        """Convert a code pattern to an improvement type."""
        pattern_mapping = {
            "handler_pattern": "event_handling",
            "manager_pattern": "resource_management",
            "engine_pattern": "processing_engine",
            "processor_pattern": "data_processor",
            "ml_pattern": "machine_learning",
            "analyzer_pattern": "deep_analysis",
            "optimizer_pattern": "performance_optimizer",
            "generator_pattern": "content_generator",
            "caching_pattern": "cache_optimizer",
            "ml_dependency": "ml_integration",
            "deep_learning": "neural_enhancement",
            "distributed_pattern": "distributed_system",
            "realtime_pattern": "realtime_processing",
            "resilience_pattern": "fault_tolerance",
            "observability_pattern": "system_observability"
        }
        
        return pattern_mapping.get(pattern)
        
    async def _analyze_success_patterns(self) -> List[Dict[str, Any]]:
        """Analyze successful improvements to find patterns."""
        discoveries = []
        
        # Load improvement history if available
        history_file = Path("improvement_history.json")
        if not history_file.exists():
            return discoveries
            
        try:
            with open(history_file) as f:
                history = json.load(f)
                
            # Analyze successful improvements
            successful = [imp for imp in history if imp.get("success", False)]
            
            # Extract features from successful improvements
            features = []
            for imp in successful:
                feature_vec = self._extract_improvement_features(imp)
                features.append(feature_vec)
                
            if len(features) > 5:
                # Cluster successful improvements
                clusters = self._cluster_improvements(features)
                
                # Each cluster represents a potential new type
                for cluster_id, cluster_features in clusters.items():
                    new_type = self._features_to_type(cluster_features)
                    if new_type and new_type not in self.discovered_types:
                        discoveries.append({
                            "type": new_type,
                            "source": "success_analysis",
                            "confidence": 0.8,
                            "metadata": {
                                "cluster_size": len(cluster_features),
                                "discovered_at": datetime.now().isoformat()
                            }
                        })
                        
        except Exception as e:
            self.logger.error(f"Error analyzing success patterns: {e}")
            
        return discoveries
        
    def _extract_improvement_features(self, improvement: Dict) -> Dict[str, Any]:
        """Extract features from an improvement for clustering."""
        features = {
            "type": improvement.get("type", "unknown"),
            "has_tests": "test" in improvement.get("code", "").lower(),
            "uses_async": "async" in improvement.get("code", ""),
            "complexity": len(improvement.get("code", "").split("\n")),
            "has_error_handling": "try:" in improvement.get("code", ""),
            "integration_count": len(improvement.get("integration_points", [])),
            "metric_count": len(improvement.get("metrics", []))
        }
        
        # Add text features
        description = improvement.get("description", "")
        features.update({
            "focuses_performance": "performance" in description.lower(),
            "focuses_quality": "quality" in description.lower(),
            "focuses_security": "security" in description.lower(),
            "focuses_ux": any(term in description.lower() for term in ["user", "experience", "interface"])
        })
        
        return features
        
    def _cluster_improvements(self, features: List[Dict]) -> Dict[int, List[Dict]]:
        """Cluster improvements to find patterns."""
        # Convert features to numerical vectors
        feature_names = set()
        for f in features:
            feature_names.update(f.keys())
            
        vectors = []
        for f in features:
            vec = [float(f.get(name, 0)) if isinstance(f.get(name, 0), bool) else f.get(name, 0) 
                   for name in sorted(feature_names)]
            vectors.append(vec)
            
        # Perform clustering
        n_clusters = min(5, max(2, len(vectors) // 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)
        
        # Group by cluster
        clusters = defaultdict(list)
        for feat, label in zip(features, labels):
            clusters[label].append(feat)
            
        return dict(clusters)
        
    def _features_to_type(self, cluster_features: List[Dict]) -> Optional[str]:
        """Convert cluster features to improvement type name."""
        # Analyze common characteristics
        avg_features = defaultdict(float)
        for features in cluster_features:
            for key, value in features.items():
                if isinstance(value, (int, float, bool)):
                    avg_features[key] += float(value)
                    
        # Normalize
        n = len(cluster_features)
        for key in avg_features:
            avg_features[key] /= n
            
        # Generate type name based on dominant features
        if avg_features.get("focuses_performance", 0) > 0.7:
            return "performance_enhancer"
        elif avg_features.get("focuses_security", 0) > 0.7:
            return "security_hardener"
        elif avg_features.get("has_tests", 0) > 0.8:
            return "test_automation"
        elif avg_features.get("uses_async", 0) > 0.7:
            return "async_optimizer"
        elif avg_features.get("integration_count", 0) > 3:
            return "integration_bridge"
            
        return None
        
    async def _analyze_system_gaps(self, system_state: Dict) -> List[Dict[str, Any]]:
        """Analyze system state to identify gaps that need new improvement types."""
        discoveries = []
        
        # Check for missing capabilities
        capabilities = system_state.get("capabilities", {})
        common_capabilities = [
            "data_visualization", "api_gateway", "message_queue",
            "search_engine", "recommendation_system", "auth_system",
            "payment_processing", "notification_service", "analytics_engine"
        ]
        
        for cap in common_capabilities:
            if cap not in capabilities and cap not in self.discovered_types:
                discoveries.append({
                    "type": cap,
                    "source": "gap_analysis",
                    "confidence": 0.7,
                    "metadata": {
                        "reason": "missing_capability",
                        "discovered_at": datetime.now().isoformat()
                    }
                })
                
        # Check for performance gaps
        metrics = system_state.get("performance_metrics", {})
        if metrics.get("error_rate", 0) > 0.1:
            if "error_recovery" not in self.discovered_types:
                discoveries.append({
                    "type": "error_recovery",
                    "source": "gap_analysis",
                    "confidence": 0.9,
                    "metadata": {
                        "reason": "high_error_rate",
                        "error_rate": metrics["error_rate"]
                    }
                })
                
        return discoveries
        
    async def _generate_novel_types(self) -> List[Dict[str, Any]]:
        """Generate novel improvement types through creative combination."""
        discoveries = []
        
        # Get existing types
        existing_types = self.config.get_improvement_types()
        
        # Creative combinations
        prefixes = ["smart", "adaptive", "intelligent", "automated", "dynamic", "hybrid"]
        suffixes = ["enhancer", "optimizer", "analyzer", "generator", "validator", "synthesizer"]
        
        for prefix in prefixes:
            for base_type in existing_types[:5]:  # Limit combinations
                novel_type = f"{prefix}_{base_type}"
                if novel_type not in self.discovered_types:
                    discoveries.append({
                        "type": novel_type,
                        "source": "creative_generation",
                        "confidence": 0.6,
                        "metadata": {
                            "base_type": base_type,
                            "modifier": prefix,
                            "discovered_at": datetime.now().isoformat()
                        }
                    })
                    
        # Domain crossovers
        domains = ["quantum", "blockchain", "edge", "serverless", "microservice"]
        for domain in domains:
            for suffix in suffixes:
                novel_type = f"{domain}_{suffix}"
                if novel_type not in self.discovered_types:
                    discoveries.append({
                        "type": novel_type,
                        "source": "domain_crossover",
                        "confidence": 0.5,
                        "metadata": {
                            "domain": domain,
                            "function": suffix,
                            "discovered_at": datetime.now().isoformat()
                        }
                    })
                    
        return discoveries[:10]  # Limit to avoid explosion
        
    def _categorize_discoveries(self, discoveries: List[Dict]) -> List[Dict]:
        """Categorize and rank discoveries."""
        # Remove duplicates
        unique_discoveries = {}
        for disc in discoveries:
            type_name = disc["type"]
            if type_name not in unique_discoveries or disc["confidence"] > unique_discoveries[type_name]["confidence"]:
                unique_discoveries[type_name] = disc
                
        # Sort by confidence
        sorted_discoveries = sorted(
            unique_discoveries.values(),
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        return sorted_discoveries[:15]  # Return top 15