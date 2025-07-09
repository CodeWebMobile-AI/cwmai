"""
Research Learning System - Tracks research outcomes and optimizes future research.

This module learns from research effectiveness to improve future topic selection,
query generation, and implementation strategies. It builds a knowledge base of
what research approaches work best for different types of problems.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter


class ResearchLearningSystem:
    """Learn from research outcomes to improve future selection and strategies."""
    
    def __init__(self, knowledge_store=None):
        self.knowledge_store = knowledge_store
        self.learning_data = []
        self.pattern_database = {}
        self.effectiveness_models = {}
        self.prediction_accuracy = {"correct": 0, "total": 0}
        
        # Learning configuration
        self.learning_config = {
            "min_samples_for_pattern": 3,
            "pattern_confidence_threshold": 0.7,
            "effectiveness_decay_factor": 0.95,  # Recent outcomes weighted more
            "max_learning_history": 1000
        }
        
        # Outcome categories for learning
        self.outcome_categories = {
            "research_quality": {
                "excellent": 0.9,
                "good": 0.7,
                "average": 0.5,
                "poor": 0.3,
                "failed": 0.1
            },
            "implementation_success": {
                "fully_implemented": 1.0,
                "partially_implemented": 0.6,
                "attempted": 0.3,
                "not_attempted": 0.1,
                "failed": 0.0
            },
            "performance_impact": {
                "high_improvement": 1.0,
                "moderate_improvement": 0.7,
                "small_improvement": 0.4,
                "no_change": 0.2,
                "regression": 0.0
            }
        }
        
    def record_research_outcome(self, research_id: str, outcome: Dict):
        """
        Record outcome of research and learn from it.
        
        Args:
            research_id: ID of the research
            outcome: Outcome data including metrics and results
        """
        # Create learning record
        learning_record = {
            "research_id": research_id,
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "effectiveness_score": self._calculate_effectiveness_score(outcome),
            "research_metadata": outcome.get("research_metadata", {}),
            "implementation_results": outcome.get("implementation_results", {}),
            "performance_changes": outcome.get("performance_changes", {}),
            "value_delivered": outcome.get("value_delivered", 0)
        }
        
        # Add to learning data
        self.learning_data.append(learning_record)
        
        # Update pattern database
        self._update_patterns(learning_record)
        
        # Update effectiveness models
        self._update_effectiveness_models(learning_record)
        
        # Update knowledge store if available
        if self.knowledge_store:
            self.knowledge_store.record_value_delivered(
                research_id, 
                learning_record["value_delivered"]
            )
            self.knowledge_store.update_quality_score(
                research_id,
                learning_record["effectiveness_score"]
            )
        
        # Maintain data size
        if len(self.learning_data) > self.learning_config["max_learning_history"]:
            self.learning_data = self.learning_data[-self.learning_config["max_learning_history"]:]
    
    def predict_research_effectiveness(self, research_proposal: Dict) -> Dict:
        """
        Predict how effective a research proposal will be.
        
        Args:
            research_proposal: Proposed research with metadata
            
        Returns:
            Prediction with confidence score
        """
        # Extract features from proposal
        features = self._extract_research_features(research_proposal)
        
        # Get predictions from different models
        predictions = {}
        
        # Pattern-based prediction
        pattern_prediction = self._predict_from_patterns(features)
        predictions["pattern_based"] = pattern_prediction
        
        # Historical effectiveness prediction
        historical_prediction = self._predict_from_history(features)
        predictions["historical"] = historical_prediction
        
        # Area-specific prediction
        area_prediction = self._predict_from_area_performance(features)
        predictions["area_specific"] = area_prediction
        
        # Combine predictions
        combined_prediction = self._combine_predictions(predictions)
        
        # Track prediction for later accuracy measurement
        self._track_prediction(research_proposal.get("id"), combined_prediction)
        
        return combined_prediction
    
    def get_research_recommendations(self, current_state: Dict) -> List[Dict]:
        """
        Get research recommendations based on learning.
        
        Args:
            current_state: Current system state and performance
            
        Returns:
            List of recommended research topics
        """
        recommendations = []
        
        # Analyze what has worked well in similar situations
        similar_situations = self._find_similar_situations(current_state)
        
        for situation in similar_situations:
            successful_research = self._get_successful_research_for_situation(situation)
            
            for research in successful_research:
                recommendation = {
                    "topic": research["topic"],
                    "area": research["area"],
                    "predicted_effectiveness": research["effectiveness_score"],
                    "similar_situation_count": situation["count"],
                    "confidence": situation["similarity"] * research["effectiveness_score"],
                    "reason": f"Similar to {situation['count']} past successful situations"
                }
                recommendations.append(recommendation)
        
        # Add high-impact areas that need more research
        gap_recommendations = self._recommend_gap_research(current_state)
        recommendations.extend(gap_recommendations)
        
        # Sort by predicted effectiveness and confidence
        recommendations.sort(
            key=lambda r: (r["predicted_effectiveness"], r["confidence"]), 
            reverse=True
        )
        
        return recommendations[:10]  # Top 10 recommendations
    
    def identify_research_patterns(self) -> Dict:
        """Identify patterns in successful vs failed research."""
        if len(self.learning_data) < self.learning_config["min_samples_for_pattern"]:
            return {"patterns": [], "confidence": 0, "sample_size": len(self.learning_data)}
        
        patterns = {
            "successful_patterns": [],
            "failure_patterns": [],
            "effectiveness_drivers": [],
            "timing_patterns": [],
            "area_insights": {}
        }
        
        # Separate successful and failed research
        successful = [r for r in self.learning_data if r["effectiveness_score"] > 0.7]
        failed = [r for r in self.learning_data if r["effectiveness_score"] < 0.3]
        
        # Identify success patterns
        patterns["successful_patterns"] = self._identify_success_patterns(successful)
        
        # Identify failure patterns
        patterns["failure_patterns"] = self._identify_failure_patterns(failed)
        
        # Identify what drives effectiveness
        patterns["effectiveness_drivers"] = self._identify_effectiveness_drivers()
        
        # Identify timing patterns
        patterns["timing_patterns"] = self._identify_timing_patterns()
        
        # Area-specific insights
        patterns["area_insights"] = self._generate_area_insights()
        
        return patterns
    
    def optimize_research_strategy(self, current_performance: Dict) -> Dict:
        """
        Optimize research strategy based on learning.
        
        Args:
            current_performance: Current system performance metrics
            
        Returns:
            Optimized research strategy
        """
        strategy = {
            "focus_areas": [],
            "timing_adjustments": {},
            "query_optimizations": {},
            "implementation_priorities": [],
            "resource_allocation": {}
        }
        
        # Determine focus areas based on performance gaps and research effectiveness
        strategy["focus_areas"] = self._determine_focus_areas(current_performance)
        
        # Optimize timing based on successful patterns
        strategy["timing_adjustments"] = self._optimize_timing_strategy()
        
        # Optimize query generation based on what works
        strategy["query_optimizations"] = self._optimize_query_strategy()
        
        # Prioritize implementations based on success patterns
        strategy["implementation_priorities"] = self._optimize_implementation_strategy()
        
        # Optimize resource allocation
        strategy["resource_allocation"] = self._optimize_resource_allocation()
        
        return strategy
    
    def _calculate_effectiveness_score(self, outcome: Dict) -> float:
        """Calculate overall effectiveness score from outcome data."""
        scores = []
        
        # Research quality score
        research_quality = outcome.get("research_quality", "average")
        if research_quality in self.outcome_categories["research_quality"]:
            scores.append(self.outcome_categories["research_quality"][research_quality])
        
        # Implementation success score
        impl_success = outcome.get("implementation_success", "not_attempted")
        if impl_success in self.outcome_categories["implementation_success"]:
            scores.append(self.outcome_categories["implementation_success"][impl_success])
        
        # Performance impact score
        perf_impact = outcome.get("performance_impact", "no_change")
        if perf_impact in self.outcome_categories["performance_impact"]:
            scores.append(self.outcome_categories["performance_impact"][perf_impact])
        
        # Value delivered (normalized to 0-1)
        value_delivered = outcome.get("value_delivered", 0)
        if value_delivered > 0:
            normalized_value = min(1.0, value_delivered / 10.0)  # Assuming max value of 10
            scores.append(normalized_value)
        
        # Calculate weighted average
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.5  # Default neutral score
    
    def _update_patterns(self, learning_record: Dict):
        """Update pattern database with new learning record."""
        research_meta = learning_record["research_metadata"]
        effectiveness = learning_record["effectiveness_score"]
        
        # Extract pattern features
        area = research_meta.get("area", "unknown")
        topic = research_meta.get("topic", "")
        severity = research_meta.get("severity", "medium")
        
        # Create pattern key
        pattern_key = f"{area}_{severity}"
        
        if pattern_key not in self.pattern_database:
            self.pattern_database[pattern_key] = {
                "samples": [],
                "average_effectiveness": 0,
                "sample_count": 0,
                "success_rate": 0
            }
        
        # Add sample
        self.pattern_database[pattern_key]["samples"].append({
            "effectiveness": effectiveness,
            "topic": topic,
            "timestamp": learning_record["timestamp"],
            "value_delivered": learning_record["value_delivered"]
        })
        
        # Update statistics
        samples = self.pattern_database[pattern_key]["samples"]
        self.pattern_database[pattern_key]["sample_count"] = len(samples)
        self.pattern_database[pattern_key]["average_effectiveness"] = \
            sum(s["effectiveness"] for s in samples) / len(samples)
        self.pattern_database[pattern_key]["success_rate"] = \
            len([s for s in samples if s["effectiveness"] > 0.7]) / len(samples)
    
    def _update_effectiveness_models(self, learning_record: Dict):
        """Update effectiveness prediction models."""
        area = learning_record["research_metadata"].get("area", "unknown")
        
        if area not in self.effectiveness_models:
            self.effectiveness_models[area] = {
                "samples": [],
                "model_weights": {},
                "last_updated": datetime.now().isoformat()
            }
        
        # Add sample to area model
        self.effectiveness_models[area]["samples"].append({
            "features": self._extract_research_features(learning_record["research_metadata"]),
            "effectiveness": learning_record["effectiveness_score"],
            "timestamp": learning_record["timestamp"]
        })
        
        # Update model weights if we have enough samples
        if len(self.effectiveness_models[area]["samples"]) >= 5:
            self._update_model_weights(area)
    
    def _extract_research_features(self, research_data: Dict) -> Dict:
        """Extract features from research data for prediction."""
        return {
            "area": research_data.get("area", "unknown"),
            "topic_length": len(research_data.get("topic", "")),
            "severity": research_data.get("severity", "medium"),
            "priority": research_data.get("priority", "medium"),
            "has_context": bool(research_data.get("context")),
            "estimated_time": research_data.get("estimated_research_time", 0),
            "provider": research_data.get("provider", "unknown")
        }
    
    def _predict_from_patterns(self, features: Dict) -> Dict:
        """Predict effectiveness based on historical patterns."""
        area = features.get("area", "unknown")
        severity = features.get("severity", "medium")
        pattern_key = f"{area}_{severity}"
        
        if pattern_key in self.pattern_database:
            pattern_data = self.pattern_database[pattern_key]
            
            if pattern_data["sample_count"] >= self.learning_config["min_samples_for_pattern"]:
                return {
                    "effectiveness": pattern_data["average_effectiveness"],
                    "confidence": min(0.9, pattern_data["sample_count"] / 10),
                    "basis": f"Pattern from {pattern_data['sample_count']} samples"
                }
        
        # Default prediction
        return {
            "effectiveness": 0.5,
            "confidence": 0.3,
            "basis": "No sufficient pattern data"
        }
    
    def _predict_from_history(self, features: Dict) -> Dict:
        """Predict based on historical research in the same area."""
        area = features.get("area", "unknown")
        
        # Get recent research in same area
        area_research = [
            r for r in self.learning_data[-50:]  # Last 50 records
            if r["research_metadata"].get("area") == area
        ]
        
        if len(area_research) >= 3:
            # Calculate weighted average (more recent = higher weight)
            total_weight = 0
            weighted_effectiveness = 0
            
            for i, research in enumerate(reversed(area_research)):
                weight = self.learning_config["effectiveness_decay_factor"] ** i
                weighted_effectiveness += research["effectiveness_score"] * weight
                total_weight += weight
            
            avg_effectiveness = weighted_effectiveness / total_weight if total_weight > 0 else 0.5
            
            return {
                "effectiveness": avg_effectiveness,
                "confidence": min(0.8, len(area_research) / 10),
                "basis": f"Historical average from {len(area_research)} recent samples"
            }
        
        return {
            "effectiveness": 0.5,
            "confidence": 0.2,
            "basis": "Insufficient historical data"
        }
    
    def _predict_from_area_performance(self, features: Dict) -> Dict:
        """Predict based on current area performance needs."""
        area = features.get("area", "unknown")
        
        # High-need areas should have higher predicted effectiveness
        area_priorities = {
            "claude_interaction": 0.9,  # Critical need
            "task_generation": 0.8,     # High need
            "outcome_learning": 0.7,    # Medium-high need
            "multi_agent_coordination": 0.6,  # Medium need
            "portfolio_management": 0.5     # Lower need
        }
        
        base_effectiveness = area_priorities.get(area, 0.5)
        
        return {
            "effectiveness": base_effectiveness,
            "confidence": 0.6,
            "basis": f"Area priority for {area}"
        }
    
    def _combine_predictions(self, predictions: Dict) -> Dict:
        """Combine multiple predictions into final prediction."""
        total_weight = 0
        weighted_effectiveness = 0
        combined_confidence = 0
        
        # Weight predictions by their confidence
        for pred_type, prediction in predictions.items():
            weight = prediction["confidence"]
            weighted_effectiveness += prediction["effectiveness"] * weight
            total_weight += weight
            combined_confidence += prediction["confidence"]
        
        if total_weight > 0:
            final_effectiveness = weighted_effectiveness / total_weight
            final_confidence = combined_confidence / len(predictions)
        else:
            final_effectiveness = 0.5
            final_confidence = 0.3
        
        return {
            "predicted_effectiveness": final_effectiveness,
            "confidence": final_confidence,
            "component_predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    def _track_prediction(self, research_id: str, prediction: Dict):
        """Track prediction for later accuracy assessment."""
        # This would be used to compare against actual outcomes
        # Implementation depends on how predictions are stored and tracked
        pass
    
    def _find_similar_situations(self, current_state: Dict) -> List[Dict]:
        """Find situations similar to current state."""
        similar_situations = []
        
        # Simple similarity based on key metrics
        current_claude_success = current_state.get("metrics", {}).get("claude_success_rate", 0)
        current_task_completion = current_state.get("metrics", {}).get("task_completion_rate", 0)
        
        # Group historical data by similar performance ranges
        performance_groups = defaultdict(list)
        
        for record in self.learning_data:
            # Extract performance context from record if available
            perf_changes = record.get("performance_changes", {})
            
            # Simple grouping by performance level
            if current_claude_success < 0.3 and current_task_completion < 0.3:
                group_key = "critical_performance"
            elif current_claude_success < 0.6 and current_task_completion < 0.6:
                group_key = "poor_performance"
            else:
                group_key = "normal_performance"
            
            performance_groups[group_key].append(record)
        
        # Find the matching group
        if current_claude_success < 0.3 and current_task_completion < 0.3:
            group_key = "critical_performance"
        elif current_claude_success < 0.6 and current_task_completion < 0.6:
            group_key = "poor_performance"
        else:
            group_key = "normal_performance"
        
        if group_key in performance_groups:
            similar_situations.append({
                "group": group_key,
                "count": len(performance_groups[group_key]),
                "similarity": 0.8,  # High similarity within group
                "records": performance_groups[group_key]
            })
        
        return similar_situations
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning system status."""
        if not self.learning_data:
            return {
                "status": "No learning data available",
                "total_records": 0,
                "patterns_identified": 0,
                "average_effectiveness": 0
            }
        
        # Calculate summary statistics
        effectiveness_scores = [r["effectiveness_score"] for r in self.learning_data]
        
        return {
            "total_records": len(self.learning_data),
            "patterns_identified": len(self.pattern_database),
            "average_effectiveness": statistics.mean(effectiveness_scores),
            "effectiveness_trend": self._calculate_trend(effectiveness_scores),
            "most_effective_areas": self._get_most_effective_areas(),
            "improvement_opportunities": self._identify_improvement_opportunities(),
            "prediction_accuracy": self._calculate_prediction_accuracy(),
            "last_update": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend in effectiveness scores."""
        if len(scores) < 10:
            return "insufficient_data"
        
        recent_scores = scores[-10:]
        older_scores = scores[-20:-10] if len(scores) >= 20 else scores[:-10]
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)
        
        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_most_effective_areas(self) -> List[Dict]:
        """Get areas with highest research effectiveness."""
        area_effectiveness = defaultdict(list)
        
        for record in self.learning_data:
            area = record["research_metadata"].get("area", "unknown")
            area_effectiveness[area].append(record["effectiveness_score"])
        
        area_averages = []
        for area, scores in area_effectiveness.items():
            if len(scores) >= 3:  # Require at least 3 samples
                area_averages.append({
                    "area": area,
                    "average_effectiveness": statistics.mean(scores),
                    "sample_count": len(scores)
                })
        
        area_averages.sort(key=lambda x: x["average_effectiveness"], reverse=True)
        return area_averages[:5]  # Top 5 areas
    
    def _get_successful_research_for_situation(self, situation: Dict) -> List[Dict]:
        """Get successful research for a similar situation."""
        successful_research = []
        
        records = situation.get("records", [])
        for record in records:
            if record.get("effectiveness_score", 0) > 0.7:
                research_info = {
                    "topic": record.get("research_metadata", {}).get("topic", "unknown"),
                    "area": record.get("research_metadata", {}).get("area", "unknown"),
                    "effectiveness_score": record.get("effectiveness_score", 0)
                }
                successful_research.append(research_info)
        
        return successful_research
    
    def _recommend_gap_research(self, current_state: Dict) -> List[Dict]:
        """Recommend research for current gaps."""
        recommendations = []
        
        # Simple gap-based recommendations
        claude_success = current_state.get("metrics", {}).get("claude_success_rate", 1.0)
        if claude_success < 0.3:
            recommendations.append({
                "topic": "Claude interaction optimization",
                "area": "claude_interaction", 
                "predicted_effectiveness": 0.8,
                "similar_situation_count": 0,
                "confidence": 0.9,
                "reason": "Critical Claude success rate issue"
            })
        
        return recommendations
    
    def _identify_improvement_opportunities(self) -> List[str]:
        """Identify opportunities for improvement."""
        opportunities = []
        
        if len(self.learning_data) < 10:
            opportunities.append("Collect more research data for better learning")
        
        if len(self.pattern_database) < 3:
            opportunities.append("Need more patterns for reliable predictions")
        
        # Check if any areas have consistently low effectiveness
        area_effectiveness = defaultdict(list)
        for record in self.learning_data:
            area = record["research_metadata"].get("area", "unknown")
            area_effectiveness[area].append(record["effectiveness_score"])
        
        for area, scores in area_effectiveness.items():
            if len(scores) >= 3 and statistics.mean(scores) < 0.5:
                opportunities.append(f"Improve research approach for {area}")
        
        return opportunities
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.prediction_accuracy["total"] == 0:
            return 0.0
        
        return self.prediction_accuracy["correct"] / self.prediction_accuracy["total"]
    
    def _determine_focus_areas(self, current_performance: Dict) -> List[str]:
        """Determine focus areas based on performance."""
        focus_areas = []
        
        # Focus on areas with poor performance
        claude_success = current_performance.get("claude_success_rate", 1.0)
        task_completion = current_performance.get("task_completion_rate", 1.0)
        
        if claude_success < 0.3:
            focus_areas.append("claude_interaction")
        
        if task_completion < 0.3:
            focus_areas.append("task_generation")
        
        # Add learning if we have few patterns
        if len(self.pattern_database) < 5:
            focus_areas.append("outcome_learning")
        
        return focus_areas
    
    def _optimize_timing_strategy(self) -> Dict:
        """Optimize timing strategy based on patterns."""
        return {
            "peak_effectiveness_hours": [9, 10, 14, 15],  # Business hours
            "avoid_hours": [0, 1, 2, 3, 4, 5],  # Night hours
            "recommended_interval": 30 * 60  # 30 minutes
        }
    
    def _optimize_query_strategy(self) -> Dict:
        """Optimize query strategy based on learning."""
        return {
            "preferred_query_length": "medium",
            "include_context": True,
            "use_specific_examples": True,
            "focus_on_actionable_insights": True
        }
    
    def _optimize_implementation_strategy(self) -> List[str]:
        """Optimize implementation strategy."""
        return [
            "Prioritize high-confidence insights",
            "Test in sandbox before deployment",
            "Monitor impact after implementation",
            "Rollback if negative impact detected"
        ]
    
    def _optimize_resource_allocation(self) -> Dict:
        """Optimize resource allocation strategy."""
        return {
            "claude_interaction": 0.4,  # 40% of resources
            "task_generation": 0.3,     # 30% of resources
            "outcome_learning": 0.2,    # 20% of resources
            "multi_agent_coordination": 0.1  # 10% of resources
        }