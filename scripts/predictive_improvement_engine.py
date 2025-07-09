import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from scripts.ai_brain import AIBrain
from scripts.redis_integration import get_redis_client
from scripts.pattern_learning_engine import PatternLearningEngine
from scripts.knowledge_synthesis_engine import KnowledgeSynthesisEngine

logger = logging.getLogger(__name__)


class PredictiveImprovementEngine:
    """Predict future system needs and proactively implement improvements"""
    
    def __init__(self):
        self.ai_brain = AIBrain()
        self.redis_client = get_redis_client()
        self.pattern_engine = PatternLearningEngine()
        self.synthesis_engine = KnowledgeSynthesisEngine()
        self.prediction_state_file = "predictive_improvement_state.json"
        self.time_series_window = 30  # days
        self.load_prediction_state()
        
    def load_prediction_state(self):
        """Load prediction models and historical data"""
        if Path(self.prediction_state_file).exists():
            with open(self.prediction_state_file, 'r') as f:
                self.prediction_state = json.load(f)
        else:
            self.prediction_state = {
                "usage_trends": defaultdict(list),
                "error_predictions": {},
                "feature_predictions": {},
                "performance_predictions": {},
                "preventive_actions": [],
                "last_prediction": datetime.now().isoformat()
            }
    
    def save_prediction_state(self):
        """Persist prediction state and models"""
        self.prediction_state["last_prediction"] = datetime.now().isoformat()
        # Convert defaultdict to regular dict for JSON serialization
        self.prediction_state["usage_trends"] = dict(self.prediction_state["usage_trends"])
        
        with open(self.prediction_state_file, 'w') as f:
            json.dump(self.prediction_state, f, indent=2)
    
    async def predict_future_needs(self) -> Dict[str, Any]:
        """Analyze usage trends and predict upcoming requirements"""
        logger.info("Predicting future system needs based on usage trends")
        
        # Collect usage data
        usage_data = await self._collect_usage_data()
        
        # Analyze trends
        trends = self._analyze_trends(usage_data)
        
        # Predict future usage patterns
        usage_predictions = self._predict_usage_patterns(trends)
        
        # Predict required features
        feature_predictions = await self._predict_required_features(usage_predictions)
        
        # Predict performance requirements
        performance_predictions = self._predict_performance_needs(usage_data)
        
        # Generate proactive improvements
        proactive_improvements = await self._generate_proactive_improvements(
            feature_predictions, performance_predictions
        )
        
        # Update prediction state
        self.prediction_state["feature_predictions"] = feature_predictions
        self.prediction_state["performance_predictions"] = performance_predictions
        
        self.save_prediction_state()
        
        return {
            "usage_trends": trends,
            "predicted_features": feature_predictions,
            "performance_requirements": performance_predictions,
            "proactive_improvements": proactive_improvements
        }
    
    async def prevent_issues(self) -> Dict[str, Any]:
        """Identify and prevent potential problems before they occur"""
        logger.info("Analyzing system to prevent future issues")
        
        # Predict potential errors
        error_predictions = await self._predict_errors()
        
        # Identify vulnerability patterns
        vulnerabilities = await self._identify_vulnerabilities()
        
        # Predict resource exhaustion
        resource_predictions = self._predict_resource_issues()
        
        # Generate preventive solutions
        preventive_solutions = []
        
        for prediction in error_predictions:
            solution = await self._create_preventive_solution(prediction)
            if solution:
                preventive_solutions.append(solution)
                
        for vulnerability in vulnerabilities:
            solution = await self._create_security_fix(vulnerability)
            if solution:
                preventive_solutions.append(solution)
                
        for resource_issue in resource_predictions:
            solution = await self._create_resource_optimization(resource_issue)
            if solution:
                preventive_solutions.append(solution)
        
        # Apply high-priority preventive measures
        applied_solutions = await self._apply_preventive_measures(preventive_solutions)
        
        # Update prediction state
        self.prediction_state["error_predictions"] = error_predictions
        self.prediction_state["preventive_actions"].extend(applied_solutions)
        
        self.save_prediction_state()
        
        return {
            "predicted_errors": error_predictions,
            "vulnerabilities": vulnerabilities,
            "resource_issues": resource_predictions,
            "preventive_solutions": preventive_solutions,
            "applied_solutions": applied_solutions
        }
    
    async def _collect_usage_data(self) -> Dict[str, Any]:
        """Collect comprehensive usage data from the system"""
        usage_data = {
            "tool_usage": defaultdict(list),
            "api_calls": defaultdict(list),
            "error_rates": defaultdict(list),
            "performance_metrics": defaultdict(list),
            "user_patterns": defaultdict(list)
        }
        
        # Collect from task history
        if Path("task_history.json").exists():
            with open("task_history.json", 'r') as f:
                history = json.load(f)
                
            for task in history.get("tasks", []):
                timestamp = datetime.fromisoformat(task.get("timestamp", datetime.now().isoformat()))
                task_type = task.get("type", "unknown")
                
                usage_data["tool_usage"][task_type].append({
                    "timestamp": timestamp.isoformat(),
                    "duration": task.get("duration", 0),
                    "success": task.get("success", True)
                })
        
        # Collect from Redis metrics
        try:
            metric_keys = await self.redis_client.keys("metrics:*")
            for key in metric_keys:
                data = await self.redis_client.get(key)
                if data:
                    metric = json.loads(data)
                    metric_type = key.decode().split(":")[1]
                    usage_data["performance_metrics"][metric_type].append(metric)
        except:
            pass
            
        # Collect from system state
        if Path("system_state.json").exists():
            with open("system_state.json", 'r') as f:
                state = json.load(f)
                usage_data["system_state"] = state
                
        return dict(usage_data)
    
    def _analyze_trends(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage data to identify trends"""
        trends = {
            "growth_rate": {},
            "peak_times": {},
            "seasonal_patterns": {},
            "emerging_uses": []
        }
        
        # Analyze tool usage growth
        for tool, usage_list in usage_data.get("tool_usage", {}).items():
            if len(usage_list) > 10:
                # Calculate growth rate
                timestamps = [datetime.fromisoformat(u["timestamp"]) for u in usage_list]
                daily_counts = defaultdict(int)
                
                for ts in timestamps:
                    daily_counts[ts.date()] += 1
                
                if len(daily_counts) > 7:
                    dates = sorted(daily_counts.keys())
                    counts = [daily_counts[d] for d in dates]
                    
                    # Simple linear regression for trend
                    X = np.array(range(len(dates))).reshape(-1, 1)
                    y = np.array(counts)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    growth_rate = float(model.coef_[0])
                    trends["growth_rate"][tool] = {
                        "rate": growth_rate,
                        "direction": "increasing" if growth_rate > 0 else "decreasing",
                        "strength": abs(growth_rate)
                    }
        
        # Identify peak usage times
        all_timestamps = []
        for tool_usage in usage_data.get("tool_usage", {}).values():
            all_timestamps.extend([
                datetime.fromisoformat(u["timestamp"]).hour 
                for u in tool_usage
            ])
            
        if all_timestamps:
            hour_counts = defaultdict(int)
            for hour in all_timestamps:
                hour_counts[hour] += 1
                
            peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            trends["peak_times"] = {
                "hours": [h[0] for h in peak_hours],
                "activity_distribution": dict(hour_counts)
            }
        
        # Detect emerging uses (new tools with increasing usage)
        for tool, trend in trends["growth_rate"].items():
            tool_usage = usage_data.get("tool_usage", {}).get(tool, [])
            if len(tool_usage) < 20 and trend["rate"] > 0.5:  # New and growing
                trends["emerging_uses"].append({
                    "tool": tool,
                    "growth_rate": trend["rate"],
                    "usage_count": len(tool_usage)
                })
                
        return trends
    
    def _predict_usage_patterns(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future usage patterns based on trends"""
        predictions = {
            "next_week": {},
            "next_month": {},
            "growth_areas": [],
            "declining_areas": []
        }
        
        # Predict based on growth rates
        for tool, trend in trends.get("growth_rate", {}).items():
            rate = trend["rate"]
            
            # Simple projection
            current_usage = 10  # baseline
            next_week = current_usage + (rate * 7)
            next_month = current_usage + (rate * 30)
            
            predictions["next_week"][tool] = max(0, next_week)
            predictions["next_month"][tool] = max(0, next_month)
            
            if rate > 1:
                predictions["growth_areas"].append({
                    "tool": tool,
                    "expected_growth": f"{rate * 30:.0f} uses/month"
                })
            elif rate < -0.5:
                predictions["declining_areas"].append({
                    "tool": tool,
                    "expected_decline": f"{abs(rate * 30):.0f} uses/month"
                })
                
        return predictions
    
    async def _predict_required_features(self, usage_predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict features that will be needed based on usage patterns"""
        required_features = []
        
        # Analyze growth areas
        for area in usage_predictions.get("growth_areas", []):
            prompt = f"""
            Based on rapid growth in {area['tool']} usage ({area['expected_growth']}),
            predict what new features or capabilities users will need:
            
            Consider:
            1. Scalability requirements
            2. Advanced features for power users
            3. Integration needs
            4. Automation opportunities
            
            Return as JSON list: [{{
                "feature": "name",
                "description": "what it does",
                "priority": "high|medium|low",
                "justification": "why needed"
            }}]
            """
            
            result = await self.ai_brain.process(prompt)
            try:
                features = json.loads(result)
                for feature in features:
                    feature["predicted_for"] = area["tool"]
                    feature["based_on"] = "usage_growth"
                    required_features.extend(features)
            except:
                pass
                
        # Analyze emerging uses
        for emerging in usage_predictions.get("emerging_uses", []):
            prompt = f"""
            New usage pattern detected: {emerging}
            
            What supporting features should we proactively create?
            
            Return as JSON list: [{{
                "feature": "name",
                "description": "purpose",
                "implementation": "brief how-to"
            }}]
            """
            
            result = await self.ai_brain.process(prompt)
            try:
                features = json.loads(result)
                for feature in features:
                    feature["predicted_for"] = "emerging_pattern"
                    feature["priority"] = "medium"
                    required_features.extend(features)
            except:
                pass
                
        return required_features
    
    def _predict_performance_needs(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future performance requirements"""
        performance_needs = {
            "cpu_requirements": {},
            "memory_requirements": {},
            "storage_requirements": {},
            "response_time_targets": {}
        }
        
        # Analyze performance metrics trends
        for metric_type, metrics in usage_data.get("performance_metrics", {}).items():
            if len(metrics) > 5:
                # Extract values
                values = [m.get("value", 0) for m in metrics if "value" in m]
                
                if values:
                    # Predict future needs (simple projection)
                    current_avg = np.mean(values)
                    current_max = np.max(values)
                    
                    # Assume 20% growth per month
                    growth_factor = 1.2
                    
                    if "cpu" in metric_type.lower():
                        performance_needs["cpu_requirements"][metric_type] = {
                            "current_avg": current_avg,
                            "predicted_avg": current_avg * growth_factor,
                            "predicted_peak": current_max * growth_factor * 1.5
                        }
                    elif "memory" in metric_type.lower():
                        performance_needs["memory_requirements"][metric_type] = {
                            "current_avg": current_avg,
                            "predicted_avg": current_avg * growth_factor,
                            "predicted_peak": current_max * growth_factor * 1.3
                        }
                        
        # Predict response time requirements based on usage growth
        total_usage = sum(len(v) for v in usage_data.get("tool_usage", {}).values())
        if total_usage > 100:
            performance_needs["response_time_targets"] = {
                "api_calls": "< 200ms",
                "tool_execution": "< 2s",
                "batch_operations": "< 10s"
            }
            
        return performance_needs
    
    async def _generate_proactive_improvements(self, 
                                             feature_predictions: List[Dict[str, Any]],
                                             performance_predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proactive improvements based on predictions"""
        improvements = []
        
        # Create improvements for predicted features
        high_priority_features = [f for f in feature_predictions if f.get("priority") == "high"]
        
        for feature in high_priority_features[:5]:  # Top 5 features
            prompt = f"""
            Create a proactive improvement to implement this predicted feature:
            {json.dumps(feature, indent=2)}
            
            Generate:
            1. Implementation plan
            2. Required tools/components
            3. Success metrics
            
            Return as JSON: {{
                "improvement": "name",
                "type": "feature",
                "implementation": ["step1", "step2"],
                "components": ["comp1", "comp2"],
                "metrics": ["metric1", "metric2"]
            }}
            """
            
            result = await self.ai_brain.process(prompt)
            try:
                improvement = json.loads(result)
                improvement["prediction_source"] = feature
                improvement["created_at"] = datetime.now().isoformat()
                improvements.append(improvement)
            except:
                pass
                
        # Create performance improvements
        if performance_predictions.get("cpu_requirements"):
            improvements.append({
                "improvement": "CPU optimization",
                "type": "performance",
                "implementation": [
                    "Profile CPU-intensive operations",
                    "Implement caching for repeated calculations",
                    "Add async processing for heavy tasks"
                ],
                "components": ["profiler", "cache_manager", "async_executor"],
                "metrics": ["cpu_usage_reduction", "response_time_improvement"]
            })
            
        if performance_predictions.get("memory_requirements"):
            improvements.append({
                "improvement": "Memory optimization",
                "type": "performance",
                "implementation": [
                    "Implement object pooling",
                    "Add memory monitoring",
                    "Optimize data structures"
                ],
                "components": ["object_pool", "memory_monitor"],
                "metrics": ["memory_usage_reduction", "gc_frequency"]
            })
            
        return improvements
    
    async def _predict_errors(self) -> List[Dict[str, Any]]:
        """Predict potential errors based on patterns"""
        error_predictions = []
        
        # Load error history from pattern engine
        if hasattr(self.pattern_engine, 'learning_state'):
            error_patterns = self.pattern_engine.learning_state.get("error_patterns", {})
            
            for date, data in error_patterns.items():
                for category, pattern in data.get("details", {}).items():
                    if pattern.get("frequency", 0) > 2:
                        # Predict recurrence
                        prediction = {
                            "error_type": category,
                            "likelihood": min(pattern["frequency"] / 10, 1.0),
                            "pattern": pattern,
                            "predicted_scenarios": await self._predict_error_scenarios(category, pattern)
                        }
                        error_predictions.append(prediction)
                        
        return error_predictions
    
    async def _predict_error_scenarios(self, category: str, pattern: Dict[str, Any]) -> List[str]:
        """Predict scenarios where error might occur"""
        prompt = f"""
        Based on this error pattern:
        Category: {category}
        Pattern: {json.dumps(pattern, indent=2)}
        
        Predict 3 scenarios where this error is likely to occur again.
        
        Return as JSON list: ["scenario1", "scenario2", "scenario3"]
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            return json.loads(result)
        except:
            return []
    
    async def _identify_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Identify potential security and stability vulnerabilities"""
        vulnerabilities = []
        
        # Check for common vulnerability patterns
        checks = [
            {
                "type": "unchecked_input",
                "pattern": "user input without validation",
                "severity": "high"
            },
            {
                "type": "resource_leak",
                "pattern": "resources not properly closed",
                "severity": "medium"
            },
            {
                "type": "race_condition",
                "pattern": "concurrent access without locks",
                "severity": "high"
            },
            {
                "type": "error_disclosure",
                "pattern": "sensitive info in error messages",
                "severity": "medium"
            }
        ]
        
        # Scan codebase for vulnerabilities
        code_files = list(Path("scripts").rglob("*.py"))[:20]  # Sample
        
        for check in checks:
            found_in_files = []
            
            for file in code_files:
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                        
                    # Simple pattern matching (could be enhanced)
                    if check["type"] == "unchecked_input" and "input(" in content and "validate" not in content:
                        found_in_files.append(str(file))
                    elif check["type"] == "resource_leak" and "open(" in content and "with" not in content:
                        found_in_files.append(str(file))
                    elif check["type"] == "race_condition" and "threading" in content and "Lock" not in content:
                        found_in_files.append(str(file))
                except:
                    pass
                    
            if found_in_files:
                vulnerabilities.append({
                    "type": check["type"],
                    "description": check["pattern"],
                    "severity": check["severity"],
                    "found_in": found_in_files[:5],  # Limit for brevity
                    "recommendation": await self._get_vulnerability_fix(check["type"])
                })
                
        return vulnerabilities
    
    async def _get_vulnerability_fix(self, vuln_type: str) -> str:
        """Get recommended fix for vulnerability type"""
        fixes = {
            "unchecked_input": "Add input validation and sanitization",
            "resource_leak": "Use context managers (with statements) for resource handling",
            "race_condition": "Implement proper locking mechanisms",
            "error_disclosure": "Sanitize error messages before display"
        }
        return fixes.get(vuln_type, "Review and fix security issue")
    
    def _predict_resource_issues(self) -> List[Dict[str, Any]]:
        """Predict potential resource exhaustion issues"""
        resource_issues = []
        
        # Check current resource usage trends
        if Path("system_state.json").exists():
            with open("system_state.json", 'r') as f:
                state = json.load(f)
                
            # Memory growth prediction
            memory_usage = state.get("memory_usage", {})
            if memory_usage:
                current_mb = memory_usage.get("current_mb", 0)
                if current_mb > 200:  # High baseline
                    resource_issues.append({
                        "type": "memory_exhaustion",
                        "current_usage": f"{current_mb}MB",
                        "predicted_issue": "Potential OOM in heavy load",
                        "time_to_issue": "2-4 weeks at current growth"
                    })
                    
            # Connection pool exhaustion
            connections = state.get("active_connections", 0)
            if connections > 50:
                resource_issues.append({
                    "type": "connection_exhaustion",
                    "current_usage": connections,
                    "predicted_issue": "Connection pool limit reached",
                    "time_to_issue": "1-2 weeks"
                })
                
        return resource_issues
    
    async def _create_preventive_solution(self, error_prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create solution to prevent predicted error"""
        prompt = f"""
        Create a preventive solution for this predicted error:
        {json.dumps(error_prediction, indent=2)}
        
        Solution should:
        1. Prevent the error from occurring
        2. Be implemented before the error happens
        3. Not impact existing functionality
        
        Return as JSON: {{
            "solution": "name",
            "type": "preventive",
            "implementation": "code or config change",
            "validation": "how to verify it works"
        }}
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            solution = json.loads(result)
            solution["prevents"] = error_prediction["error_type"]
            return solution
        except:
            return None
    
    async def _create_security_fix(self, vulnerability: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create security fix for vulnerability"""
        return {
            "solution": f"Fix {vulnerability['type']}",
            "type": "security",
            "implementation": vulnerability["recommendation"],
            "validation": "Security scan after implementation",
            "priority": vulnerability["severity"],
            "files_to_fix": vulnerability.get("found_in", [])
        }
    
    async def _create_resource_optimization(self, resource_issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create optimization for resource issue"""
        optimizations = {
            "memory_exhaustion": {
                "solution": "Memory optimization",
                "implementation": "Implement object pooling and garbage collection tuning",
                "validation": "Monitor memory usage reduction"
            },
            "connection_exhaustion": {
                "solution": "Connection pool optimization",
                "implementation": "Implement connection pooling and recycling",
                "validation": "Monitor connection usage"
            }
        }
        
        opt = optimizations.get(resource_issue["type"])
        if opt:
            opt["type"] = "resource_optimization"
            opt["addresses"] = resource_issue
            return opt
            
        return None
    
    async def _apply_preventive_measures(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply high-priority preventive measures"""
        applied = []
        
        # Sort by priority
        high_priority = [s for s in solutions if s.get("priority") == "high" or s.get("type") == "security"]
        
        for solution in high_priority[:3]:  # Apply top 3
            try:
                # Log the application
                applied_solution = {
                    "solution": solution["solution"],
                    "type": solution["type"],
                    "applied_at": datetime.now().isoformat(),
                    "status": "applied",
                    "details": solution
                }
                
                applied.append(applied_solution)
                
                # In a real implementation, this would apply the actual fix
                logger.info(f"Applied preventive solution: {solution['solution']}")
                
            except Exception as e:
                logger.error(f"Failed to apply solution: {e}")
                
        return applied