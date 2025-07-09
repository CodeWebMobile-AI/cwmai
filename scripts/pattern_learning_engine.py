import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio
from pathlib import Path
import re
import ast

from scripts.ai_brain import AIBrain
from scripts.redis_integration import get_redis_client

logger = logging.getLogger(__name__)


class PatternLearningEngine:
    """Learn from system behavior patterns to improve autonomously"""
    
    def __init__(self):
        self.ai_brain = AIBrain()
        self.redis_client = get_redis_client()
        self.pattern_cache = defaultdict(list)
        self.learning_state_file = "pattern_learning_state.json"
        self.load_learning_state()
        
    def load_learning_state(self):
        """Load previously learned patterns"""
        if Path(self.learning_state_file).exists():
            with open(self.learning_state_file, 'r') as f:
                self.learning_state = json.load(f)
        else:
            self.learning_state = {
                "workflow_patterns": {},
                "error_patterns": {},
                "optimization_patterns": {},
                "success_patterns": {},
                "learned_at": datetime.now().isoformat()
            }
    
    def save_learning_state(self):
        """Persist learned patterns"""
        self.learning_state["learned_at"] = datetime.now().isoformat()
        with open(self.learning_state_file, 'w') as f:
            json.dump(self.learning_state, f, indent=2)
    
    async def learn_workflow_patterns(self) -> Dict[str, Any]:
        """Identify common sequences of actions and create automated workflows"""
        logger.info("Learning workflow patterns from system behavior")
        
        # Analyze task history
        task_sequences = await self._get_task_sequences()
        
        # Find frequent patterns
        frequent_patterns = self._find_frequent_sequences(task_sequences)
        
        # Generate workflow proposals
        new_workflows = []
        for pattern in frequent_patterns:
            workflow = await self._generate_workflow_from_pattern(pattern)
            if workflow:
                new_workflows.append(workflow)
                
        # Store learned patterns
        self.learning_state["workflow_patterns"][datetime.now().isoformat()] = {
            "patterns_found": len(frequent_patterns),
            "workflows_created": len(new_workflows),
            "patterns": frequent_patterns
        }
        
        self.save_learning_state()
        return {
            "patterns_discovered": frequent_patterns,
            "workflows_created": new_workflows
        }
    
    async def learn_error_patterns(self) -> Dict[str, Any]:
        """Identify recurring errors and generate preventive measures"""
        logger.info("Learning error patterns to prevent future issues")
        
        # Get error history
        error_logs = await self._get_error_history()
        
        # Categorize errors
        error_categories = self._categorize_errors(error_logs)
        
        # Find patterns in error occurrences
        error_patterns = {}
        preventive_measures = []
        
        for category, errors in error_categories.items():
            pattern = self._analyze_error_pattern(errors)
            if pattern["frequency"] > 3:  # Recurring error
                measure = await self._generate_preventive_measure(category, pattern)
                if measure:
                    preventive_measures.append(measure)
                    error_patterns[category] = pattern
        
        # Store learned patterns
        self.learning_state["error_patterns"][datetime.now().isoformat()] = {
            "patterns_found": len(error_patterns),
            "preventive_measures": len(preventive_measures),
            "details": error_patterns
        }
        
        self.save_learning_state()
        return {
            "error_patterns": error_patterns,
            "preventive_measures": preventive_measures
        }
    
    async def learn_optimization_patterns(self) -> Dict[str, Any]:
        """Find inefficient code patterns and generate optimized replacements"""
        logger.info("Learning optimization patterns from codebase")
        
        # Analyze code performance metrics
        performance_data = await self._get_performance_metrics()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(performance_data)
        
        # Find common inefficient patterns
        inefficient_patterns = []
        optimizations = []
        
        for bottleneck in bottlenecks:
            pattern = await self._analyze_code_pattern(bottleneck)
            if pattern:
                optimization = await self._generate_optimization(pattern)
                if optimization:
                    inefficient_patterns.append(pattern)
                    optimizations.append(optimization)
        
        # Store learned patterns
        self.learning_state["optimization_patterns"][datetime.now().isoformat()] = {
            "patterns_found": len(inefficient_patterns),
            "optimizations_generated": len(optimizations),
            "performance_improvement": self._calculate_improvement(optimizations)
        }
        
        self.save_learning_state()
        return {
            "inefficient_patterns": inefficient_patterns,
            "optimizations": optimizations
        }
    
    async def _get_task_sequences(self) -> List[List[str]]:
        """Extract task sequences from history"""
        sequences = []
        
        # Read task history
        if Path("task_history.json").exists():
            with open("task_history.json", 'r') as f:
                history = json.load(f)
                
            # Group tasks by session/time window
            current_sequence = []
            last_time = None
            
            for task in history.get("tasks", []):
                task_time = datetime.fromisoformat(task.get("timestamp", datetime.now().isoformat()))
                
                if last_time and (task_time - last_time) > timedelta(hours=1):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = []
                
                current_sequence.append(task.get("type", "unknown"))
                last_time = task_time
                
            if current_sequence:
                sequences.append(current_sequence)
                
        return sequences
    
    def _find_frequent_sequences(self, sequences: List[List[str]], min_support: int = 3) -> List[Dict[str, Any]]:
        """Find frequently occurring task sequences"""
        sequence_counts = Counter()
        
        # Count subsequences of length 2-5
        for seq in sequences:
            for length in range(2, min(6, len(seq) + 1)):
                for i in range(len(seq) - length + 1):
                    subseq = tuple(seq[i:i+length])
                    sequence_counts[subseq] += 1
        
        # Filter by minimum support
        frequent = []
        for seq, count in sequence_counts.items():
            if count >= min_support:
                frequent.append({
                    "sequence": list(seq),
                    "frequency": count,
                    "confidence": count / len(sequences)
                })
        
        return sorted(frequent, key=lambda x: x["frequency"], reverse=True)[:10]
    
    async def _generate_workflow_from_pattern(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a workflow from a discovered pattern"""
        prompt = f"""
        Create a workflow from this frequently occurring task pattern:
        
        Pattern: {pattern['sequence']}
        Frequency: {pattern['frequency']} times
        Confidence: {pattern['confidence']:.2%}
        
        Generate a workflow that:
        1. Automates this sequence of tasks
        2. Includes error handling
        3. Has clear triggers and conditions
        4. Provides value to the user
        
        Return as JSON with: name, description, trigger, steps, error_handling
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            workflow = json.loads(result)
            workflow["generated_from_pattern"] = pattern
            workflow["created_at"] = datetime.now().isoformat()
            return workflow
        except:
            return None
    
    async def _get_error_history(self) -> List[Dict[str, Any]]:
        """Retrieve error logs from various sources"""
        errors = []
        
        # Check log files
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if "ERROR" in line or "Exception" in line:
                            errors.append({
                                "source": log_file.name,
                                "message": line.strip(),
                                "timestamp": datetime.now().isoformat()
                            })
            except:
                pass
                
        # Check Redis for error events
        try:
            error_keys = await self.redis_client.keys("error:*")
            for key in error_keys:
                error_data = await self.redis_client.get(key)
                if error_data:
                    errors.append(json.loads(error_data))
        except:
            pass
            
        return errors
    
    def _categorize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize errors by type"""
        categories = defaultdict(list)
        
        for error in errors:
            msg = error.get("message", "")
            
            if "ImportError" in msg or "ModuleNotFoundError" in msg:
                categories["import_errors"].append(error)
            elif "AttributeError" in msg:
                categories["attribute_errors"].append(error)
            elif "TypeError" in msg:
                categories["type_errors"].append(error)
            elif "connection" in msg.lower() or "timeout" in msg.lower():
                categories["connection_errors"].append(error)
            elif "permission" in msg.lower() or "access" in msg.lower():
                categories["permission_errors"].append(error)
            else:
                categories["other_errors"].append(error)
                
        return dict(categories)
    
    def _analyze_error_pattern(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pattern in error occurrences"""
        if not errors:
            return {"frequency": 0}
            
        # Extract common elements
        messages = [e.get("message", "") for e in errors]
        
        # Find common substrings
        common_parts = []
        if len(messages) > 1:
            first_msg = messages[0]
            for i in range(len(first_msg)):
                for j in range(i + 10, len(first_msg) + 1):  # Min substring length 10
                    substring = first_msg[i:j]
                    if all(substring in msg for msg in messages[1:]):
                        common_parts.append(substring)
        
        # Find timing patterns
        timestamps = []
        for error in errors:
            try:
                timestamps.append(datetime.fromisoformat(error.get("timestamp", "")))
            except:
                pass
                
        time_pattern = "random"
        if len(timestamps) > 2:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals, timedelta()) / len(intervals)
            if all(abs(interval - avg_interval) < timedelta(minutes=5) for interval in intervals):
                time_pattern = f"periodic ({avg_interval})"
        
        return {
            "frequency": len(errors),
            "common_elements": max(common_parts, key=len) if common_parts else "",
            "time_pattern": time_pattern,
            "sources": list(set(e.get("source", "unknown") for e in errors))
        }
    
    async def _generate_preventive_measure(self, category: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate preventive measure for error pattern"""
        prompt = f"""
        Generate a preventive measure for this recurring error pattern:
        
        Category: {category}
        Pattern: {json.dumps(pattern, indent=2)}
        
        Create a solution that:
        1. Prevents the error from occurring
        2. Can be implemented automatically
        3. Doesn't break existing functionality
        4. Includes monitoring to verify effectiveness
        
        Return as JSON with: name, description, implementation, monitoring
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            measure = json.loads(result)
            measure["category"] = category
            measure["pattern"] = pattern
            return measure
        except:
            return None
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from system"""
        metrics = {
            "tool_execution_times": {},
            "memory_usage": {},
            "api_response_times": {},
            "task_completion_times": {}
        }
        
        # Read from performance logs if available
        if Path("performance_metrics.json").exists():
            with open("performance_metrics.json", 'r') as f:
                metrics.update(json.load(f))
                
        # Get Redis performance data
        try:
            perf_keys = await self.redis_client.keys("performance:*")
            for key in perf_keys:
                data = await self.redis_client.get(key)
                if data:
                    metric_type = key.decode().split(":")[1]
                    metrics[metric_type] = json.loads(data)
        except:
            pass
            
        return metrics
    
    def _identify_bottlenecks(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze execution times
        for category, times in performance_data.get("tool_execution_times", {}).items():
            if isinstance(times, list) and times:
                avg_time = sum(times) / len(times)
                if avg_time > 5.0:  # More than 5 seconds average
                    bottlenecks.append({
                        "type": "slow_execution",
                        "category": category,
                        "avg_time": avg_time,
                        "samples": len(times)
                    })
        
        # Analyze memory usage
        for component, usage in performance_data.get("memory_usage", {}).items():
            if isinstance(usage, dict) and usage.get("peak_mb", 0) > 500:  # More than 500MB
                bottlenecks.append({
                    "type": "high_memory",
                    "component": component,
                    "peak_mb": usage["peak_mb"]
                })
                
        return bottlenecks
    
    async def _analyze_code_pattern(self, bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze code causing performance bottleneck"""
        # Find relevant code files
        if bottleneck["type"] == "slow_execution":
            # Look for the tool/function implementation
            tool_name = bottleneck.get("category", "")
            tool_file = Path(f"scripts/custom_tools/{tool_name}.py")
            
            if tool_file.exists():
                with open(tool_file, 'r') as f:
                    code = f.read()
                    
                # Analyze code structure
                try:
                    tree = ast.parse(code)
                    
                    # Look for common inefficiencies
                    inefficiencies = []
                    
                    for node in ast.walk(tree):
                        # Nested loops
                        if isinstance(node, ast.For):
                            for child in ast.walk(node):
                                if isinstance(child, ast.For) and child != node:
                                    inefficiencies.append({
                                        "type": "nested_loops",
                                        "line": node.lineno
                                    })
                                    
                        # Multiple similar API calls
                        if isinstance(node, ast.Call):
                            if hasattr(node.func, 'attr') and 'request' in str(node.func.attr):
                                inefficiencies.append({
                                    "type": "repeated_api_calls",
                                    "line": node.lineno
                                })
                    
                    if inefficiencies:
                        return {
                            "file": str(tool_file),
                            "bottleneck": bottleneck,
                            "inefficiencies": inefficiencies,
                            "code_snippet": code[:500]
                        }
                except:
                    pass
                    
        return None
    
    async def _generate_optimization(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate optimization for inefficient pattern"""
        prompt = f"""
        Generate an optimization for this performance issue:
        
        Pattern: {json.dumps(pattern, indent=2)}
        
        Create an optimization that:
        1. Addresses the specific inefficiency
        2. Maintains functionality
        3. Is easy to implement
        4. Provides measurable improvement
        
        Return as JSON with: description, before_code, after_code, expected_improvement
        """
        
        result = await self.ai_brain.process(prompt)
        try:
            optimization = json.loads(result)
            optimization["pattern"] = pattern
            optimization["generated_at"] = datetime.now().isoformat()
            return optimization
        except:
            return None
    
    def _calculate_improvement(self, optimizations: List[Dict[str, Any]]) -> float:
        """Calculate expected performance improvement"""
        if not optimizations:
            return 0.0
            
        total_improvement = 0.0
        for opt in optimizations:
            expected = opt.get("expected_improvement", "0%")
            try:
                # Extract percentage
                improvement = float(re.search(r'(\d+)', expected).group(1))
                total_improvement += improvement
            except:
                pass
                
        return total_improvement / len(optimizations) if optimizations else 0.0
    
    async def apply_learned_patterns(self) -> Dict[str, Any]:
        """Apply all learned patterns to improve the system"""
        results = {
            "workflows_implemented": 0,
            "preventive_measures_applied": 0,
            "optimizations_applied": 0,
            "improvements": []
        }
        
        # Apply workflow patterns
        for date, data in self.learning_state.get("workflow_patterns", {}).items():
            for pattern in data.get("patterns", []):
                if pattern.get("confidence", 0) > 0.7:  # High confidence patterns
                    # Implement the workflow
                    improvement = {
                        "type": "workflow",
                        "pattern": pattern,
                        "applied_at": datetime.now().isoformat()
                    }
                    results["improvements"].append(improvement)
                    results["workflows_implemented"] += 1
        
        # Apply preventive measures
        for date, data in self.learning_state.get("error_patterns", {}).items():
            for category, pattern in data.get("details", {}).items():
                if pattern.get("frequency", 0) > 5:  # Frequent errors
                    # Implement preventive measure
                    improvement = {
                        "type": "error_prevention",
                        "category": category,
                        "pattern": pattern,
                        "applied_at": datetime.now().isoformat()
                    }
                    results["improvements"].append(improvement)
                    results["preventive_measures_applied"] += 1
        
        # Apply optimizations
        for date, data in self.learning_state.get("optimization_patterns", {}).items():
            if data.get("performance_improvement", 0) > 10:  # Significant improvement
                # Apply optimization
                improvement = {
                    "type": "optimization",
                    "expected_improvement": data["performance_improvement"],
                    "applied_at": datetime.now().isoformat()
                }
                results["improvements"].append(improvement)
                results["optimizations_applied"] += 1
        
        return results