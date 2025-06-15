"""
Task Type Registry and Learning System

This module manages task type definitions, tracks their usage,
and learns which task types work best for different contexts.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import os

from scripts.task_types import SmartTaskType, TaskCategory, ArchitectureType, TaskTypeMetadata


@dataclass
class TaskTypeUsage:
    """Track usage and outcomes of a task type."""
    task_type: SmartTaskType
    repository: str
    lifecycle_stage: str
    architecture: Optional[ArchitectureType]
    success: bool
    duration_cycles: int
    value_created: float  # 0-10 scale
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failure_reason: Optional[str] = None
    

@dataclass 
class TaskTypePattern:
    """Learned pattern about task type effectiveness."""
    task_type: SmartTaskType
    context: Dict[str, Any]  # stage, architecture, etc.
    success_rate: float
    avg_duration: float
    avg_value: float
    usage_count: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskTypeRegistry:
    """Central registry for task types with learning capabilities."""
    
    def __init__(self, storage_path: str = "task_type_registry.json"):
        """Initialize the task type registry.
        
        Args:
            storage_path: Path to persist registry data
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        
        # Usage tracking
        self.usage_history: List[TaskTypeUsage] = []
        self.learned_patterns: Dict[str, TaskTypePattern] = {}
        
        # Success metrics by context
        self.success_by_stage: Dict[str, Dict[SmartTaskType, float]] = defaultdict(lambda: defaultdict(float))
        self.success_by_architecture: Dict[ArchitectureType, Dict[SmartTaskType, float]] = defaultdict(lambda: defaultdict(float))
        
        # Dynamic task type discovery
        self.discovered_task_types: Dict[str, Dict[str, Any]] = {}
        
        # Load persisted data
        self._load_registry_data()
        
    def record_task_outcome(self, task_type: SmartTaskType, outcome: Dict[str, Any]):
        """Record the outcome of a task execution.
        
        Args:
            task_type: The task type that was executed
            outcome: Execution outcome details
        """
        usage = TaskTypeUsage(
            task_type=task_type,
            repository=outcome.get('repository', 'unknown'),
            lifecycle_stage=outcome.get('lifecycle_stage', 'unknown'),
            architecture=outcome.get('architecture'),
            success=outcome.get('success', False),
            duration_cycles=outcome.get('duration_cycles', 1),
            value_created=outcome.get('value_created', 5.0),
            failure_reason=outcome.get('failure_reason')
        )
        
        self.usage_history.append(usage)
        
        # Update patterns
        self._update_learned_patterns(usage)
        
        # Persist data
        self._save_registry_data()
        
    def get_best_task_types(self, context: Dict[str, Any], limit: int = 5) -> List[Tuple[SmartTaskType, float]]:
        """Get best task types for given context based on learned patterns.
        
        Args:
            context: Current context (stage, architecture, needs)
            limit: Maximum number of task types to return
            
        Returns:
            List of (task_type, confidence_score) tuples
        """
        stage = context.get('lifecycle_stage', 'unknown')
        architecture = context.get('architecture')
        current_needs = context.get('current_needs', [])
        
        # Score each task type
        scores: Dict[SmartTaskType, float] = {}
        
        for task_type in SmartTaskType:
            score = self._calculate_task_type_score(task_type, stage, architecture, current_needs)
            if score > 0:
                scores[task_type] = score
                
        # Sort by score and return top N
        sorted_tasks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_tasks[:limit]
        
    def _calculate_task_type_score(self, task_type: SmartTaskType, stage: str, 
                                 architecture: Optional[ArchitectureType], 
                                 current_needs: List[str]) -> float:
        """Calculate effectiveness score for a task type in given context."""
        score = 0.0
        
        # Base score from metadata
        from scripts.task_types import TASK_TYPE_REGISTRY
        metadata = TASK_TYPE_REGISTRY.get(task_type)
        if metadata:
            # Check if applicable to stage
            if stage in metadata.applicable_stages:
                score += 0.3
                
            # Check if applicable to architecture
            if architecture and architecture in metadata.applicable_architectures:
                score += 0.2
            elif not metadata.applicable_architectures:  # Universal task
                score += 0.1
                
        # Boost from learned success patterns
        pattern_key = f"{stage}:{task_type.value}"
        if pattern_key in self.learned_patterns:
            pattern = self.learned_patterns[pattern_key]
            score += pattern.success_rate * 0.5
            
        # Boost if addresses current needs
        task_keywords = task_type.value.lower().split('_')
        for need in current_needs:
            if any(keyword in need.lower() for keyword in task_keywords):
                score += 0.2
                
        return score
        
    def _update_learned_patterns(self, usage: TaskTypeUsage):
        """Update learned patterns based on new usage data."""
        # Create pattern key
        pattern_key = f"{usage.lifecycle_stage}:{usage.task_type.value}"
        
        # Get or create pattern
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = TaskTypePattern(
                task_type=usage.task_type,
                context={
                    'lifecycle_stage': usage.lifecycle_stage,
                    'architecture': usage.architecture.value if usage.architecture else None
                },
                success_rate=0.0,
                avg_duration=0.0,
                avg_value=0.0,
                usage_count=0
            )
            
        pattern = self.learned_patterns[pattern_key]
        
        # Update pattern with exponential moving average
        alpha = 0.3  # Learning rate
        pattern.usage_count += 1
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * (1.0 if usage.success else 0.0)
        pattern.avg_duration = (1 - alpha) * pattern.avg_duration + alpha * usage.duration_cycles
        pattern.avg_value = (1 - alpha) * pattern.avg_value + alpha * usage.value_created
        pattern.last_updated = datetime.now(timezone.utc)
        
        # Update success metrics
        if usage.success:
            self.success_by_stage[usage.lifecycle_stage][usage.task_type] += 1
            if usage.architecture:
                self.success_by_architecture[usage.architecture][usage.task_type] += 1
                
    def suggest_new_task_type(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest a new task type based on patterns and gaps.
        
        Args:
            context: Current context including unmet needs
            
        Returns:
            Suggested new task type definition or None
        """
        unmet_needs = context.get('unmet_needs', [])
        if not unmet_needs:
            return None
            
        # Analyze patterns in successful tasks
        successful_patterns = [p for p in self.learned_patterns.values() if p.success_rate > 0.7]
        
        # Look for gaps
        for need in unmet_needs:
            # Check if existing task types cover this need
            covered = False
            for task_type in SmartTaskType:
                if need.lower() in task_type.value.lower():
                    covered = True
                    break
                    
            if not covered:
                # Suggest new task type
                suggestion = {
                    'name': f"custom_{need.lower().replace(' ', '_')}",
                    'category': self._infer_category(need),
                    'description': f"Custom task type for: {need}",
                    'applicable_stages': [context.get('lifecycle_stage', 'unknown')],
                    'estimated_complexity': 'moderate',
                    'suggested_by_context': context
                }
                
                # Store in discovered types
                self.discovered_task_types[suggestion['name']] = suggestion
                return suggestion
                
        return None
        
    def _infer_category(self, need: str) -> str:
        """Infer task category from need description."""
        need_lower = need.lower()
        
        if any(word in need_lower for word in ['test', 'spec', 'coverage']):
            return TaskCategory.TESTING.value
        elif any(word in need_lower for word in ['doc', 'readme', 'guide']):
            return TaskCategory.DOCUMENTATION.value
        elif any(word in need_lower for word in ['deploy', 'ci', 'cd', 'docker']):
            return TaskCategory.INFRASTRUCTURE.value
        elif any(word in need_lower for word in ['optimize', 'performance', 'speed']):
            return TaskCategory.OPTIMIZATION.value
        elif any(word in need_lower for word in ['security', 'vulnerability', 'auth']):
            return TaskCategory.SECURITY.value
        else:
            return TaskCategory.DEVELOPMENT.value
            
    def get_task_type_analytics(self) -> Dict[str, Any]:
        """Get analytics about task type usage and effectiveness."""
        total_usage = len(self.usage_history)
        successful_tasks = sum(1 for u in self.usage_history if u.success)
        
        # Calculate success rate by category
        category_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        for usage in self.usage_history:
            from scripts.task_types import TASK_TYPE_REGISTRY
            metadata = TASK_TYPE_REGISTRY.get(usage.task_type)
            if metadata:
                category = metadata.category.value
                category_stats[category]['total'] += 1
                if usage.success:
                    category_stats[category]['successful'] += 1
                    
        # Calculate category success rates
        category_success_rates = {}
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                category_success_rates[category] = stats['successful'] / stats['total']
                
        # Find most effective task types per stage
        effective_by_stage = {}
        for stage, task_success in self.success_by_stage.items():
            if task_success:
                best_task = max(task_success.items(), key=lambda x: x[1])
                effective_by_stage[stage] = best_task[0].value
                
        return {
            'total_tasks_executed': total_usage,
            'overall_success_rate': successful_tasks / total_usage if total_usage > 0 else 0,
            'category_success_rates': category_success_rates,
            'most_effective_by_stage': effective_by_stage,
            'discovered_custom_types': len(self.discovered_task_types),
            'patterns_learned': len(self.learned_patterns),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
    def _save_registry_data(self):
        """Persist registry data to storage."""
        try:
            data = {
                'usage_history': [
                    {
                        'task_type': u.task_type.value,
                        'repository': u.repository,
                        'lifecycle_stage': u.lifecycle_stage,
                        'architecture': u.architecture.value if u.architecture else None,
                        'success': u.success,
                        'duration_cycles': u.duration_cycles,
                        'value_created': u.value_created,
                        'timestamp': u.timestamp.isoformat(),
                        'failure_reason': u.failure_reason
                    }
                    for u in self.usage_history[-1000:]  # Keep last 1000 entries
                ],
                'learned_patterns': {
                    key: {
                        'task_type': p.task_type.value,
                        'context': p.context,
                        'success_rate': p.success_rate,
                        'avg_duration': p.avg_duration,
                        'avg_value': p.avg_value,
                        'usage_count': p.usage_count,
                        'last_updated': p.last_updated.isoformat()
                    }
                    for key, p in self.learned_patterns.items()
                },
                'discovered_task_types': self.discovered_task_types
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving registry data: {e}")
            
    def _load_registry_data(self):
        """Load persisted registry data."""
        if not os.path.exists(self.storage_path):
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            # Load usage history
            for item in data.get('usage_history', []):
                try:
                    task_type = SmartTaskType(item['task_type'])
                    architecture = ArchitectureType(item['architecture']) if item.get('architecture') else None
                    
                    usage = TaskTypeUsage(
                        task_type=task_type,
                        repository=item['repository'],
                        lifecycle_stage=item['lifecycle_stage'],
                        architecture=architecture,
                        success=item['success'],
                        duration_cycles=item['duration_cycles'],
                        value_created=item['value_created'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        failure_reason=item.get('failure_reason')
                    )
                    self.usage_history.append(usage)
                except Exception as e:
                    self.logger.debug(f"Error loading usage item: {e}")
                    
            # Load learned patterns
            for key, item in data.get('learned_patterns', {}).items():
                try:
                    task_type = SmartTaskType(item['task_type'])
                    pattern = TaskTypePattern(
                        task_type=task_type,
                        context=item['context'],
                        success_rate=item['success_rate'],
                        avg_duration=item['avg_duration'],
                        avg_value=item['avg_value'],
                        usage_count=item['usage_count'],
                        last_updated=datetime.fromisoformat(item['last_updated'])
                    )
                    self.learned_patterns[key] = pattern
                except Exception as e:
                    self.logger.debug(f"Error loading pattern: {e}")
                    
            # Load discovered task types
            self.discovered_task_types = data.get('discovered_task_types', {})
            
            self.logger.info(f"Loaded {len(self.usage_history)} usage records and {len(self.learned_patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading registry data: {e}")