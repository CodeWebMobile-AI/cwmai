"""
Resource Optimizer

Optimizes resource allocation across research functions, tasks, and self-improvement.
Uses AI to dynamically balance resources for maximum system effectiveness.
"""

import os
import json
import time
import psutil
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np


class ResourceOptimizer:
    """Optimizes resource allocation across system functions."""
    
    def __init__(self, ai_brain):
        """Initialize resource optimizer.
        
        Args:
            ai_brain: AI brain for optimization decisions
        """
        self.ai_brain = ai_brain
        self.resource_pools = self._initialize_resource_pools()
        self.allocation_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: {'successes': 0, 'failures': 0})
        self.optimization_insights = []
        self.current_allocations = {}
        self.resource_limits = self._get_resource_limits()
        
    def _initialize_resource_pools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource pools.
        
        Returns:
            Resource pool configuration
        """
        return {
            'compute': {
                'total': 100.0,  # Percentage
                'available': 100.0,
                'reserved': 0.0,
                'unit': 'percent'
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024**3),  # GB
                'available': psutil.virtual_memory().available / (1024**3),
                'reserved': 0.0,
                'unit': 'GB'
            },
            'ai_tokens': {
                'total': 1000000,  # Daily limit
                'available': 1000000,
                'reserved': 0,
                'unit': 'tokens',
                'reset_time': datetime.now(timezone.utc) + timedelta(days=1)
            },
            'api_calls': {
                'total': 10000,  # Daily limit
                'available': 10000,
                'reserved': 0,
                'unit': 'calls',
                'reset_time': datetime.now(timezone.utc) + timedelta(days=1)
            },
            'time_allocation': {
                'total': 86400,  # Seconds in a day
                'available': 86400,
                'reserved': 0,
                'unit': 'seconds'
            }
        }
    
    def _get_resource_limits(self) -> Dict[str, Any]:
        """Get system resource limits.
        
        Returns:
            Resource limits
        """
        return {
            'max_concurrent_tasks': 5,
            'max_memory_per_task_gb': 2.0,
            'max_compute_per_task_percent': 25.0,
            'max_ai_tokens_per_task': 50000,
            'priority_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
        }
    
    async def allocate_resources(self,
                                request: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate resources for a request.
        
        Args:
            request: Resource request with requirements
            
        Returns:
            Allocation result
        """
        request_id = request.get('id', f"req_{int(time.time())}")
        request_type = request.get('type', 'task')
        priority = request.get('priority', 'medium')
        
        # Check current availability
        availability = self._check_availability(request)
        
        if not availability['sufficient']:
            # Try optimization
            optimization = await self._optimize_allocation(request, availability)
            
            if optimization['successful']:
                availability = self._check_availability(request)
            else:
                return {
                    'allocated': False,
                    'reason': optimization['reason'],
                    'suggestions': optimization.get('suggestions', [])
                }
        
        # Allocate resources
        allocation = self._perform_allocation(request, request_id)
        
        # Record allocation
        self._record_allocation(request_id, request, allocation)
        
        # Update current allocations
        self.current_allocations[request_id] = {
            'request': request,
            'allocation': allocation,
            'start_time': datetime.now(timezone.utc),
            'status': 'active'
        }
        
        # Schedule monitoring
        asyncio.create_task(self._monitor_allocation(request_id))
        
        return {
            'allocated': True,
            'allocation_id': request_id,
            'resources': allocation,
            'estimated_duration': request.get('estimated_duration', 300)
        }
    
    def _check_availability(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource availability.
        
        Args:
            request: Resource request
            
        Returns:
            Availability status
        """
        requirements = request.get('requirements', {})
        insufficient = []
        
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                if pool['available'] < amount:
                    insufficient.append({
                        'resource': resource_type,
                        'required': amount,
                        'available': pool['available'],
                        'unit': pool['unit']
                    })
        
        return {
            'sufficient': len(insufficient) == 0,
            'insufficient_resources': insufficient,
            'total_available': {
                k: v['available'] 
                for k, v in self.resource_pools.items()
            }
        }
    
    async def _optimize_allocation(self,
                                  request: Dict[str, Any],
                                  availability: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation using AI.
        
        Args:
            request: Resource request
            availability: Current availability
            
        Returns:
            Optimization result
        """
        prompt = f"""
        Optimize resource allocation for this request:
        
        Request:
        {json.dumps(request, indent=2)}
        
        Current Availability:
        {json.dumps(availability, indent=2)}
        
        Active Allocations:
        {json.dumps([
            {
                'id': aid,
                'type': a['request']['type'],
                'priority': a['request']['priority'],
                'resources': a['allocation'],
                'duration': (datetime.now(timezone.utc) - a['start_time']).seconds
            }
            for aid, a in self.current_allocations.items()
            if a['status'] == 'active'
        ], indent=2)}
        
        Options:
        1. Defer lower priority tasks
        2. Reduce resource allocation for active tasks
        3. Queue this request
        4. Suggest alternative resource configuration
        
        Return optimization plan as JSON with:
        - successful: true/false
        - actions: List of actions to take
        - reason: Explanation
        - suggestions: Alternative approaches
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        optimization = self._parse_json_response(response)
        
        # Execute optimization actions
        if optimization.get('successful') and optimization.get('actions'):
            for action in optimization['actions']:
                await self._execute_optimization_action(action)
        
        return optimization
    
    async def _execute_optimization_action(self, action: Dict[str, Any]):
        """Execute an optimization action.
        
        Args:
            action: Action to execute
        """
        action_type = action.get('type')
        
        if action_type == 'defer_task':
            task_id = action.get('task_id')
            if task_id in self.current_allocations:
                # Mark for deferral
                self.current_allocations[task_id]['status'] = 'deferred'
                await self._release_resources(task_id)
                
        elif action_type == 'reduce_allocation':
            task_id = action.get('task_id')
            reduction = action.get('reduction', {})
            if task_id in self.current_allocations:
                await self._reduce_allocation(task_id, reduction)
                
        elif action_type == 'rebalance':
            await self._rebalance_all_allocations()
    
    def _perform_allocation(self,
                           request: Dict[str, Any],
                           allocation_id: str) -> Dict[str, Any]:
        """Perform actual resource allocation.
        
        Args:
            request: Resource request
            allocation_id: Allocation identifier
            
        Returns:
            Allocated resources
        """
        requirements = request.get('requirements', {})
        allocated = {}
        
        for resource_type, amount in requirements.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                
                # Allocate requested amount
                allocated_amount = min(amount, pool['available'])
                pool['available'] -= allocated_amount
                pool['reserved'] += allocated_amount
                
                allocated[resource_type] = {
                    'amount': allocated_amount,
                    'unit': pool['unit']
                }
        
        return allocated
    
    async def release_resources(self, allocation_id: str) -> Dict[str, Any]:
        """Release allocated resources.
        
        Args:
            allocation_id: Allocation to release
            
        Returns:
            Release result
        """
        if allocation_id not in self.current_allocations:
            return {'released': False, 'reason': 'Allocation not found'}
        
        allocation_data = self.current_allocations[allocation_id]
        
        # Release resources back to pools
        for resource_type, resource_data in allocation_data['allocation'].items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                amount = resource_data['amount']
                
                pool['available'] += amount
                pool['reserved'] -= amount
        
        # Update allocation status
        allocation_data['status'] = 'completed'
        allocation_data['end_time'] = datetime.now(timezone.utc)
        
        # Calculate effectiveness
        effectiveness = await self._calculate_allocation_effectiveness(allocation_data)
        
        # Update performance metrics
        request_type = allocation_data['request']['type']
        if effectiveness > 0.7:
            self.performance_metrics[request_type]['successes'] += 1
        else:
            self.performance_metrics[request_type]['failures'] += 1
        
        # Remove from active allocations
        del self.current_allocations[allocation_id]
        
        return {
            'released': True,
            'effectiveness': effectiveness,
            'duration': (allocation_data['end_time'] - allocation_data['start_time']).seconds
        }
    
    async def _monitor_allocation(self, allocation_id: str):
        """Monitor resource allocation.
        
        Args:
            allocation_id: Allocation to monitor
        """
        start_time = time.time()
        check_interval = 30  # seconds
        
        while allocation_id in self.current_allocations:
            await asyncio.sleep(check_interval)
            
            allocation = self.current_allocations.get(allocation_id)
            if not allocation or allocation['status'] != 'active':
                break
            
            # Check resource usage
            usage = await self._check_resource_usage(allocation_id)
            
            # Detect anomalies
            if usage.get('anomaly_detected'):
                await self._handle_resource_anomaly(allocation_id, usage)
            
            # Check timeout
            elapsed = time.time() - start_time
            max_duration = allocation['request'].get('max_duration', 3600)
            
            if elapsed > max_duration:
                print(f"Allocation {allocation_id} exceeded max duration")
                await self.release_resources(allocation_id)
                break
    
    async def _check_resource_usage(self, allocation_id: str) -> Dict[str, Any]:
        """Check actual resource usage.
        
        Args:
            allocation_id: Allocation to check
            
        Returns:
            Usage information
        """
        # In production, would check actual usage
        # For now, simulate
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'anomaly_detected': False
        }
    
    async def _handle_resource_anomaly(self,
                                      allocation_id: str,
                                      usage: Dict[str, Any]):
        """Handle resource usage anomaly.
        
        Args:
            allocation_id: Allocation with anomaly
            usage: Usage data
        """
        print(f"Resource anomaly detected for {allocation_id}: {usage}")
        
        # Could implement automatic scaling or alerts
        pass
    
    async def _calculate_allocation_effectiveness(self,
                                                allocation_data: Dict[str, Any]) -> float:
        """Calculate how effectively resources were used.
        
        Args:
            allocation_data: Allocation data
            
        Returns:
            Effectiveness score 0.0-1.0
        """
        # Simple calculation based on completion and duration
        request = allocation_data['request']
        actual_duration = (allocation_data['end_time'] - allocation_data['start_time']).seconds
        expected_duration = request.get('estimated_duration', 300)
        
        # Efficiency score
        if actual_duration > 0:
            efficiency = min(1.0, expected_duration / actual_duration)
        else:
            efficiency = 0.0
        
        # Completion score (would check actual completion in production)
        completion = 0.9 if allocation_data['status'] == 'completed' else 0.3
        
        # Resource utilization (simulated)
        utilization = 0.8
        
        # Weighted average
        effectiveness = (efficiency * 0.3 + completion * 0.5 + utilization * 0.2)
        
        return effectiveness
    
    def _record_allocation(self,
                          allocation_id: str,
                          request: Dict[str, Any],
                          allocation: Dict[str, Any]):
        """Record allocation for analysis.
        
        Args:
            allocation_id: Allocation identifier
            request: Original request
            allocation: Allocated resources
        """
        record = {
            'id': allocation_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request': request,
            'allocation': allocation,
            'pools_snapshot': {
                k: {'available': v['available'], 'reserved': v['reserved']}
                for k, v in self.resource_pools.items()
            }
        }
        
        self.allocation_history.append(record)
    
    async def optimize_global_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation globally.
        
        Returns:
            Optimization result
        """
        print("Optimizing global resource allocation...")
        
        # Analyze current state
        analysis = self._analyze_resource_state()
        
        # Get optimization recommendations
        recommendations = await self._get_optimization_recommendations(analysis)
        
        # Apply optimizations
        applied = []
        for rec in recommendations:
            if await self._apply_recommendation(rec):
                applied.append(rec)
        
        # Generate insights
        insights = await self._generate_optimization_insights(analysis, applied)
        self.optimization_insights.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'insights': insights
        })
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'applied': applied,
            'insights': insights
        }
    
    def _analyze_resource_state(self) -> Dict[str, Any]:
        """Analyze current resource state.
        
        Returns:
            State analysis
        """
        total_allocations = len(self.current_allocations)
        active_allocations = sum(
            1 for a in self.current_allocations.values()
            if a['status'] == 'active'
        )
        
        # Resource utilization
        utilization = {}
        for resource_type, pool in self.resource_pools.items():
            if pool['total'] > 0:
                utilization[resource_type] = {
                    'utilized': (pool['reserved'] / pool['total']) * 100,
                    'available': (pool['available'] / pool['total']) * 100
                }
        
        # Performance by type
        performance = {}
        for task_type, metrics in self.performance_metrics.items():
            total = metrics['successes'] + metrics['failures']
            if total > 0:
                performance[task_type] = {
                    'success_rate': metrics['successes'] / total,
                    'total_tasks': total
                }
        
        return {
            'total_allocations': total_allocations,
            'active_allocations': active_allocations,
            'utilization': utilization,
            'performance': performance,
            'bottlenecks': self._identify_bottlenecks(utilization)
        }
    
    def _identify_bottlenecks(self, utilization: Dict[str, Any]) -> List[str]:
        """Identify resource bottlenecks.
        
        Args:
            utilization: Resource utilization data
            
        Returns:
            List of bottlenecks
        """
        bottlenecks = []
        
        for resource, usage in utilization.items():
            if usage['utilized'] > 80:
                bottlenecks.append(f"{resource} (at {usage['utilized']:.1f}% capacity)")
        
        return bottlenecks
    
    async def _get_optimization_recommendations(self,
                                               analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI-powered optimization recommendations.
        
        Args:
            analysis: Resource state analysis
            
        Returns:
            List of recommendations
        """
        prompt = f"""
        Recommend resource allocation optimizations:
        
        Current State:
        {json.dumps(analysis, indent=2)}
        
        Resource Limits:
        {json.dumps(self.resource_limits, indent=2)}
        
        Recent Allocation History (last 10):
        {json.dumps([
            {
                'type': r['request']['type'],
                'priority': r['request']['priority'],
                'resources': list(r['allocation'].keys())
            }
            for r in list(self.allocation_history)[-10:]
        ], indent=2)}
        
        Provide recommendations to:
        1. Eliminate bottlenecks
        2. Improve task success rates
        3. Balance resource usage
        4. Prepare for future demand
        
        Format as JSON array with:
        - action: Specific action to take
        - target: What to optimize
        - expected_impact: Expected improvement
        - priority: high/medium/low
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        recommendations = self._parse_json_response(response)
        
        if isinstance(recommendations, list):
            return recommendations
        
        return []
    
    async def _apply_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Apply optimization recommendation.
        
        Args:
            recommendation: Recommendation to apply
            
        Returns:
            Success status
        """
        action = recommendation.get('action', '').lower()
        
        if 'rebalance' in action:
            return await self._rebalance_all_allocations()
        elif 'increase' in action and 'limit' in action:
            return self._adjust_resource_limits(recommendation)
        elif 'prioritize' in action:
            return self._adjust_priorities(recommendation)
        
        return False
    
    async def _rebalance_all_allocations(self) -> bool:
        """Rebalance all active allocations.
        
        Returns:
            Success status
        """
        active_allocations = [
            (aid, a) for aid, a in self.current_allocations.items()
            if a['status'] == 'active'
        ]
        
        if not active_allocations:
            return True
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        active_allocations.sort(
            key=lambda x: priority_order.get(x[1]['request'].get('priority', 'medium'), 2)
        )
        
        # Recalculate fair share
        total_weight = sum(
            1.0 / (priority_order.get(a[1]['request'].get('priority', 'medium'), 2) + 1)
            for a in active_allocations
        )
        
        # Rebalance each resource type
        rebalanced = True
        for resource_type in self.resource_pools:
            if not self._rebalance_resource_type(resource_type, active_allocations, total_weight):
                rebalanced = False
        
        return rebalanced
    
    def _rebalance_resource_type(self,
                                resource_type: str,
                                allocations: List[Tuple[str, Dict[str, Any]]],
                                total_weight: float) -> bool:
        """Rebalance specific resource type.
        
        Args:
            resource_type: Resource to rebalance
            allocations: Active allocations
            total_weight: Total priority weight
            
        Returns:
            Success status
        """
        # This is simplified - in production would be more sophisticated
        return True
    
    def _adjust_resource_limits(self, recommendation: Dict[str, Any]) -> bool:
        """Adjust resource limits based on recommendation.
        
        Args:
            recommendation: Adjustment recommendation
            
        Returns:
            Success status
        """
        target = recommendation.get('target', '')
        
        # Adjust limits conservatively
        if 'memory' in target.lower():
            self.resource_limits['max_memory_per_task_gb'] *= 1.1
        elif 'compute' in target.lower():
            self.resource_limits['max_compute_per_task_percent'] *= 1.1
        
        return True
    
    def _adjust_priorities(self, recommendation: Dict[str, Any]) -> bool:
        """Adjust priority thresholds.
        
        Args:
            recommendation: Priority adjustment
            
        Returns:
            Success status
        """
        # Adjust priority thresholds
        self.resource_limits['priority_thresholds']['high'] *= 0.95
        self.resource_limits['priority_thresholds']['medium'] *= 0.95
        
        return True
    
    async def _reduce_allocation(self,
                               task_id: str,
                               reduction: Dict[str, float]):
        """Reduce resource allocation for a task.
        
        Args:
            task_id: Task to reduce
            reduction: Reduction amounts by resource type
        """
        if task_id not in self.current_allocations:
            return
        
        allocation = self.current_allocations[task_id]['allocation']
        
        for resource_type, reduction_percent in reduction.items():
            if resource_type in allocation and resource_type in self.resource_pools:
                current_amount = allocation[resource_type]['amount']
                reduction_amount = current_amount * (reduction_percent / 100)
                
                # Return resources to pool
                self.resource_pools[resource_type]['available'] += reduction_amount
                self.resource_pools[resource_type]['reserved'] -= reduction_amount
                
                # Update allocation
                allocation[resource_type]['amount'] -= reduction_amount
    
    async def _release_resources(self, task_id: str):
        """Release all resources for a task.
        
        Args:
            task_id: Task to release
        """
        await self.release_resources(task_id)
    
    async def _generate_optimization_insights(self,
                                            analysis: Dict[str, Any],
                                            applied: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from optimization.
        
        Args:
            analysis: Resource analysis
            applied: Applied optimizations
            
        Returns:
            Optimization insights
        """
        prompt = f"""
        Generate insights from resource optimization:
        
        Analysis:
        {json.dumps(analysis, indent=2)}
        
        Applied Optimizations:
        {json.dumps(applied, indent=2)}
        
        Provide insights on:
        1. Resource usage patterns
        2. Optimization effectiveness
        3. Future resource needs
        4. System scalability
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            
            # Look for JSON array
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                return json.loads(array_match.group())
            
            # Look for JSON object
            obj_match = re.search(r'\{[\s\S]*\}', content)
            if obj_match:
                return json.loads(obj_match.group())
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        
        return {}
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource allocation summary.
        
        Returns:
            Resource summary
        """
        return {
            'pools': {
                k: {
                    'total': v['total'],
                    'available': v['available'],
                    'reserved': v['reserved'],
                    'utilization': (v['reserved'] / v['total'] * 100) if v['total'] > 0 else 0
                }
                for k, v in self.resource_pools.items()
            },
            'active_allocations': len(self.current_allocations),
            'performance': dict(self.performance_metrics),
            'recent_optimizations': self.optimization_insights[-3:] if self.optimization_insights else []
        }
    
    def reset_daily_limits(self):
        """Reset daily resource limits."""
        now = datetime.now(timezone.utc)
        
        for resource_type, pool in self.resource_pools.items():
            if 'reset_time' in pool and now >= pool['reset_time']:
                pool['available'] = pool['total'] - pool['reserved']
                pool['reset_time'] = now + timedelta(days=1)
                print(f"Reset daily limit for {resource_type}")


async def demonstrate_resource_optimizer():
    """Demonstrate resource optimization."""
    print("=== Resource Optimizer Demo ===\n")
    
    # Mock AI brain
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt):
            if "Optimize resource allocation" in prompt:
                return {
                    'content': '''{
                        "successful": true,
                        "actions": [{"type": "rebalance"}],
                        "reason": "Rebalancing can free up resources"
                    }'''
                }
            return {'content': '[]'}
    
    ai_brain = MockAIBrain()
    optimizer = ResourceOptimizer(ai_brain)
    
    # Allocate resources
    print("Allocating resources for task...")
    result = await optimizer.allocate_resources({
        'id': 'task_001',
        'type': 'ai_research',
        'priority': 'high',
        'requirements': {
            'compute': 20.0,
            'memory': 1.5,
            'ai_tokens': 10000
        },
        'estimated_duration': 300
    })
    
    print(f"Allocation result: {result['allocated']}")
    if result['allocated']:
        print(f"Allocation ID: {result['allocation_id']}")
        print(f"Resources: {result['resources']}")
    
    # Show resource state
    print("\n=== Resource State ===")
    summary = optimizer.get_resource_summary()
    for resource, data in summary['pools'].items():
        print(f"{resource}: {data['utilization']:.1f}% utilized")
    
    # Optimize globally
    print("\nOptimizing global allocation...")
    optimization = await optimizer.optimize_global_allocation()
    print(f"Applied {len(optimization['applied'])} optimizations")
    
    # Release resources
    if result['allocated']:
        print("\nReleasing resources...")
        release_result = await optimizer.release_resources(result['allocation_id'])
        print(f"Released: {release_result['released']}")
        print(f"Effectiveness: {release_result['effectiveness']:.2f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_resource_optimizer())