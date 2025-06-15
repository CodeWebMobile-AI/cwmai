"""
External Knowledge Integrator

Safely integrates extracted capabilities from external AI agent repositories
into CWMAI's architecture with comprehensive testing, validation, and rollback
mechanisms.
"""

import os
import json
import asyncio
import tempfile
import shutil
import subprocess
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# Import CWMAI components
from capability_extractor import ExtractedCapability, IntegrationComplexity
from safe_self_improver import SafeSelfImprover, ModificationType
from state_manager import StateManager


class IntegrationStatus(Enum):
    """Status of capability integration."""
    PENDING = "pending"
    TESTING = "testing"
    VALIDATED = "validated"
    INTEGRATED = "integrated"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class IntegrationStrategy(Enum):
    """Strategies for integrating capabilities."""
    DIRECT_COPY = "direct_copy"           # Copy code with minimal changes
    ADAPTER_PATTERN = "adapter_pattern"   # Create adapter to integrate
    WRAPPER_CLASS = "wrapper_class"       # Wrap external code in CWMAI interface
    REFACTOR_INTEGRATE = "refactor_integrate"  # Refactor code to fit CWMAI patterns
    PLUGIN_SYSTEM = "plugin_system"       # Integrate as plugin
    MICROSERVICE = "microservice"         # Deploy as separate service


@dataclass
class IntegrationPlan:
    """Plan for integrating a capability."""
    capability_id: str
    capability_name: str
    integration_strategy: IntegrationStrategy
    target_modules: List[str]
    modification_steps: List[Dict[str, Any]]
    test_requirements: List[str]
    rollback_plan: Dict[str, Any]
    estimated_effort_hours: float
    risk_assessment: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IntegrationResult:
    """Result of capability integration."""
    capability_id: str
    integration_plan_id: str
    status: IntegrationStatus
    integration_strategy: IntegrationStrategy
    
    # Implementation details
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    code_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Testing results
    test_results: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    compatibility_check: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    integration_time_seconds: float = 0.0
    rollback_commit: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class ExternalKnowledgeIntegrator:
    """Safely integrates external capabilities into CWMAI."""
    
    def __init__(self, 
                 safe_improver: Optional[SafeSelfImprover] = None,
                 state_manager: Optional[StateManager] = None):
        """Initialize the external knowledge integrator.
        
        Args:
            safe_improver: Safe self-improver instance
            state_manager: State manager instance
        """
        self.safe_improver = safe_improver or SafeSelfImprover()
        self.state_manager = state_manager or StateManager()
        
        # Integration state
        self.integration_plans: Dict[str, IntegrationPlan] = {}
        self.integration_results: Dict[str, IntegrationResult] = {}
        self.active_integrations: Set[str] = set()
        
        # Configuration
        self.config = {
            'max_concurrent_integrations': 3,
            'integration_timeout_minutes': 30,
            'test_timeout_seconds': 300,
            'backup_before_integration': True,
            'require_manual_approval': False,
            'auto_rollback_on_failure': True
        }
        
        # CWMAI architecture knowledge
        self.cwmai_modules = self._load_cwmai_architecture()
        
        # Integration cache and workspace
        self.cache_dir = Path('.integration_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        self.workspace_dir = Path('.integration_workspace')
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.integration_stats = {
            'total_integrations_attempted': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'rolled_back_integrations': 0,
            'total_integration_time': 0.0,
            'integrations_by_strategy': {},
            'integrations_by_complexity': {}
        }
        
        # Load previous state
        self._load_integration_state()
    
    async def create_integration_plan(self, capability: ExtractedCapability) -> IntegrationPlan:
        """Create an integration plan for a capability.
        
        Args:
            capability: Extracted capability to integrate
            
        Returns:
            Integration plan
        """
        self.logger.info(f"Creating integration plan for: {capability.name}")
        
        # Determine integration strategy
        strategy = self._select_integration_strategy(capability)
        
        # Identify target modules
        target_modules = self._identify_target_modules(capability)
        
        # Create modification steps
        modification_steps = await self._plan_modification_steps(capability, strategy)
        
        # Assess risks
        risk_assessment = self._assess_integration_risks(capability, strategy)
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(capability, strategy)
        
        # Estimate effort
        effort_hours = self._estimate_integration_effort(capability, strategy)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(capability)
        
        # Create plan
        plan_id = self._generate_plan_id(capability)
        
        plan = IntegrationPlan(
            capability_id=capability.id,
            capability_name=capability.name,
            integration_strategy=strategy,
            target_modules=target_modules,
            modification_steps=modification_steps,
            test_requirements=self._define_test_requirements(capability),
            rollback_plan=rollback_plan,
            estimated_effort_hours=effort_hours,
            risk_assessment=risk_assessment,
            success_criteria=success_criteria
        )
        
        self.integration_plans[plan_id] = plan
        self._save_integration_state()
        
        self.logger.info(f"Created integration plan {plan_id} using {strategy.value} strategy")
        
        return plan
    
    async def execute_integration(self, 
                                plan: IntegrationPlan,
                                capability: ExtractedCapability,
                                auto_approve: bool = False) -> IntegrationResult:
        """Execute a capability integration plan.
        
        Args:
            plan: Integration plan to execute
            capability: Capability to integrate
            auto_approve: Whether to auto-approve without manual review
            
        Returns:
            Integration result
        """
        if len(self.active_integrations) >= self.config['max_concurrent_integrations']:
            raise RuntimeError("Maximum concurrent integrations reached")
        
        result_id = f"result_{plan.capability_id}_{int(time.time())}"
        
        result = IntegrationResult(
            capability_id=capability.id,
            integration_plan_id=result_id,
            status=IntegrationStatus.PENDING,
            integration_strategy=plan.integration_strategy
        )
        
        self.integration_results[result_id] = result
        self.active_integrations.add(result_id)
        
        try:
            self.logger.info(f"Starting integration of {capability.name} using {plan.integration_strategy.value}")
            
            # Manual approval check
            if self.config['require_manual_approval'] and not auto_approve:
                approval = await self._request_manual_approval(plan, capability)
                if not approval:
                    result.status = IntegrationStatus.FAILED
                    result.errors.append("Manual approval denied")
                    return result
            
            # Create backup if configured
            if self.config['backup_before_integration']:
                backup_commit = await self._create_backup()
                result.rollback_commit = backup_commit
            
            # Execute integration steps
            result.status = IntegrationStatus.TESTING
            
            # Step 1: Prepare integration environment
            workspace = await self._prepare_integration_workspace(capability, plan)
            
            # Step 2: Apply modifications
            modification_results = await self._apply_modifications(capability, plan, workspace)
            result.files_modified.extend(modification_results.get('modified_files', []))
            result.files_created.extend(modification_results.get('created_files', []))
            result.code_changes.extend(modification_results.get('code_changes', []))
            
            # Step 3: Run tests
            test_results = await self._run_integration_tests(plan, workspace)
            result.test_results = test_results
            
            # Step 4: Check compatibility
            compatibility_results = await self._check_compatibility(capability, workspace)
            result.compatibility_check = compatibility_results
            
            # Step 5: Measure performance impact
            performance_results = await self._measure_performance_impact(workspace)
            result.performance_impact = performance_results
            
            # Step 6: Validate integration
            validation_success = await self._validate_integration(plan, result)
            
            if validation_success:
                # Step 7: Apply to production
                await self._apply_to_production(capability, plan, workspace)
                result.status = IntegrationStatus.INTEGRATED
                
                # Update statistics
                self.integration_stats['successful_integrations'] += 1
                strategy_key = plan.integration_strategy.value
                self.integration_stats['integrations_by_strategy'][strategy_key] = \
                    self.integration_stats['integrations_by_strategy'].get(strategy_key, 0) + 1
                
                self.logger.info(f"Successfully integrated {capability.name}")
            else:
                # Integration failed validation
                result.status = IntegrationStatus.FAILED
                result.errors.append("Integration failed validation")
                
                if self.config['auto_rollback_on_failure'] and result.rollback_commit:
                    await self._execute_rollback(result.rollback_commit)
                    result.status = IntegrationStatus.ROLLED_BACK
                
                self.integration_stats['failed_integrations'] += 1
            
        except Exception as e:
            self.logger.error(f"Error during integration of {capability.name}: {e}")
            result.status = IntegrationStatus.FAILED
            result.errors.append(str(e))
            
            # Auto-rollback on error
            if self.config['auto_rollback_on_failure'] and result.rollback_commit:
                try:
                    await self._execute_rollback(result.rollback_commit)
                    result.status = IntegrationStatus.ROLLED_BACK
                    self.integration_stats['rolled_back_integrations'] += 1
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
                    result.errors.append(f"Rollback failed: {rollback_error}")
            
            self.integration_stats['failed_integrations'] += 1
        
        finally:
            # Cleanup and finalize
            result.completed_at = datetime.now(timezone.utc)
            result.integration_time_seconds = (result.completed_at - result.started_at).total_seconds()
            
            self.integration_stats['total_integrations_attempted'] += 1
            self.integration_stats['total_integration_time'] += result.integration_time_seconds
            
            self.active_integrations.discard(result_id)
            self._save_integration_state()
            
            # Cleanup workspace
            await self._cleanup_workspace(workspace if 'workspace' in locals() else None)
        
        return result
    
    async def rollback_integration(self, result_id: str) -> bool:
        """Rollback a previously integrated capability.
        
        Args:
            result_id: ID of integration result to rollback
            
        Returns:
            Success status
        """
        if result_id not in self.integration_results:
            self.logger.error(f"Integration result {result_id} not found")
            return False
        
        result = self.integration_results[result_id]
        
        if result.status != IntegrationStatus.INTEGRATED:
            self.logger.error(f"Cannot rollback integration in status: {result.status.value}")
            return False
        
        try:
            if result.rollback_commit:
                await self._execute_rollback(result.rollback_commit)
                result.status = IntegrationStatus.ROLLED_BACK
                self.integration_stats['rolled_back_integrations'] += 1
                self._save_integration_state()
                
                self.logger.info(f"Successfully rolled back integration {result_id}")
                return True
            else:
                self.logger.error(f"No rollback commit available for {result_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error during rollback of {result_id}: {e}")
            return False
    
    def get_integration_recommendations(self, capabilities: List[ExtractedCapability]) -> List[Dict[str, Any]]:
        """Get recommendations for which capabilities to integrate.
        
        Args:
            capabilities: List of extracted capabilities
            
        Returns:
            List of integration recommendations
        """
        recommendations = []
        
        for capability in capabilities:
            # Calculate integration score
            score = self._calculate_integration_score(capability)
            
            if score > 0.5:  # Minimum threshold
                recommendation = {
                    'capability_id': capability.id,
                    'capability_name': capability.name,
                    'capability_type': capability.capability_type.value,
                    'integration_score': score,
                    'recommended_strategy': self._select_integration_strategy(capability).value,
                    'estimated_effort_hours': self._estimate_integration_effort(
                        capability, self._select_integration_strategy(capability)
                    ),
                    'risk_level': self._assess_risk_level(capability),
                    'expected_benefits': self._identify_expected_benefits(capability),
                    'priority': self._calculate_priority(capability, score)
                }
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    # Integration strategy selection
    
    def _select_integration_strategy(self, capability: ExtractedCapability) -> IntegrationStrategy:
        """Select the best integration strategy for a capability."""
        # Simple capabilities can be directly copied
        if capability.integration_complexity == IntegrationComplexity.SIMPLE:
            if len(capability.dependencies) == 0:
                return IntegrationStrategy.DIRECT_COPY
            else:
                return IntegrationStrategy.WRAPPER_CLASS
        
        # Moderate complexity capabilities
        elif capability.integration_complexity == IntegrationComplexity.MODERATE:
            # Check for interface patterns
            if capability.interfaces:
                return IntegrationStrategy.ADAPTER_PATTERN
            else:
                return IntegrationStrategy.REFACTOR_INTEGRATE
        
        # Complex capabilities
        else:
            # Architecture patterns might work as plugins
            if capability.patterns and any('plugin' in p.get('pattern_name', '') for p in capability.patterns):
                return IntegrationStrategy.PLUGIN_SYSTEM
            else:
                return IntegrationStrategy.MICROSERVICE
    
    def _identify_target_modules(self, capability: ExtractedCapability) -> List[str]:
        """Identify CWMAI modules that should be modified for integration."""
        target_modules = []
        
        # Map capability types to target modules
        capability_module_map = {
            'task_orchestration': ['task_manager.py', 'dynamic_swarm.py'],
            'multi_agent_coordination': ['swarm_intelligence.py', 'multi_repo_coordinator.py'],
            'performance_optimization': ['ai_brain.py', 'http_ai_client.py', 'production_orchestrator.py'],
            'error_handling': ['production_orchestrator.py', 'state_manager.py'],
            'api_integration': ['http_ai_client.py'],
            'data_processing': ['state_manager.py', 'ai_brain.py']
        }
        
        capability_type = capability.capability_type.value
        if capability_type in capability_module_map:
            target_modules.extend(capability_module_map[capability_type])
        
        # Add integration points specified in capability
        if capability.cwmai_integration_points:
            target_modules.extend(capability.cwmai_integration_points)
        
        # Remove duplicates and return
        return list(set(target_modules))
    
    async def _plan_modification_steps(self, 
                                     capability: ExtractedCapability,
                                     strategy: IntegrationStrategy) -> List[Dict[str, Any]]:
        """Plan the specific modification steps for integration."""
        steps = []
        
        if strategy == IntegrationStrategy.DIRECT_COPY:
            steps.extend(await self._plan_direct_copy_steps(capability))
        
        elif strategy == IntegrationStrategy.ADAPTER_PATTERN:
            steps.extend(await self._plan_adapter_steps(capability))
        
        elif strategy == IntegrationStrategy.WRAPPER_CLASS:
            steps.extend(await self._plan_wrapper_steps(capability))
        
        elif strategy == IntegrationStrategy.REFACTOR_INTEGRATE:
            steps.extend(await self._plan_refactor_steps(capability))
        
        elif strategy == IntegrationStrategy.PLUGIN_SYSTEM:
            steps.extend(await self._plan_plugin_steps(capability))
        
        elif strategy == IntegrationStrategy.MICROSERVICE:
            steps.extend(await self._plan_microservice_steps(capability))
        
        return steps
    
    async def _plan_direct_copy_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for direct copy integration."""
        steps = []
        
        # Step 1: Create new module file
        module_name = f"external_{capability.name.lower().replace(' ', '_')}.py"
        steps.append({
            'action': 'create_file',
            'target': f"scripts/{module_name}",
            'content_source': 'capability_classes_and_functions',
            'description': f'Create new module for {capability.name}'
        })
        
        # Step 2: Add imports to target modules
        for target_module in capability.cwmai_integration_points:
            steps.append({
                'action': 'add_import',
                'target': f"scripts/{target_module}",
                'import_statement': f"from {module_name[:-3]} import *",
                'description': f'Add import to {target_module}'
            })
        
        return steps
    
    async def _plan_adapter_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for adapter pattern integration."""
        steps = []
        
        # Create adapter class
        adapter_name = f"{capability.name.replace(' ', '')}Adapter"
        
        steps.append({
            'action': 'create_adapter',
            'target': f"scripts/{capability.name.lower().replace(' ', '_')}_adapter.py",
            'adapter_class': adapter_name,
            'external_interface': capability.interfaces[0] if capability.interfaces else None,
            'cwmai_interface': self._identify_cwmai_interface(capability),
            'description': f'Create adapter for {capability.name}'
        })
        
        return steps
    
    async def _plan_wrapper_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for wrapper class integration."""
        steps = []
        
        wrapper_name = f"{capability.name.replace(' ', '')}Wrapper"
        
        steps.append({
            'action': 'create_wrapper',
            'target': f"scripts/{capability.name.lower().replace(' ', '_')}_wrapper.py",
            'wrapper_class': wrapper_name,
            'external_classes': capability.classes,
            'description': f'Create wrapper for {capability.name}'
        })
        
        return steps
    
    async def _plan_refactor_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for refactor integration."""
        steps = []
        
        # This would require more sophisticated analysis
        # For now, create a basic refactoring plan
        
        steps.append({
            'action': 'refactor_integrate',
            'target': capability.cwmai_integration_points[0] if capability.cwmai_integration_points else 'ai_brain.py',
            'capability_code': capability.classes + capability.functions,
            'integration_method': 'merge_into_existing_class',
            'description': f'Refactor and integrate {capability.name}'
        })
        
        return steps
    
    async def _plan_plugin_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for plugin system integration."""
        steps = []
        
        # Create plugin structure
        plugin_dir = f"plugins/{capability.name.lower().replace(' ', '_')}"
        
        steps.extend([
            {
                'action': 'create_directory',
                'target': plugin_dir,
                'description': f'Create plugin directory for {capability.name}'
            },
            {
                'action': 'create_plugin_manifest',
                'target': f"{plugin_dir}/manifest.json",
                'capability': capability,
                'description': f'Create plugin manifest for {capability.name}'
            },
            {
                'action': 'create_plugin_code',
                'target': f"{plugin_dir}/plugin.py",
                'capability_code': capability,
                'description': f'Create plugin implementation for {capability.name}'
            }
        ])
        
        return steps
    
    async def _plan_microservice_steps(self, capability: ExtractedCapability) -> List[Dict[str, Any]]:
        """Plan steps for microservice integration."""
        steps = []
        
        service_dir = f"services/{capability.name.lower().replace(' ', '_')}_service"
        
        steps.extend([
            {
                'action': 'create_directory',
                'target': service_dir,
                'description': f'Create microservice directory for {capability.name}'
            },
            {
                'action': 'create_service_api',
                'target': f"{service_dir}/api.py",
                'capability': capability,
                'description': f'Create API interface for {capability.name} service'
            },
            {
                'action': 'create_docker_config',
                'target': f"{service_dir}/Dockerfile",
                'capability': capability,
                'description': f'Create Docker configuration for {capability.name} service'
            }
        ])
        
        return steps
    
    # Risk assessment and validation
    
    def _assess_integration_risks(self, capability: ExtractedCapability, strategy: IntegrationStrategy) -> Dict[str, Any]:
        """Assess risks of integrating a capability."""
        risks = {
            'technical_risks': [],
            'security_risks': [],
            'performance_risks': [],
            'maintenance_risks': [],
            'overall_risk_level': 'low'
        }
        
        # Technical risks
        if capability.integration_complexity == IntegrationComplexity.COMPLEX:
            risks['technical_risks'].append('High integration complexity')
        
        if len(capability.dependencies) > 5:
            risks['technical_risks'].append('Many external dependencies')
        
        if capability.external_apis:
            risks['technical_risks'].append('External API dependencies')
        
        # Security risks
        if capability.security_score < 0.7:
            risks['security_risks'].append('Low security score')
        
        if any('exec' in func['name'] or 'eval' in func['name'] for func in capability.functions):
            risks['security_risks'].append('Potentially unsafe functions detected')
        
        # Performance risks
        if strategy == IntegrationStrategy.MICROSERVICE:
            risks['performance_risks'].append('Network latency from microservice calls')
        
        # Maintenance risks
        if capability.documentation_score < 0.5:
            risks['maintenance_risks'].append('Poor documentation')
        
        if capability.test_coverage < 0.3:
            risks['maintenance_risks'].append('Low test coverage')
        
        # Calculate overall risk level
        total_risks = len(risks['technical_risks']) + len(risks['security_risks']) + \
                     len(risks['performance_risks']) + len(risks['maintenance_risks'])
        
        if total_risks == 0:
            risks['overall_risk_level'] = 'low'
        elif total_risks <= 2:
            risks['overall_risk_level'] = 'medium'
        else:
            risks['overall_risk_level'] = 'high'
        
        return risks
    
    def _create_rollback_plan(self, capability: ExtractedCapability, strategy: IntegrationStrategy) -> Dict[str, Any]:
        """Create rollback plan for integration."""
        return {
            'rollback_method': 'git_reset',
            'backup_required': True,
            'files_to_restore': capability.cwmai_integration_points,
            'cleanup_actions': [
                'Remove created files',
                'Restore modified files',
                'Clear integration cache',
                'Reset configuration'
            ]
        }
    
    def _estimate_integration_effort(self, capability: ExtractedCapability, strategy: IntegrationStrategy) -> float:
        """Estimate effort required for integration in hours."""
        base_effort = {
            IntegrationStrategy.DIRECT_COPY: 2.0,
            IntegrationStrategy.ADAPTER_PATTERN: 4.0,
            IntegrationStrategy.WRAPPER_CLASS: 3.0,
            IntegrationStrategy.REFACTOR_INTEGRATE: 6.0,
            IntegrationStrategy.PLUGIN_SYSTEM: 8.0,
            IntegrationStrategy.MICROSERVICE: 12.0
        }
        
        effort = base_effort.get(strategy, 4.0)
        
        # Adjust based on complexity
        if capability.integration_complexity == IntegrationComplexity.COMPLEX:
            effort *= 1.5
        elif capability.integration_complexity == IntegrationComplexity.SIMPLE:
            effort *= 0.7
        
        # Adjust based on size
        code_size = len(capability.classes) + len(capability.functions)
        if code_size > 10:
            effort *= 1.3
        elif code_size < 3:
            effort *= 0.8
        
        return effort
    
    def _define_success_criteria(self, capability: ExtractedCapability) -> List[str]:
        """Define success criteria for integration."""
        criteria = [
            'All tests pass',
            'No performance regression',
            'Integration compiles without errors',
            'No security vulnerabilities introduced'
        ]
        
        # Add capability-specific criteria
        if capability.capability_type.value == 'task_orchestration':
            criteria.append('Task completion rate improves or maintains')
        
        elif capability.capability_type.value == 'performance_optimization':
            criteria.append('Performance metrics show improvement')
        
        elif capability.capability_type.value == 'error_handling':
            criteria.append('Error recovery mechanisms function correctly')
        
        return criteria
    
    def _define_test_requirements(self, capability: ExtractedCapability) -> List[str]:
        """Define test requirements for integration."""
        requirements = [
            'Unit tests for integrated code',
            'Integration tests with CWMAI components',
            'Performance benchmarks',
            'Security scan'
        ]
        
        # Add capability-specific test requirements
        if capability.external_apis:
            requirements.append('Mock external API calls in tests')
        
        if capability.capability_type.value == 'multi_agent_coordination':
            requirements.append('Multi-agent communication tests')
        
        return requirements
    
    # Integration execution methods
    
    async def _prepare_integration_workspace(self, capability: ExtractedCapability, plan: IntegrationPlan) -> str:
        """Prepare workspace for integration."""
        workspace_id = f"workspace_{capability.id}_{int(time.time())}"
        workspace_path = self.workspace_dir / workspace_id
        workspace_path.mkdir(exist_ok=True)
        
        # Copy CWMAI source to workspace
        cwmai_workspace = workspace_path / 'cwmai'
        shutil.copytree('scripts', cwmai_workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        
        # Create capability workspace
        capability_workspace = workspace_path / 'capability'
        capability_workspace.mkdir()
        
        # Save capability data
        with open(capability_workspace / 'capability.json', 'w') as f:
            capability_data = {
                'id': capability.id,
                'name': capability.name,
                'type': capability.capability_type.value,
                'classes': capability.classes,
                'functions': capability.functions,
                'patterns': capability.patterns,
                'interfaces': capability.interfaces
            }
            json.dump(capability_data, f, indent=2)
        
        return str(workspace_path)
    
    async def _apply_modifications(self, 
                                 capability: ExtractedCapability,
                                 plan: IntegrationPlan,
                                 workspace: str) -> Dict[str, Any]:
        """Apply modification steps to integrate capability."""
        results = {
            'modified_files': [],
            'created_files': [],
            'code_changes': []
        }
        
        workspace_path = Path(workspace)
        cwmai_path = workspace_path / 'cwmai'
        
        for step in plan.modification_steps:
            try:
                if step['action'] == 'create_file':
                    file_path = cwmai_path / step['target'].replace('scripts/', '')
                    content = await self._generate_file_content(step, capability)
                    
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    results['created_files'].append(str(file_path))
                    results['code_changes'].append({
                        'action': 'create',
                        'file': str(file_path),
                        'description': step['description']
                    })
                
                elif step['action'] == 'add_import':
                    file_path = cwmai_path / step['target'].replace('scripts/', '')
                    await self._add_import_to_file(file_path, step['import_statement'])
                    
                    results['modified_files'].append(str(file_path))
                    results['code_changes'].append({
                        'action': 'modify',
                        'file': str(file_path),
                        'description': step['description']
                    })
                
                # Add other action types as needed
                
            except Exception as e:
                self.logger.error(f"Error applying modification step {step['action']}: {e}")
                raise
        
        return results
    
    async def _run_integration_tests(self, plan: IntegrationPlan, workspace: str) -> Dict[str, Any]:
        """Run tests for the integration."""
        test_results = {
            'unit_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'integration_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'performance_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'security_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'overall_success': False
        }
        
        workspace_path = Path(workspace)
        cwmai_path = workspace_path / 'cwmai'
        
        # Run Python syntax check
        try:
            result = subprocess.run(
                ['python', '-m', 'py_compile'] + [str(f) for f in cwmai_path.glob('*.py')],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=cwmai_path
            )
            
            if result.returncode == 0:
                test_results['unit_tests']['passed'] += 1
            else:
                test_results['unit_tests']['failed'] += 1
                test_results['unit_tests']['errors'].append(f"Syntax errors: {result.stderr}")
        
        except Exception as e:
            test_results['unit_tests']['errors'].append(f"Syntax check failed: {e}")
        
        # Run basic import tests
        try:
            for test_requirement in plan.test_requirements:
                if 'unit test' in test_requirement.lower():
                    # Run simple import test
                    test_passed = await self._run_import_test(cwmai_path)
                    if test_passed:
                        test_results['integration_tests']['passed'] += 1
                    else:
                        test_results['integration_tests']['failed'] += 1
        
        except Exception as e:
            test_results['integration_tests']['errors'].append(f"Integration test failed: {e}")
        
        # Overall success
        total_failed = (test_results['unit_tests']['failed'] + 
                       test_results['integration_tests']['failed'] +
                       test_results['performance_tests']['failed'] +
                       test_results['security_tests']['failed'])
        
        test_results['overall_success'] = total_failed == 0
        
        return test_results
    
    async def _check_compatibility(self, capability: ExtractedCapability, workspace: str) -> Dict[str, Any]:
        """Check compatibility of integrated capability."""
        compatibility_results = {
            'name_conflicts': [],
            'dependency_conflicts': [],
            'interface_compatibility': True,
            'architecture_compatibility': True,
            'overall_compatible': True
        }
        
        # Check for name conflicts
        cwmai_names = self._get_cwmai_names()
        capability_names = [cls['name'] for cls in capability.classes] + [func['name'] for func in capability.functions]
        
        conflicts = set(cwmai_names) & set(capability_names)
        if conflicts:
            compatibility_results['name_conflicts'] = list(conflicts)
            compatibility_results['overall_compatible'] = False
        
        # Check dependency conflicts
        # This would require more sophisticated analysis
        
        return compatibility_results
    
    async def _measure_performance_impact(self, workspace: str) -> Dict[str, Any]:
        """Measure performance impact of integration."""
        performance_results = {
            'import_time_impact': 0.0,
            'memory_usage_impact': 0.0,
            'startup_time_impact': 0.0,
            'acceptable_impact': True
        }
        
        # Measure import time
        workspace_path = Path(workspace)
        cwmai_path = workspace_path / 'cwmai'
        
        try:
            # Simple import time test
            start_time = time.time()
            
            result = subprocess.run(
                ['python', '-c', 'import sys; sys.path.insert(0, "."); import ai_brain'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwmai_path
            )
            
            import_time = time.time() - start_time
            performance_results['import_time_impact'] = import_time
            
            # Consider acceptable if under 2 seconds
            if import_time > 2.0:
                performance_results['acceptable_impact'] = False
        
        except Exception as e:
            performance_results['acceptable_impact'] = False
            self.logger.warning(f"Performance measurement failed: {e}")
        
        return performance_results
    
    async def _validate_integration(self, plan: IntegrationPlan, result: IntegrationResult) -> bool:
        """Validate that integration meets success criteria."""
        validation_success = True
        
        # Check test results
        if not result.test_results.get('overall_success', False):
            validation_success = False
            result.warnings.append("Tests failed")
        
        # Check performance impact
        if not result.performance_impact.get('acceptable_impact', True):
            validation_success = False
            result.warnings.append("Performance impact too high")
        
        # Check compatibility
        if not result.compatibility_check.get('overall_compatible', True):
            validation_success = False
            result.warnings.append("Compatibility issues detected")
        
        # Check against success criteria
        for criteria in plan.success_criteria:
            if not await self._check_success_criteria(criteria, result):
                validation_success = False
                result.warnings.append(f"Success criteria not met: {criteria}")
        
        return validation_success
    
    async def _apply_to_production(self, capability: ExtractedCapability, plan: IntegrationPlan, workspace: str):
        """Apply the integration to production CWMAI."""
        workspace_path = Path(workspace)
        cwmai_workspace = workspace_path / 'cwmai'
        
        # Copy modified/created files to production
        for step in plan.modification_steps:
            if step['action'] == 'create_file':
                source_file = cwmai_workspace / step['target'].replace('scripts/', '')
                target_file = Path('scripts') / step['target'].replace('scripts/', '')
                
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                    self.logger.info(f"Copied {source_file} to {target_file}")
            
            elif step['action'] == 'add_import':
                # Apply import modifications to production files
                target_file = Path('scripts') / step['target'].replace('scripts/', '')
                await self._add_import_to_file(target_file, step['import_statement'])
                self.logger.info(f"Added import to {target_file}")
    
    # Helper methods
    
    async def _create_backup(self) -> str:
        """Create a backup commit for rollback."""
        try:
            # Use git to create a commit
            result = subprocess.run(
                ['git', 'add', '.'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                commit_result = subprocess.run(
                    ['git', 'commit', '-m', f'Backup before integration at {datetime.now().isoformat()}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if commit_result.returncode == 0:
                    # Get commit hash
                    hash_result = subprocess.run(
                        ['git', 'rev-parse', 'HEAD'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if hash_result.returncode == 0:
                        return hash_result.stdout.strip()
            
            return ""
        
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            return ""
    
    async def _execute_rollback(self, commit_hash: str):
        """Execute rollback to a specific commit."""
        try:
            result = subprocess.run(
                ['git', 'reset', '--hard', commit_hash],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Git rollback failed: {result.stderr}")
            
            self.logger.info(f"Successfully rolled back to commit {commit_hash}")
        
        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            raise
    
    async def _cleanup_workspace(self, workspace: Optional[str]):
        """Clean up integration workspace."""
        if workspace and os.path.exists(workspace):
            try:
                shutil.rmtree(workspace)
                self.logger.debug(f"Cleaned up workspace: {workspace}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup workspace {workspace}: {e}")
    
    def _calculate_integration_score(self, capability: ExtractedCapability) -> float:
        """Calculate integration score for prioritization."""
        score = 0.5  # Base score
        
        # Quality factors
        score += capability.code_quality_score * 0.2
        score += capability.documentation_score * 0.1
        score += capability.extraction_confidence * 0.2
        
        # Complexity factor (inverse)
        if capability.integration_complexity == IntegrationComplexity.SIMPLE:
            score += 0.3
        elif capability.integration_complexity == IntegrationComplexity.MODERATE:
            score += 0.1
        else:
            score -= 0.1
        
        # CWMAI compatibility
        if capability.cwmai_integration_points:
            score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _assess_risk_level(self, capability: ExtractedCapability) -> str:
        """Assess risk level of integrating capability."""
        risk_factors = 0
        
        if capability.integration_complexity == IntegrationComplexity.COMPLEX:
            risk_factors += 2
        elif capability.integration_complexity == IntegrationComplexity.MODERATE:
            risk_factors += 1
        
        if len(capability.dependencies) > 5:
            risk_factors += 1
        
        if capability.security_score < 0.7:
            risk_factors += 1
        
        if capability.external_apis:
            risk_factors += 1
        
        if risk_factors <= 1:
            return 'low'
        elif risk_factors <= 3:
            return 'medium'
        else:
            return 'high'
    
    def _identify_expected_benefits(self, capability: ExtractedCapability) -> List[str]:
        """Identify expected benefits from integrating capability."""
        benefits = []
        
        capability_benefits = {
            'task_orchestration': ['Improved task management', 'Better workflow orchestration'],
            'multi_agent_coordination': ['Enhanced agent communication', 'Better distributed processing'],
            'performance_optimization': ['Faster execution', 'Reduced resource usage'],
            'error_handling': ['Better error recovery', 'Improved system stability'],
            'api_integration': ['Enhanced external connectivity', 'Better API management']
        }
        
        capability_type = capability.capability_type.value
        if capability_type in capability_benefits:
            benefits.extend(capability_benefits[capability_type])
        
        return benefits
    
    def _calculate_priority(self, capability: ExtractedCapability, integration_score: float) -> float:
        """Calculate priority for integration."""
        priority = integration_score
        
        # High-value capability types get priority boost
        high_value_types = ['task_orchestration', 'performance_optimization', 'error_handling']
        if capability.capability_type.value in high_value_types:
            priority += 0.2
        
        # Simple integrations get priority boost
        if capability.integration_complexity == IntegrationComplexity.SIMPLE:
            priority += 0.1
        
        return min(1.0, priority)
    
    # Utility methods for file operations and content generation
    
    async def _generate_file_content(self, step: Dict[str, Any], capability: ExtractedCapability) -> str:
        """Generate content for new files during integration."""
        if step['content_source'] == 'capability_classes_and_functions':
            content = f'''"""
{capability.name} - Integrated from external repository

Extracted capability: {capability.description}
Integration method: {step.get('description', 'Direct integration')}
"""

'''
            
            # Add imports
            if capability.dependencies:
                for dep in capability.dependencies:
                    content += f"import {dep}\n"
                content += "\n"
            
            # Add classes
            for cls in capability.classes:
                content += f"class {cls['name']}:\n"
                class_name = cls.get('name', 'Unknown')
                docstring = cls.get("docstring", f"Integrated class {class_name}")
                content += f'    """{docstring}"""\n'
                content += "    pass\n\n"
            
            # Add functions
            for func in capability.functions:
                args_str = ', '.join(func.get('args', []))
                content += f"def {func['name']}({args_str}):\n"
                func_name = func.get('name', 'unknown')
                docstring = func.get("docstring", f"Integrated function {func_name}")
                content += f'    """{docstring}"""\n'
                content += "    pass\n\n"
            
            return content
        
        return "# Generated content placeholder\n"
    
    async def _add_import_to_file(self, file_path: Path, import_statement: str):
        """Add import statement to a file."""
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add import after existing imports
            lines = content.split('\n')
            import_index = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip().startswith('#'):
                    import_index = i + 1
                elif line.strip() == '':
                    continue
                else:
                    break
            
            lines.insert(import_index, import_statement)
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
        
        except Exception as e:
            self.logger.error(f"Failed to add import to {file_path}: {e}")
    
    async def _run_import_test(self, cwmai_path: Path) -> bool:
        """Run basic import test."""
        try:
            result = subprocess.run(
                ['python', '-c', 'import ai_brain; print("Import successful")'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwmai_path
            )
            
            return result.returncode == 0 and "Import successful" in result.stdout
        
        except Exception:
            return False
    
    async def _check_success_criteria(self, criteria: str, result: IntegrationResult) -> bool:
        """Check if a success criteria is met."""
        if 'tests pass' in criteria.lower():
            return result.test_results.get('overall_success', False)
        
        elif 'performance' in criteria.lower():
            return result.performance_impact.get('acceptable_impact', True)
        
        elif 'compiles' in criteria.lower():
            return len(result.errors) == 0
        
        elif 'security' in criteria.lower():
            return result.compatibility_check.get('overall_compatible', True)
        
        # Default to True for unrecognized criteria
        return True
    
    def _load_cwmai_architecture(self) -> Dict[str, Any]:
        """Load CWMAI architecture information."""
        return {
            'core_modules': [
                'ai_brain.py', 'task_manager.py', 'state_manager.py',
                'production_orchestrator.py', 'swarm_intelligence.py',
                'http_ai_client.py', 'safe_self_improver.py'
            ],
            'integration_interfaces': {
                'task_management': 'TaskManager',
                'ai_coordination': 'AIBrain',
                'state_management': 'StateManager'
            }
        }
    
    def _get_cwmai_names(self) -> List[str]:
        """Get list of existing CWMAI class and function names."""
        # This would be populated by scanning CWMAI source files
        return [
            'AIBrain', 'TaskManager', 'StateManager', 'ProductionOrchestrator',
            'SwarmIntelligence', 'DynamicSwarm', 'HTTPAIClient', 'SafeSelfImprover'
        ]
    
    def _identify_cwmai_interface(self, capability: ExtractedCapability) -> Optional[str]:
        """Identify the appropriate CWMAI interface for adapter pattern."""
        capability_type = capability.capability_type.value
        
        interface_map = {
            'task_orchestration': 'TaskManager',
            'multi_agent_coordination': 'SwarmIntelligence',
            'performance_optimization': 'AIBrain',
            'error_handling': 'StateManager'
        }
        
        return interface_map.get(capability_type)
    
    async def _request_manual_approval(self, plan: IntegrationPlan, capability: ExtractedCapability) -> bool:
        """Request manual approval for integration (placeholder)."""
        # In a real implementation, this would prompt for user approval
        # For now, auto-approve simple integrations
        return capability.integration_complexity == IntegrationComplexity.SIMPLE
    
    def _generate_plan_id(self, capability: ExtractedCapability) -> str:
        """Generate unique plan ID."""
        content = f"{capability.id}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _load_integration_state(self):
        """Load previous integration state."""
        state_file = self.cache_dir / 'integration_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.integration_stats = state.get('stats', self.integration_stats)
                # Load other state as needed
                
            except Exception as e:
                self.logger.error(f"Error loading integration state: {e}")
    
    def _save_integration_state(self):
        """Save integration state."""
        state_file = self.cache_dir / 'integration_state.json'
        
        try:
            state = {
                'stats': self.integration_stats,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving integration state: {e}")
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self.integration_stats.copy()
        
        if stats['total_integrations_attempted'] > 0:
            stats['success_rate'] = stats['successful_integrations'] / stats['total_integrations_attempted']
            stats['average_integration_time'] = stats['total_integration_time'] / stats['total_integrations_attempted']
        else:
            stats['success_rate'] = 0.0
            stats['average_integration_time'] = 0.0
        
        return stats


async def demonstrate_external_knowledge_integration():
    """Demonstrate external knowledge integration."""
    print("=== External Knowledge Integration Demo ===\n")
    
    # Create integrator
    integrator = ExternalKnowledgeIntegrator()
    
    print("Note: This is a demonstration using simulated data")
    print("In real usage, would integrate actual extracted capabilities")
    
    # Show integration statistics
    print("\n=== Integration Statistics ===")
    stats = integrator.get_integration_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nIntegration system ready for capability integration")


if __name__ == "__main__":
    asyncio.run(demonstrate_external_knowledge_integration())