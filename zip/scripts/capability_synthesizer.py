"""
Capability Synthesizer

Adapts external patterns and capabilities to CWMAI's specific architecture patterns,
enabling seamless integration while maintaining system coherence and design principles.
"""

import os
import ast
import json
import re
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# Import CWMAI and external capability components
from capability_extractor import ExtractedCapability, IntegrationComplexity
from external_knowledge_integrator import IntegrationStrategy, IntegrationPlan
from state_manager import StateManager


class SynthesisStrategy(Enum):
    """Strategies for synthesizing external capabilities."""
    DIRECT_ADAPTATION = "direct_adaptation"         # Adapt with minimal changes
    PATTERN_TRANSLATION = "pattern_translation"    # Translate to CWMAI patterns
    HYBRID_SYNTHESIS = "hybrid_synthesis"          # Combine multiple approaches
    INTERFACE_BRIDGING = "interface_bridging"      # Create interface bridges
    ARCHITECTURE_MAPPING = "architecture_mapping"  # Map to CWMAI architecture


class SynthesisComplexity(Enum):
    """Complexity levels for synthesis operations."""
    TRIVIAL = "trivial"        # Simple copy-paste adaptation
    SIMPLE = "simple"          # Minor architectural adjustments
    MODERATE = "moderate"      # Significant pattern translation
    COMPLEX = "complex"        # Major architectural changes
    IMPOSSIBLE = "impossible"  # Cannot be synthesized safely


@dataclass
class SynthesisPattern:
    """Represents a synthesis pattern for adapting capabilities."""
    pattern_name: str
    external_pattern: str
    cwmai_equivalent: str
    transformation_rules: List[Dict[str, Any]]
    compatibility_score: float
    usage_examples: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class SynthesizedCapability:
    """Result of synthesizing an external capability."""
    original_capability: ExtractedCapability
    synthesis_strategy: SynthesisStrategy
    synthesis_complexity: SynthesisComplexity
    
    # Synthesized components
    synthesized_classes: List[Dict[str, Any]] = field(default_factory=list)
    synthesized_functions: List[Dict[str, Any]] = field(default_factory=list)
    synthesized_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    
    # CWMAI integration
    cwmai_module_mappings: Dict[str, str] = field(default_factory=dict)
    required_imports: List[str] = field(default_factory=list)
    configuration_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Synthesis metadata
    synthesis_confidence: float = 0.0
    quality_preservation: float = 0.0
    architectural_alignment: float = 0.0
    performance_impact_estimate: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation guidance
    implementation_notes: List[str] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    rollback_considerations: List[str] = field(default_factory=list)
    
    synthesized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SynthesisResult:
    """Result of a synthesis operation."""
    capability_id: str
    synthesis_success: bool
    synthesized_capability: Optional[SynthesizedCapability]
    synthesis_time_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    synthesis_summary: Dict[str, Any] = field(default_factory=dict)


class CapabilitySynthesizer:
    """Synthesizes external capabilities to fit CWMAI architecture."""
    
    def __init__(self, state_manager: Optional[StateManager] = None):
        """Initialize the capability synthesizer.
        
        Args:
            state_manager: State manager instance
        """
        self.state_manager = state_manager or StateManager()
        
        # CWMAI architecture knowledge
        self.cwmai_patterns = self._load_cwmai_patterns()
        self.cwmai_interfaces = self._load_cwmai_interfaces()
        self.cwmai_conventions = self._load_cwmai_conventions()
        
        # Synthesis patterns for common transformations
        self.synthesis_patterns = self._initialize_synthesis_patterns()
        
        # Synthesis cache
        self.synthesis_cache: Dict[str, SynthesisResult] = {}
        
        # Statistics
        self.synthesis_stats = {
            'total_syntheses_attempted': 0,
            'successful_syntheses': 0,
            'failed_syntheses': 0,
            'synthesis_strategies_used': {},
            'average_synthesis_time': 0.0,
            'total_synthesis_time': 0.0
        }
        
        # Configuration
        self.config = {
            'max_synthesis_attempts': 3,
            'min_confidence_threshold': 0.6,
            'preserve_original_semantics': True,
            'enforce_cwmai_conventions': True,
            'enable_pattern_optimization': True
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def synthesize_capability(self, capability: ExtractedCapability) -> SynthesisResult:
        """Synthesize an external capability to fit CWMAI architecture.
        
        Args:
            capability: External capability to synthesize
            
        Returns:
            Synthesis result with adapted capability
        """
        start_time = datetime.now()
        self.logger.info(f"Starting synthesis of capability: {capability.name}")
        
        result = SynthesisResult(
            capability_id=capability.id,
            synthesis_success=False,
            synthesized_capability=None,
            synthesis_time_seconds=0.0
        )
        
        try:
            # Step 1: Analyze compatibility with CWMAI
            compatibility_analysis = await self._analyze_cwmai_compatibility(capability)
            
            # Step 2: Select synthesis strategy
            synthesis_strategy = self._select_synthesis_strategy(capability, compatibility_analysis)
            
            # Step 3: Determine synthesis complexity
            synthesis_complexity = self._assess_synthesis_complexity(capability, synthesis_strategy)
            
            if synthesis_complexity == SynthesisComplexity.IMPOSSIBLE:
                result.errors.append("Capability cannot be synthesized safely")
                return result
            
            # Step 4: Execute synthesis
            synthesized_capability = await self._execute_synthesis(
                capability, synthesis_strategy, synthesis_complexity, compatibility_analysis
            )
            
            if synthesized_capability:
                # Step 5: Validate synthesis
                validation_result = await self._validate_synthesis(synthesized_capability)
                
                if validation_result['valid']:
                    result.synthesis_success = True
                    result.synthesized_capability = synthesized_capability
                    
                    # Update statistics
                    self.synthesis_stats['successful_syntheses'] += 1
                    strategy_key = synthesis_strategy.value
                    self.synthesis_stats['synthesis_strategies_used'][strategy_key] = \
                        self.synthesis_stats['synthesis_strategies_used'].get(strategy_key, 0) + 1
                else:
                    result.errors.extend(validation_result.get('errors', []))
                    result.warnings.extend(validation_result.get('warnings', []))
            else:
                result.errors.append("Synthesis execution failed")
        
        except Exception as e:
            self.logger.error(f"Error synthesizing capability {capability.name}: {e}")
            result.errors.append(str(e))
            self.synthesis_stats['failed_syntheses'] += 1
        
        # Finalize result
        processing_time = (datetime.now() - start_time).total_seconds()
        result.synthesis_time_seconds = processing_time
        
        self.synthesis_stats['total_syntheses_attempted'] += 1
        self.synthesis_stats['total_synthesis_time'] += processing_time
        
        if self.synthesis_stats['total_syntheses_attempted'] > 0:
            self.synthesis_stats['average_synthesis_time'] = \
                self.synthesis_stats['total_synthesis_time'] / self.synthesis_stats['total_syntheses_attempted']
        
        # Cache result
        self.synthesis_cache[capability.id] = result
        
        self.logger.info(f"Completed synthesis of {capability.name}: {result.synthesis_success}")
        
        return result
    
    async def synthesize_multiple_capabilities(self, capabilities: List[ExtractedCapability]) -> List[SynthesisResult]:
        """Synthesize multiple capabilities in batch.
        
        Args:
            capabilities: List of capabilities to synthesize
            
        Returns:
            List of synthesis results
        """
        results = []
        
        # Process capabilities in order of synthesis complexity (simple first)
        sorted_capabilities = sorted(capabilities, key=self._estimate_synthesis_complexity)
        
        for capability in sorted_capabilities:
            try:
                result = await self.synthesize_capability(capability)
                results.append(result)
                
                # Learn from each synthesis to improve subsequent ones
                if result.synthesis_success and result.synthesized_capability:
                    await self._learn_from_synthesis(result.synthesized_capability)
                
            except Exception as e:
                self.logger.error(f"Error in batch synthesis for {capability.name}: {e}")
                error_result = SynthesisResult(
                    capability_id=capability.id,
                    synthesis_success=False,
                    synthesized_capability=None,
                    synthesis_time_seconds=0.0,
                    errors=[str(e)]
                )
                results.append(error_result)
        
        return results
    
    async def _analyze_cwmai_compatibility(self, capability: ExtractedCapability) -> Dict[str, Any]:
        """Analyze compatibility between external capability and CWMAI architecture."""
        compatibility = {
            'pattern_matches': [],
            'interface_compatibility': {},
            'naming_conflicts': [],
            'architectural_alignment': 0.0,
            'integration_points': [],
            'required_adaptations': []
        }
        
        # Check pattern compatibility
        for pattern in capability.patterns:
            # Handle both dict and string patterns
            if isinstance(pattern, str):
                pattern_name = pattern.lower()
                pattern_dict = {'pattern_name': pattern_name}
            else:
                pattern_name = pattern.get('pattern_name', '').lower()
                pattern_dict = pattern
                
            for cwmai_pattern in self.cwmai_patterns:
                similarity = self._calculate_pattern_similarity(pattern_dict, cwmai_pattern)
                if similarity > 0.6:
                    compatibility['pattern_matches'].append({
                        'external_pattern': pattern_name,
                        'cwmai_pattern': cwmai_pattern['name'],
                        'similarity': similarity,
                        'adaptation_required': similarity < 0.9
                    })
        
        # Check interface compatibility
        for interface in capability.interfaces:
            # Handle both dict and string interfaces
            if isinstance(interface, str):
                interface_name = interface
                interface_dict = {'name': interface, 'methods': []}
            else:
                interface_name = interface.get('name', '')
                interface_dict = interface
                
            cwmai_interface = self._find_compatible_cwmai_interface(interface_dict)
            if cwmai_interface:
                compatibility['interface_compatibility'][interface_name] = {
                    'cwmai_interface': cwmai_interface['name'],
                    'compatibility_score': self._calculate_interface_compatibility(interface_dict, cwmai_interface),
                    'required_adaptations': self._identify_interface_adaptations(interface_dict, cwmai_interface)
                }
        
        # Check for naming conflicts
        capability_names = [cls['name'] for cls in capability.classes] + [func['name'] for func in capability.functions]
        cwmai_names = self._get_cwmai_component_names()
        
        conflicts = set(capability_names) & set(cwmai_names)
        compatibility['naming_conflicts'] = list(conflicts)
        
        # Calculate overall architectural alignment
        alignment_factors = [
            len(compatibility['pattern_matches']) / max(1, len(capability.patterns)),
            len(compatibility['interface_compatibility']) / max(1, len(capability.interfaces)),
            1.0 - (len(conflicts) / max(1, len(capability_names)))
        ]
        compatibility['architectural_alignment'] = sum(alignment_factors) / len(alignment_factors)
        
        # Identify integration points
        capability_type = capability.capability_type.value
        integration_map = {
            'task_orchestration': ['task_manager.py', 'dynamic_swarm.py'],
            'multi_agent_coordination': ['swarm_intelligence.py', 'multi_repo_coordinator.py'],
            'performance_optimization': ['ai_brain.py', 'http_ai_client.py'],
            'error_handling': ['production_orchestrator.py', 'state_manager.py']
        }
        compatibility['integration_points'] = integration_map.get(capability_type, [])
        
        return compatibility
    
    def _select_synthesis_strategy(self, capability: ExtractedCapability, compatibility: Dict[str, Any]) -> SynthesisStrategy:
        """Select the best synthesis strategy based on capability and compatibility analysis."""
        alignment_score = compatibility['architectural_alignment']
        pattern_matches = len(compatibility['pattern_matches'])
        interface_compatibility = compatibility['interface_compatibility']
        
        # High alignment - direct adaptation
        if alignment_score > 0.8 and pattern_matches > 0:
            return SynthesisStrategy.DIRECT_ADAPTATION
        
        # Good pattern matches - pattern translation
        elif pattern_matches > 0 and alignment_score > 0.6:
            return SynthesisStrategy.PATTERN_TRANSLATION
        
        # Interface compatibility - interface bridging
        elif interface_compatibility and alignment_score > 0.5:
            return SynthesisStrategy.INTERFACE_BRIDGING
        
        # Complex capability - architecture mapping
        elif capability.integration_complexity == IntegrationComplexity.COMPLEX:
            return SynthesisStrategy.ARCHITECTURE_MAPPING
        
        # Default - hybrid synthesis
        else:
            return SynthesisStrategy.HYBRID_SYNTHESIS
    
    def _assess_synthesis_complexity(self, capability: ExtractedCapability, strategy: SynthesisStrategy) -> SynthesisComplexity:
        """Assess the complexity of synthesizing a capability with the given strategy."""
        complexity_factors = []
        
        # Base complexity from capability
        if capability.integration_complexity == IntegrationComplexity.SIMPLE:
            complexity_factors.append(0.2)
        elif capability.integration_complexity == IntegrationComplexity.MODERATE:
            complexity_factors.append(0.5)
        else:
            complexity_factors.append(0.8)
        
        # Strategy complexity
        strategy_complexity = {
            SynthesisStrategy.DIRECT_ADAPTATION: 0.2,
            SynthesisStrategy.PATTERN_TRANSLATION: 0.4,
            SynthesisStrategy.INTERFACE_BRIDGING: 0.5,
            SynthesisStrategy.HYBRID_SYNTHESIS: 0.6,
            SynthesisStrategy.ARCHITECTURE_MAPPING: 0.8
        }
        complexity_factors.append(strategy_complexity.get(strategy, 0.5))
        
        # Code structure complexity
        total_components = len(capability.classes) + len(capability.functions)
        if total_components > 20:
            complexity_factors.append(0.8)
        elif total_components > 10:
            complexity_factors.append(0.5)
        else:
            complexity_factors.append(0.2)
        
        # Dependency complexity
        if len(capability.dependencies) > 10:
            complexity_factors.append(0.7)
        elif len(capability.dependencies) > 5:
            complexity_factors.append(0.4)
        else:
            complexity_factors.append(0.1)
        
        # Calculate overall complexity
        avg_complexity = sum(complexity_factors) / len(complexity_factors)
        
        if avg_complexity < 0.3:
            return SynthesisComplexity.TRIVIAL
        elif avg_complexity < 0.5:
            return SynthesisComplexity.SIMPLE
        elif avg_complexity < 0.7:
            return SynthesisComplexity.MODERATE
        elif avg_complexity < 0.9:
            return SynthesisComplexity.COMPLEX
        else:
            return SynthesisComplexity.IMPOSSIBLE
    
    async def _execute_synthesis(self,
                                capability: ExtractedCapability,
                                strategy: SynthesisStrategy,
                                complexity: SynthesisComplexity,
                                compatibility: Dict[str, Any]) -> Optional[SynthesizedCapability]:
        """Execute the synthesis using the selected strategy."""
        try:
            synthesized = SynthesizedCapability(
                original_capability=capability,
                synthesis_strategy=strategy,
                synthesis_complexity=complexity
            )
            
            # Execute strategy-specific synthesis
            if strategy == SynthesisStrategy.DIRECT_ADAPTATION:
                await self._synthesize_direct_adaptation(synthesized, compatibility)
            
            elif strategy == SynthesisStrategy.PATTERN_TRANSLATION:
                await self._synthesize_pattern_translation(synthesized, compatibility)
            
            elif strategy == SynthesisStrategy.INTERFACE_BRIDGING:
                await self._synthesize_interface_bridging(synthesized, compatibility)
            
            elif strategy == SynthesisStrategy.ARCHITECTURE_MAPPING:
                await self._synthesize_architecture_mapping(synthesized, compatibility)
            
            elif strategy == SynthesisStrategy.HYBRID_SYNTHESIS:
                await self._synthesize_hybrid_approach(synthesized, compatibility)
            
            # Calculate quality metrics
            synthesized.synthesis_confidence = self._calculate_synthesis_confidence(synthesized)
            synthesized.quality_preservation = self._calculate_quality_preservation(synthesized)
            synthesized.architectural_alignment = compatibility['architectural_alignment']
            
            # Generate implementation guidance
            synthesized.implementation_notes = self._generate_implementation_notes(synthesized)
            synthesized.testing_requirements = self._generate_testing_requirements(synthesized)
            synthesized.rollback_considerations = self._generate_rollback_considerations(synthesized)
            
            return synthesized
        
        except Exception as e:
            self.logger.error(f"Error in synthesis execution: {e}")
            return None
    
    async def _synthesize_direct_adaptation(self, synthesized: SynthesizedCapability, compatibility: Dict[str, Any]):
        """Synthesize using direct adaptation strategy."""
        capability = synthesized.original_capability
        
        # Adapt classes with minimal changes
        for cls in capability.classes:
            adapted_class = self._adapt_class_to_cwmai(cls, compatibility)
            if adapted_class:
                synthesized.synthesized_classes.append(adapted_class)
        
        # Adapt functions with minimal changes
        for func in capability.functions:
            adapted_function = self._adapt_function_to_cwmai(func, compatibility)
            if adapted_function:
                synthesized.synthesized_functions.append(adapted_function)
        
        # Handle naming conflicts
        self._resolve_naming_conflicts(synthesized, compatibility['naming_conflicts'])
        
        # Set module mappings
        capability_type = capability.capability_type.value
        target_modules = compatibility.get('integration_points', [])
        for module in target_modules:
            synthesized.cwmai_module_mappings[module] = f"Integrate {capability.name} components"
    
    async def _synthesize_pattern_translation(self, synthesized: SynthesizedCapability, compatibility: Dict[str, Any]):
        """Synthesize using pattern translation strategy."""
        capability = synthesized.original_capability
        
        # Translate patterns to CWMAI equivalents
        for pattern_match in compatibility['pattern_matches']:
            external_pattern = pattern_match['external_pattern']
            cwmai_pattern = pattern_match['cwmai_pattern']
            
            # Find synthesis pattern for this translation
            synthesis_pattern = self._find_synthesis_pattern(external_pattern, cwmai_pattern)
            if synthesis_pattern:
                await self._apply_synthesis_pattern(synthesized, synthesis_pattern)
        
        # Transform components using patterns
        for cls in capability.classes:
            transformed_class = await self._transform_class_with_patterns(cls, synthesized)
            if transformed_class:
                synthesized.synthesized_classes.append(transformed_class)
        
        for func in capability.functions:
            transformed_function = await self._transform_function_with_patterns(func, synthesized)
            if transformed_function:
                synthesized.synthesized_functions.append(transformed_function)
    
    async def _synthesize_interface_bridging(self, synthesized: SynthesizedCapability, compatibility: Dict[str, Any]):
        """Synthesize using interface bridging strategy."""
        capability = synthesized.original_capability
        
        # Create interface bridges
        for interface_name, interface_info in compatibility['interface_compatibility'].items():
            cwmai_interface = interface_info['cwmai_interface']
            adaptations = interface_info['required_adaptations']
            
            # Create bridge interface
            bridge_interface = self._create_interface_bridge(
                interface_name, cwmai_interface, adaptations
            )
            synthesized.synthesized_interfaces.append(bridge_interface)
        
        # Adapt components to use bridges
        for cls in capability.classes:
            bridged_class = self._adapt_class_for_bridging(cls, synthesized.synthesized_interfaces)
            if bridged_class:
                synthesized.synthesized_classes.append(bridged_class)
        
        # Add required imports for bridges
        synthesized.required_imports.extend([
            f"from {interface['module']} import {interface['name']}"
            for interface in synthesized.synthesized_interfaces
            if interface.get('module')
        ])
    
    async def _synthesize_architecture_mapping(self, synthesized: SynthesizedCapability, compatibility: Dict[str, Any]):
        """Synthesize using architecture mapping strategy."""
        capability = synthesized.original_capability
        
        # Map capability to CWMAI architecture
        architecture_mapping = self._create_architecture_mapping(capability)
        
        # Transform components according to mapping
        for cls in capability.classes:
            mapped_class = await self._map_class_to_architecture(cls, architecture_mapping)
            if mapped_class:
                synthesized.synthesized_classes.append(mapped_class)
        
        # Create configuration changes
        for config_change in architecture_mapping.get('configuration_changes', []):
            synthesized.configuration_changes.append(config_change)
        
        # Set module mappings from architecture
        synthesized.cwmai_module_mappings.update(architecture_mapping.get('module_mappings', {}))
    
    async def _synthesize_hybrid_approach(self, synthesized: SynthesizedCapability, compatibility: Dict[str, Any]):
        """Synthesize using hybrid approach combining multiple strategies."""
        # Use direct adaptation for simple components
        simple_classes = [cls for cls in synthesized.original_capability.classes 
                         if len(cls.get('methods', [])) <= 5]
        for cls in simple_classes:
            adapted_class = self._adapt_class_to_cwmai(cls, compatibility)
            if adapted_class:
                synthesized.synthesized_classes.append(adapted_class)
        
        # Use pattern translation for complex components
        complex_classes = [cls for cls in synthesized.original_capability.classes 
                          if len(cls.get('methods', [])) > 5]
        for cls in complex_classes:
            transformed_class = await self._transform_class_with_patterns(cls, synthesized)
            if transformed_class:
                synthesized.synthesized_classes.append(transformed_class)
        
        # Use interface bridging where applicable
        if compatibility['interface_compatibility']:
            await self._synthesize_interface_bridging(synthesized, compatibility)
    
    async def _validate_synthesis(self, synthesized: SynthesizedCapability) -> Dict[str, Any]:
        """Validate the synthesized capability."""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        # Check synthesis confidence
        if synthesized.synthesis_confidence < self.config['min_confidence_threshold']:
            validation['valid'] = False
            validation['errors'].append(f"Synthesis confidence too low: {synthesized.synthesis_confidence}")
        
        # Check component consistency
        if not (synthesized.synthesized_classes or synthesized.synthesized_functions):
            validation['valid'] = False
            validation['errors'].append("No components were successfully synthesized")
        
        # Check naming conventions
        naming_issues = self._validate_naming_conventions(synthesized)
        if naming_issues:
            validation['warnings'].extend(naming_issues)
        
        # Check integration consistency
        integration_issues = self._validate_integration_consistency(synthesized)
        if integration_issues:
            validation['warnings'].extend(integration_issues)
        
        # Calculate quality metrics
        validation['quality_metrics'] = {
            'confidence': synthesized.synthesis_confidence,
            'quality_preservation': synthesized.quality_preservation,
            'architectural_alignment': synthesized.architectural_alignment,
            'component_count': len(synthesized.synthesized_classes) + len(synthesized.synthesized_functions)
        }
        
        return validation
    
    def _adapt_class_to_cwmai(self, cls: Dict[str, Any], compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adapt a class to CWMAI conventions."""
        try:
            adapted_class = cls.copy()
            
            # Apply CWMAI naming conventions
            adapted_class['name'] = self._apply_cwmai_naming(cls['name'])
            
            # Add CWMAI-style documentation
            if not adapted_class.get('docstring'):
                adapted_class['docstring'] = f"Adapted {cls['name']} for CWMAI integration"
            
            # Adapt methods to CWMAI patterns
            if 'methods' in adapted_class:
                adapted_methods = []
                for method in adapted_class['methods']:
                    adapted_method = self._adapt_method_to_cwmai(method)
                    if adapted_method:
                        adapted_methods.append(adapted_method)
                adapted_class['methods'] = adapted_methods
            
            # Add CWMAI integration hooks
            adapted_class['cwmai_integration'] = {
                'adapted_from': cls['name'],
                'integration_points': compatibility.get('integration_points', []),
                'requires_configuration': bool(compatibility.get('required_adaptations'))
            }
            
            return adapted_class
        
        except Exception as e:
            self.logger.error(f"Error adapting class {cls.get('name', 'unknown')}: {e}")
            return None
    
    def _adapt_function_to_cwmai(self, func: Dict[str, Any], compatibility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adapt a function to CWMAI conventions."""
        try:
            adapted_func = func.copy()
            
            # Apply CWMAI naming conventions
            adapted_func['name'] = self._apply_cwmai_naming(func['name'])
            
            # Add CWMAI-style documentation
            if not adapted_func.get('docstring'):
                adapted_func['docstring'] = f"Adapted {func['name']} for CWMAI integration"
            
            # Add async support if beneficial
            if self._should_make_async(func):
                adapted_func['is_async'] = True
                adapted_func['name'] = adapted_func['name'] if adapted_func['name'].startswith('async_') else f"async_{adapted_func['name']}"
            
            # Add CWMAI integration metadata
            adapted_func['cwmai_integration'] = {
                'adapted_from': func['name'],
                'async_converted': adapted_func.get('is_async', False),
                'integration_notes': self._generate_function_integration_notes(func)
            }
            
            return adapted_func
        
        except Exception as e:
            self.logger.error(f"Error adapting function {func.get('name', 'unknown')}: {e}")
            return None
    
    def _apply_cwmai_naming(self, name: str) -> str:
        """Apply CWMAI naming conventions."""
        # Convert to snake_case if not already
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        
        # Add CWMAI prefix if it's a potential conflict
        cwmai_names = self._get_cwmai_component_names()
        if snake_case in cwmai_names:
            snake_case = f"external_{snake_case}"
        
        return snake_case
    
    def _should_make_async(self, func: Dict[str, Any]) -> bool:
        """Determine if a function should be made async for CWMAI integration."""
        # Functions that typically benefit from async in CWMAI
        async_indicators = [
            'request', 'fetch', 'get', 'post', 'send', 'receive',
            'process', 'execute', 'run', 'perform', 'handle'
        ]
        
        func_name = func.get('name', '').lower()
        return any(indicator in func_name for indicator in async_indicators)
    
    def _resolve_naming_conflicts(self, synthesized: SynthesizedCapability, conflicts: List[str]):
        """Resolve naming conflicts in synthesized components."""
        for conflict in conflicts:
            # Rename conflicting classes
            for cls in synthesized.synthesized_classes:
                if cls['name'] == conflict:
                    cls['name'] = f"external_{conflict}"
                    cls['original_name'] = conflict
            
            # Rename conflicting functions
            for func in synthesized.synthesized_functions:
                if func['name'] == conflict:
                    func['name'] = f"external_{conflict}"
                    func['original_name'] = conflict
    
    def _create_interface_bridge(self, external_interface: str, cwmai_interface: str, adaptations: List[str]) -> Dict[str, Any]:
        """Create an interface bridge for compatibility."""
        return {
            'name': f"{external_interface}Bridge",
            'type': 'interface_bridge',
            'external_interface': external_interface,
            'cwmai_interface': cwmai_interface,
            'adaptations': adaptations,
            'methods': [
                {
                    'name': 'adapt_call',
                    'description': 'Adapt external interface calls to CWMAI interface',
                    'parameters': ['method_name', 'args', 'kwargs'],
                    'returns': 'adapted_result'
                },
                {
                    'name': 'convert_response',
                    'description': 'Convert CWMAI response to external format',
                    'parameters': ['cwmai_response'],
                    'returns': 'external_format_response'
                }
            ],
            'module': 'external_interfaces'
        }
    
    def _load_cwmai_patterns(self) -> List[Dict[str, Any]]:
        """Load CWMAI architectural patterns."""
        return [
            {
                'name': 'task_orchestration',
                'description': 'CWMAI task orchestration pattern',
                'components': ['TaskManager', 'DynamicSwarm', 'WorkflowExecutor'],
                'interfaces': ['execute_task', 'manage_workflow', 'coordinate_agents'],
                'conventions': ['async operations', 'state management', 'error handling']
            },
            {
                'name': 'ai_brain_pattern',
                'description': 'CWMAI AI brain coordination pattern',
                'components': ['AIBrain', 'HttpAIClient', 'ResponseCache'],
                'interfaces': ['generate_response', 'process_request', 'manage_context'],
                'conventions': ['retry logic', 'response validation', 'context preservation']
            },
            {
                'name': 'state_management',
                'description': 'CWMAI state management pattern',
                'components': ['StateManager', 'AsyncStateManager'],
                'interfaces': ['save_state', 'load_state', 'update_state'],
                'conventions': ['atomic operations', 'backup strategies', 'validation']
            },
            {
                'name': 'self_improvement',
                'description': 'CWMAI self-improvement pattern',
                'components': ['SafeSelfImprover', 'ResearchEvolutionEngine'],
                'interfaces': ['propose_improvement', 'analyze_impact', 'apply_modification'],
                'conventions': ['safety checks', 'rollback capability', 'validation']
            }
        ]
    
    def _load_cwmai_interfaces(self) -> List[Dict[str, Any]]:
        """Load CWMAI interface definitions."""
        return [
            {
                'name': 'TaskManagerInterface',
                'methods': ['create_task', 'execute_task', 'monitor_progress', 'handle_completion'],
                'async_methods': ['execute_task', 'monitor_progress'],
                'required_parameters': {'create_task': ['task_type', 'description']}
            },
            {
                'name': 'AIBrainInterface',
                'methods': ['generate_response', 'process_context', 'validate_response'],
                'async_methods': ['generate_response'],
                'required_parameters': {'generate_response': ['prompt', 'context']}
            },
            {
                'name': 'StateManagerInterface',
                'methods': ['save_state', 'load_state', 'update_state', 'backup_state'],
                'async_methods': ['backup_state'],
                'required_parameters': {'save_state': ['state_data']}
            }
        ]
    
    def _load_cwmai_conventions(self) -> Dict[str, Any]:
        """Load CWMAI coding conventions."""
        return {
            'naming': {
                'classes': 'PascalCase',
                'functions': 'snake_case',
                'constants': 'UPPER_SNAKE_CASE',
                'files': 'snake_case'
            },
            'async': {
                'use_async_for': ['io_operations', 'api_calls', 'long_running_tasks'],
                'error_handling': 'try_except_with_logging',
                'timeout_handling': 'asyncio_wait_for'
            },
            'documentation': {
                'required_for': ['public_methods', 'classes', 'modules'],
                'format': 'google_style',
                'include': ['parameters', 'returns', 'raises']
            },
            'error_handling': {
                'strategy': 'explicit_exception_handling',
                'logging': 'structured_logging',
                'recovery': 'graceful_degradation'
            }
        }
    
    def _initialize_synthesis_patterns(self) -> List[SynthesisPattern]:
        """Initialize common synthesis patterns."""
        return [
            SynthesisPattern(
                pattern_name='task_queue_to_orchestration',
                external_pattern='task_queue',
                cwmai_equivalent='task_orchestration',
                transformation_rules=[
                    {'from': 'Queue.enqueue', 'to': 'TaskManager.create_task'},
                    {'from': 'Queue.dequeue', 'to': 'TaskManager.execute_next_task'},
                    {'from': 'Queue.size', 'to': 'TaskManager.get_pending_count'}
                ],
                compatibility_score=0.8
            ),
            SynthesisPattern(
                pattern_name='simple_agent_to_ai_brain',
                external_pattern='simple_agent',
                cwmai_equivalent='ai_brain_pattern',
                transformation_rules=[
                    {'from': 'Agent.think', 'to': 'AIBrain.generate_response'},
                    {'from': 'Agent.act', 'to': 'AIBrain.execute_action'},
                    {'from': 'Agent.perceive', 'to': 'AIBrain.process_context'}
                ],
                compatibility_score=0.7
            ),
            SynthesisPattern(
                pattern_name='config_to_state_management',
                external_pattern='configuration_management',
                cwmai_equivalent='state_management',
                transformation_rules=[
                    {'from': 'Config.get', 'to': 'StateManager.load_state'},
                    {'from': 'Config.set', 'to': 'StateManager.update_state'},
                    {'from': 'Config.save', 'to': 'StateManager.save_state'}
                ],
                compatibility_score=0.9
            )
        ]
    
    def _calculate_pattern_similarity(self, external_pattern: Dict[str, Any], cwmai_pattern: Dict[str, Any]) -> float:
        """Calculate similarity between external and CWMAI patterns."""
        external_name = external_pattern.get('pattern_name', '').lower()
        cwmai_name = cwmai_pattern.get('name', '').lower()
        
        # Simple similarity based on name matching and component overlap
        name_similarity = len(set(external_name.split('_')) & set(cwmai_name.split('_'))) / max(1, len(external_name.split('_')))
        
        # Check for common concepts
        common_concepts = ['task', 'agent', 'orchestrat', 'coordinat', 'manag', 'execut', 'process']
        external_concepts = sum(1 for concept in common_concepts if concept in external_name)
        cwmai_concepts = sum(1 for concept in common_concepts if concept in cwmai_name)
        
        concept_similarity = min(external_concepts, cwmai_concepts) / max(1, max(external_concepts, cwmai_concepts))
        
        return (name_similarity + concept_similarity) / 2
    
    def _find_compatible_cwmai_interface(self, external_interface: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a compatible CWMAI interface for an external interface."""
        external_methods = set(external_interface.get('methods', []))
        
        best_match = None
        best_score = 0.0
        
        for cwmai_interface in self.cwmai_interfaces:
            cwmai_methods = set(cwmai_interface.get('methods', []))
            
            # Calculate method overlap
            overlap = len(external_methods & cwmai_methods)
            total = len(external_methods | cwmai_methods)
            
            if total > 0:
                score = overlap / total
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_match = cwmai_interface
        
        return best_match
    
    def _calculate_interface_compatibility(self, external_interface: Dict[str, Any], cwmai_interface: Dict[str, Any]) -> float:
        """Calculate compatibility score between interfaces."""
        external_methods = set(external_interface.get('methods', []))
        cwmai_methods = set(cwmai_interface.get('methods', []))
        
        if not external_methods or not cwmai_methods:
            return 0.0
        
        overlap = len(external_methods & cwmai_methods)
        total = len(external_methods | cwmai_methods)
        
        return overlap / total if total > 0 else 0.0
    
    def _identify_interface_adaptations(self, external_interface: Dict[str, Any], cwmai_interface: Dict[str, Any]) -> List[str]:
        """Identify required adaptations between interfaces."""
        adaptations = []
        
        external_methods = set(external_interface.get('methods', []))
        cwmai_methods = set(cwmai_interface.get('methods', []))
        
        # Methods that need to be added
        missing_methods = external_methods - cwmai_methods
        if missing_methods:
            adaptations.append(f"Add methods: {', '.join(missing_methods)}")
        
        # Methods that need parameter adaptation
        for method in external_methods & cwmai_methods:
            # This would require more detailed analysis in a real implementation
            adaptations.append(f"Adapt parameters for {method}")
        
        return adaptations
    
    def _get_cwmai_component_names(self) -> List[str]:
        """Get list of CWMAI component names."""
        return [
            'AIBrain', 'TaskManager', 'StateManager', 'ProductionOrchestrator',
            'SwarmIntelligence', 'DynamicSwarm', 'HTTPAIClient', 'SafeSelfImprover',
            'ResearchEvolutionEngine', 'ExternalAgentDiscoverer', 'CapabilityExtractor',
            'ExternalKnowledgeIntegrator', 'CapabilitySynthesizer'
        ]
    
    def _estimate_synthesis_complexity(self, capability: ExtractedCapability) -> float:
        """Estimate synthesis complexity for sorting."""
        factors = [
            len(capability.classes) / 10.0,
            len(capability.functions) / 20.0,
            len(capability.dependencies) / 15.0,
            1.0 if capability.integration_complexity == IntegrationComplexity.COMPLEX else 0.5
        ]
        return sum(factors) / len(factors)
    
    async def _learn_from_synthesis(self, synthesized: SynthesizedCapability):
        """Learn from synthesis results to improve future syntheses."""
        # Record successful patterns and strategies
        if synthesized.synthesis_confidence > 0.8:
            strategy = synthesized.synthesis_strategy.value
            self.logger.info(f"Successful synthesis using {strategy} strategy")
            
            # This could update synthesis patterns or strategies based on success
            # For now, just log the success
    
    def _calculate_synthesis_confidence(self, synthesized: SynthesizedCapability) -> float:
        """Calculate confidence in the synthesis result."""
        factors = []
        
        # Component success rate
        original_components = len(synthesized.original_capability.classes) + len(synthesized.original_capability.functions)
        synthesized_components = len(synthesized.synthesized_classes) + len(synthesized.synthesized_functions)
        
        if original_components > 0:
            component_success = synthesized_components / original_components
            factors.append(component_success)
        
        # Strategy appropriateness
        strategy_confidence = {
            SynthesisStrategy.DIRECT_ADAPTATION: 0.9,
            SynthesisStrategy.PATTERN_TRANSLATION: 0.8,
            SynthesisStrategy.INTERFACE_BRIDGING: 0.7,
            SynthesisStrategy.HYBRID_SYNTHESIS: 0.6,
            SynthesisStrategy.ARCHITECTURE_MAPPING: 0.5
        }
        factors.append(strategy_confidence.get(synthesized.synthesis_strategy, 0.5))
        
        # Integration point coverage
        if synthesized.cwmai_module_mappings:
            factors.append(0.8)
        else:
            factors.append(0.4)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _calculate_quality_preservation(self, synthesized: SynthesizedCapability) -> float:
        """Calculate how well the original quality was preserved."""
        original = synthesized.original_capability
        
        # Compare component counts
        original_count = len(original.classes) + len(original.functions)
        synthesized_count = len(synthesized.synthesized_classes) + len(synthesized.synthesized_functions)
        
        if original_count == 0:
            return 1.0
        
        # Basic preservation metric
        preservation = min(1.0, synthesized_count / original_count)
        
        # Adjust based on original quality
        if hasattr(original, 'code_quality_score'):
            preservation *= original.code_quality_score
        
        return preservation
    
    def _generate_implementation_notes(self, synthesized: SynthesizedCapability) -> List[str]:
        """Generate implementation notes for the synthesized capability."""
        notes = []
        
        if synthesized.synthesis_strategy == SynthesisStrategy.DIRECT_ADAPTATION:
            notes.append("Direct adaptation used - minimal changes required")
        elif synthesized.synthesis_strategy == SynthesisStrategy.PATTERN_TRANSLATION:
            notes.append("Pattern translation applied - verify pattern compliance")
        elif synthesized.synthesis_strategy == SynthesisStrategy.INTERFACE_BRIDGING:
            notes.append("Interface bridges created - test bridge functionality")
        
        if synthesized.cwmai_module_mappings:
            notes.append(f"Integration points: {', '.join(synthesized.cwmai_module_mappings.keys())}")
        
        if synthesized.required_imports:
            notes.append(f"Required imports: {len(synthesized.required_imports)} additional imports needed")
        
        return notes
    
    def _generate_testing_requirements(self, synthesized: SynthesizedCapability) -> List[str]:
        """Generate testing requirements for the synthesized capability."""
        requirements = [
            "Unit tests for all synthesized components",
            "Integration tests with CWMAI modules",
            "Performance impact testing"
        ]
        
        if synthesized.synthesized_interfaces:
            requirements.append("Interface bridge testing")
        
        if synthesized.configuration_changes:
            requirements.append("Configuration change validation")
        
        if synthesized.synthesis_strategy == SynthesisStrategy.ARCHITECTURE_MAPPING:
            requirements.append("Architecture compliance testing")
        
        return requirements
    
    def _generate_rollback_considerations(self, synthesized: SynthesizedCapability) -> List[str]:
        """Generate rollback considerations for the synthesized capability."""
        considerations = [
            "Backup original configuration before integration",
            "Test rollback procedure in safe environment"
        ]
        
        if synthesized.configuration_changes:
            considerations.append("Document all configuration changes for rollback")
        
        if synthesized.cwmai_module_mappings:
            considerations.append("Prepare module restoration scripts")
        
        return considerations
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return self.synthesis_stats.copy()
    
    def get_synthesis_patterns(self) -> List[SynthesisPattern]:
        """Get available synthesis patterns."""
        return self.synthesis_patterns.copy()


async def demonstrate_capability_synthesis():
    """Demonstrate capability synthesis."""
    print("=== Capability Synthesis Demo ===\n")
    
    # Create synthesizer
    synthesizer = CapabilitySynthesizer()
    
    print("Capability Synthesizer initialized")
    print(f"Available synthesis patterns: {len(synthesizer.synthesis_patterns)}")
    print(f"CWMAI patterns loaded: {len(synthesizer.cwmai_patterns)}")
    print(f"CWMAI interfaces loaded: {len(synthesizer.cwmai_interfaces)}")
    
    # Show synthesis statistics
    print("\n=== Synthesis Statistics ===")
    stats = synthesizer.get_synthesis_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nCapability Synthesizer ready for external capability adaptation")


if __name__ == "__main__":
    asyncio.run(demonstrate_capability_synthesis())