"""
Capability Extractor

Analyzes external AI agent repositories to extract reusable capabilities, patterns,
and code structures that can be safely integrated into CWMAI's architecture.
"""

import os
import ast
import json
import re
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# Import external agent discoverer types
from external_agent_discoverer import CapabilityType, RepositoryAnalysis


class ExtractionMethod(Enum):
    """Methods for extracting capabilities."""
    PATTERN_MATCHING = "pattern_matching"
    AST_ANALYSIS = "ast_analysis"
    INTERFACE_EXTRACTION = "interface_extraction"
    ARCHITECTURE_MAPPING = "architecture_mapping"
    DEPENDENCY_ANALYSIS = "dependency_analysis"


class IntegrationComplexity(Enum):
    """Complexity levels for integration."""
    SIMPLE = "simple"        # Copy-paste with minor modifications
    MODERATE = "moderate"    # Requires adaptation and integration
    COMPLEX = "complex"      # Significant refactoring needed
    IMPOSSIBLE = "impossible" # Cannot be integrated safely


@dataclass
class ExtractedCapability:
    """Represents an extracted capability from external code."""
    id: str
    name: str
    capability_type: CapabilityType
    description: str
    source_repository: str
    source_files: List[str]
    extraction_method: ExtractionMethod
    integration_complexity: IntegrationComplexity
    
    # Code components
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dependencies and requirements
    dependencies: List[str] = field(default_factory=list)
    external_apis: List[str] = field(default_factory=list)
    configuration_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Integration metadata
    cwmai_integration_points: List[str] = field(default_factory=list)
    required_modifications: List[str] = field(default_factory=list)
    compatibility_issues: List[str] = field(default_factory=list)
    performance_considerations: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    code_quality_score: float = 0.0
    test_coverage: float = 0.0
    documentation_score: float = 0.0
    security_score: float = 0.0
    
    # Extraction metadata
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Results from capability extraction process."""
    repository_url: str
    total_capabilities_found: int
    extracted_capabilities: List[ExtractedCapability]
    extraction_summary: Dict[str, Any]
    processing_time_seconds: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CapabilityExtractor:
    """Extracts reusable capabilities from external AI agent repositories."""
    
    def __init__(self):
        """Initialize the capability extractor."""
        self.logger = logging.getLogger(__name__)
        
        # Extraction patterns for different capability types
        self.capability_patterns = self._initialize_capability_patterns()
        
        # CWMAI architecture knowledge for integration assessment
        self.cwmai_architecture = self._load_cwmai_architecture_info()
        
        # Cache for extracted capabilities
        self.extraction_cache: Dict[str, ExtractionResult] = {}
        
        # Statistics
        self.extraction_stats = {
            'total_repositories_processed': 0,
            'total_capabilities_extracted': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_time_total': 0.0
        }
    
    async def extract_capabilities_from_repository(self, 
                                                  repository_analysis: RepositoryAnalysis,
                                                  repo_path: str) -> ExtractionResult:
        """Extract capabilities from a cloned repository.
        
        Args:
            repository_analysis: Analysis results from ExternalAgentDiscoverer
            repo_path: Path to cloned repository
            
        Returns:
            Extraction results with found capabilities
        """
        start_time = datetime.now()
        self.logger.info(f"Extracting capabilities from: {repository_analysis.name}")
        
        result = ExtractionResult(
            repository_url=repository_analysis.url,
            total_capabilities_found=0,
            extracted_capabilities=[],
            extraction_summary={},
            processing_time_seconds=0.0
        )
        
        try:
            # Extract capabilities using multiple methods
            capabilities = []
            
            # Method 1: Pattern-based extraction
            pattern_caps = await self._extract_via_patterns(repo_path, repository_analysis)
            capabilities.extend(pattern_caps)
            
            # Method 2: AST analysis
            ast_caps = await self._extract_via_ast_analysis(repo_path, repository_analysis)
            capabilities.extend(ast_caps)
            
            # Method 3: Interface extraction
            interface_caps = await self._extract_interfaces(repo_path, repository_analysis)
            capabilities.extend(interface_caps)
            
            # Method 4: Architecture mapping
            arch_caps = await self._extract_architecture_patterns(repo_path, repository_analysis)
            capabilities.extend(arch_caps)
            
            # Deduplicate and filter capabilities
            unique_capabilities = self._deduplicate_capabilities(capabilities)
            filtered_capabilities = self._filter_capabilities(unique_capabilities)
            
            # Assess integration complexity for each capability
            for capability in filtered_capabilities:
                await self._assess_integration_complexity(capability, repo_path)
                await self._assess_cwmai_compatibility(capability)
                self._calculate_quality_scores(capability)
            
            # Update result
            result.extracted_capabilities = filtered_capabilities
            result.total_capabilities_found = len(filtered_capabilities)
            
            # Generate extraction summary
            result.extraction_summary = self._generate_extraction_summary(filtered_capabilities)
            
            self.extraction_stats['successful_extractions'] += 1
            
        except Exception as e:
            self.logger.error(f"Error extracting capabilities from {repository_analysis.name}: {e}")
            result.errors.append(str(e))
            self.extraction_stats['failed_extractions'] += 1
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        result.processing_time_seconds = processing_time
        
        self.extraction_stats['total_repositories_processed'] += 1
        self.extraction_stats['total_capabilities_extracted'] += len(result.extracted_capabilities)
        self.extraction_stats['extraction_time_total'] += processing_time
        
        # Cache result
        self.extraction_cache[repository_analysis.url] = result
        
        self.logger.info(f"Extracted {len(result.extracted_capabilities)} capabilities from {repository_analysis.name}")
        
        return result
    
    async def extract_specific_capability(self, 
                                        repo_path: str,
                                        capability_type: CapabilityType,
                                        target_files: Optional[List[str]] = None) -> List[ExtractedCapability]:
        """Extract a specific type of capability from repository.
        
        Args:
            repo_path: Path to repository
            capability_type: Type of capability to extract
            target_files: Specific files to analyze (optional)
            
        Returns:
            List of extracted capabilities of the specified type
        """
        self.logger.info(f"Extracting {capability_type.value} capabilities")
        
        capabilities = []
        
        # Get extraction patterns for this capability type
        patterns = self.capability_patterns.get(capability_type, [])
        
        # Determine files to analyze
        if target_files:
            files_to_analyze = target_files
        else:
            files_to_analyze = self._find_relevant_files(repo_path, capability_type)
        
        # Extract from each file
        for file_path in files_to_analyze:
            try:
                file_capabilities = await self._extract_from_file(
                    os.path.join(repo_path, file_path),
                    capability_type,
                    patterns
                )
                capabilities.extend(file_capabilities)
                
            except Exception as e:
                self.logger.warning(f"Error extracting from {file_path}: {e}")
        
        return capabilities
    
    async def _extract_via_patterns(self, 
                                  repo_path: str,
                                  repository_analysis: RepositoryAnalysis) -> List[ExtractedCapability]:
        """Extract capabilities using pattern matching."""
        capabilities = []
        
        for capability_type in repository_analysis.capabilities:
            try:
                type_capabilities = await self.extract_specific_capability(
                    repo_path, capability_type
                )
                capabilities.extend(type_capabilities)
                
            except Exception as e:
                self.logger.warning(f"Pattern extraction error for {capability_type.value}: {e}")
        
        return capabilities
    
    async def _extract_via_ast_analysis(self,
                                      repo_path: str,
                                      repository_analysis: RepositoryAnalysis) -> List[ExtractedCapability]:
        """Extract capabilities using AST analysis."""
        capabilities = []
        
        # Find Python files for AST analysis
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # Analyze key files first
        key_python_files = [f for f in python_files if any(
            keyword in os.path.basename(f).lower() 
            for keyword in ['agent', 'task', 'orchestrat', 'coordinat', 'manag', 'brain', 'swarm']
        )]
        
        # Limit analysis to prevent excessive processing
        files_to_analyze = key_python_files[:10] + python_files[:20]
        files_to_analyze = list(set(files_to_analyze))  # Remove duplicates
        
        for file_path in files_to_analyze:
            try:
                file_capabilities = await self._analyze_file_ast(file_path, repository_analysis)
                capabilities.extend(file_capabilities)
                
            except Exception as e:
                self.logger.debug(f"AST analysis error for {file_path}: {e}")
        
        return capabilities
    
    async def _extract_interfaces(self,
                                repo_path: str,
                                repository_analysis: RepositoryAnalysis) -> List[ExtractedCapability]:
        """Extract interface patterns and API designs."""
        capabilities = []
        
        # Look for interface-like patterns
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if any(keyword in file.lower() for keyword in ['interface', 'api', 'protocol', 'abc']):
                        python_files.append(file_path)
        
        for file_path in python_files[:5]:  # Limit to 5 interface files
            try:
                interfaces = await self._extract_interfaces_from_file(file_path)
                
                if interfaces:
                    capability = ExtractedCapability(
                        id=self._generate_capability_id(file_path, "interface"),
                        name=f"Interface patterns from {os.path.basename(file_path)}",
                        capability_type=CapabilityType.API_INTEGRATION,
                        description="Extracted interface patterns and API designs",
                        source_repository=repository_analysis.url,
                        source_files=[os.path.relpath(file_path, repo_path)],
                        extraction_method=ExtractionMethod.INTERFACE_EXTRACTION,
                        integration_complexity=IntegrationComplexity.MODERATE,
                        interfaces=interfaces
                    )
                    capabilities.append(capability)
                    
            except Exception as e:
                self.logger.debug(f"Interface extraction error for {file_path}: {e}")
        
        return capabilities
    
    async def _extract_architecture_patterns(self,
                                           repo_path: str,
                                           repository_analysis: RepositoryAnalysis) -> List[ExtractedCapability]:
        """Extract architectural patterns and design structures."""
        capabilities = []
        
        # Analyze architecture patterns found in the repository
        for pattern in repository_analysis.architecture_patterns:
            try:
                pattern_capability = await self._extract_architecture_pattern(
                    repo_path, pattern, repository_analysis
                )
                
                if pattern_capability:
                    capabilities.append(pattern_capability)
                    
            except Exception as e:
                self.logger.debug(f"Architecture pattern extraction error for {pattern}: {e}")
        
        return capabilities
    
    async def _analyze_file_ast(self, file_path: str, repository_analysis: RepositoryAnalysis) -> List[ExtractedCapability]:
        """Analyze a Python file using AST."""
        capabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Extract classes and functions
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, source)
                    classes.append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, source)
                    functions.append(func_info)
            
            # Create capabilities based on found patterns
            if classes or functions:
                # Determine capability type based on file content
                capability_type = self._infer_capability_type_from_ast(classes, functions, source)
                
                if capability_type:
                    capability = ExtractedCapability(
                        id=self._generate_capability_id(file_path, "ast"),
                        name=f"Code structures from {os.path.basename(file_path)}",
                        capability_type=capability_type,
                        description=f"Extracted classes and functions for {capability_type.value}",
                        source_repository=repository_analysis.url,
                        source_files=[os.path.relpath(file_path, os.path.dirname(file_path))],
                        extraction_method=ExtractionMethod.AST_ANALYSIS,
                        integration_complexity=IntegrationComplexity.MODERATE,
                        classes=classes,
                        functions=functions
                    )
                    capabilities.append(capability)
        
        except Exception as e:
            self.logger.debug(f"AST analysis failed for {file_path}: {e}")
        
        return capabilities
    
    async def _extract_from_file(self,
                               file_path: str,
                               capability_type: CapabilityType,
                               patterns: List[Dict[str, Any]]) -> List[ExtractedCapability]:
        """Extract capabilities from a specific file using patterns."""
        capabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Apply patterns to find capability implementations
            for pattern in patterns:
                matches = self._apply_pattern(content, pattern)
                
                if matches:
                    capability = ExtractedCapability(
                        id=self._generate_capability_id(file_path, pattern.get('name', 'pattern')),
                        name=f"{pattern.get('name', 'Pattern')} from {os.path.basename(file_path)}",
                        capability_type=capability_type,
                        description=pattern.get('description', 'Pattern-based capability'),
                        source_repository="",  # Will be set by caller
                        source_files=[os.path.basename(file_path)],
                        extraction_method=ExtractionMethod.PATTERN_MATCHING,
                        integration_complexity=IntegrationComplexity.SIMPLE,
                        patterns=[{
                            'pattern_name': pattern.get('name', 'Unknown'),
                            'matches': matches,
                            'pattern_type': pattern.get('type', 'regex')
                        }]
                    )
                    capabilities.append(capability)
        
        except Exception as e:
            self.logger.debug(f"Pattern extraction failed for {file_path}: {e}")
        
        return capabilities
    
    async def _extract_interfaces_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract interface definitions from a file."""
        interfaces = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's an interface-like class
                    if self._is_interface_class(node):
                        interface_info = {
                            'name': node.name,
                            'methods': [],
                            'properties': [],
                            'docstring': ast.get_docstring(node) or "",
                            'line_number': node.lineno
                        }
                        
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                interface_info['methods'].append({
                                    'name': item.name,
                                    'args': [arg.arg for arg in item.args.args],
                                    'docstring': ast.get_docstring(item) or "",
                                    'is_abstract': self._is_abstract_method(item)
                                })
                        
                        interfaces.append(interface_info)
        
        except Exception as e:
            self.logger.debug(f"Interface extraction failed for {file_path}: {e}")
        
        return interfaces
    
    async def _extract_architecture_pattern(self,
                                          repo_path: str,
                                          pattern_name: str,
                                          repository_analysis: RepositoryAnalysis) -> Optional[ExtractedCapability]:
        """Extract a specific architecture pattern."""
        # Map pattern names to extraction strategies
        pattern_extractors = {
            'plugin_architecture': self._extract_plugin_pattern,
            'pipeline_pattern': self._extract_pipeline_pattern,
            'observer_pattern': self._extract_observer_pattern,
            'strategy_pattern': self._extract_strategy_pattern,
            'factory_pattern': self._extract_factory_pattern,
            'microservices': self._extract_microservices_pattern
        }
        
        extractor = pattern_extractors.get(pattern_name)
        if not extractor:
            return None
        
        try:
            pattern_data = await extractor(repo_path)
            
            if pattern_data:
                capability = ExtractedCapability(
                    id=self._generate_capability_id(repo_path, pattern_name),
                    name=f"{pattern_name.replace('_', ' ').title()} Pattern",
                    capability_type=CapabilityType.TASK_ORCHESTRATION,  # Default, may be overridden
                    description=f"Extracted {pattern_name} architectural pattern",
                    source_repository=repository_analysis.url,
                    source_files=pattern_data.get('source_files', []),
                    extraction_method=ExtractionMethod.ARCHITECTURE_MAPPING,
                    integration_complexity=IntegrationComplexity.COMPLEX,
                    patterns=[pattern_data]
                )
                return capability
        
        except Exception as e:
            self.logger.debug(f"Pattern extraction failed for {pattern_name}: {e}")
        
        return None
    
    async def _assess_integration_complexity(self, capability: ExtractedCapability, repo_path: str):
        """Assess how complex it would be to integrate this capability."""
        complexity_factors = []
        
        # Check dependencies
        dependencies = await self._analyze_dependencies(capability, repo_path)
        capability.dependencies = dependencies
        
        if len(dependencies) > 10:
            complexity_factors.append("High dependency count")
        
        # Check external API usage
        external_apis = self._find_external_apis(capability)
        capability.external_apis = external_apis
        
        if external_apis:
            complexity_factors.append("External API dependencies")
        
        # Check for complex patterns
        if capability.extraction_method == ExtractionMethod.ARCHITECTURE_MAPPING:
            complexity_factors.append("Complex architectural pattern")
        
        # Assess code complexity
        code_complexity = self._assess_code_complexity(capability)
        if code_complexity > 0.7:
            complexity_factors.append("High code complexity")
        
        # Set integration complexity
        if len(complexity_factors) == 0:
            capability.integration_complexity = IntegrationComplexity.SIMPLE
        elif len(complexity_factors) <= 2:
            capability.integration_complexity = IntegrationComplexity.MODERATE
        else:
            capability.integration_complexity = IntegrationComplexity.COMPLEX
        
        capability.compatibility_issues = complexity_factors
    
    async def _assess_cwmai_compatibility(self, capability: ExtractedCapability):
        """Assess compatibility with CWMAI architecture."""
        integration_points = []
        modifications = []
        
        # Check for CWMAI integration opportunities
        if capability.capability_type == CapabilityType.TASK_ORCHESTRATION:
            integration_points.extend(['task_manager.py', 'dynamic_swarm.py'])
            modifications.append("Adapt to CWMAI task management interface")
        
        elif capability.capability_type == CapabilityType.MULTI_AGENT_COORDINATION:
            integration_points.extend(['swarm_intelligence.py', 'multi_repo_coordinator.py'])
            modifications.append("Integrate with existing swarm coordination")
        
        elif capability.capability_type == CapabilityType.PERFORMANCE_OPTIMIZATION:
            integration_points.extend(['ai_brain.py', 'http_ai_client.py'])
            modifications.append("Adapt to CWMAI performance monitoring")
        
        elif capability.capability_type == CapabilityType.ERROR_HANDLING:
            integration_points.extend(['production_orchestrator.py', 'state_manager.py'])
            modifications.append("Integrate with existing error handling")
        
        # Check for naming conflicts
        cwmai_classes = self._get_cwmai_class_names()
        capability_classes = [c['name'] for c in capability.classes]
        
        conflicts = set(cwmai_classes) & set(capability_classes)
        if conflicts:
            modifications.append(f"Rename conflicting classes: {list(conflicts)}")
        
        capability.cwmai_integration_points = integration_points
        capability.required_modifications = modifications
    
    def _calculate_quality_scores(self, capability: ExtractedCapability):
        """Calculate quality scores for the capability."""
        # Code quality based on structure and documentation
        quality_score = 0.5  # Base score
        
        # Documentation score
        doc_score = 0.0
        total_items = len(capability.classes) + len(capability.functions)
        
        if total_items > 0:
            documented_items = 0
            
            for cls in capability.classes:
                if cls.get('docstring'):
                    documented_items += 1
            
            for func in capability.functions:
                if func.get('docstring'):
                    documented_items += 1
            
            doc_score = documented_items / total_items
        
        # Complexity assessment
        complexity_penalty = 0.0
        if capability.integration_complexity == IntegrationComplexity.COMPLEX:
            complexity_penalty = 0.3
        elif capability.integration_complexity == IntegrationComplexity.MODERATE:
            complexity_penalty = 0.1
        
        # Update scores
        capability.code_quality_score = max(0.0, quality_score - complexity_penalty)
        capability.documentation_score = doc_score
        capability.security_score = 0.8  # Default, would need deeper analysis
        capability.extraction_confidence = quality_score + doc_score * 0.3
    
    def _deduplicate_capabilities(self, capabilities: List[ExtractedCapability]) -> List[ExtractedCapability]:
        """Remove duplicate capabilities."""
        seen_ids = set()
        unique_capabilities = []
        
        for capability in capabilities:
            # Create a content-based ID for deduplication
            content_id = hashlib.md5(
                f"{capability.name}_{capability.capability_type.value}_{len(capability.classes)}_{len(capability.functions)}".encode()
            ).hexdigest()[:8]
            
            if content_id not in seen_ids:
                seen_ids.add(content_id)
                unique_capabilities.append(capability)
        
        return unique_capabilities
    
    def _filter_capabilities(self, capabilities: List[ExtractedCapability]) -> List[ExtractedCapability]:
        """Filter capabilities based on quality and relevance."""
        filtered = []
        
        for capability in capabilities:
            # Filter out low-quality capabilities
            if capability.extraction_confidence < 0.3:
                continue
            
            # Filter out capabilities that are too complex to integrate
            if capability.integration_complexity == IntegrationComplexity.IMPOSSIBLE:
                continue
            
            # Must have some extractable content
            if not (capability.classes or capability.functions or capability.patterns):
                continue
            
            filtered.append(capability)
        
        return filtered
    
    def _generate_extraction_summary(self, capabilities: List[ExtractedCapability]) -> Dict[str, Any]:
        """Generate summary of extraction results."""
        if not capabilities:
            return {'total_capabilities': 0, 'extraction_methods': {}, 'capability_types': {}}
        
        # Count by extraction method
        method_counts = {}
        for cap in capabilities:
            method = cap.extraction_method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Count by capability type
        type_counts = {}
        for cap in capabilities:
            cap_type = cap.capability_type.value
            type_counts[cap_type] = type_counts.get(cap_type, 0) + 1
        
        # Count by integration complexity
        complexity_counts = {}
        for cap in capabilities:
            complexity = cap.integration_complexity.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Calculate average scores
        avg_confidence = sum(cap.extraction_confidence for cap in capabilities) / len(capabilities)
        avg_quality = sum(cap.code_quality_score for cap in capabilities) / len(capabilities)
        
        return {
            'total_capabilities': len(capabilities),
            'extraction_methods': method_counts,
            'capability_types': type_counts,
            'integration_complexity': complexity_counts,
            'average_confidence': avg_confidence,
            'average_quality': avg_quality,
            'high_confidence_count': len([c for c in capabilities if c.extraction_confidence > 0.7]),
            'simple_integration_count': len([c for c in capabilities if c.integration_complexity == IntegrationComplexity.SIMPLE])
        }
    
    # Helper methods for pattern matching and analysis
    
    def _initialize_capability_patterns(self) -> Dict[CapabilityType, List[Dict[str, Any]]]:
        """Initialize capability detection patterns."""
        return {
            CapabilityType.TASK_ORCHESTRATION: [
                {
                    'name': 'task_queue_pattern',
                    'description': 'Task queue implementation',
                    'type': 'regex',
                    'patterns': [r'class.*Queue.*:', r'def.*enqueue.*:', r'def.*dequeue.*:']
                },
                {
                    'name': 'workflow_pattern',
                    'description': 'Workflow orchestration',
                    'type': 'regex',
                    'patterns': [r'class.*Workflow.*:', r'def.*execute.*workflow.*:', r'class.*Pipeline.*:']
                }
            ],
            CapabilityType.MULTI_AGENT_COORDINATION: [
                {
                    'name': 'agent_communication',
                    'description': 'Agent communication patterns',
                    'type': 'regex',
                    'patterns': [r'def.*send_message.*:', r'def.*broadcast.*:', r'class.*AgentNetwork.*:']
                }
            ],
            CapabilityType.PERFORMANCE_OPTIMIZATION: [
                {
                    'name': 'caching_pattern',
                    'description': 'Caching mechanisms',
                    'type': 'regex',
                    'patterns': [r'@cache', r'@lru_cache', r'class.*Cache.*:']
                }
            ]
        }
    
    def _load_cwmai_architecture_info(self) -> Dict[str, Any]:
        """Load CWMAI architecture information for compatibility assessment."""
        return {
            'core_modules': [
                'ai_brain.py', 'task_manager.py', 'state_manager.py',
                'production_orchestrator.py', 'swarm_intelligence.py'
            ],
            'integration_points': {
                'task_management': ['task_manager.py', 'dynamic_swarm.py'],
                'ai_coordination': ['ai_brain.py', 'swarm_intelligence.py'],
                'performance': ['production_orchestrator.py', 'http_ai_client.py'],
                'state_management': ['state_manager.py', 'async_state_manager.py']
            }
        }
    
    def _find_relevant_files(self, repo_path: str, capability_type: CapabilityType) -> List[str]:
        """Find files relevant to a specific capability type."""
        relevant_files = []
        
        # Keywords for different capability types
        type_keywords = {
            CapabilityType.TASK_ORCHESTRATION: ['task', 'queue', 'workflow', 'pipeline', 'orchestrat'],
            CapabilityType.MULTI_AGENT_COORDINATION: ['agent', 'coordinat', 'communication', 'network'],
            CapabilityType.SWARM_INTELLIGENCE: ['swarm', 'collective', 'emergent'],
            CapabilityType.PERFORMANCE_OPTIMIZATION: ['cache', 'optimize', 'performance', 'speed'],
            CapabilityType.ERROR_HANDLING: ['error', 'exception', 'retry', 'recovery']
        }
        
        keywords = type_keywords.get(capability_type, [])
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_lower = file.lower()
                    if any(keyword in file_lower for keyword in keywords):
                        relevant_files.append(os.path.relpath(os.path.join(root, file), repo_path))
        
        return relevant_files[:10]  # Limit to 10 most relevant files
    
    def _apply_pattern(self, content: str, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a pattern to content and return matches."""
        matches = []
        
        if pattern.get('type') == 'regex':
            for regex_pattern in pattern.get('patterns', []):
                for match in re.finditer(regex_pattern, content, re.MULTILINE | re.IGNORECASE):
                    matches.append({
                        'match': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'line': content[:match.start()].count('\n') + 1
                    })
        
        return matches
    
    def _extract_class_info(self, node: ast.ClassDef, source: str) -> Dict[str, Any]:
        """Extract information from a class definition."""
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'line_number': node.lineno,
            'methods': [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
            'base_classes': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
        }
    
    def _extract_function_info(self, node: ast.FunctionDef, source: str) -> Dict[str, Any]:
        """Extract information from a function definition."""
        return {
            'name': node.name,
            'docstring': ast.get_docstring(node) or "",
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
    
    def _infer_capability_type_from_ast(self, 
                                      classes: List[Dict[str, Any]], 
                                      functions: List[Dict[str, Any]], 
                                      source: str) -> Optional[CapabilityType]:
        """Infer capability type from AST analysis."""
        source_lower = source.lower()
        
        # Check for task orchestration patterns
        if any('task' in cls['name'].lower() or 'queue' in cls['name'].lower() for cls in classes):
            return CapabilityType.TASK_ORCHESTRATION
        
        # Check for agent coordination patterns
        if any('agent' in cls['name'].lower() or 'coordinat' in cls['name'].lower() for cls in classes):
            return CapabilityType.MULTI_AGENT_COORDINATION
        
        # Check for performance optimization patterns
        if 'cache' in source_lower or '@lru_cache' in source_lower:
            return CapabilityType.PERFORMANCE_OPTIMIZATION
        
        # Check for error handling patterns
        if 'exception' in source_lower or 'retry' in source_lower:
            return CapabilityType.ERROR_HANDLING
        
        # Default based on common patterns
        if classes and functions:
            return CapabilityType.TASK_ORCHESTRATION
        
        return None
    
    def _is_interface_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is interface-like."""
        # Check for ABC inheritance
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ['ABC', 'Protocol']:
                return True
        
        # Check for abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for decorator in item.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        return True
        
        return False
    
    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if a method is abstract."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                return True
        return False
    
    # Architecture pattern extractors
    
    async def _extract_plugin_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract plugin architecture pattern."""
        # Look for plugin-related files and structures
        plugin_files = []
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if 'plugin' in file.lower() and file.endswith('.py'):
                    plugin_files.append(os.path.relpath(os.path.join(root, file), repo_path))
        
        if plugin_files:
            return {
                'pattern_name': 'plugin_architecture',
                'source_files': plugin_files,
                'description': 'Plugin architecture implementation',
                'components': ['plugin_loader', 'plugin_interface', 'plugin_registry']
            }
        
        return None
    
    async def _extract_pipeline_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract pipeline pattern."""
        pipeline_files = []
        
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if 'pipeline' in file.lower() and file.endswith('.py'):
                    pipeline_files.append(os.path.relpath(os.path.join(root, file), repo_path))
        
        if pipeline_files:
            return {
                'pattern_name': 'pipeline_pattern',
                'source_files': pipeline_files,
                'description': 'Data/task pipeline implementation',
                'components': ['pipeline_stage', 'pipeline_executor', 'data_transformer']
            }
        
        return None
    
    async def _extract_observer_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract observer pattern."""
        # Look for observer-related code patterns
        return None  # Placeholder
    
    async def _extract_strategy_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract strategy pattern."""
        return None  # Placeholder
    
    async def _extract_factory_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract factory pattern."""
        return None  # Placeholder
    
    async def _extract_microservices_pattern(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """Extract microservices pattern."""
        return None  # Placeholder
    
    # Analysis helper methods
    
    async def _analyze_dependencies(self, capability: ExtractedCapability, repo_path: str) -> List[str]:
        """Analyze dependencies for a capability."""
        dependencies = set()
        
        # Check source files for imports
        for source_file in capability.source_files:
            file_path = os.path.join(repo_path, source_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract imports
                    import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
                    
                    for import_line in import_lines:
                        # Parse import statements
                        if import_line.startswith('import '):
                            module = import_line.split('import ')[1].split()[0].split('.')[0]
                            dependencies.add(module)
                        elif import_line.startswith('from '):
                            module = import_line.split('from ')[1].split()[0].split('.')[0]
                            dependencies.add(module)
                
                except Exception as e:
                    self.logger.debug(f"Error analyzing dependencies in {file_path}: {e}")
        
        # Filter out standard library modules
        stdlib_modules = {'os', 'sys', 'json', 'time', 'datetime', 're', 'collections', 'itertools'}
        external_deps = [dep for dep in dependencies if dep not in stdlib_modules]
        
        return external_deps
    
    def _find_external_apis(self, capability: ExtractedCapability) -> List[str]:
        """Find external API calls in capability code."""
        apis = []
        
        # Look for common API patterns in functions and classes
        for func in capability.functions:
            if 'http' in func['name'].lower() or 'request' in func['name'].lower():
                apis.append(f"HTTP API in function {func['name']}")
        
        for cls in capability.classes:
            if 'client' in cls['name'].lower() or 'api' in cls['name'].lower():
                apis.append(f"API client class {cls['name']}")
        
        return apis
    
    def _assess_code_complexity(self, capability: ExtractedCapability) -> float:
        """Assess code complexity of a capability."""
        complexity_score = 0.0
        
        # Base complexity on number of classes and functions
        total_items = len(capability.classes) + len(capability.functions)
        
        if total_items == 0:
            return 0.0
        
        # Calculate complexity based on structure
        class_complexity = sum(len(cls.get('methods', [])) for cls in capability.classes)
        function_complexity = len(capability.functions)
        
        complexity_score = (class_complexity + function_complexity) / (total_items * 5)  # Normalize
        
        return min(1.0, complexity_score)
    
    def _get_cwmai_class_names(self) -> List[str]:
        """Get list of existing CWMAI class names to check for conflicts."""
        # This would be populated by scanning CWMAI source files
        return [
            'AIBrain', 'TaskManager', 'StateManager', 'ProductionOrchestrator',
            'SwarmIntelligence', 'DynamicSwarm', 'HTTPAIClient', 'SafeSelfImprover'
        ]
    
    def _generate_capability_id(self, source: str, prefix: str) -> str:
        """Generate unique ID for a capability."""
        content = f"{source}_{prefix}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        if self.extraction_stats['total_repositories_processed'] == 0:
            return self.extraction_stats
        
        avg_time = self.extraction_stats['extraction_time_total'] / self.extraction_stats['total_repositories_processed']
        success_rate = self.extraction_stats['successful_extractions'] / self.extraction_stats['total_repositories_processed']
        
        return {
            **self.extraction_stats,
            'average_extraction_time': avg_time,
            'success_rate': success_rate,
            'capabilities_per_repository': self.extraction_stats['total_capabilities_extracted'] / max(1, self.extraction_stats['successful_extractions'])
        }


async def demonstrate_capability_extraction():
    """Demonstrate capability extraction."""
    print("=== Capability Extraction Demo ===\n")
    
    # Create capability extractor
    extractor = CapabilityExtractor()
    
    # Create sample repository analysis (would come from ExternalAgentDiscoverer)
    sample_analysis = RepositoryAnalysis(
        url="https://github.com/example/ai-agent",
        name="example-ai-agent",
        description="Sample AI agent repository",
        language="Python",
        stars=100,
        forks=20,
        last_updated="2024-01-01",
        health_score=0.8,
        capabilities=[CapabilityType.TASK_ORCHESTRATION, CapabilityType.MULTI_AGENT_COORDINATION],
        architecture_patterns=['plugin_architecture', 'pipeline_pattern'],
        key_files=[],
        integration_difficulty=0.5,
        license="MIT",
        documentation_quality=0.7,
        test_coverage=0.6,
        performance_indicators={},
        security_assessment={},
        compatibility_score=0.8
    )
    
    # Simulate extraction (would normally use real repository path)
    print("Note: This is a demonstration using simulated data")
    print("In real usage, would extract from cloned repository:")
    print(f"Repository: {sample_analysis.name}")
    print(f"Capabilities detected: {[c.value for c in sample_analysis.capabilities]}")
    print(f"Architecture patterns: {sample_analysis.architecture_patterns}")
    
    # Show extraction statistics
    print("\n=== Extraction Statistics ===")
    stats = extractor.get_extraction_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(demonstrate_capability_extraction())