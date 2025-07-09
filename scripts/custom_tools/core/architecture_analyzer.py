"""
Architecture Analyzer

Unified tool for all architecture operations including checking, ensuring,
generating, and documenting system architecture. Consolidates 8 architecture tools.

Capabilities:
- Check for architecture files and completeness
- Ensure proper architecture documentation exists
- Generate architecture diagrams and documentation
- Analyze architecture quality and patterns
- Update and maintain architecture docs
- Validate architecture decisions
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
import re

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class ArchitectureAnalyzer:
    """Comprehensive architecture analysis and management system."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.project_root = Path(__file__).parent.parent.parent
        
        # Common architecture files
        self.architecture_files = {
            'readme': ['README.md', 'readme.md', 'Readme.md'],
            'architecture': ['ARCHITECTURE.md', 'architecture.md', 'docs/ARCHITECTURE.md'],
            'design': ['DESIGN.md', 'design.md', 'docs/design.md'],
            'api': ['API.md', 'api.md', 'docs/API.md', 'openapi.yaml', 'swagger.yaml'],
            'database': ['DATABASE.md', 'database.md', 'schema.sql', 'migrations/'],
            'deployment': ['DEPLOYMENT.md', 'deployment.md', 'docker-compose.yml', 'Dockerfile'],
            'security': ['SECURITY.md', 'security.md', 'docs/security.md'],
            'contributing': ['CONTRIBUTING.md', 'contributing.md'],
            'changelog': ['CHANGELOG.md', 'changelog.md', 'HISTORY.md']
        }
        
        # Architecture patterns
        self.patterns = {
            'mvc': ['model', 'view', 'controller'],
            'microservices': ['service', 'api', 'gateway'],
            'layered': ['presentation', 'business', 'data', 'layer'],
            'hexagonal': ['port', 'adapter', 'domain'],
            'event_driven': ['event', 'handler', 'subscriber', 'publisher'],
            'clean': ['entity', 'usecase', 'interface', 'infrastructure']
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute architecture operation."""
        operations = {
            'check': self.check_architecture,
            'ensure': self.ensure_architecture,
            'analyze': self.analyze_architecture,
            'generate': self.generate_architecture,
            'update': self.update_architecture,
            'validate': self.validate_architecture,
            'completeness': self.check_completeness,
            'quality': self.assess_quality,
            'patterns': self.detect_patterns,
            'business': self.generate_business_architecture
        }
        
        handler = operations.get(operation)
        if not handler:
            # Fuzzy match
            for op, func in operations.items():
                if op in operation or operation in op:
                    handler = func
                    break
                    
        if not handler:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": list(operations.keys())
            }
            
        return await handler(**kwargs)
        
    async def check_architecture(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Check for architecture files and documentation."""
        check_path = Path(path) if path else self.project_root
        
        if not check_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        found_files = {}
        missing_files = []
        
        for category, files in self.architecture_files.items():
            found = False
            for file_name in files:
                file_path = check_path / file_name
                if file_path.exists():
                    found_files[category] = {
                        'file': str(file_path.relative_to(check_path)),
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    found = True
                    break
                    
            if not found:
                missing_files.append(category)
                
        # Calculate completeness score
        total_categories = len(self.architecture_files)
        found_categories = len(found_files)
        completeness = round((found_categories / total_categories) * 100, 1)
        
        return {
            "success": True,
            "path": str(check_path),
            "completeness_score": completeness,
            "found_files": found_files,
            "missing_categories": missing_files,
            "recommendations": self._generate_architecture_recommendations(missing_files)
        }
        
    async def ensure_architecture(self, path: str = None, create_missing: bool = True, **kwargs) -> Dict[str, Any]:
        """Ensure architecture documentation exists."""
        check_result = await self.check_architecture(path)
        
        if check_result.get('error'):
            return check_result
            
        actions_taken = []
        
        if create_missing and check_result['missing_categories']:
            base_path = Path(path) if path else self.project_root
            
            for category in check_result['missing_categories']:
                # Determine what to create
                if category == 'readme':
                    content = await self._generate_readme(base_path)
                    file_path = base_path / 'README.md'
                elif category == 'architecture':
                    content = await self._generate_architecture_doc(base_path)
                    file_path = base_path / 'ARCHITECTURE.md'
                else:
                    content = self._get_template_for_category(category)
                    file_path = base_path / f'{category.upper()}.md'
                    
                actions_taken.append({
                    'category': category,
                    'action': 'would_create',
                    'file': str(file_path.relative_to(base_path)),
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                })
                
        return {
            "success": True,
            "current_completeness": check_result['completeness_score'],
            "actions_proposed": len(actions_taken),
            "actions": actions_taken,
            "new_completeness": min(100, check_result['completeness_score'] + len(actions_taken) * 10)
        }
        
    async def analyze_architecture(self, path: str = None, deep: bool = False, **kwargs) -> Dict[str, Any]:
        """Analyze architecture quality and patterns."""
        analyze_path = Path(path) if path else self.project_root
        
        # Check what exists
        check_result = await self.check_architecture(str(analyze_path))
        
        # Analyze code structure
        structure_analysis = self._analyze_code_structure(analyze_path)
        
        # Detect patterns
        detected_patterns = await self.detect_patterns(str(analyze_path))
        
        # Quality assessment
        quality_score = await self.assess_quality(str(analyze_path))
        
        analysis = {
            "documentation_completeness": check_result['completeness_score'],
            "structure": structure_analysis,
            "patterns": detected_patterns['patterns'],
            "quality_score": quality_score['score'],
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Identify strengths and weaknesses
        if check_result['completeness_score'] > 80:
            analysis['strengths'].append("Good documentation coverage")
        else:
            analysis['weaknesses'].append("Incomplete documentation")
            
        if len(detected_patterns['patterns']) > 0:
            analysis['strengths'].append(f"Clear architecture pattern: {detected_patterns['patterns'][0]}")
        else:
            analysis['weaknesses'].append("No clear architecture pattern detected")
            
        # Deep analysis with AI
        if deep:
            ai_analysis = await self._get_ai_architecture_analysis(analyze_path, analysis)
            analysis['ai_insights'] = ai_analysis
            
        return {
            "success": True,
            "path": str(analyze_path),
            "analysis": analysis
        }
        
    async def generate_architecture(self, path: str = None, style: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Generate architecture documentation."""
        gen_path = Path(path) if path else self.project_root
        
        # Analyze existing code
        structure = self._analyze_code_structure(gen_path)
        patterns = await self.detect_patterns(str(gen_path))
        
        # Generate based on style
        if style == 'comprehensive':
            content = await self._generate_comprehensive_architecture(gen_path, structure, patterns)
        elif style == 'minimal':
            content = await self._generate_minimal_architecture(gen_path, structure)
        elif style == 'business':
            content = await self.generate_business_architecture(str(gen_path))
        else:
            content = await self._generate_architecture_doc(gen_path)
            
        return {
            "success": True,
            "style": style,
            "content": content,
            "file_path": "ARCHITECTURE.md",
            "sections": self._extract_sections(content)
        }
        
    async def update_architecture(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Update existing architecture documentation."""
        update_path = Path(path) if path else self.project_root
        arch_file = None
        
        # Find existing architecture file
        for file_name in self.architecture_files['architecture']:
            file_path = update_path / file_name
            if file_path.exists():
                arch_file = file_path
                break
                
        if not arch_file:
            return {"error": "No architecture documentation found to update"}
            
        # Read current content
        current_content = arch_file.read_text()
        
        # Analyze what's changed
        structure = self._analyze_code_structure(update_path)
        patterns = await self.detect_patterns(str(update_path))
        
        # Generate updates
        updates = await self._generate_architecture_updates(current_content, structure, patterns)
        
        return {
            "success": True,
            "file": str(arch_file.relative_to(update_path)),
            "current_sections": self._extract_sections(current_content),
            "proposed_updates": updates,
            "update_summary": f"Found {len(updates)} sections to update"
        }
        
    async def validate_architecture(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Validate architecture against best practices."""
        val_path = Path(path) if path else self.project_root
        
        violations = []
        warnings = []
        passed_checks = []
        
        # Check documentation exists
        check_result = await self.check_architecture(str(val_path))
        if check_result['completeness_score'] < 50:
            violations.append({
                'rule': 'documentation_required',
                'message': 'Architecture documentation is incomplete',
                'severity': 'high'
            })
        else:
            passed_checks.append('documentation_exists')
            
        # Check for anti-patterns
        structure = self._analyze_code_structure(val_path)
        
        # Check for circular dependencies
        if self._has_circular_dependencies(structure):
            violations.append({
                'rule': 'no_circular_dependencies',
                'message': 'Circular dependencies detected',
                'severity': 'high'
            })
        else:
            passed_checks.append('no_circular_dependencies')
            
        # Check for proper layering
        if not self._has_proper_layering(structure):
            warnings.append({
                'rule': 'proper_layering',
                'message': 'Layer violations detected',
                'severity': 'medium'
            })
        else:
            passed_checks.append('proper_layering')
            
        # Calculate validation score
        total_checks = len(violations) + len(warnings) + len(passed_checks)
        score = round((len(passed_checks) / total_checks) * 100, 1) if total_checks > 0 else 0
        
        return {
            "success": True,
            "validation_score": score,
            "passed": len(passed_checks),
            "violations": violations,
            "warnings": warnings,
            "passed_checks": passed_checks
        }
        
    async def check_completeness(self, path: str = None, project_type: str = None, **kwargs) -> Dict[str, Any]:
        """Check architecture completeness for specific project type."""
        check_path = Path(path) if path else self.project_root
        
        # Detect project type if not specified
        if not project_type:
            project_type = self._detect_project_type(check_path)
            
        # Get requirements for project type
        requirements = self._get_architecture_requirements(project_type)
        
        # Check each requirement
        met_requirements = []
        missing_requirements = []
        
        for req in requirements:
            if self._check_requirement(check_path, req):
                met_requirements.append(req)
            else:
                missing_requirements.append(req)
                
        completeness = round((len(met_requirements) / len(requirements)) * 100, 1) if requirements else 0
        
        return {
            "success": True,
            "project_type": project_type,
            "completeness_percentage": completeness,
            "met_requirements": met_requirements,
            "missing_requirements": missing_requirements,
            "next_steps": self._suggest_next_steps(missing_requirements)
        }
        
    async def assess_quality(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Assess architecture quality."""
        assess_path = Path(path) if path else self.project_root
        
        quality_metrics = {
            'modularity': 0,
            'cohesion': 0,
            'coupling': 0,
            'complexity': 0,
            'documentation': 0
        }
        
        # Check modularity
        structure = self._analyze_code_structure(assess_path)
        quality_metrics['modularity'] = min(100, len(structure['modules']) * 10)
        
        # Check documentation
        check_result = await self.check_architecture(str(assess_path))
        quality_metrics['documentation'] = check_result['completeness_score']
        
        # Check cohesion and coupling (simplified)
        quality_metrics['cohesion'] = 80 if len(structure['modules']) > 5 else 60
        quality_metrics['coupling'] = 70 if not self._has_circular_dependencies(structure) else 30
        
        # Calculate overall score
        overall_score = round(sum(quality_metrics.values()) / len(quality_metrics), 1)
        
        return {
            "success": True,
            "score": overall_score,
            "metrics": quality_metrics,
            "grade": self._calculate_grade(overall_score),
            "improvements": self._suggest_quality_improvements(quality_metrics)
        }
        
    async def detect_patterns(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Detect architecture patterns in use."""
        detect_path = Path(path) if path else self.project_root
        
        detected = []
        evidence = {}
        
        # Analyze file structure
        for pattern_name, indicators in self.patterns.items():
            pattern_score = 0
            pattern_evidence = []
            
            for root, dirs, files in os.walk(detect_path):
                root_path = Path(root)
                
                # Check directory names
                for dir_name in dirs:
                    if any(indicator in dir_name.lower() for indicator in indicators):
                        pattern_score += 1
                        pattern_evidence.append(f"Directory: {dir_name}")
                        
                # Check file names
                for file_name in files:
                    if any(indicator in file_name.lower() for indicator in indicators):
                        pattern_score += 0.5
                        pattern_evidence.append(f"File: {file_name}")
                        
            if pattern_score >= 2:  # Threshold for detection
                detected.append(pattern_name)
                evidence[pattern_name] = pattern_evidence[:5]  # Top 5 evidence
                
        return {
            "success": True,
            "patterns": detected,
            "evidence": evidence,
            "primary_pattern": detected[0] if detected else None,
            "confidence": "high" if len(detected) == 1 else "medium" if detected else "low"
        }
        
    async def generate_business_architecture(self, path: str = None, **kwargs) -> Dict[str, Any]:
        """Generate business-focused architecture documentation."""
        gen_path = Path(path) if path else self.project_root
        
        # Analyze project
        structure = self._analyze_code_structure(gen_path)
        
        # Generate business architecture
        prompt = f"""Generate a business architecture document for a project with this structure:
{json.dumps(structure, indent=2)}

Include:
1. Business Overview
2. Key Components and Their Business Value
3. Integration Points
4. Data Flow
5. Security Considerations
6. Scalability Plan
7. Cost Considerations"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            content = response.get('content', '')
            return {
                "success": True,
                "type": "business_architecture",
                "content": content,
                "sections": self._extract_sections(content)
            }
            
        return {"error": "Failed to generate business architecture"}
        
    def _analyze_code_structure(self, path: Path) -> Dict[str, Any]:
        """Analyze code structure."""
        structure = {
            'modules': [],
            'depth': 0,
            'total_files': 0,
            'by_type': {}
        }
        
        for root, dirs, files in os.walk(path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            rel_path = Path(root).relative_to(path)
            depth = len(rel_path.parts)
            structure['depth'] = max(structure['depth'], depth)
            
            # Count files by type
            for file in files:
                if not file.startswith('.'):
                    ext = Path(file).suffix
                    structure['by_type'][ext] = structure['by_type'].get(ext, 0) + 1
                    structure['total_files'] += 1
                    
            # Identify modules
            if any(f.endswith('.py') for f in files) and depth <= 2:
                structure['modules'].append(str(rel_path))
                
        return structure
        
    def _generate_architecture_recommendations(self, missing: List[str]) -> List[str]:
        """Generate recommendations for missing architecture."""
        recommendations = []
        
        if 'readme' in missing:
            recommendations.append("Create a README.md with project overview and setup instructions")
        if 'architecture' in missing:
            recommendations.append("Document your architecture decisions and system design")
        if 'api' in missing:
            recommendations.append("Add API documentation for better integration")
        if 'security' in missing:
            recommendations.append("Document security considerations and practices")
            
        return recommendations
        
    async def _generate_readme(self, path: Path) -> str:
        """Generate README content."""
        project_name = path.name
        
        return f"""# {project_name}

## Overview
{project_name} is a [brief description of what the project does].

## Features
- Feature 1
- Feature 2
- Feature 3

## Installation
```bash
# Installation instructions
```

## Usage
```bash
# Usage examples
```

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License
[License information]
"""
        
    async def _generate_architecture_doc(self, path: Path) -> str:
        """Generate architecture documentation."""
        structure = self._analyze_code_structure(path)
        
        return f"""# Architecture Documentation

## Overview
This document describes the architecture of {path.name}.

## System Architecture
### Components
{chr(10).join(f'- {module}' for module in structure['modules'][:10])}

### Technology Stack
- Programming Languages: {', '.join(ext for ext in structure['by_type'] if ext in ['.py', '.js', '.java', '.go'])}
- Total Files: {structure['total_files']}
- Max Depth: {structure['depth']} levels

## Design Decisions
[Document key design decisions here]

## Data Flow
[Describe how data flows through the system]

## Security Architecture
[Document security measures and considerations]

## Deployment Architecture
[Describe deployment strategy and infrastructure]

## Future Considerations
[Document planned improvements and scalability considerations]
"""
        
    def _get_template_for_category(self, category: str) -> str:
        """Get template content for a category."""
        templates = {
            'contributing': "# Contributing Guidelines\n\n## How to Contribute\n\n## Code Style\n\n## Pull Request Process",
            'security': "# Security Policy\n\n## Reporting Vulnerabilities\n\n## Security Measures",
            'changelog': "# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - {}\n### Added\n### Changed\n### Fixed".format(datetime.now().strftime('%Y-%m-%d'))
        }
        
        return templates.get(category, f"# {category.title()}\n\n[Content to be added]")
        
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from markdown."""
        sections = []
        for line in content.split('\n'):
            if line.startswith('#'):
                sections.append(line.strip('# ').strip())
        return sections
        
    def _has_circular_dependencies(self, structure: Dict[str, Any]) -> bool:
        """Check for circular dependencies (simplified)."""
        # This is a simplified check - real implementation would analyze imports
        return False
        
    def _has_proper_layering(self, structure: Dict[str, Any]) -> bool:
        """Check for proper architectural layering."""
        # Simplified check - look for common layer names
        layers = ['presentation', 'business', 'data', 'domain', 'infrastructure']
        found_layers = sum(1 for module in structure['modules'] if any(layer in module.lower() for layer in layers))
        return found_layers >= 2
        
    def _detect_project_type(self, path: Path) -> str:
        """Detect project type from files."""
        if (path / 'package.json').exists():
            return 'nodejs'
        elif (path / 'requirements.txt').exists() or (path / 'setup.py').exists():
            return 'python'
        elif (path / 'pom.xml').exists():
            return 'java'
        elif (path / 'go.mod').exists():
            return 'golang'
        else:
            return 'generic'
            
    def _get_architecture_requirements(self, project_type: str) -> List[str]:
        """Get architecture requirements for project type."""
        base_requirements = ['README', 'Architecture Overview', 'Setup Instructions']
        
        type_specific = {
            'nodejs': ['Package Structure', 'API Documentation', 'Build Process'],
            'python': ['Module Structure', 'Dependencies', 'Testing Strategy'],
            'java': ['Package Structure', 'Build Configuration', 'Deployment'],
            'golang': ['Module Structure', 'Build Instructions', 'Docker Support']
        }
        
        return base_requirements + type_specific.get(project_type, [])
        
    def _check_requirement(self, path: Path, requirement: str) -> bool:
        """Check if a requirement is met."""
        # Simplified check - real implementation would be more thorough
        requirement_lower = requirement.lower()
        
        # Check in common files
        for file_list in self.architecture_files.values():
            for file_name in file_list:
                file_path = path / file_name
                if file_path.exists():
                    content = file_path.read_text().lower()
                    if requirement_lower in content:
                        return True
                        
        return False
        
    def _suggest_next_steps(self, missing: List[str]) -> List[str]:
        """Suggest next steps for missing requirements."""
        steps = []
        for req in missing[:3]:  # Top 3
            steps.append(f"Document {req} in appropriate architecture file")
        return steps
        
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
            
    def _suggest_quality_improvements(self, metrics: Dict[str, float]) -> List[str]:
        """Suggest improvements based on quality metrics."""
        improvements = []
        
        for metric, score in metrics.items():
            if score < 60:
                if metric == 'modularity':
                    improvements.append("Break down large modules into smaller, focused components")
                elif metric == 'documentation':
                    improvements.append("Improve architecture documentation coverage")
                elif metric == 'coupling':
                    improvements.append("Reduce dependencies between modules")
                    
        return improvements
        
    async def _generate_comprehensive_architecture(self, path: Path, structure: Dict, patterns: Dict) -> str:
        """Generate comprehensive architecture documentation."""
        # Use AI to generate detailed architecture
        prompt = f"""Generate comprehensive architecture documentation for a project:
Path: {path.name}
Structure: {json.dumps(structure, indent=2)}
Detected Patterns: {patterns.get('patterns', [])}

Include all standard sections plus:
- Detailed component descriptions
- Interaction diagrams (in text/ASCII)
- Decision rationale
- Trade-offs
- Evolution roadmap"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        return response.get('content', '') if response.get('success') else await self._generate_architecture_doc(path)
        
    async def _generate_minimal_architecture(self, path: Path, structure: Dict) -> str:
        """Generate minimal architecture documentation."""
        return f"""# {path.name} Architecture

## Overview
Minimal architecture documentation for {path.name}.

## Structure
- **Modules**: {len(structure['modules'])}
- **Files**: {structure['total_files']}
- **Depth**: {structure['depth']} levels

## Key Components
{chr(10).join(f'- {module}' for module in structure['modules'][:5])}

## Quick Start
See README.md for setup instructions.
"""
        
    async def _generate_architecture_updates(self, current: str, structure: Dict, patterns: Dict) -> List[Dict[str, str]]:
        """Generate updates for existing architecture."""
        updates = []
        
        # Check if structure section needs update
        if 'modules' in structure and str(len(structure['modules'])) not in current:
            updates.append({
                'section': 'System Structure',
                'update': f"Update module count to {len(structure['modules'])}"
            })
            
        # Check if patterns section needs update
        if patterns['patterns'] and not any(p in current for p in patterns['patterns']):
            updates.append({
                'section': 'Architecture Patterns',
                'update': f"Document {patterns['patterns'][0]} pattern usage"
            })
            
        return updates
        
    async def _get_ai_architecture_analysis(self, path: Path, analysis: Dict) -> Dict[str, Any]:
        """Get AI-powered architecture analysis."""
        prompt = f"""Analyze this architecture data and provide insights:
Path: {path.name}
Analysis: {json.dumps(analysis, indent=2)}

Provide:
1. Architecture strengths and how to leverage them
2. Critical weaknesses and remediation steps
3. Scalability assessment
4. Maintenance complexity rating
5. Recommended improvements priority list"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            return {
                "insights": response.get('content', ''),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        return {}


async def main(operation: str = None, **kwargs):
    """Entry point for architecture analyzer."""
    analyzer = ArchitectureAnalyzer()
    
    if operation:
        return await analyzer.execute(operation, **kwargs)
    else:
        # Default to check
        return await analyzer.check_architecture(**kwargs)
        

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test():
        analyzer = ArchitectureAnalyzer()
        
        # Test operations
        print("Check:", await analyzer.check_architecture())
        print("\nPatterns:", await analyzer.detect_patterns())
        print("\nQuality:", await analyzer.assess_quality())
        
    asyncio.run(test())