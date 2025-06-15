"""
CodeAgent - Code Generation and Refactoring Expert

Specializes in writing, refactoring, and optimizing code across multiple languages.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.append('..')

from base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class CodeAgent(BaseAgent):
    """Agent specialized in code generation and refactoring."""
    
    @property
    def agent_type(self) -> str:
        return "coder"
    
    @property
    def persona(self) -> str:
        return """You are a master software engineer with expertise in multiple programming languages 
        and frameworks. You write clean, efficient, and maintainable code following best practices 
        and design patterns. You think about scalability, performance, and code reusability. 
        You always consider edge cases, error handling, and write self-documenting code. 
        You are proficient in Python, JavaScript, TypeScript, Go, Java, and modern frameworks."""
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_GENERATION,
            AgentCapability.CODE_REVIEW,
            AgentCapability.OPTIMIZATION,
            AgentCapability.ARCHITECTURE
        ]
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task from a coding perspective."""
        work_item = context.work_item
        
        # Check for existing plan
        project_plan = context.get_artifact('project_plan')
        
        prompt = f"""
        Analyze this task from a software engineering perspective:
        
        Task: {work_item.title}
        Description: {work_item.description}
        Type: {work_item.task_type}
        
        {f"Project Plan: {json.dumps(project_plan, indent=2)}" if project_plan else ""}
        
        Provide analysis including:
        1. Technical approach and architecture
        2. Programming language and framework selection
        3. Key components/modules to implement
        4. Design patterns to apply
        5. Performance considerations
        6. Error handling strategy
        7. Testing approach
        8. Code quality metrics to target
        
        Format as JSON with keys: technical_approach, language_selection, 
        components, design_patterns, performance_considerations, 
        error_handling, testing_strategy, quality_targets
        """
        
        response = await self._call_ai_model(prompt)
        
        expected_format = {
            'technical_approach': str,
            'language_selection': dict,
            'components': list,
            'design_patterns': list,
            'performance_considerations': list,
            'error_handling': dict,
            'testing_strategy': str,
            'quality_targets': dict
        }
        
        return self._parse_ai_response(response, expected_format)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Generate or refactor code for the task."""
        start_time = time.time()
        
        try:
            # Analyze the task
            analysis = await self.analyze(context)
            
            # Determine code action
            if context.work_item.task_type in ['code_generation', 'feature_implementation']:
                code_output = await self._generate_code(context, analysis)
            elif context.work_item.task_type in ['refactoring', 'optimization']:
                code_output = await self._refactor_code(context, analysis)
            else:
                code_output = await self._generate_code(context, analysis)
            
            # Generate supporting code (interfaces, types, etc.)
            supporting_code = await self._generate_supporting_code(context, analysis, code_output)
            
            # Create code review checklist
            review_checklist = await self._create_review_checklist(context, code_output)
            
            # Store artifacts
            artifacts_created = []
            
            if context.blackboard:
                # Store in blackboard
                await context.blackboard.write_artifact(
                    f"main_code_{context.work_item.id}",
                    code_output,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"supporting_code_{context.work_item.id}",
                    supporting_code,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"code_review_checklist_{context.work_item.id}",
                    review_checklist,
                    self.agent_id
                )
                artifacts_created = [
                    f"main_code_{context.work_item.id}",
                    f"supporting_code_{context.work_item.id}",
                    f"code_review_checklist_{context.work_item.id}"
                ]
            else:
                # Store in context
                context.add_artifact('main_code', code_output, self.agent_id)
                context.add_artifact('supporting_code', supporting_code, self.agent_id)
                context.add_artifact('code_review_checklist', review_checklist, self.agent_id)
                artifacts_created = ['main_code', 'supporting_code', 'code_review_checklist']
            
            # Generate insights
            insights = [
                f"Generated {len(code_output.get('files', []))} code files",
                f"Primary language: {analysis.get('language_selection', {}).get('primary', 'Unknown')}",
                f"Applied {len(analysis.get('design_patterns', []))} design patterns",
                f"Code complexity: {code_output.get('metrics', {}).get('complexity', 'Medium')}"
            ]
            
            # Generate recommendations
            recommendations = []
            if code_output.get('metrics', {}).get('complexity', '') == 'High':
                recommendations.append("Consider breaking down complex functions")
            if len(code_output.get('todos', [])) > 0:
                recommendations.append(f"Complete {len(code_output.get('todos', []))} TODO items before deployment")
            recommendations.append("Run TestAgent to ensure comprehensive test coverage")
            recommendations.append("Execute SecurityAgent scan before merging")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                output={
                    'code': code_output,
                    'supporting_code': supporting_code,
                    'review_checklist': review_checklist,
                    'analysis': analysis
                },
                artifacts_created=artifacts_created,
                insights=insights,
                recommendations=recommendations,
                confidence=0.9,
                execution_time=execution_time,
                metadata={
                    'files_count': len(code_output.get('files', [])),
                    'total_lines': sum(f.get('lines', 0) for f in code_output.get('files', []))
                }
            )
            
        except Exception as e:
            self.logger.error(f"CodeAgent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                output={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _generate_code(self, context: AgentContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new code based on requirements."""
        prompt = f"""
        Generate production-quality code for:
        Task: {context.work_item.title}
        Description: {context.work_item.description}
        
        Technical approach: {analysis.get('technical_approach')}
        Language: {analysis.get('language_selection', {}).get('primary', 'Python')}
        Components: {json.dumps(analysis.get('components', []))}
        
        Requirements:
        1. Follow best practices and coding standards
        2. Include comprehensive error handling
        3. Make the code modular and reusable
        4. Add inline documentation
        5. Consider performance optimizations
        6. Include TODO comments for future enhancements
        
        Format the response as JSON with structure:
        {{
            "files": [
                {{
                    "filename": "...",
                    "language": "...",
                    "content": "...",
                    "lines": 100,
                    "purpose": "..."
                }}
            ],
            "metrics": {{
                "complexity": "Low/Medium/High",
                "maintainability": "score",
                "test_coverage_target": "percentage"
            }},
            "todos": ["..."],
            "dependencies": ["..."]
        }}
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            code_output = json.loads(response)
            return code_output
        except:
            # Fallback code structure
            return {
                "files": [{
                    "filename": "main.py",
                    "language": "python",
                    "content": "# Generated code placeholder\n\ndef main():\n    pass\n",
                    "lines": 4,
                    "purpose": "Main entry point"
                }],
                "metrics": {
                    "complexity": "Low",
                    "maintainability": "8/10",
                    "test_coverage_target": "80%"
                },
                "todos": ["Implement main functionality"],
                "dependencies": []
            }
    
    async def _refactor_code(self, context: AgentContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor existing code."""
        # In a real implementation, this would analyze existing code
        # For now, we'll simulate refactoring
        return await self._generate_code(context, analysis)
    
    async def _generate_supporting_code(self, context: AgentContext, analysis: Dict[str, Any], 
                                      main_code: Dict[str, Any]) -> Dict[str, Any]:
        """Generate supporting code like interfaces, types, utilities."""
        supporting = {
            "interfaces": [],
            "types": [],
            "utilities": [],
            "configurations": []
        }
        
        # Generate based on main code needs
        for file in main_code.get('files', []):
            if file.get('language') in ['typescript', 'java']:
                supporting['interfaces'].append({
                    "name": f"I{file['filename'].split('.')[0].title()}",
                    "content": f"// Interface for {file['filename']}"
                })
        
        return supporting
    
    async def _create_review_checklist(self, context: AgentContext, code_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a code review checklist."""
        checklist = [
            {
                "category": "Code Quality",
                "items": [
                    "Code follows language conventions",
                    "No code duplication",
                    "Functions are single-purpose",
                    "Clear variable/function names"
                ]
            },
            {
                "category": "Error Handling",
                "items": [
                    "All exceptions are caught appropriately",
                    "Error messages are informative",
                    "Graceful degradation implemented"
                ]
            },
            {
                "category": "Performance",
                "items": [
                    "No obvious performance bottlenecks",
                    "Efficient algorithms used",
                    "Database queries optimized"
                ]
            },
            {
                "category": "Security",
                "items": [
                    "Input validation implemented",
                    "No hardcoded secrets",
                    "SQL injection prevention"
                ]
            },
            {
                "category": "Testing",
                "items": [
                    "Unit tests needed",
                    "Integration tests needed",
                    "Edge cases covered"
                ]
            }
        ]
        
        return checklist
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review artifacts from other agents from a coding perspective."""
        review = await super().review_artifact(artifact_key, artifact_value, created_by, context)
        
        # Specific code reviews
        if 'test' in artifact_key:
            review['feedback'].append("Verify tests cover all code paths")
            review['feedback'].append("Check test naming conventions")
        elif 'security' in artifact_key:
            review['feedback'].append("Update code to address security findings")
        elif 'docs' in artifact_key:
            review['feedback'].append("Ensure documentation matches implementation")
        
        review['confidence'] = 0.85
        return review