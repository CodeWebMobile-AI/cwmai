"""
DocsAgent - Documentation and Technical Writing Expert

Specializes in creating comprehensive documentation, API docs, user guides,
and maintaining up-to-date documentation for code changes.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.append('..')

from base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class DocsAgent(BaseAgent):
    """Agent specialized in documentation and technical writing."""
    
    @property
    def agent_type(self) -> str:
        return "documenter"
    
    @property
    def persona(self) -> str:
        return """You are an expert technical writer with extensive experience in creating 
        clear, comprehensive, and user-friendly documentation. You excel at explaining 
        complex technical concepts in simple terms while maintaining technical accuracy. 
        You understand different documentation needs - from API references to user guides, 
        from README files to architectural documents. You always consider the target 
        audience and create documentation that is well-structured, searchable, and 
        maintainable. You follow documentation best practices and standards."""
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.DOCUMENTATION,
            AgentCapability.ANALYSIS
        ]
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task from a documentation perspective."""
        work_item = context.work_item
        
        # Check for existing artifacts
        main_code = context.get_artifact('main_code')
        project_plan = context.get_artifact('project_plan')
        
        prompt = f"""
        Analyze this task from a documentation perspective:
        
        Task: {work_item.title}
        Description: {work_item.description}
        Type: {work_item.task_type}
        
        {f"Code to document: {json.dumps(main_code, indent=2)}" if main_code else ""}
        {f"Project plan: {json.dumps(project_plan, indent=2)}" if project_plan else ""}
        
        Provide documentation analysis including:
        1. Documentation types needed (README, API docs, guides, etc.)
        2. Target audiences (developers, users, administrators)
        3. Key sections to include
        4. Documentation standards to follow
        5. Diagrams and visualizations needed
        6. Code examples and tutorials required
        7. Maintenance strategy
        8. Documentation tools/formats to use
        
        Format as JSON with keys: doc_types, target_audiences, key_sections, 
        standards, diagrams_needed, examples_needed, maintenance_strategy, 
        tools_formats
        """
        
        response = await self._call_ai_model(prompt)
        
        expected_format = {
            'doc_types': list,
            'target_audiences': list,
            'key_sections': list,
            'standards': list,
            'diagrams_needed': list,
            'examples_needed': list,
            'maintenance_strategy': str,
            'tools_formats': dict
        }
        
        return self._parse_ai_response(response, expected_format)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Create comprehensive documentation for the task."""
        start_time = time.time()
        
        try:
            # Analyze the task
            analysis = await self.analyze(context)
            
            # Get all available artifacts
            main_code = context.get_artifact('main_code')
            test_suite = context.get_artifact('test_suite')
            security_audit = context.get_artifact('security_audit')
            project_plan = context.get_artifact('project_plan')
            
            # Generate main documentation
            main_docs = await self._generate_main_documentation(context, analysis, 
                                                               main_code, project_plan)
            
            # Generate API documentation
            api_docs = await self._generate_api_documentation(context, main_code, analysis)
            
            # Create user guide
            user_guide = await self._create_user_guide(context, analysis, main_code)
            
            # Generate developer documentation
            dev_docs = await self._generate_developer_docs(context, main_code, test_suite, 
                                                         security_audit, analysis)
            
            # Create documentation index
            doc_index = await self._create_documentation_index(main_docs, api_docs, 
                                                             user_guide, dev_docs)
            
            # Store artifacts
            artifacts_created = []
            
            if context.blackboard:
                # Store in blackboard
                await context.blackboard.write_artifact(
                    f"main_docs_{context.work_item.id}",
                    main_docs,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"api_docs_{context.work_item.id}",
                    api_docs,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"user_guide_{context.work_item.id}",
                    user_guide,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"dev_docs_{context.work_item.id}",
                    dev_docs,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"doc_index_{context.work_item.id}",
                    doc_index,
                    self.agent_id
                )
                artifacts_created = [
                    f"main_docs_{context.work_item.id}",
                    f"api_docs_{context.work_item.id}",
                    f"user_guide_{context.work_item.id}",
                    f"dev_docs_{context.work_item.id}",
                    f"doc_index_{context.work_item.id}"
                ]
            else:
                # Store in context
                context.add_artifact('main_docs', main_docs, self.agent_id)
                context.add_artifact('api_docs', api_docs, self.agent_id)
                context.add_artifact('user_guide', user_guide, self.agent_id)
                context.add_artifact('dev_docs', dev_docs, self.agent_id)
                context.add_artifact('doc_index', doc_index, self.agent_id)
                artifacts_created = ['main_docs', 'api_docs', 'user_guide', 'dev_docs', 'doc_index']
            
            # Generate insights
            insights = [
                f"Created {len(analysis.get('doc_types', []))} documentation types",
                f"Targeting {len(analysis.get('target_audiences', []))} audience groups",
                f"Documentation completeness: {self._calculate_doc_completeness(main_docs, api_docs, user_guide)}%",
                f"Includes {len(analysis.get('diagrams_needed', []))} diagrams"
            ]
            
            # Generate recommendations
            recommendations = []
            if not main_code:
                recommendations.append("Waiting for CodeAgent output to create specific documentation")
            recommendations.append("Set up automated documentation generation in CI/CD")
            recommendations.append("Implement documentation versioning")
            recommendations.append("Create documentation review process")
            if len(analysis.get('diagrams_needed', [])) > 0:
                recommendations.append("Use diagram generation tools for visual documentation")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                output={
                    'main_docs': main_docs,
                    'api_docs': api_docs,
                    'user_guide': user_guide,
                    'dev_docs': dev_docs,
                    'doc_index': doc_index,
                    'analysis': analysis
                },
                artifacts_created=artifacts_created,
                insights=insights,
                recommendations=recommendations,
                confidence=0.85,
                execution_time=execution_time,
                metadata={
                    'total_pages': self._estimate_total_pages(main_docs, api_docs, user_guide, dev_docs),
                    'doc_types_count': len(analysis.get('doc_types', []))
                }
            )
            
        except Exception as e:
            self.logger.error(f"DocsAgent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                output={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _generate_main_documentation(self, context: AgentContext, analysis: Dict[str, Any],
                                         main_code: Optional[Dict[str, Any]], 
                                         project_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate main project documentation."""
        prompt = f"""
        Generate comprehensive project documentation for:
        Task: {context.work_item.title}
        Description: {context.work_item.description}
        
        Documentation analysis: {json.dumps(analysis, indent=2)}
        {f"Code structure: {json.dumps(main_code, indent=2)}" if main_code else ""}
        {f"Project plan: {json.dumps(project_plan, indent=2)}" if project_plan else ""}
        
        Create a main documentation including:
        1. Project overview and purpose
        2. Installation and setup instructions
        3. Configuration guide
        4. Architecture overview
        5. Key features and functionality
        6. Usage examples
        7. Troubleshooting guide
        8. Contributing guidelines
        
        Format as JSON with structure:
        {{
            "title": "...",
            "version": "...",
            "last_updated": "...",
            "sections": [
                {{
                    "title": "...",
                    "content": "...",
                    "subsections": [...]
                }}
            ],
            "metadata": {{...}}
        }}
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            docs = json.loads(response)
            docs['last_updated'] = datetime.now(timezone.utc).isoformat()
            return docs
        except:
            # Fallback documentation structure
            return {
                "title": f"Documentation: {context.work_item.title}",
                "version": "1.0.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "sections": [
                    {
                        "title": "Overview",
                        "content": f"Documentation for {context.work_item.title}",
                        "subsections": []
                    }
                ],
                "metadata": {"status": "draft"}
            }
    
    async def _generate_api_documentation(self, context: AgentContext, 
                                        main_code: Optional[Dict[str, Any]], 
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API documentation."""
        if not main_code:
            return {
                "openapi": "3.0.0",
                "info": {
                    "title": f"{context.work_item.title} API",
                    "version": "1.0.0",
                    "description": "API documentation pending code generation"
                },
                "paths": {}
            }
        
        prompt = f"""
        Generate API documentation for the code:
        {json.dumps(main_code, indent=2)}
        
        Create OpenAPI/Swagger documentation including:
        1. API endpoints and methods
        2. Request/response schemas
        3. Authentication requirements
        4. Error responses
        5. Example requests
        
        Format as OpenAPI 3.0 specification JSON.
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            api_doc = json.loads(response)
            return api_doc
        except:
            return {
                "openapi": "3.0.0",
                "info": {
                    "title": f"{context.work_item.title} API",
                    "version": "1.0.0"
                },
                "paths": {}
            }
    
    async def _create_user_guide(self, context: AgentContext, analysis: Dict[str, Any],
                                main_code: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create user-friendly guide."""
        user_guide = {
            "title": f"{context.work_item.title} User Guide",
            "version": "1.0.0",
            "chapters": [
                {
                    "title": "Getting Started",
                    "sections": [
                        {"title": "What is this?", "content": "Introduction to the project"},
                        {"title": "Quick Start", "content": "Get up and running in 5 minutes"},
                        {"title": "Prerequisites", "content": "What you need before starting"}
                    ]
                },
                {
                    "title": "Basic Usage",
                    "sections": [
                        {"title": "Common Tasks", "content": "How to perform common operations"},
                        {"title": "Best Practices", "content": "Recommended ways to use the system"}
                    ]
                },
                {
                    "title": "Advanced Features",
                    "sections": [
                        {"title": "Power User Tips", "content": "Advanced techniques"},
                        {"title": "Customization", "content": "How to customize for your needs"}
                    ]
                },
                {
                    "title": "Troubleshooting",
                    "sections": [
                        {"title": "Common Issues", "content": "Solutions to frequent problems"},
                        {"title": "Getting Help", "content": "Where to find additional support"}
                    ]
                }
            ],
            "glossary": {},
            "faq": []
        }
        
        return user_guide
    
    async def _generate_developer_docs(self, context: AgentContext, 
                                     main_code: Optional[Dict[str, Any]],
                                     test_suite: Optional[Dict[str, Any]],
                                     security_audit: Optional[Dict[str, Any]],
                                     analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate developer-focused documentation."""
        dev_docs = {
            "title": "Developer Documentation",
            "sections": {
                "architecture": {
                    "title": "System Architecture",
                    "content": "Technical architecture overview",
                    "diagrams": analysis.get('diagrams_needed', [])
                },
                "development_setup": {
                    "title": "Development Environment Setup",
                    "content": "How to set up local development",
                    "requirements": [],
                    "steps": []
                },
                "coding_standards": {
                    "title": "Coding Standards",
                    "content": "Code style and best practices",
                    "languages": {}
                },
                "testing": {
                    "title": "Testing Guide",
                    "content": "How to write and run tests",
                    "test_types": test_suite.get('test_types', []) if test_suite else [],
                    "coverage_requirements": {}
                },
                "security": {
                    "title": "Security Guidelines",
                    "content": "Security best practices",
                    "checklist": security_audit.get('security_checklist', []) if security_audit else []
                },
                "deployment": {
                    "title": "Deployment Guide",
                    "content": "How to deploy to production",
                    "environments": [],
                    "ci_cd": {}
                }
            }
        }
        
        return dev_docs
    
    async def _create_documentation_index(self, main_docs: Dict[str, Any], 
                                        api_docs: Dict[str, Any],
                                        user_guide: Dict[str, Any], 
                                        dev_docs: Dict[str, Any]) -> Dict[str, Any]:
        """Create an index of all documentation."""
        doc_index = {
            "title": "Documentation Index",
            "generated": datetime.now(timezone.utc).isoformat(),
            "documents": [
                {
                    "type": "main",
                    "title": main_docs.get('title', 'Main Documentation'),
                    "version": main_docs.get('version', '1.0.0'),
                    "sections": len(main_docs.get('sections', []))
                },
                {
                    "type": "api",
                    "title": "API Documentation",
                    "format": "OpenAPI 3.0",
                    "endpoints": len(api_docs.get('paths', {}))
                },
                {
                    "type": "user_guide",
                    "title": user_guide.get('title', 'User Guide'),
                    "chapters": len(user_guide.get('chapters', []))
                },
                {
                    "type": "developer",
                    "title": dev_docs.get('title', 'Developer Documentation'),
                    "sections": len(dev_docs.get('sections', {}))
                }
            ],
            "search_keywords": [],
            "quick_links": []
        }
        
        return doc_index
    
    def _calculate_doc_completeness(self, main_docs: Dict[str, Any], 
                                  api_docs: Dict[str, Any], 
                                  user_guide: Dict[str, Any]) -> int:
        """Calculate documentation completeness percentage."""
        total_sections = 0
        completed_sections = 0
        
        # Check main docs
        for section in main_docs.get('sections', []):
            total_sections += 1
            if section.get('content') and len(section['content']) > 50:
                completed_sections += 1
        
        # Check API docs
        if api_docs.get('paths'):
            total_sections += 1
            completed_sections += 1
        
        # Check user guide
        for chapter in user_guide.get('chapters', []):
            total_sections += len(chapter.get('sections', []))
            completed_sections += len([s for s in chapter.get('sections', []) 
                                     if s.get('content')])
        
        return int((completed_sections / max(total_sections, 1)) * 100)
    
    def _estimate_total_pages(self, *docs) -> int:
        """Estimate total documentation pages."""
        total_chars = 0
        for doc in docs:
            total_chars += len(json.dumps(doc))
        
        # Assume ~2000 chars per page
        return max(1, total_chars // 2000)
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review artifacts from other agents from a documentation perspective."""
        review = await super().review_artifact(artifact_key, artifact_value, created_by, context)
        
        # Specific documentation reviews
        if 'code' in artifact_key:
            review['feedback'].append("Code needs inline documentation and comments")
            review['feedback'].append("Add docstrings to all functions and classes")
        elif 'test' in artifact_key:
            review['feedback'].append("Test documentation should explain test scenarios")
            review['feedback'].append("Include test data documentation")
        elif 'security' in artifact_key:
            review['feedback'].append("Security findings should be documented for developers")
            review['feedback'].append("Create security implementation guide")
        elif 'plan' in artifact_key:
            review['feedback'].append("Project plan should be reflected in documentation")
        
        review['confidence'] = 0.8
        return review