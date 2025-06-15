"""
SecurityAgent - Security Analysis and Vulnerability Detection Expert

Specializes in scanning code for vulnerabilities, ensuring security best practices,
and providing security recommendations before code is committed.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.append('..')

from base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class SecurityAgent(BaseAgent):
    """Agent specialized in security analysis and vulnerability detection."""
    
    @property
    def agent_type(self) -> str:
        return "security"
    
    @property
    def persona(self) -> str:
        return """You are a cybersecurity expert and ethical hacker with extensive experience 
        in application security, penetration testing, and secure coding practices. You think 
        like an attacker to defend better. You are well-versed in OWASP Top 10, security 
        frameworks, and compliance requirements. You excel at identifying vulnerabilities, 
        security misconfigurations, and potential attack vectors. You provide actionable 
        recommendations to harden applications and protect against threats."""
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.SECURITY_ANALYSIS,
            AgentCapability.CODE_REVIEW,
            AgentCapability.ANALYSIS
        ]
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task from a security perspective."""
        work_item = context.work_item
        
        # Check for existing code
        main_code = context.get_artifact('main_code')
        
        prompt = f"""
        Analyze this task from a security perspective:
        
        Task: {work_item.title}
        Description: {work_item.description}
        Type: {work_item.task_type}
        
        {f"Code to analyze: {json.dumps(main_code, indent=2)}" if main_code else ""}
        
        Provide security analysis including:
        1. Potential security risks and vulnerabilities
        2. Attack vectors to consider
        3. OWASP Top 10 relevance
        4. Authentication and authorization requirements
        5. Data protection needs (encryption, hashing)
        6. Input validation requirements
        7. Security testing approach
        8. Compliance considerations (GDPR, PCI DSS, HIPAA, etc.)
        
        Format as JSON with keys: security_risks, attack_vectors, owasp_relevance, 
        auth_requirements, data_protection, input_validation, 
        security_testing, compliance_requirements
        """
        
        response = await self._call_ai_model(prompt)
        
        expected_format = {
            'security_risks': list,
            'attack_vectors': list,
            'owasp_relevance': list,
            'auth_requirements': dict,
            'data_protection': dict,
            'input_validation': list,
            'security_testing': dict,
            'compliance_requirements': list
        }
        
        return self._parse_ai_response(response, expected_format)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Perform security analysis and generate security recommendations."""
        start_time = time.time()
        
        try:
            # Analyze the task
            analysis = await self.analyze(context)
            
            # Get code artifacts if available
            main_code = context.get_artifact('main_code')
            
            # Perform security audit
            security_audit = await self._perform_security_audit(context, analysis, main_code)
            
            # Generate security fixes
            security_fixes = await self._generate_security_fixes(context, security_audit)
            
            # Create security checklist
            security_checklist = await self._create_security_checklist(context, analysis, security_audit)
            
            # Generate threat model
            threat_model = await self._generate_threat_model(context, analysis)
            
            # Store artifacts
            artifacts_created = []
            
            if context.blackboard:
                # Store in blackboard
                await context.blackboard.write_artifact(
                    f"security_audit_{context.work_item.id}",
                    security_audit,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"security_fixes_{context.work_item.id}",
                    security_fixes,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"security_checklist_{context.work_item.id}",
                    security_checklist,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"threat_model_{context.work_item.id}",
                    threat_model,
                    self.agent_id
                )
                artifacts_created = [
                    f"security_audit_{context.work_item.id}",
                    f"security_fixes_{context.work_item.id}",
                    f"security_checklist_{context.work_item.id}",
                    f"threat_model_{context.work_item.id}"
                ]
            else:
                # Store in context
                context.add_artifact('security_audit', security_audit, self.agent_id)
                context.add_artifact('security_fixes', security_fixes, self.agent_id)
                context.add_artifact('security_checklist', security_checklist, self.agent_id)
                context.add_artifact('threat_model', threat_model, self.agent_id)
                artifacts_created = ['security_audit', 'security_fixes', 'security_checklist', 'threat_model']
            
            # Generate insights
            insights = [
                f"Identified {len(security_audit.get('vulnerabilities', []))} potential vulnerabilities",
                f"Security score: {security_audit.get('security_score', 'N/A')}/10",
                f"Critical issues: {len([v for v in security_audit.get('vulnerabilities', []) if v.get('severity') == 'critical'])}",
                f"OWASP coverage: {len(analysis.get('owasp_relevance', []))} categories"
            ]
            
            # Generate recommendations
            recommendations = []
            critical_vulns = [v for v in security_audit.get('vulnerabilities', []) if v.get('severity') == 'critical']
            if critical_vulns:
                recommendations.append(f"Fix {len(critical_vulns)} critical vulnerabilities before deployment")
            if security_audit.get('security_score', 10) < 7:
                recommendations.append("Implement security hardening measures to improve score")
            if not main_code:
                recommendations.append("Waiting for CodeAgent output to perform detailed security analysis")
            recommendations.append("Implement security testing in CI/CD pipeline")
            recommendations.append("Schedule regular security audits")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                output={
                    'security_audit': security_audit,
                    'security_fixes': security_fixes,
                    'security_checklist': security_checklist,
                    'threat_model': threat_model,
                    'analysis': analysis
                },
                artifacts_created=artifacts_created,
                insights=insights,
                recommendations=recommendations,
                confidence=0.9,
                execution_time=execution_time,
                metadata={
                    'vulnerabilities_count': len(security_audit.get('vulnerabilities', [])),
                    'security_score': security_audit.get('security_score', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"SecurityAgent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                output={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _perform_security_audit(self, context: AgentContext, analysis: Dict[str, Any], 
                                    main_code: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        prompt = f"""
        Perform a security audit for:
        Task: {context.work_item.title}
        
        Security analysis: {json.dumps(analysis, indent=2)}
        {f"Code to audit: {json.dumps(main_code, indent=2)}" if main_code else ""}
        
        Identify:
        1. Security vulnerabilities (with severity: critical/high/medium/low)
        2. Insecure coding practices
        3. Missing security controls
        4. Configuration issues
        5. Dependency vulnerabilities
        6. Hardcoded secrets or credentials
        7. Insufficient logging/monitoring
        
        Format as JSON with structure:
        {{
            "vulnerabilities": [
                {{
                    "id": "...",
                    "type": "...",
                    "severity": "critical/high/medium/low",
                    "description": "...",
                    "location": "...",
                    "impact": "...",
                    "likelihood": "...",
                    "cwe_id": "..."
                }}
            ],
            "insecure_practices": [...],
            "missing_controls": [...],
            "configuration_issues": [...],
            "dependency_risks": [...],
            "secrets_found": [...],
            "security_score": 7.5,
            "summary": "..."
        }}
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            audit = json.loads(response)
            return audit
        except:
            # Fallback audit structure
            return {
                "vulnerabilities": [],
                "insecure_practices": ["Generic security assessment needed"],
                "missing_controls": ["Authentication", "Authorization", "Input validation"],
                "configuration_issues": [],
                "dependency_risks": [],
                "secrets_found": [],
                "security_score": 5.0,
                "summary": "Initial security assessment - detailed code analysis pending"
            }
    
    async def _generate_security_fixes(self, context: AgentContext, security_audit: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fixes for identified security issues."""
        fixes = {
            "immediate_fixes": [],
            "short_term_fixes": [],
            "long_term_fixes": [],
            "code_patches": []
        }
        
        # Categorize fixes based on vulnerability severity
        for vuln in security_audit.get('vulnerabilities', []):
            fix = {
                "vulnerability_id": vuln['id'],
                "fix_description": f"Fix for {vuln['type']}",
                "implementation_guide": f"Implement security control for {vuln['description']}",
                "estimated_effort": "Medium",
                "priority": vuln['severity']
            }
            
            if vuln['severity'] == 'critical':
                fixes['immediate_fixes'].append(fix)
            elif vuln['severity'] == 'high':
                fixes['short_term_fixes'].append(fix)
            else:
                fixes['long_term_fixes'].append(fix)
        
        return fixes
    
    async def _create_security_checklist(self, context: AgentContext, analysis: Dict[str, Any], 
                                       security_audit: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create security checklist for the project."""
        checklist = [
            {
                "category": "Authentication & Authorization",
                "items": [
                    "Implement secure authentication mechanism",
                    "Use strong password policies",
                    "Implement MFA where appropriate",
                    "Secure session management",
                    "Proper authorization checks"
                ]
            },
            {
                "category": "Input Validation & Sanitization",
                "items": [
                    "Validate all user inputs",
                    "Sanitize data before processing",
                    "Implement parameterized queries",
                    "Prevent XSS attacks",
                    "Prevent SQL injection"
                ]
            },
            {
                "category": "Data Protection",
                "items": [
                    "Encrypt sensitive data at rest",
                    "Use TLS for data in transit",
                    "Implement proper key management",
                    "Secure password storage (hashing)",
                    "PII data protection"
                ]
            },
            {
                "category": "Security Headers & Configuration",
                "items": [
                    "Set security headers (CSP, HSTS, etc.)",
                    "Disable unnecessary features",
                    "Secure default configurations",
                    "Regular security updates",
                    "Secure error handling"
                ]
            },
            {
                "category": "Monitoring & Logging",
                "items": [
                    "Log security events",
                    "Monitor for anomalies",
                    "Implement intrusion detection",
                    "Regular security audits",
                    "Incident response plan"
                ]
            }
        ]
        
        return checklist
    
    async def _generate_threat_model(self, context: AgentContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate threat model using STRIDE methodology."""
        threat_model = {
            "assets": [
                {"name": "User Data", "value": "High", "threats": []},
                {"name": "Application Code", "value": "Medium", "threats": []},
                {"name": "Configuration", "value": "Medium", "threats": []}
            ],
            "stride_analysis": {
                "spoofing": analysis.get('attack_vectors', []),
                "tampering": ["Data modification", "Code injection"],
                "repudiation": ["Lack of audit trail", "Insufficient logging"],
                "information_disclosure": ["Data leakage", "Error messages"],
                "denial_of_service": ["Resource exhaustion", "Rate limiting"],
                "elevation_of_privilege": ["Privilege escalation", "Authorization bypass"]
            },
            "threat_scenarios": [],
            "mitigations": {},
            "risk_matrix": {
                "high_impact_high_likelihood": [],
                "high_impact_low_likelihood": [],
                "low_impact_high_likelihood": [],
                "low_impact_low_likelihood": []
            }
        }
        
        return threat_model
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review artifacts from other agents from a security perspective."""
        review = await super().review_artifact(artifact_key, artifact_value, created_by, context)
        
        # Specific security reviews
        if 'code' in artifact_key:
            review['feedback'].append("Code must pass security scan before approval")
            review['feedback'].append("Implement input validation for all user inputs")
            review['feedback'].append("Ensure no hardcoded secrets or credentials")
            review['approval'] = False  # Don't approve code without security review
        elif 'test' in artifact_key:
            review['feedback'].append("Include security test cases")
            review['feedback'].append("Test for common vulnerabilities (XSS, injection, etc.)")
        elif 'docs' in artifact_key:
            review['feedback'].append("Documentation should include security considerations")
            review['feedback'].append("Add security best practices section")
        
        review['confidence'] = 0.95
        return review