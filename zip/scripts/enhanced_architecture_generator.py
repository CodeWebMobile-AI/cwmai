#!/usr/bin/env python3
"""Enhanced Architecture Generator with complete sections for AI-driven development."""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedArchitectureGenerator:
    """Generates comprehensive architecture documentation with all required sections."""
    
    def __init__(self, ai_brain):
        self.ai_brain = ai_brain
        self.logger = logger
        
    async def generate_complete_architecture(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete architecture document with all sections."""
        self.logger.info("🏗️ Generating enhanced architecture documentation...")
        
        architecture = {}
        
        try:
            # Generate each section with focused prompts
            architecture['executive_summary'] = await self._generate_executive_summary(details)
            architecture['system_context'] = await self._generate_system_context(details)
            
            # Add key features from project details
            if details.get('initial_features'):
                architecture['key_features'] = details.get('initial_features', [])
            architecture['business_rules'] = await self._generate_business_rules(details)
            architecture['domain_models'] = await self._generate_complete_domain_models(details)
            architecture['portal_separation'] = await self._generate_portal_separation(details)
            architecture['api_documentation'] = await self._generate_api_documentation(details)
            architecture['implementation_examples'] = await self._generate_code_examples(details)
            architecture['state_machines'] = await self._generate_state_machines(details)
            architecture['data_architecture'] = await self._generate_data_architecture(details)
            architecture['ai_integration'] = await self._generate_ai_integration(details)
            architecture['infrastructure'] = await self._generate_infrastructure(details)
            architecture['testing_strategy'] = await self._generate_testing_strategy(details)
            architecture['security'] = await self._generate_security_details(details)
            architecture['performance'] = await self._generate_performance_details(details)
            architecture['monitoring'] = await self._generate_monitoring_details(details)
            architecture['implementation_roadmap'] = await self._generate_roadmap(details)
            
            # Validate completeness
            self._validate_architecture_completeness(architecture)
            
            # Balance section sizes
            architecture = self._balance_section_detail(architecture)
            
            # Add metadata
            architecture['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'version': '2.0',
                'generator': 'enhanced_architecture_generator'
            }
            
            return architecture
            
        except Exception as e:
            self.logger.error(f"Failed to generate complete architecture: {e}")
            raise
    
    async def _generate_business_rules(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive business rules section."""
        prompt = f"""
        Generate comprehensive business rules for the {details.get('name')} project.
        
        Context:
        - Problem: {details.get('problem_statement')}
        - Target Users: {details.get('target_audience')}
        - Core Features: {json.dumps(details.get('initial_features', []))}
        
        Generate JSON with:
        {{
            "core_rules": {{
                "user_management": {{
                    "registration": ["email_verification_required", "unique_email", "password_requirements"],
                    "authentication": ["session_timeout", "remember_me_duration", "oauth_providers"],
                    "authorization": ["role_based_access", "permission_levels"]
                }},
                "[main_entity]_rules": {{
                    "creation": ["validation_rules", "required_fields", "defaults"],
                    "modification": ["who_can_edit", "audit_trail", "version_control"],
                    "deletion": ["soft_delete", "cascade_rules", "recovery_period"]
                }},
                "business_logic": {{
                    "pricing": ["calculation_method", "discount_rules", "tax_handling"],
                    "scheduling": ["time_zones", "availability_windows", "conflict_resolution"],
                    "notifications": ["trigger_events", "delivery_channels", "opt_out_rules"]
                }}
            }},
            "validation_rules": {{
                "[field_name]": {{
                    "type": "string|number|date",
                    "required": true,
                    "min": 0,
                    "max": 100,
                    "pattern": "regex_if_applicable",
                    "custom": "custom_validation_description"
                }}
            }},
            "edge_cases": [
                {{
                    "scenario": "User account suspended",
                    "handling": "Preserve data, block access, notify user",
                    "recovery": "Admin review process"
                }},
                {{
                    "scenario": "Payment failure",
                    "handling": "Retry logic, grace period, service suspension",
                    "recovery": "Payment update flow"
                }}
            ],
            "business_constraints": {{
                "rate_limits": {{"api_calls": "100/hour", "file_uploads": "10/day"}},
                "data_retention": {{"user_data": "7 years", "logs": "90 days"}},
                "compliance": ["GDPR", "CCPA", "industry_specific"]
            }}
        }}
        
        Be specific to this project's domain and use cases.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        return self._parse_json_response(response)
    
    async def _generate_complete_domain_models(self, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ALL domain models, not just core ones."""
        prompt = f"""
        Generate ALL domain models for the {details.get('name')} project.
        Include every model that would be needed for a complete implementation.
        
        Context:
        - Problem: {details.get('problem_statement')}
        - Features: {json.dumps(details.get('initial_features', []))}
        
        Generate JSON array with AT LEAST 6-8 models including:
        1. Core business entities (2-3)
        2. User-related models (User extensions, Teams, Roles)
        3. Supporting models (Notifications, ActivityLog, Settings)
        4. Junction tables for many-to-many relationships
        
        For EACH model provide:
        {{
            "name": "ModelName",
            "purpose": "What this model represents and why it's needed",
            "fields": [
                {{
                    "name": "id",
                    "type": "bigint",
                    "description": "Primary key",
                    "validation": {{"required": true, "auto_increment": true}}
                }},
                {{
                    "name": "field_name",
                    "type": "string|integer|boolean|json|date|decimal",
                    "description": "What this field stores",
                    "validation": {{
                        "required": true|false,
                        "unique": true|false,
                        "min": "for numbers",
                        "max": "for numbers/strings",
                        "pattern": "for strings",
                        "default": "default value"
                    }}
                }}
            ],
            "relationships": [
                "belongsTo:User:user_id",
                "hasMany:RelatedModel:foreign_key",
                "belongsToMany:OtherModel:through:pivot_table"
            ],
            "key_methods": [
                {{
                    "name": "methodName",
                    "description": "What this method does",
                    "returns": "return type"
                }}
            ],
            "indexes": [
                "idx_user_id",
                "idx_created_at",
                "unique_email"
            ],
            "traits": ["SoftDeletes", "HasTimestamps", "Auditable"]
        }}
        
        Include all models referenced in relationships!
        """
        
        models = await self._generate_with_retry(prompt, "domain_models")
        
        # Ensure we got a list
        if isinstance(models, dict) and "models" in models:
            models = models["models"]
        elif not isinstance(models, list):
            self.logger.error("Domain models response was not a list")
            models = []
        
        # Validate we have enough models
        if len(models) < 6:
            self.logger.warning(f"Only generated {len(models)} models, need at least 6")
            # Could retry or enhance here
        
        return models
    
    async def _generate_state_machines(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate state machine definitions for stateful entities."""
        prompt = f"""
        Generate state machine definitions for the {details.get('name')} project.
        
        Context:
        - Features: {json.dumps(details.get('initial_features', []))}
        
        For each stateful entity, define:
        {{
            "[entity_name]": {{
                "description": "What this state machine controls",
                "initial_state": "starting_state",
                "states": [
                    {{
                        "name": "state_name",
                        "description": "What this state means",
                        "allowed_actions": ["action1", "action2"]
                    }}
                ],
                "transitions": [
                    {{
                        "from": "state1",
                        "to": "state2",
                        "trigger": "event_name",
                        "conditions": ["condition1", "condition2"],
                        "actions": ["send_notification", "update_timestamp"],
                        "guard": "user.can_perform_action"
                    }}
                ],
                "error_handling": {{
                    "invalid_transition": "Log and notify admin",
                    "stuck_state": "Timeout and auto-recover"
                }}
            }}
        }}
        
        Include state machines for:
        - User account lifecycle
        - Main entity workflow
        - Payment/subscription states
        - Any async processes
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        return self._parse_json_response(response)
    
    async def _generate_api_documentation(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete API documentation with examples."""
        prompt = f"""
        Generate complete API documentation for the {details.get('name')} project.
        
        For each endpoint provide:
        {{
            "endpoints": [
                {{
                    "path": "/api/v1/resource",
                    "method": "GET|POST|PUT|DELETE",
                    "description": "What this endpoint does",
                    "authentication": "Bearer token|API key|Public",
                    "rate_limit": "60/hour",
                    "request": {{
                        "headers": {{
                            "Authorization": "Bearer {{token}}",
                            "Content-Type": "application/json"
                        }},
                        "params": {{
                            "param_name": {{
                                "type": "string",
                                "required": true,
                                "description": "What this param is for"
                            }}
                        }},
                        "body": {{
                            "field_name": {{
                                "type": "string",
                                "required": true,
                                "validation": "email"
                            }}
                        }},
                        "example": {{}}
                    }},
                    "responses": {{
                        "200": {{
                            "description": "Success response",
                            "schema": {{}},
                            "example": {{}}
                        }},
                        "400": {{
                            "description": "Validation error",
                            "schema": {{}},
                            "example": {{}}
                        }},
                        "401": {{
                            "description": "Unauthorized",
                            "schema": {{}},
                            "example": {{}}
                        }}
                    }},
                    "curl_example": "curl -X GET..."
                }}
            ],
            "error_codes": {{
                "VALIDATION_ERROR": "Invalid input data",
                "RESOURCE_NOT_FOUND": "Requested resource doesn't exist",
                "RATE_LIMIT_EXCEEDED": "Too many requests"
            }},
            "versioning": {{
                "strategy": "URL path versioning",
                "current": "v1",
                "deprecated": [],
                "sunset_policy": "6 months notice"
            }}
        }}
        
        Include endpoints for all CRUD operations and special actions.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
        return self._parse_json_response(response)
    
    async def _generate_code_examples(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation code examples."""
        prompt = f"""
        Generate code examples for key implementations in the {details.get('name')} project.
        
        IMPORTANT: Return ONLY valid JSON. Code examples should be strings with escaped newlines and quotes.
        
        Return JSON with this structure:
        {{
            "controller_example": {{
                "description": "Typical controller with CRUD operations",
                "code": "<?php\\n\\nnamespace App\\\\Http\\\\Controllers;\\n\\nclass ExampleController extends Controller\\n{{\\n    public function index()\\n    {{\\n        // Implementation\\n    }}\\n}}"
            }},
            "service_layer": {{
                "description": "Business logic service",
                "code": "<?php\\n\\nnamespace App\\\\Services;\\n\\nclass ExampleService\\n{{\\n    public function processData($data)\\n    {{\\n        // Validation\\n        // Business logic\\n        // Return result\\n    }}\\n}}"
            }},
            "ai_integration": {{
                "description": "AI service integration example",
                "code": "<?php\\n\\npublic function analyzeContent($content)\\n{{\\n    try {{\\n        $result = $this->aiService->analyze($content);\\n        Cache::put($cacheKey, $result, 3600);\\n        return $result;\\n    }} catch (RateLimitException $e) {{\\n        return Cache::get($cacheKey, null);\\n    }}\\n}}"
            }},
            "queue_job": {{
                "description": "Background job example",
                "code": "<?php\\n\\nclass ProcessDataJob implements ShouldQueue\\n{{\\n    use Dispatchable, InteractsWithQueue;\\n\\n    public function handle()\\n    {{\\n        // Job logic\\n    }}\\n}}"
            }},
            "react_component": {{
                "description": "React component with TypeScript",
                "code": "interface Props {{\\n    data: DataType;\\n}}\\n\\nexport default function Component({{ data }}: Props) {{\\n    return <div>{{data.value}}</div>;\\n}}"
            }},
            "api_test": {{
                "description": "Feature test example",
                "code": "<?php\\n\\npublic function test_api_endpoint()\\n{{\\n    $response = $this->postJson('/api/v1/resource', [\\n        'field' => 'value'\\n    ]);\\n\\n    $response->assertStatus(201)\\n        ->assertJson(['success' => true]);\\n}}"
            }}
        }}
        
        Make examples specific to this project's domain.
        """
        
        return await self._generate_with_retry(prompt, "code_examples")
    
    async def _generate_infrastructure(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate infrastructure and deployment details."""
        prompt = f"""
        Generate infrastructure configuration for the {details.get('name')} project.
        
        IMPORTANT: Return ONLY valid JSON. Do not include any code examples or markdown formatting.
        
        Return JSON with this structure:
        {{
            "environments": {{
                "local": {{
                    "setup": "Docker Compose configuration",
                    "services": ["web", "database", "redis", "queue"],
                    "volumes": ["./src:/app", "db_data:/var/lib/mysql"]
                }},
                "staging": {{
                    "provider": "AWS/GCP/Azure",
                    "services": ["EC2/GCE", "RDS", "ElastiCache"],
                    "configuration": "Auto-scaling, load balanced"
                }},
                "production": {{
                    "provider": "AWS/GCP/Azure",
                    "services": ["ECS/GKE", "RDS Multi-AZ", "ElastiCache cluster"],
                    "configuration": "HA, auto-scaling, CDN"
                }}
            }},
            "ci_cd": {{
                "pipeline": "GitHub Actions",
                "stages": ["lint", "test", "build", "deploy"],
                "deployment_strategy": "Blue-green deployment"
            }},
            "scaling": {{
                "application": "Horizontal pod autoscaling",
                "database": "Read replicas, connection pooling",
                "cache": "Redis cluster with persistence",
                "queue": "Multiple workers, priority queues"
            }},
            "monitoring": {{
                "metrics": "Prometheus + Grafana",
                "logs": "ELK stack or CloudWatch",
                "alerts": "PagerDuty integration",
                "apm": "New Relic or DataDog"
            }}
        }}
        """
        
        return await self._generate_with_retry(prompt, "infrastructure")
    
    async def _generate_testing_strategy(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive testing strategy."""
        prompt = f"""
        Generate testing strategy for the {details.get('name')} project.
        
        IMPORTANT: Return ONLY valid JSON. Do not include any code examples or markdown formatting.
        
        Return JSON with this structure:
        {{
            "test_types": {{
                "unit": {{
                    "coverage_target": "80%",
                    "tools": ["PHPUnit", "Jest"],
                    "focus_areas": ["Models", "Services", "Utilities"],
                    "example": "test code snippet"
                }},
                "integration": {{
                    "coverage_target": "70%",
                    "tools": ["PHPUnit", "Cypress"],
                    "focus_areas": ["API endpoints", "Database operations"],
                    "example": "test code snippet"
                }},
                "e2e": {{
                    "coverage_target": "Critical paths",
                    "tools": ["Cypress", "Playwright"],
                    "scenarios": ["User registration", "Core workflow", "Payment"],
                    "example": "test code snippet"
                }}
            }},
            "testing_data": {{
                "factories": "Model factories for test data",
                "seeders": "Database seeders for scenarios",
                "fixtures": "Static test data files"
            }},
            "ci_integration": {{
                "pre_commit": ["Linting", "Unit tests"],
                "pull_request": ["All tests", "Coverage check"],
                "deployment": ["E2E tests", "Performance tests"]
            }}
        }}
        """
        
        return await self._generate_with_retry(prompt, "testing_strategy")
    
    async def _generate_with_retry(self, prompt: str, section_name: str) -> Dict[str, Any]:
        """Generate AI response with retry logic for JSON parsing failures."""
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    # Make the prompt progressively more explicit
                    if attempt == 2:
                        enhanced_prompt = f"""
{prompt}

IMPORTANT: Return ONLY valid JSON without any markdown formatting or code blocks.
"""
                    else:  # attempt 3
                        enhanced_prompt = f"""
You MUST return ONLY valid JSON for the {section_name} section.

{prompt}

CRITICAL REQUIREMENTS:
1. Start your response with {{ and end with }}
2. Do NOT include ```json or ``` markers
3. Do NOT include any explanatory text
4. Ensure all strings are properly escaped
5. No trailing commas before }} or ]
6. Return ONLY the JSON object, nothing else
"""
                else:
                    enhanced_prompt = prompt
                
                response = await self.ai_brain.generate_enhanced_response(enhanced_prompt, model='gemini')
                result = self._parse_json_response(response)
                
                # Check if retry is needed
                if isinstance(result, dict) and result.get("_retry_needed"):
                    self.logger.warning(f"[{section_name}] Retry needed for JSON parsing (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(1)  # Brief pause before retry
                    continue
                    
                # Check if there was an error
                if isinstance(result, dict) and result.get("_error"):
                    if attempt < max_attempts:
                        self.logger.warning(f"[{section_name}] JSON parsing error, retrying (attempt {attempt}/{max_attempts})")
                        self.logger.debug(f"[{section_name}] Error details: {result.get('_error_message')}")
                        await asyncio.sleep(1)  # Brief pause before retry
                        continue
                    else:
                        self.logger.error(f"[{section_name}] Failed to generate valid JSON after {max_attempts} attempts")
                        self.logger.error(f"[{section_name}] Final error: {result.get('_error_message')}")
                        # Raise exception instead of returning error dict
                        raise ValueError(f"Failed to generate {section_name}: AI returned invalid JSON format")
                
                # Success!
                if hasattr(self, '_retry_count'):
                    self._retry_count = 0  # Reset retry count
                self.logger.info(f"[{section_name}] Successfully generated content on attempt {attempt}")
                return result
                
            except ValueError:
                # Re-raise ValueError to propagate the specific error
                raise
            except Exception as e:
                self.logger.error(f"[{section_name}] Error in attempt {attempt}: {e}")
                if attempt == max_attempts:
                    raise RuntimeError(f"Failed to generate {section_name} after {max_attempts} attempts: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before retry
        
        # Should not reach here
        raise RuntimeError(f"Unexpected error generating {section_name}")
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response with better error handling."""
        content = response.get('content', '')
        
        # Try to extract JSON from markdown code blocks
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            content = content[json_start:json_end].strip()
        elif '```' in content:
            # Check if it's a code block with a language specifier (not JSON)
            code_block_start = content.find('```')
            next_newline = content.find('\n', code_block_start)
            if next_newline > code_block_start:
                language_hint = content[code_block_start + 3:next_newline].strip().lower()
                # If it's clearly not JSON (e.g., php, javascript, etc.), skip extraction
                if language_hint and language_hint not in ['json', '{', '']:
                    # AI returned code instead of JSON, return retry signal
                    self.logger.warning(f"Expected JSON but got {language_hint} code")
                    return {"_retry_needed": True, "_original_content": content}
            
            json_start = content.find('```') + 3
            json_end = content.find('```', json_start)
            content = content[json_start:json_end].strip()
        
        # Clean up common JSON issues
        content = content.strip()
        
        # Handle case where AI returns raw code instead of JSON
        if content.startswith('<?php') or content.startswith('php\n'):
            self.logger.warning("Expected JSON but got PHP code")
            return {"_retry_needed": True, "_original_content": content}
        
        try:
            # First attempt: direct parsing
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            try:
                # Remove trailing commas before closing braces/brackets
                import re
                fixed_content = re.sub(r',(\s*[}\]])', r'\1', content)
                
                # Try to parse the fixed content
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                # If still failing, try to extract just the JSON object
                try:
                    # Find the first { and last }
                    first_brace = content.find('{')
                    last_brace = content.rfind('}')
                    if first_brace >= 0 and last_brace > first_brace:
                        json_content = content[first_brace:last_brace + 1]
                        return json.loads(json_content)
                except:
                    pass
                
                self.logger.error(f"Failed to parse JSON: {e}")
                self.logger.error(f"Content preview: {content[:500]}...")
                
                # Option 1: Retry with a more explicit prompt
                if hasattr(self, '_retry_count'):
                    self._retry_count += 1
                else:
                    self._retry_count = 1
                
                if self._retry_count <= 2:
                    self.logger.warning(f"Retrying with more explicit JSON instructions (attempt {self._retry_count}/2)")
                    # Return a sentinel value to trigger retry
                    return {"_retry_needed": True, "_original_content": content}
                
                # Option 2: Return a structure with error information
                return {
                    "_error": True,
                    "_error_message": str(e),
                    "_partial_content": content[:200],
                    "_fallback_content": {
                        "note": "AI response could not be parsed as JSON",
                        "manual_review_required": True
                    }
                }
    
    def _validate_architecture_completeness(self, architecture: Dict[str, Any]) -> None:
        """Ensure all required sections are present and complete."""
        required_sections = [
            'executive_summary', 'system_context', 'business_rules',
            'domain_models', 'api_documentation', 'infrastructure',
            'testing_strategy', 'state_machines'
        ]
        
        missing = []
        incomplete = []
        
        for section in required_sections:
            if section not in architecture:
                missing.append(section)
            elif not architecture[section]:
                incomplete.append(section)
        
        if missing:
            self.logger.warning(f"Missing sections: {missing}")
        
        if incomplete:
            self.logger.warning(f"Incomplete sections: {incomplete}")
        
        # Validate domain models specifically
        models = architecture.get('domain_models', [])
        if len(models) < 5:
            self.logger.warning(f"Only {len(models)} domain models defined, should have at least 5")
        
        # Check for missing model references
        all_relationships = []
        for model in models:
            all_relationships.extend(model.get('relationships', []))
        
        model_names = [m['name'] for m in models]
        for rel in all_relationships:
            # Extract model name from relationship
            parts = rel.split(':')
            if len(parts) > 1:
                related_model = parts[1]
                if related_model not in model_names and related_model != 'User':
                    self.logger.warning(f"Referenced model '{related_model}' not defined")
    
    def _balance_section_detail(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure no section dominates the document."""
        # Calculate total size
        total_size = 0
        section_sizes = {}
        
        for section, content in architecture.items():
            size = len(json.dumps(content))
            section_sizes[section] = size
            total_size += size
        
        # Check if any section is too large (>30% of total)
        for section, size in section_sizes.items():
            if total_size > 0 and size / total_size > 0.3:
                self.logger.warning(f"Section '{section}' is {size/total_size:.1%} of document (too large)")
                # In a real implementation, we might trim or summarize this section
        
        return architecture
    
    # Add remaining generation methods...
    async def _generate_executive_summary(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "project": f"{details.get('name')} - {details.get('value_proposition', 'AI-powered solution')}",
            "problem_domain": details.get('problem_statement', ''),
            "solution_type": f"AI-powered {details.get('category', 'platform')}",
            "architecture_pattern": "Inertia.js Monolith with Event-Driven AI Services",
            "deployment_target": "Local Development → Cloud Production"
        }
    
    async def _generate_system_context(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system context."""
        metadata = details.get('metadata', {})
        market_research = metadata.get('market_research', {})
        
        return {
            "problem_analysis": {
                "problem": market_research.get('problem_description', details.get('problem_statement', '')),
                "current_solutions": market_research.get('current_solutions', 'Existing solutions lack key features'),
                "our_approach": market_research.get('unique_angle', 'AI-powered solution with unique approach'),
                "key_innovation": market_research.get('key_innovation', 'Leveraging AI for automation and insights')
            },
            "user_journey": [
                f"Discovery → Users find us through {details.get('marketing_channels', 'organic search and referrals')}",
                "Onboarding → Simple signup with email/OAuth, quick profile setup",
                f"First Value → Experience {details.get('key_feature', 'core feature')} within 2 minutes",
                "Engagement → Daily/weekly usage for core workflows",
                f"Retention → Users stay for {details.get('retention_driver', 'time savings and unique insights')}"
            ]
        }
    
    async def _generate_portal_separation(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portal separation details."""
        # This is already quite detailed in the existing implementation
        # We'll keep the comprehensive lists but ensure they're relevant
        feature_name = details.get('name', 'Feature').replace('-', ' ').title()
        
        return {
            "admin_controllers": [
                "DashboardController.php - Platform metrics and KPIs",
                "UsersController.php - User management CRUD",
                "AnalyticsController.php - Business intelligence",
                "RevenueController.php - Financial metrics",
                f"{feature_name}ManagementController.php - Manage {feature_name.lower()} entities",
                "SystemHealthController.php - Infrastructure monitoring",
                "AuditLogController.php - Security and compliance",
                "ConfigurationController.php - System settings"
            ],
            "user_controllers": [
                "DashboardController.php - User overview",
                f"{feature_name}Controller.php - Core {feature_name.lower()} features",
                "SettingsController.php - User preferences",
                "ProfileController.php - Profile management",
                "BillingController.php - Subscription management",
                "NotificationController.php - Notification center"
            ],
            "shared_components": [
                "DataTable.tsx - Reusable data grid",
                "Chart.tsx - Analytics charts",
                "Modal.tsx - Modal dialogs",
                f"{feature_name}Card.tsx - Display {feature_name.lower()} items",
                "SearchBar.tsx - Global search",
                "FilterPanel.tsx - Advanced filters"
            ],
            "api_endpoints": {
                "admin": [
                    "/api/admin/metrics - Platform metrics",
                    "/api/admin/users - User management",
                    f"/api/admin/{feature_name.lower()} - {feature_name} management"
                ],
                "user": [
                    f"/api/v1/{feature_name.lower()} - {feature_name} CRUD",
                    "/api/v1/user - User profile",
                    "/api/v1/settings - User settings"
                ]
            }
        }
    
    async def _generate_data_architecture(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data architecture details."""
        # This should work with the domain models to create complete schemas
        return {
            "database": {
                "type": "MySQL/PostgreSQL",
                "version": "8.0+/13+",
                "configuration": "InnoDB, UTF8MB4"
            },
            "caching": {
                "provider": "Redis",
                "strategy": "Write-through with TTL",
                "key_patterns": {
                    "user_data": "user:{id}:* - 5 min TTL",
                    "api_responses": "api:{endpoint}:{params} - 1 hour TTL",
                    "analytics": "analytics:{type}:{period} - 1 hour TTL"
                }
            },
            "queues": {
                "driver": "Redis/Database",
                "workers": 3,
                "queues": ["default", "emails", "ai-processing", "analytics"]
            },
            "search": {
                "engine": "Database full-text / Elasticsearch",
                "indexed_models": ["Main entities", "Users", "Activity logs"]
            }
        }
    
    async def _generate_ai_integration(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI integration details."""
        return {
            "providers": {
                "primary": "OpenAI GPT-4",
                "fallback": ["Google Gemini", "Anthropic Claude"],
                "specialized": {
                    "vision": "GPT-4 Vision",
                    "embeddings": "OpenAI Embeddings"
                }
            },
            "use_cases": [
                {
                    "name": "Content Analysis",
                    "description": f"Analyze {details.get('analysis_target', 'user content')}",
                    "model": "gpt-4",
                    "max_tokens": 1000
                },
                {
                    "name": "Recommendation Engine",
                    "description": "Generate personalized recommendations",
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 500
                }
            ],
            "optimization": {
                "caching": "Cache AI responses for 24 hours",
                "batching": "Batch similar requests",
                "fallback": "Use cached results on API failure"
            },
            "prompts": {
                "system_context": f"You are an AI assistant for {details.get('name')}, helping users with {details.get('problem_statement')}",
                "templates": ["analysis_prompt", "recommendation_prompt", "summary_prompt"]
            }
        }
    
    async def _generate_security_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security implementation details."""
        return {
            "authentication": {
                "method": "Laravel Sanctum + OAuth",
                "providers": ["Email/Password", "Google", "GitHub"],
                "session": "Secure, HTTP-only cookies",
                "mfa": "Optional TOTP support"
            },
            "authorization": {
                "model": "Role-based (RBAC)",
                "roles": ["admin", "user", "premium_user"],
                "policies": "Resource-based policies for fine-grained control"
            },
            "data_protection": {
                "encryption": {
                    "at_rest": "AES-256 for sensitive fields",
                    "in_transit": "TLS 1.3+ everywhere",
                    "keys": "Managed via environment variables"
                },
                "pii_handling": {
                    "storage": "Encrypted and access-logged",
                    "retention": "As per GDPR requirements",
                    "deletion": "Soft delete with purge after 30 days"
                }
            },
            "security_headers": [
                "Content-Security-Policy",
                "X-Frame-Options: DENY",
                "X-Content-Type-Options: nosniff",
                "Strict-Transport-Security"
            ],
            "compliance": {
                "standards": ["GDPR", "CCPA", "SOC 2 Type II ready"],
                "features": ["Data export", "Right to deletion", "Consent management"]
            }
        }
    
    async def _generate_performance_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance optimization details."""
        return {
            "targets": {
                "page_load": "< 2 seconds (LCP)",
                "api_response": "< 200ms p95",
                "ai_processing": "< 5 seconds",
                "concurrent_users": "1,000 initial, 10,000 target"
            },
            "optimization_strategies": {
                "database": [
                    "Proper indexing on all foreign keys and search fields",
                    "Query optimization with EXPLAIN analysis",
                    "Connection pooling",
                    "Read replicas for analytics"
                ],
                "application": [
                    "Eager loading to prevent N+1 queries",
                    "Response caching with Redis",
                    "CDN for static assets",
                    "Image optimization and lazy loading"
                ],
                "frontend": [
                    "Code splitting with dynamic imports",
                    "Service worker for offline support",
                    "Optimized bundle size < 200KB initial",
                    "Preload critical resources"
                ]
            },
            "monitoring": {
                "apm": "New Relic or DataDog",
                "metrics": ["Response time", "Error rate", "Throughput", "Apdex score"],
                "alerts": "PagerDuty integration for critical issues"
            }
        }
    
    async def _generate_monitoring_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring and observability details."""
        return {
            "logging": {
                "structure": "JSON formatted logs",
                "levels": ["ERROR", "WARN", "INFO", "DEBUG"],
                "retention": {
                    "errors": "90 days",
                    "info": "30 days",
                    "debug": "7 days"
                },
                "channels": ["application", "security", "ai_operations", "user_activity"]
            },
            "metrics": {
                "business": [
                    "User signups per day",
                    "Feature adoption rate",
                    "Churn rate",
                    "Revenue metrics"
                ],
                "technical": [
                    "API response times",
                    "Error rates by endpoint",
                    "Database query performance",
                    "Queue job processing time"
                ],
                "ai_specific": [
                    "Tokens used per request",
                    "AI API costs",
                    "Cache hit rate",
                    "Fallback usage"
                ]
            },
            "alerting": {
                "critical": [
                    "Application down",
                    "Payment processing failures",
                    "Data breach attempts",
                    "AI service outages"
                ],
                "warning": [
                    "High error rate (>5%)",
                    "Slow response times",
                    "Queue backlog",
                    "Low disk space"
                ]
            },
            "dashboards": [
                "Executive overview",
                "Technical health",
                "User behavior",
                "AI usage and costs"
            ]
        }
    
    async def _generate_roadmap(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation roadmap."""
        return {
            "phase1": {
                "name": "Foundation",
                "duration": "Week 1",
                "tasks": [
                    "Set up development environment",
                    "Initialize Laravel + React starter kit",
                    "Configure authentication",
                    "Create domain models and migrations",
                    "Build basic CRUD for core entities",
                    "Set up AI service integration"
                ],
                "deliverables": [
                    "Working authentication system",
                    "Database schema implemented",
                    "Basic feature operational",
                    "AI service connected"
                ]
            },
            "phase2": {
                "name": "Core Features",
                "duration": "Weeks 2-3",
                "tasks": [
                    "Implement main business logic",
                    "Build user dashboard",
                    "Create admin panel",
                    "Integrate payment processing",
                    "Add real-time features",
                    "Implement search and filtering"
                ],
                "deliverables": [
                    "3-5 core features complete",
                    "Payment system integrated",
                    "Real-time updates working",
                    "Admin can manage platform"
                ]
            },
            "phase3": {
                "name": "Polish & Launch",
                "duration": "Week 4",
                "tasks": [
                    "Performance optimization",
                    "Security audit",
                    "UI/UX polish",
                    "Write tests (80% coverage)",
                    "Documentation",
                    "Deployment setup"
                ],
                "deliverables": [
                    "Production-ready application",
                    "Comprehensive test suite",
                    "Deployment pipeline",
                    "User documentation"
                ]
            }
        }