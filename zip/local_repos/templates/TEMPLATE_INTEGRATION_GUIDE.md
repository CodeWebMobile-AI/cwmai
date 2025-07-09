# Template Integration Guide for Enhanced Documentation

## Overview

This guide explains how to integrate the enhanced documentation templates (ARCHITECTURE_TEMPLATE.md, MONETIZATION_STRATEGY_TEMPLATE.md, README_TEMPLATE.md) into the project creator system to generate comprehensive, AI-ready documentation.

## Key Improvements Over Current System

### Current Issues
1. **Sparse Documentation**: Current ARCHITECTURE.md files lack detail and specific implementation guidance
2. **No Starter Kit Acknowledgment**: Doesn't recognize or build upon the Laravel React starter kit
3. **Generic Content**: Same boilerplate text for all projects without problem-specific details
4. **Missing Implementation Details**: No code examples, file paths, or concrete guidance

### Enhanced Template Benefits
1. **Comprehensive Detail**: 50+ pages of actionable content per project
2. **Starter Kit Integration**: Explicitly builds upon existing authentication, UI components, and Inertia.js
3. **Problem-Specific**: Tailored to each project's unique requirements
4. **Implementation-Ready**: Complete code examples, exact file paths, and step-by-step guidance

## Integration Steps

### 1. Update Documentation Generation Prompts

Replace the current simple prompts with comprehensive, template-based prompts:

```python
# In project_creator.py - Update _generate_structured_architecture method

async def _generate_structured_architecture(self, details: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive architecture following enhanced template."""
    
    # Extract market research data
    metadata = details.get('metadata', {})
    market_research = metadata.get('market_research', {})
    
    prompt = f"""
    Generate a comprehensive system architecture document following this exact structure.
    This project builds on the Laravel React starter kit with Inertia.js.
    
    Project Context:
    - Name: {details.get('name')}
    - Problem: {market_research.get('problem_description')}
    - Audience: {market_research.get('affected_audience')}
    - Market Size: {market_research.get('market_size')}
    - Validation Score: {metadata.get('validation_metrics', {}).get('validation_score')}/10
    
    IMPORTANT: The Laravel React starter kit already provides:
    - Complete authentication system (login, register, password reset, email verification)
    - User model and profile management
    - shadcn/ui component library with all UI elements
    - Inertia.js for server-side routing with React frontend
    - TypeScript and Tailwind CSS configuration
    
    Generate JSON with this EXACT structure:
    {{
        "executive_summary": {{
            "project": "{details.get('name')} - [One-line value proposition based on problem]",
            "problem_domain": "[Specific problem from research]",
            "solution_type": "AI-powered [category] platform",
            "architecture_pattern": "Inertia.js Monolith with Event-Driven AI Services",
            "deployment_target": "Local Development → Cloud Production"
        }},
        "system_context": {{
            "problem_analysis": {{
                "problem": "[Detailed from research]",
                "current_solutions": "[From market research]",
                "our_approach": "[How AI solves it differently]",
                "key_innovation": "[Unique technical approach]"
            }},
            "user_journey": [
                "Discovery → [How users find us]",
                "Onboarding → [OAuth connections, setup]",
                "First Value → [Key feature within 2 minutes]",
                "Engagement → [Daily/weekly usage patterns]",
                "Retention → [Why users stay]"
            ]
        }},
        "building_blocks": {{
            "starter_kit_foundation": {{
                "authentication": ["Login/Register ✓", "Password Reset ✓", "Email Verification ✓"],
                "ui_components": ["shadcn/ui forms ✓", "Layouts ✓", "Theme system ✓"],
                "architecture": ["Inertia.js ✓", "Server routing ✓", "TypeScript ✓"]
            }},
            "domain_models": [
                {{
                    "name": "[Core Model 1]",
                    "purpose": "[What it represents]",
                    "fields": [
                        {{"name": "field1", "type": "string", "description": "Purpose"}},
                        {{"name": "field2", "type": "bigint", "description": "Foreign key to users"}}
                    ],
                    "relationships": ["belongsTo:User", "hasMany:[Model2]"],
                    "key_methods": ["calculateValue()", "needsAttention()"]
                }},
                {{
                    "name": "[Core Model 2]",
                    "purpose": "[What it represents]",
                    "fields": [...],
                    "relationships": [...],
                    "key_methods": [...]
                }}
            ]
        }},
        "portal_separation": {{
            "admin_controllers": [
                "DashboardController.php - Platform metrics",
                "UsersController.php - User management",
                "AnalyticsController.php - System analytics"
            ],
            "user_controllers": [
                "DashboardController.php - User overview",
                "[Feature]Controller.php - Core feature",
                "SettingsController.php - User preferences"
            ],
            "admin_pages": [
                "Dashboard.tsx - Admin metrics",
                "Users/Index.tsx - User list",
                "Analytics/Overview.tsx - Platform stats"
            ],
            "user_pages": [
                "Dashboard.tsx - User home",
                "[Feature]/Index.tsx - Feature list",
                "[Feature]/Create.tsx - Create new"
            ],
            "middleware": {{
                "admin_check": "EnsureUserIsAdmin - Checks is_admin flag",
                "route_groups": "Separate /admin and /app prefixes"
            }}
        }},
        "implementation_roadmap": {{
            "phase1_foundation": {{
                "week": 1,
                "tasks": [
                    "Set up Laravel React starter kit",
                    "Implement domain models and migrations",
                    "Create admin/user portal separation",
                    "Build basic CRUD for core entities"
                ],
                "deliverables": [
                    "Working auth system (from starter)",
                    "Database schema implemented",
                    "Portal routing configured",
                    "Basic feature working"
                ]
            }},
            "phase2_core": {{
                "week": "2-3",
                "tasks": [
                    "Integrate AI services (OpenAI/Gemini)",
                    "Build real-time features with Echo",
                    "Implement core business logic",
                    "Create analytics dashboard"
                ],
                "deliverables": [
                    "AI integration working",
                    "Real-time updates",
                    "3-5 core features complete",
                    "Admin can see metrics"
                ]
            }},
            "phase3_enhancement": {{
                "week": 4,
                "tasks": [
                    "Performance optimization",
                    "Advanced features",
                    "Comprehensive testing",
                    "UI/UX polish"
                ],
                "deliverables": [
                    "Sub-2s page loads",
                    "80% test coverage",
                    "Production ready"
                ]
            }}
        }},
        "data_architecture": {{
            "schema": {{
                "users_extension": "Add is_admin, subscription_tier, trial_ends_at to existing users table",
                "[table1]": {{
                    "fields": [
                        "id BIGINT PRIMARY KEY",
                        "user_id BIGINT FOREIGN KEY",
                        "[domain_field] VARCHAR(255)",
                        "status ENUM('active','pending','completed')",
                        "metadata JSON",
                        "created_at TIMESTAMP"
                    ],
                    "indexes": [
                        "idx_user_status (user_id, status)",
                        "idx_created (created_at)"
                    ]
                }},
                "[table2]": {{...}}
            }},
            "caching": {{
                "strategy": "Redis with strategic TTLs",
                "keys": [
                    "user:{{id}}:stats - 5 min TTL",
                    "[feature]:{{id}}:data - 1 hour TTL",
                    "analytics:{{type}}:{{period}} - 1 hour TTL"
                ]
            }},
            "queues": [
                "Process[Feature] - Handles AI processing",
                "Send[Notification] - Email/push notifications",
                "Generate[Report] - Background analytics"
            ]
        }},
        "ai_integration": {{
            "service_interface": {{
                "methods": [
                    "analyze(data): AIResult",
                    "generateContent(prompt, context): string",
                    "extractInsights(content): array"
                ],
                "providers": ["OpenAI (primary)", "Gemini (fallback)", "Local models (future)"]
            }},
            "prompts": {{
                "[feature]_prompt": "Specialized prompt for [specific use case]",
                "system_context": "You are an AI assistant for [domain]..."
            }},
            "error_handling": [
                "Rate limit retry with exponential backoff",
                "Fallback to cached responses",
                "User notification on failures"
            ]
        }},
        "security": {{
            "authentication": {{
                "method": "Laravel Sanctum (from starter kit)",
                "enhancements": "Add API tokens for external access"
            }},
            "authorization": {{
                "policies": [
                    "[Model]Policy - Row-level security",
                    "Feature access by subscription tier"
                ],
                "middleware": [
                    "CheckSubscriptionLimits",
                    "ValidateAPIAccess"
                ]
            }},
            "data_protection": [
                "HTTPS everywhere",
                "Encrypted sensitive fields",
                "GDPR compliance features"
            ]
        }},
        "performance": {{
            "targets": {{
                "page_load": "< 2 seconds",
                "api_response": "< 200ms p95",
                "ai_processing": "< 5 seconds",
                "concurrent_users": "1,000 initial"
            }},
            "optimization": [
                "Eager loading with select fields",
                "Database query optimization",
                "CDN for static assets",
                "Redis caching layer"
            ]
        }},
        "monitoring": {{
            "health_checks": "/health endpoint with component status",
            "logging": {{
                "channels": ["user_activity", "ai_operations", "errors"],
                "retention": "30 days",
                "alerts": ["High error rate", "Slow responses", "AI failures"]
            }},
            "metrics": [
                "Response times by endpoint",
                "AI token usage and costs",
                "User engagement metrics"
            ]
        }}
    }}
    
    Return ONLY valid JSON. Be specific to this project's domain and problem.
    """
    
    response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
    return self._extract_json_from_response(response.get('content', ''))
```

### 2. Enhance Monetization Strategy Generation

```python
async def _generate_structured_monetization(self, details: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive monetization strategy with implementation details."""
    
    prompt = f"""
    Generate a comprehensive monetization strategy that directly ties to technical implementation.
    
    Project: {details.get('name')}
    Market Research:
    - Problem: {market_research.get('problem_description')}
    - Market Size: {market_research.get('market_size')}
    - Competition: {market_research.get('competition_level')}
    - Revenue Potential: {validation_metrics.get('revenue_potential')}
    
    Generate JSON with implementation-focused monetization:
    {{
        "executive_summary": {{
            "business_model": "Subscription SaaS",
            "primary_revenue": "Monthly/annual subscriptions",
            "secondary_revenue": ["API access", "White-label", "Professional services"],
            "target_ltv": "$1,200",
            "payback_period": "3 months"
        }},
        "pricing_tiers": [
            {{
                "name": "Free",
                "price": 0,
                "purpose": "User acquisition",
                "features": [
                    "[Limited feature 1] - up to 3 [items]",
                    "[Limited feature 2] - basic only"
                ],
                "implementation": {{
                    "database_flag": "subscription_tier = 'free'",
                    "middleware_check": "CheckFreeTierLimits",
                    "usage_tracking": "track in usage_records table"
                }},
                "conversion_target": "5% to paid in 30 days"
            }},
            {{
                "name": "Starter",
                "price": 29,
                "features": [
                    "Everything in Free",
                    "[Feature 1] - up to 100 [items]",
                    "[Feature 2] - standard access"
                ],
                "implementation": {{
                    "database_flag": "subscription_tier = 'starter'",
                    "feature_flags": {{"[feature1]_limit": 100, "[feature2]_enabled": true}},
                    "api_rate_limit": "100 requests/hour"
                }}
            }},
            {{
                "name": "Professional",
                "price": 99,
                "features": [
                    "Everything in Starter",
                    "[Advanced feature 1]",
                    "[Advanced feature 2]",
                    "API access"
                ],
                "implementation": {{
                    "database_flag": "subscription_tier = 'professional'",
                    "feature_flags": {{"[feature1]_limit": 1000, "api_enabled": true}},
                    "api_rate_limit": "1000 requests/hour"
                }}
            }}
        ]},
        "implementation_code": {{
            "subscription_check": "Example middleware code for checking limits",
            "usage_tracking": "Example code for tracking feature usage",
            "upgrade_flow": "Example code for upgrade prompts"
        }},
        "metrics_queries": {{
            "mrr_calculation": "SQL query for monthly recurring revenue",
            "churn_analysis": "SQL query for cohort retention",
            "feature_adoption": "SQL query for feature usage by tier"
        }}
    }}
    """
```

### 3. Create Helper Methods for Template-Based Generation

```python
def _get_architecture_template_prompt(self, details: Dict[str, Any]) -> str:
    """Get the full architecture template prompt with project specifics."""
    
    # Load the template
    template_path = Path(__file__).parent.parent / "local_repos" / "templates" / "ARCHITECTURE_TEMPLATE.md"
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders with project-specific details
    replacements = {
        "[Project Name]": details.get('name'),
        "[One-line value proposition]": self._generate_value_proposition(details),
        "[Specific problem being solved]": details.get('metadata', {}).get('market_research', {}).get('problem_description'),
        "[category]": self._determine_project_category(details),
        # ... more replacements
    }
    
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value or placeholder)
    
    return f"""
    Using this architecture template as a guide, generate a complete architecture document.
    Fill in all sections with specific details for this project.
    
    Template:
    {template}
    
    Project Details:
    {json.dumps(details, indent=2)}
    
    Return a complete architecture document following the template structure exactly.
    """
```

### 4. Implement Validation

```python
def _validate_generated_documentation(self, docs: Dict[str, Any]) -> List[str]:
    """Validate that generated documentation meets quality standards."""
    
    issues = []
    
    # Check architecture completeness
    arch = docs.get('architecture', {})
    required_sections = [
        'executive_summary', 'system_context', 'building_blocks',
        'portal_separation', 'implementation_roadmap', 'data_architecture',
        'ai_integration', 'security', 'performance', 'monitoring'
    ]
    
    for section in required_sections:
        if not arch.get(section):
            issues.append(f"Architecture missing required section: {section}")
    
    # Check for starter kit acknowledgment
    if 'starter_kit_foundation' not in str(arch):
        issues.append("Architecture doesn't acknowledge Laravel React starter kit")
    
    # Check for code examples
    if arch and 'class ' not in str(arch) and 'function ' not in str(arch):
        issues.append("Architecture lacks code examples")
    
    # Check monetization completeness
    monetization = docs.get('monetization', {})
    if monetization:
        if not monetization.get('pricing_tiers'):
            issues.append("Monetization missing pricing tiers")
        if not monetization.get('implementation_code'):
            issues.append("Monetization missing implementation details")
    
    return issues
```

### 5. Update the Main Create Method

```python
async def create_project(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new project with enhanced documentation."""
    
    try:
        # ... existing setup code ...
        
        # Generate comprehensive structured documentation
        self.logger.info("📚 Generating enhanced documentation...")
        structured_docs = await self._generate_enhanced_documentation(repo, details)
        
        # Validate documentation quality
        issues = self._validate_generated_documentation(structured_docs)
        if issues:
            self.logger.warning(f"Documentation validation issues: {issues}")
            # Optionally retry generation for critical issues
        
        # Save comprehensive metadata first
        await self._save_enhanced_metadata(repo, details, structured_docs, results)
        
        # Transform to markdown with enhanced formatting
        await self._create_enhanced_markdown_files(repo, structured_docs, results)
        
        # Create implementation-focused issues
        issue_numbers = await self._create_implementation_issues(repo, structured_docs)
        
        # ... rest of the method ...
```

## Expected Outcomes

### Before (Current System)
```markdown
# System Architecture

## Overview
**Description**: Clear system architecture overview for review-insight-ai
**Architecture Style**: monolithic

## Core Entities
[Empty - not populated]

## System Components
### Laravel API
**Type**: backend
**Description**: RESTful API server
```

### After (Enhanced Templates)
```markdown
# System Architecture - ReviewInsight AI

## Executive Summary
**Project**: ReviewInsight AI - Transform customer feedback into business growth  
**Problem Domain**: Small businesses struggle to manage reviews across 50+ platforms
**Solution Type**: AI-powered review management and reputation enhancement platform
**Architecture Pattern**: Inertia.js Monolith with Event-Driven AI Services

## Building Blocks

### Laravel React Starter Kit Foundation
What We Inherit:
├── Authentication System (complete)
│   ├── Login/Register/Password Reset ✓
│   ├── Email Verification ✓
│   └── Profile Management ✓

### Domain-Specific Extensions

#### 1. Models & Relationships
```php
// app/Models/Review.php
class Review extends Model
{
    protected $fillable = [
        'business_id',    // bigint: Business that owns this
        'platform_id',    // bigint: Source platform  
        'rating',         // tinyint: 1-5 stars
        'content',        // text: Review content
        'sentiment',      // enum: positive/negative/neutral
    ];
    
    public function business(): BelongsTo
    {
        return $this->belongsTo(Business::class);
    }
    
    public function needsUrgentResponse(): bool
    {
        return $this->rating <= 2 && !$this->response;
    }
}
```

[... continues with full implementation details ...]
```

## Integration Checklist

- [ ] Update project_creator.py to use enhanced prompts
- [ ] Add template validation methods
- [ ] Implement chunked generation for long documents
- [ ] Add quality checks for generated content
- [ ] Test with multiple project types
- [ ] Monitor token usage and optimize prompts
- [ ] Add fallback handling for generation failures
- [ ] Create metrics for documentation quality

## Best Practices

1. **Always acknowledge the starter kit** - Don't regenerate what already exists
2. **Be specific, not generic** - Use market research data in prompts
3. **Include implementation details** - Code examples, file paths, database schemas
4. **Link features to monetization** - Show how technical features enable pricing
5. **Progressive enhancement** - Build phases that deliver value incrementally
6. **Test the output** - Ensure AI agents can build from the documentation alone

This integration will transform the project creation system from generating sparse, generic documentation to creating comprehensive, implementation-ready guides that enable true AI-driven development.