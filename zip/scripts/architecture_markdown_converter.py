#!/usr/bin/env python3
"""Convert structured architecture data to markdown format."""

import json
from typing import Dict, Any, List
from datetime import datetime


class ArchitectureMarkdownConverter:
    """Converts structured architecture data to markdown format."""
    
    @staticmethod
    def convert_to_markdown(architecture: Dict[str, Any]) -> str:
        """Convert architecture dict to markdown string."""
        md_lines = []
        
        # Title
        md_lines.append("# System Architecture\n")
        
        # Executive Summary
        if 'executive_summary' in architecture:
            md_lines.append("## Executive Summary\n")
            summary = architecture['executive_summary']
            md_lines.append(f"**Project**: {summary.get('project', 'N/A')}\n")
            md_lines.append(f"**Problem Domain**: {summary.get('problem_domain', 'N/A')}\n")
            md_lines.append(f"**Solution Type**: {summary.get('solution_type', 'N/A')}\n")
            md_lines.append(f"**Architecture Pattern**: {summary.get('architecture_pattern', 'N/A')}\n")
            md_lines.append(f"**Deployment Target**: {summary.get('deployment_target', 'N/A')}\n")
        
        # System Context
        if 'system_context' in architecture:
            md_lines.append("## System Context\n")
            context = architecture['system_context']
            
            if 'problem_analysis' in context:
                md_lines.append("### Problem Analysis\n")
                analysis = context['problem_analysis']
                md_lines.append(f"**Problem**: {analysis.get('problem', 'N/A')}\n")
                md_lines.append(f"**Current Solutions**: {analysis.get('current_solutions', 'N/A')}\n")
                md_lines.append(f"**Our Approach**: {analysis.get('our_approach', 'N/A')}\n")
                md_lines.append(f"**Key Innovation**: {analysis.get('key_innovation', 'N/A')}\n")
            
            if 'user_journey' in context:
                md_lines.append("### User Journey\n")
                for step in context['user_journey']:
                    md_lines.append(f"- {step}\n")
                md_lines.append("\n")
        
        # Key Features
        if 'key_features' in architecture:
            md_lines.append("## Key Features\n\n")
            md_lines.append("### AI-Generated Features from Market Research\n")
            md_lines.append("The following features were identified through AI-powered market research and analysis:\n\n")
            
            features = architecture.get('key_features', [])
            if isinstance(features, list):
                for i, feature in enumerate(features, 1):
                    if isinstance(feature, dict):
                        md_lines.append(f"{i}. **{feature.get('name', feature)}**\n")
                        if 'description' in feature:
                            md_lines.append(f"   - {feature['description']}\n")
                    else:
                        md_lines.append(f"{i}. **{feature}**\n")
                md_lines.append("\n")
        
        # Business Rules
        if 'business_rules' in architecture:
            md_lines.append("## Business Rules\n")
            rules = architecture['business_rules']
            
            if 'core_rules' in rules:
                md_lines.append("### Core Business Rules\n")
                for category, category_rules in rules['core_rules'].items():
                    md_lines.append(f"#### {category.replace('_', ' ').title()}\n")
                    for rule_type, rule_details in category_rules.items():
                        md_lines.append(f"**{rule_type.replace('_', ' ').title()}**:\n")
                        if isinstance(rule_details, list):
                            for item in rule_details:
                                md_lines.append(f"- {item}\n")
                        elif isinstance(rule_details, dict):
                            for key, value in rule_details.items():
                                md_lines.append(f"- {key}: {value}\n")
                        else:
                            md_lines.append(f"- {rule_details}\n")
                        md_lines.append("\n")
            
            if 'validation_rules' in rules:
                md_lines.append("### Validation Rules\n")
                md_lines.append("```json\n")
                md_lines.append(json.dumps(rules['validation_rules'], indent=2))
                md_lines.append("\n```\n")
            
            if 'edge_cases' in rules:
                md_lines.append("### Edge Cases\n")
                for edge_case in rules['edge_cases']:
                    md_lines.append(f"**{edge_case.get('scenario', 'N/A')}**:\n")
                    md_lines.append(f"- Handling: {edge_case.get('handling', 'N/A')}\n")
                    md_lines.append(f"- Recovery: {edge_case.get('recovery', 'N/A')}\n\n")
        
        # State Machines
        if 'state_machines' in architecture:
            md_lines.append("## State Machines\n")
            for entity, machine in architecture['state_machines'].items():
                md_lines.append(f"### {entity.replace('_', ' ').title()}\n")
                md_lines.append(f"{machine.get('description', '')}\n\n")
                
                if 'states' in machine:
                    md_lines.append("**States**:\n")
                    for state in machine['states']:
                        md_lines.append(f"- **{state['name']}**: {state.get('description', '')}\n")
                    md_lines.append("\n")
                
                if 'transitions' in machine:
                    md_lines.append("**Transitions**:\n")
                    md_lines.append("```\n")
                    for transition in machine['transitions']:
                        conditions = ", ".join(transition.get('conditions', []))
                        md_lines.append(f"{transition['from']} → {transition['to']} (trigger: {transition['trigger']})")
                        if conditions:
                            md_lines.append(f" [if: {conditions}]")
                        md_lines.append("\n")
                    md_lines.append("```\n")
        
        # Domain Models
        if 'domain_models' in architecture:
            md_lines.append("## Domain Models\n")
            for model in architecture['domain_models']:
                md_lines.append(f"### {model['name']}\n")
                md_lines.append(f"**Purpose**: {model.get('purpose', 'N/A')}\n\n")
                
                if 'fields' in model:
                    md_lines.append("**Fields**:\n")
                    for field in model['fields']:
                        validation = field.get('validation', {})
                        val_str = ""
                        if validation:
                            val_parts = []
                            if validation.get('required'):
                                val_parts.append("required")
                            if validation.get('unique'):
                                val_parts.append("unique")
                            if 'min' in validation:
                                val_parts.append(f"min: {validation['min']}")
                            if 'max' in validation:
                                val_parts.append(f"max: {validation['max']}")
                            if val_parts:
                                val_str = f" ({', '.join(val_parts)})"
                        
                        md_lines.append(f"- `{field['name']}` ({field['type']}): {field.get('description', '')}{val_str}\n")
                    md_lines.append("\n")
                
                if 'relationships' in model:
                    md_lines.append("**Relationships**:\n")
                    for rel in model['relationships']:
                        md_lines.append(f"- {rel}\n")
                    md_lines.append("\n")
                
                if 'key_methods' in model:
                    md_lines.append("**Key Methods**:\n")
                    for method in model['key_methods']:
                        if isinstance(method, dict):
                            md_lines.append(f"- `{method['name']}()`: {method.get('description', '')} → {method.get('returns', 'void')}\n")
                        else:
                            md_lines.append(f"- `{method}()`\n")
                    md_lines.append("\n")
        
        # API Documentation
        if 'api_documentation' in architecture:
            md_lines.append("## API Documentation\n")
            api_docs = architecture['api_documentation']
            
            if 'endpoints' in api_docs:
                md_lines.append("### Endpoints\n")
                for endpoint in api_docs['endpoints']:
                    md_lines.append(f"#### {endpoint['method']} {endpoint['path']}\n")
                    md_lines.append(f"{endpoint.get('description', '')}\n\n")
                    
                    if 'authentication' in endpoint:
                        md_lines.append(f"**Authentication**: {endpoint['authentication']}\n")
                    if 'rate_limit' in endpoint:
                        md_lines.append(f"**Rate Limit**: {endpoint['rate_limit']}\n")
                    md_lines.append("\n")
                    
                    if 'request' in endpoint:
                        md_lines.append("**Request**:\n")
                        if 'example' in endpoint['request']:
                            md_lines.append("```json\n")
                            md_lines.append(json.dumps(endpoint['request']['example'], indent=2))
                            md_lines.append("\n```\n")
                    
                    if 'responses' in endpoint:
                        md_lines.append("**Responses**:\n")
                        for code, response in endpoint['responses'].items():
                            md_lines.append(f"- **{code}**: {response.get('description', '')}\n")
                            if 'example' in response:
                                md_lines.append("  ```json\n")
                                md_lines.append("  " + json.dumps(response['example'], indent=2).replace('\n', '\n  '))
                                md_lines.append("\n  ```\n")
                    
                    if 'curl_example' in endpoint:
                        md_lines.append("**Example**:\n")
                        md_lines.append("```bash\n")
                        md_lines.append(endpoint['curl_example'])
                        md_lines.append("\n```\n")
        
        # Implementation Examples
        if 'implementation_examples' in architecture:
            md_lines.append("## Implementation Examples\n")
            for example_name, example in architecture['implementation_examples'].items():
                md_lines.append(f"### {example_name.replace('_', ' ').title()}\n")
                md_lines.append(f"{example.get('description', '')}\n\n")
                if 'code' in example:
                    # Determine language from code
                    lang = "php"
                    if "interface" in example['code'] or "export default" in example['code']:
                        lang = "typescript"
                    md_lines.append(f"```{lang}\n")
                    md_lines.append(example['code'])
                    md_lines.append("\n```\n")
        
        # Infrastructure
        if 'infrastructure' in architecture:
            md_lines.append("## Infrastructure\n")
            infra = architecture['infrastructure']
            
            if 'environments' in infra:
                md_lines.append("### Environments\n")
                for env_name, env_config in infra['environments'].items():
                    md_lines.append(f"#### {env_name.capitalize()}\n")
                    for key, value in env_config.items():
                        if isinstance(value, list):
                            md_lines.append(f"**{key.replace('_', ' ').title()}**:\n")
                            for item in value:
                                md_lines.append(f"- {item}\n")
                        else:
                            md_lines.append(f"**{key.replace('_', ' ').title()}**: {value}\n")
                    md_lines.append("\n")
        
        # Testing Strategy
        if 'testing_strategy' in architecture:
            md_lines.append("## Testing Strategy\n")
            testing = architecture['testing_strategy']
            
            if 'test_types' in testing:
                for test_type, config in testing['test_types'].items():
                    md_lines.append(f"### {test_type.capitalize()} Tests\n")
                    for key, value in config.items():
                        if key == 'example' and value:
                            md_lines.append(f"**Example**:\n")
                            md_lines.append("```\n")
                            md_lines.append(value)
                            md_lines.append("\n```\n")
                        elif isinstance(value, list):
                            md_lines.append(f"**{key.replace('_', ' ').title()}**:\n")
                            for item in value:
                                md_lines.append(f"- {item}\n")
                        else:
                            md_lines.append(f"**{key.replace('_', ' ').title()}**: {value}\n")
                    md_lines.append("\n")
        
        # Add remaining sections (Portal Separation, etc.) if they exist
        # These would follow similar patterns
        
        return ''.join(md_lines)