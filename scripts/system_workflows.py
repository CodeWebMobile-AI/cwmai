"""
System-Wide Workflow Definitions
================================

This module contains comprehensive workflow definitions for automating
various system operations using the workflow orchestrator.
"""

SYSTEM_WORKFLOWS = {
    # 1. System Health Check & Auto-Fix Workflow
    "system_health_check": {
        "name": "System Health Check & Auto-Fix",
        "description": "Comprehensive health check with automatic issue resolution",
        "steps": [
            {
                "id": "run_health_check",
                "name": "Run Comprehensive Health Check",
                "type": "tool",
                "tool": "run_tool_health_check",
                "params": {}
            },
            {
                "id": "check_tools",
                "name": "Check All Tools",
                "type": "tool",
                "tool": "list_tools",
                "params": {}
            },
            {
                "id": "identify_broken",
                "name": "Identify Broken Tools",
                "type": "transform",
                "transform": "filter",
                "input": "{{step_check_tools_result.tools}}",
                "output": "broken_tools",
                "filter_condition": {
                    "operator": "eq",
                    "left": "{{item.status}}",
                    "right": "error"
                }
            },
            {
                "id": "fix_tools",
                "name": "Fix Broken Tools",
                "type": "loop",
                "items": "{{broken_tools}}",
                "item_var": "tool",
                "body": {
                    "type": "tool",
                    "tool": "fix_broken_tool",
                    "params": {
                        "tool_name": "{{tool.name}}",
                        "error": "{{tool.error}}"
                    }
                }
            },
            {
                "id": "verify_fixes",
                "name": "Verify All Tools Working",
                "type": "tool",
                "tool": "verify_tool_health",
                "params": {}
            },
            {
                "id": "generate_report",
                "name": "Generate Health Report",
                "type": "tool",
                "tool": "generate_report",
                "params": {
                    "report_type": "health_check",
                    "data": {
                        "proactive_check": "{{step_run_health_check_result}}",
                        "total_tools": "{{step_check_tools_result.count}}",
                        "broken_fixed": "{{step_fix_tools_result.results}}",
                        "final_status": "{{step_verify_fixes_result}}"
                    }
                }
            }
        ]
    },

    # 2. Repository Analysis & Issue Creation Workflow
    "repo_analysis_pipeline": {
        "name": "Repository Analysis & Issue Creation",
        "description": "Analyze multiple repositories and create issues for problems",
        "steps": [
            {
                "id": "get_repos",
                "name": "Get Repository List",
                "type": "tool",
                "tool": "list_repositories",
                "params": {
                    "organization": "{{organization_name}}"
                }
            },
            {
                "id": "parallel_analysis",
                "name": "Analyze All Repositories",
                "type": "parallel",
                "steps": [
                    {
                        "type": "loop",
                        "items": "{{step_get_repos_result.repositories}}",
                        "item_var": "repo",
                        "body": {
                            "type": "tool",
                            "tool": "analyze_repository",
                            "params": {
                                "repo_name": "{{repo.name}}",
                                "include_metrics": True,
                                "check_security": True
                            }
                        }
                    }
                ]
            },
            {
                "id": "aggregate_issues",
                "name": "Aggregate All Issues",
                "type": "transform",
                "transform": "aggregate",
                "input": "{{step_parallel_analysis_result}}",
                "output": "total_issues",
                "aggregation": "merge"
            },
            {
                "id": "create_issues",
                "name": "Create GitHub Issues",
                "type": "loop",
                "items": "{{total_issues}}",
                "item_var": "issue",
                "body": {
                    "type": "conditional",
                    "condition": {
                        "operator": "gte",
                        "left": "{{issue.severity}}",
                        "right": "medium"
                    },
                    "then": {
                        "type": "tool",
                        "tool": "create_issue",
                        "params": {
                            "repo": "{{issue.repo}}",
                            "title": "[Auto] {{issue.title}}",
                            "body": "{{issue.description}}",
                            "labels": ["auto-generated", "{{issue.severity}}"]
                        }
                    }
                }
            },
            {
                "id": "notify_critical",
                "name": "Notify on Critical Issues",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{total_issues.critical_count}}",
                    "right": 0
                },
                "then": {
                    "type": "tool",
                    "tool": "send_notification",
                    "params": {
                        "message": "Critical issues found in repositories",
                        "severity": "high"
                    }
                }
            }
        ]
    },

    # 3. Daily Maintenance Workflow
    "daily_maintenance": {
        "name": "Daily System Maintenance",
        "description": "Automated daily maintenance tasks",
        "steps": [
            {
                "id": "cleanup_logs",
                "name": "Clean Up Old Logs",
                "type": "tool",
                "tool": "cleanup_logs",
                "params": {
                    "older_than_days": 7
                }
            },
            {
                "id": "cleanup_temp",
                "name": "Clean Temporary Files",
                "type": "tool",
                "tool": "cleanup_temp_files",
                "params": {}
            },
            {
                "id": "check_redis",
                "name": "Check Redis Health",
                "type": "tool",
                "tool": "check_redis_health",
                "params": {}
            },
            {
                "id": "verify_apis",
                "name": "Verify API Keys",
                "type": "parallel",
                "steps": [
                    {"type": "tool", "tool": "verify_api_key", "params": {"provider": "anthropic"}},
                    {"type": "tool", "tool": "verify_api_key", "params": {"provider": "openai"}},
                    {"type": "tool", "tool": "verify_api_key", "params": {"provider": "github"}}
                ]
            },
            {
                "id": "update_metrics",
                "name": "Update System Metrics",
                "type": "tool",
                "tool": "update_system_metrics",
                "params": {
                    "cleanup_results": "{{step_cleanup_logs_result}}",
                    "redis_health": "{{step_check_redis_result}}",
                    "api_status": "{{step_verify_apis_result}}"
                }
            },
            {
                "id": "daily_report",
                "name": "Generate Daily Report",
                "type": "tool",
                "tool": "generate_report",
                "params": {
                    "report_type": "daily_maintenance",
                    "timestamp": "{{current_timestamp}}"
                }
            }
        ]
    },

    # 4. Research & Knowledge Update Workflow
    "research_knowledge_update": {
        "name": "Research & Knowledge Update",
        "description": "Automated research and knowledge base updates",
        "steps": [
            {
                "id": "define_topics",
                "name": "Get Research Topics",
                "type": "tool",
                "tool": "get_research_topics",
                "params": {}
            },
            {
                "id": "search_papers",
                "name": "Search for Papers",
                "type": "parallel",
                "steps": [
                    {
                        "type": "loop",
                        "items": "{{step_define_topics_result.topics}}",
                        "item_var": "topic",
                        "body": {
                            "type": "tool",
                            "tool": "search_papers",
                            "params": {
                                "query": "{{topic.query}}",
                                "limit": 10,
                                "sort": "relevance"
                            }
                        }
                    }
                ]
            },
            {
                "id": "extract_insights",
                "name": "Extract Key Insights",
                "type": "loop",
                "items": "{{step_search_papers_result.papers}}",
                "item_var": "paper",
                "body": {
                    "type": "tool",
                    "tool": "analyze_paper",
                    "params": {
                        "paper_url": "{{paper.url}}",
                        "extract_insights": True,
                        "summarize": True
                    }
                }
            },
            {
                "id": "update_knowledge",
                "name": "Update Knowledge Base",
                "type": "tool",
                "tool": "update_knowledge_base",
                "params": {
                    "insights": "{{step_extract_insights_result.results}}",
                    "category": "research"
                }
            },
            {
                "id": "create_summary",
                "name": "Create Summary Document",
                "type": "tool",
                "tool": "create_document",
                "params": {
                    "title": "Research Update - {{current_date}}",
                    "content": "{{step_extract_insights_result.summary}}"
                }
            },
            {
                "id": "notify_team",
                "name": "Notify Team Members",
                "type": "tool",
                "tool": "notify_team",
                "params": {
                    "message": "New research insights available",
                    "document_link": "{{step_create_summary_result.url}}"
                }
            }
        ]
    },

    # 5. Code Quality Pipeline Workflow
    "code_quality_pipeline": {
        "name": "Code Quality Pipeline",
        "description": "Comprehensive code quality checks and improvements",
        "steps": [
            {
                "id": "find_python_files",
                "name": "Find All Python Files",
                "type": "tool",
                "tool": "find_files",
                "params": {
                    "pattern": "**/*.py",
                    "exclude": ["venv", "__pycache__"]
                }
            },
            {
                "id": "run_linting",
                "name": "Run Linting",
                "type": "parallel",
                "steps": [
                    {
                        "type": "tool",
                        "tool": "run_linter",
                        "params": {
                            "linter": "pylint",
                            "files": "{{step_find_python_files_result.files}}"
                        }
                    },
                    {
                        "type": "tool",
                        "tool": "run_linter",
                        "params": {
                            "linter": "flake8",
                            "files": "{{step_find_python_files_result.files}}"
                        }
                    }
                ]
            },
            {
                "id": "security_check",
                "name": "Security Vulnerability Check",
                "type": "tool",
                "tool": "run_security_scan",
                "params": {
                    "scan_type": "dependencies"
                }
            },
            {
                "id": "complexity_analysis",
                "name": "Analyze Code Complexity",
                "type": "tool",
                "tool": "analyze_complexity",
                "params": {
                    "files": "{{step_find_python_files_result.files}}",
                    "threshold": 10
                }
            },
            {
                "id": "generate_suggestions",
                "name": "Generate Improvement Suggestions",
                "type": "tool",
                "tool": "ai_code_review",
                "params": {
                    "lint_results": "{{step_run_linting_result}}",
                    "complexity_results": "{{step_complexity_analysis_result}}"
                }
            },
            {
                "id": "create_issues",
                "name": "Create Issues for Problems",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{step_generate_suggestions_result.critical_count}}",
                    "right": 0
                },
                "then": {
                    "type": "loop",
                    "items": "{{step_generate_suggestions_result.critical_issues}}",
                    "item_var": "issue",
                    "body": {
                        "type": "tool",
                        "tool": "create_issue",
                        "params": {
                            "title": "Code Quality: {{issue.title}}",
                            "body": "{{issue.description}}",
                            "labels": ["code-quality", "{{issue.type}}"]
                        }
                    }
                }
            }
        ]
    },

    # 6. Multi-Tool Data Processing Workflow
    "data_processing_pipeline": {
        "name": "Multi-Tool Data Processing",
        "description": "Complex data processing using multiple tools",
        "steps": [
            {
                "id": "fetch_data",
                "name": "Fetch Data from APIs",
                "type": "parallel",
                "steps": [
                    {
                        "type": "tool",
                        "tool": "fetch_api_data",
                        "params": {
                            "endpoint": "{{api1_endpoint}}",
                            "auth": "{{api1_key}}"
                        }
                    },
                    {
                        "type": "tool",
                        "tool": "fetch_api_data",
                        "params": {
                            "endpoint": "{{api2_endpoint}}",
                            "auth": "{{api2_key}}"
                        }
                    }
                ]
            },
            {
                "id": "clean_data",
                "name": "Clean and Transform Data",
                "type": "tool",
                "tool": "clean_data",
                "params": {
                    "raw_data": "{{step_fetch_data_result}}",
                    "remove_nulls": True,
                    "standardize_format": True
                }
            },
            {
                "id": "analyze_patterns",
                "name": "Analyze Patterns with AI",
                "type": "tool",
                "tool": "ai_pattern_analysis",
                "params": {
                    "data": "{{step_clean_data_result.cleaned_data}}",
                    "analysis_type": "trends",
                    "confidence_threshold": 0.8
                }
            },
            {
                "id": "store_results",
                "name": "Store in Redis",
                "type": "tool",
                "tool": "store_in_redis",
                "params": {
                    "key": "analysis_{{current_timestamp}}",
                    "data": "{{step_analyze_patterns_result}}",
                    "ttl": 86400
                }
            },
            {
                "id": "generate_viz",
                "name": "Generate Visualization Recommendations",
                "type": "tool",
                "tool": "recommend_visualizations",
                "params": {
                    "data": "{{step_analyze_patterns_result}}",
                    "chart_types": ["line", "bar", "scatter", "heatmap"]
                }
            },
            {
                "id": "notify_complete",
                "name": "Notify Completion",
                "type": "tool",
                "tool": "send_notification",
                "params": {
                    "message": "Data processing complete",
                    "redis_key": "{{step_store_results_result.key}}",
                    "visualizations": "{{step_generate_viz_result}}"
                }
            }
        ]
    },

    # 7. Intelligent Task Distribution Workflow
    "task_distribution": {
        "name": "Intelligent Task Distribution",
        "description": "Smart assignment of tasks based on skills and availability",
        "steps": [
            {
                "id": "get_issues",
                "name": "Get Pending Issues",
                "type": "tool",
                "tool": "get_github_issues",
                "params": {
                    "state": "open",
                    "labels": ["unassigned"]
                }
            },
            {
                "id": "categorize",
                "name": "Categorize Issues",
                "type": "tool",
                "tool": "ai_categorize_issues",
                "params": {
                    "issues": "{{step_get_issues_result.issues}}",
                    "categories": ["bug", "feature", "documentation", "refactor"]
                }
            },
            {
                "id": "get_team",
                "name": "Get Team Members",
                "type": "tool",
                "tool": "get_team_members",
                "params": {
                    "include_skills": True,
                    "include_availability": True
                }
            },
            {
                "id": "match_skills",
                "name": "Match Issues to Skills",
                "type": "tool",
                "tool": "match_tasks_to_members",
                "params": {
                    "tasks": "{{step_categorize_result}}",
                    "members": "{{step_get_team_result.members}}"
                }
            },
            {
                "id": "assign_tasks",
                "name": "Assign Tasks",
                "type": "loop",
                "items": "{{step_match_skills_result.assignments}}",
                "item_var": "assignment",
                "body": {
                    "type": "tool",
                    "tool": "assign_issue",
                    "params": {
                        "issue_number": "{{assignment.issue_number}}",
                        "assignee": "{{assignment.member}}",
                        "comment": "Assigned based on skill match: {{assignment.reason}}"
                    }
                }
            },
            {
                "id": "create_lists",
                "name": "Create Task Lists",
                "type": "tool",
                "tool": "create_task_lists",
                "params": {
                    "assignments": "{{step_match_skills_result.assignments}}",
                    "format": "markdown"
                }
            },
            {
                "id": "monitor_setup",
                "name": "Setup Progress Monitoring",
                "type": "tool",
                "tool": "setup_task_monitoring",
                "params": {
                    "assignments": "{{step_match_skills_result.assignments}}",
                    "check_interval": 3600
                }
            }
        ]
    },

    # 8. System Performance Monitoring Workflow
    "performance_monitoring": {
        "name": "System Performance Monitoring",
        "description": "Continuous monitoring of system performance",
        "steps": [
            {
                "id": "measure_tools",
                "name": "Measure Tool Performance",
                "type": "tool",
                "tool": "measure_tool_performance",
                "params": {
                    "duration": 300,
                    "include_memory": True
                }
            },
            {
                "id": "check_redis",
                "name": "Monitor Redis Performance",
                "type": "tool",
                "tool": "monitor_redis",
                "params": {
                    "metrics": ["latency", "memory", "connections", "operations"]
                }
            },
            {
                "id": "check_apis",
                "name": "Check API Rate Limits",
                "type": "parallel",
                "steps": [
                    {"type": "tool", "tool": "check_rate_limit", "params": {"api": "anthropic"}},
                    {"type": "tool", "tool": "check_rate_limit", "params": {"api": "openai"}},
                    {"type": "tool", "tool": "check_rate_limit", "params": {"api": "github"}}
                ]
            },
            {
                "id": "detect_anomalies",
                "name": "Detect Anomalies",
                "type": "tool",
                "tool": "detect_performance_anomalies",
                "params": {
                    "tool_metrics": "{{step_measure_tools_result}}",
                    "redis_metrics": "{{step_check_redis_result}}",
                    "api_metrics": "{{step_check_apis_result}}"
                }
            },
            {
                "id": "alert_issues",
                "name": "Alert on Issues",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{step_detect_anomalies_result.anomaly_count}}",
                    "right": 0
                },
                "then": {
                    "type": "tool",
                    "tool": "send_alert",
                    "params": {
                        "severity": "{{step_detect_anomalies_result.max_severity}}",
                        "anomalies": "{{step_detect_anomalies_result.anomalies}}"
                    }
                }
            },
            {
                "id": "optimize",
                "name": "Generate Optimizations",
                "type": "tool",
                "tool": "generate_optimizations",
                "params": {
                    "performance_data": "{{step_measure_tools_result}}",
                    "anomalies": "{{step_detect_anomalies_result}}"
                }
            },
            {
                "id": "save_metrics",
                "name": "Save Performance Metrics",
                "type": "tool",
                "tool": "save_metrics",
                "params": {
                    "timestamp": "{{current_timestamp}}",
                    "metrics": {
                        "tools": "{{step_measure_tools_result}}",
                        "redis": "{{step_check_redis_result}}",
                        "apis": "{{step_check_apis_result}}"
                    }
                }
            }
        ]
    },

    # 9. Documentation Auto-Generation Workflow
    "documentation_generator": {
        "name": "Documentation Auto-Generation",
        "description": "Automatically generate and update documentation",
        "steps": [
            {
                "id": "scan_tools",
                "name": "Scan All Custom Tools",
                "type": "tool",
                "tool": "scan_custom_tools",
                "params": {
                    "include_params": True,
                    "include_examples": True
                }
            },
            {
                "id": "scan_modules",
                "name": "Scan Python Modules",
                "type": "tool",
                "tool": "scan_modules",
                "params": {
                    "extract_docstrings": True,
                    "extract_signatures": True
                }
            },
            {
                "id": "generate_docs",
                "name": "Generate Documentation",
                "type": "parallel",
                "steps": [
                    {
                        "type": "tool",
                        "tool": "generate_tool_docs",
                        "params": {
                            "tools": "{{step_scan_tools_result.tools}}",
                            "format": "markdown"
                        }
                    },
                    {
                        "type": "tool",
                        "tool": "generate_api_docs",
                        "params": {
                            "modules": "{{step_scan_modules_result.modules}}",
                            "format": "markdown"
                        }
                    }
                ]
            },
            {
                "id": "check_outdated",
                "name": "Check for Outdated Docs",
                "type": "tool",
                "tool": "compare_documentation",
                "params": {
                    "new_docs": "{{step_generate_docs_result}}",
                    "existing_docs_path": "docs/"
                }
            },
            {
                "id": "update_docs",
                "name": "Update Documentation Files",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{step_check_outdated_result.changes_count}}",
                    "right": 0
                },
                "then": {
                    "type": "tool",
                    "tool": "update_documentation",
                    "params": {
                        "changes": "{{step_check_outdated_result.changes}}",
                        "backup": True
                    }
                }
            },
            {
                "id": "create_pr",
                "name": "Create Documentation PR",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{step_check_outdated_result.changes_count}}",
                    "right": 0
                },
                "then": {
                    "type": "tool",
                    "tool": "create_pull_request",
                    "params": {
                        "title": "Documentation Update - {{current_date}}",
                        "body": "Automated documentation updates\n\nChanges: {{step_check_outdated_result.summary}}",
                        "branch": "docs-update-{{current_timestamp}}"
                    }
                }
            }
        ]
    },

    # 10. Emergency Response Workflow
    "emergency_response": {
        "name": "Emergency Response",
        "description": "Critical error handling and recovery workflow",
        "steps": [
            {
                "id": "detect_failure",
                "name": "Detect System Failures",
                "type": "tool",
                "tool": "detect_system_failures",
                "params": {
                    "check_all_services": True
                }
            },
            {
                "id": "assess_severity",
                "name": "Assess Severity",
                "type": "tool",
                "tool": "assess_failure_severity",
                "params": {
                    "failures": "{{step_detect_failure_result.failures}}"
                }
            },
            {
                "id": "attempt_recovery",
                "name": "Attempt Automatic Recovery",
                "type": "loop",
                "items": "{{step_detect_failure_result.failures}}",
                "item_var": "failure",
                "body": {
                    "type": "conditional",
                    "condition": {
                        "operator": "lte",
                        "left": "{{failure.severity}}",
                        "right": "medium"
                    },
                    "then": {
                        "type": "tool",
                        "tool": "auto_recover",
                        "params": {
                            "failure_type": "{{failure.type}}",
                            "component": "{{failure.component}}"
                        }
                    }
                }
            },
            {
                "id": "escalate",
                "name": "Escalate Critical Issues",
                "type": "conditional",
                "condition": {
                    "operator": "or",
                    "conditions": [
                        {
                            "operator": "eq",
                            "left": "{{step_assess_severity_result.max_severity}}",
                            "right": "critical"
                        },
                        {
                            "operator": "lt",
                            "left": "{{step_attempt_recovery_result.success_rate}}",
                            "right": 0.5
                        }
                    ]
                },
                "then": {
                    "type": "parallel",
                    "steps": [
                        {
                            "type": "tool",
                            "tool": "page_oncall",
                            "params": {
                                "severity": "critical",
                                "message": "{{step_assess_severity_result.summary}}"
                            }
                        },
                        {
                            "type": "tool",
                            "tool": "create_incident",
                            "params": {
                                "title": "Critical System Failure",
                                "severity": "{{step_assess_severity_result.max_severity}}",
                                "affected_components": "{{step_detect_failure_result.failures}}"
                            }
                        }
                    ]
                }
            },
            {
                "id": "incident_report",
                "name": "Create Incident Report",
                "type": "tool",
                "tool": "create_incident_report",
                "params": {
                    "failures": "{{step_detect_failure_result.failures}}",
                    "recovery_attempts": "{{step_attempt_recovery_result}}",
                    "escalation": "{{step_escalate_result}}"
                }
            },
            {
                "id": "notify_team",
                "name": "Notify Team Members",
                "type": "tool",
                "tool": "notify_team",
                "params": {
                    "channels": ["email", "slack"],
                    "incident_report": "{{step_incident_report_result.url}}"
                }
            },
            {
                "id": "schedule_postmortem",
                "name": "Schedule Post-Mortem",
                "type": "tool",
                "tool": "schedule_meeting",
                "params": {
                    "title": "Incident Post-Mortem: {{step_incident_report_result.incident_id}}",
                    "attendees": "{{step_assess_severity_result.stakeholders}}",
                    "when": "next_business_day"
                }
            }
        ]
    }
}


def get_workflow_definition(workflow_name: str) -> dict:
    """Get a specific workflow definition by name."""
    return SYSTEM_WORKFLOWS.get(workflow_name)


def list_all_workflows() -> list:
    """List all available system workflows."""
    return [
        {
            "name": key,
            "display_name": workflow["name"],
            "description": workflow["description"]
        }
        for key, workflow in SYSTEM_WORKFLOWS.items()
    ]