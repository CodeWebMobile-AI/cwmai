"""
Central Configuration for Project Discovery System

All system-wide settings and defaults in one place.
"""

# Technology Stack Configuration
TECH_STACK = {
    "backend": {
        "language": "php",
        "framework": "laravel",
        "version": "11+"
    },
    "frontend": {
        "language": "typescript", 
        "framework": "react",
        "version": "18+"
    },
    "database": {
        "system": "mysql",
        "version": "8.0+"
    },
    "cache": {
        "system": "redis",
        "purpose": ["caching", "queues", "sessions"]
    },
    "realtime": {
        "system": "laravel-echo-server",
        "purpose": "websockets"
    },
    "styling": {
        "framework": "tailwind-css",
        "version": "3.0+"
    },
    "auth": {
        "system": "laravel-sanctum",
        "type": "token-based"
    }
}

# Project Defaults
PROJECT_DEFAULTS = {
    "type": "Web Application",
    "architecture": "laravel-react",
    "starter_kit": "https://github.com/laravel/react-starter-kit",
    "complexity": "medium",
    "initial_dev_time": "3 months",
    "team_size": "1-3 developers"
}

# Business Model Templates
MONETIZATION_TEMPLATES = {
    "saas": {
        "tiers": [
            {"name": "Free", "price": 0, "users": 1},
            {"name": "Pro", "price": 29, "users": 5},
            {"name": "Team", "price": 99, "users": 20},
            {"name": "Enterprise", "price": "custom", "users": "unlimited"}
        ],
        "billing": "monthly",
        "trial_days": 14
    },
    "marketplace": {
        "commission": "10-20%",
        "listing_fee": "optional",
        "premium_features": True
    },
    "freemium": {
        "free_tier_limits": "basic features only",
        "conversion_target": "2-5%",
        "upsell_triggers": ["usage limits", "advanced features"]
    }
}

# Market Research Settings
RESEARCH_CONFIG = {
    "min_problem_severity": 7.0,  # 1-10 scale
    "min_market_size": "10000 users",
    "validation_sources_required": 3,
    "competition_analysis_depth": "top 5 competitors"
}

# Project Generation Rules
GENERATION_RULES = {
    "name_max_length": 30,
    "description_max_length": 200,
    "features_per_project": 5,
    "initial_api_endpoints": 6,
    "database_tables_limit": 10
}

# File Structure Template
PROJECT_STRUCTURE = {
    "backend": [
        "app/Http/Controllers/",
        "app/Models/",
        "app/Services/",
        "routes/api.php",
        "database/migrations/"
    ],
    "frontend": [
        "resources/js/components/",
        "resources/js/pages/",
        "resources/js/hooks/",
        "resources/js/services/",
        "resources/js/types/"
    ],
    "config": [
        ".env.example",
        "docker-compose.yml",
        ".github/workflows/ci.yml"
    ]
}

# Excluded Technologies (never use these)
EXCLUDED_TECH = [
    "pusher",  # Use laravel-echo-server instead
    "postgresql",  # Use MySQL instead
    "vue",  # Use React instead
    "angular",  # Use React instead
    "firebase",  # Use Laravel backend instead
    "supabase",  # Use Laravel backend instead
    "nextjs",  # Use Laravel + React instead
    "django",  # Use Laravel instead
    "flask",  # Use Laravel instead
    "express",  # Use Laravel instead
]


def get_full_tech_stack() -> list:
    """Get the complete tech stack as a flat list."""
    stack = []
    for category, config in TECH_STACK.items():
        if isinstance(config, dict):
            if "language" in config:
                stack.append(config["language"])
            if "framework" in config:
                stack.append(config["framework"]) 
            if "system" in config:
                stack.append(config["system"])
    return stack


def validate_tech_choice(technology: str) -> bool:
    """Check if a technology choice is allowed."""
    return technology.lower() not in [t.lower() for t in EXCLUDED_TECH]


def get_monetization_template(model_type: str) -> dict:
    """Get monetization template by type."""
    return MONETIZATION_TEMPLATES.get(model_type, MONETIZATION_TEMPLATES["saas"])