"""
Fixed schema for duplicate detection that works with all AI providers
"""

def get_fixed_duplicate_analysis_schema():
    """Get a schema that works with Gemini, OpenAI, DeepSeek, and Anthropic."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_duplicate": {
                "type": "boolean",
                "description": "Whether this is a duplicate or very similar project"
            },
            "duplicate_confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0"
            },
            "duplicate_analysis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "matched_projects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "similarity_score": {"type": "number"},
                                "overlap_areas": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "key_differences": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "could_merge": {"type": "boolean"}
                            },
                            "required": ["name", "similarity_score", "overlap_areas", "key_differences", "could_merge"]
                        }
                    }
                },
                "required": ["matched_projects"]
            },
            "recommendation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_new", "enhance_existing", "merge_projects", "pivot_idea"]
                    },
                    "explanation": {"type": "string"},
                    "suggested_approach": {"type": "string"}
                },
                "required": ["action", "explanation", "suggested_approach"]
            },
            "unique_value_analysis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "unique_aspects": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "market_gaps_filled": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "innovation_score": {"type": "number"}
                },
                "required": ["unique_aspects", "market_gaps_filled", "innovation_score"]
            },
            "differentiation_suggestions": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "is_duplicate",
            "duplicate_confidence",
            "duplicate_analysis",
            "recommendation",
            "unique_value_analysis",
            "differentiation_suggestions"
        ]
    }


def get_simple_duplicate_check_schema():
    """A simpler schema for basic duplicate checking."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_duplicate": {
                "type": "boolean",
                "description": "Is this project a duplicate?"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level from 0 to 1"
            },
            "reason": {
                "type": "string",
                "description": "Explanation of the decision"
            },
            "similar_to": {
                "type": "string",
                "description": "Name of the most similar existing project"
            },
            "recommendation": {
                "type": "string",
                "enum": ["create_new", "enhance_existing", "exact_duplicate"],
                "description": "What action to take"
            }
        },
        "required": ["is_duplicate", "confidence", "reason", "similar_to", "recommendation"]
    }