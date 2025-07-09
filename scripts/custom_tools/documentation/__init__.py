"""
Documentation tools for CWMAI system.

Available tools:
- enhance_architecture_simple: Enhance existing ARCHITECTURE.md files to comprehensive standards
"""

# Only import what exists
try:
    from .enhance_architecture_simple import enhance_architecture_documentation
    __all__ = ['enhance_architecture_documentation']
except ImportError:
    __all__ = []