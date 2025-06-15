"""
Simple file for self-improvement demonstration.
Contains basic patterns that can be optimized.
"""

def double_numbers(numbers):
    """Double each number in the list."""
    # Simple optimization opportunity
    doubled = [n * 2 for n in numbers]
    return doubled


def add_prefix(words):
    """Add prefix to each word."""
    # Another simple pattern
    prefixed = ["pre_" + word for word in words]
    return prefixed


def square_values(values):
    # Missing docstring
    squares = [v ** 2 for v in values]
    return squares