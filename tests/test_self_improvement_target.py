"""
Test file for self-improvement demonstration.
This file contains patterns that can be optimized.
"""

def process_data(items):
    """Process a list of items."""
    # This can be converted to list comprehension
    results = []
    for item in items:
        results.append(item * 2)
    return results


def filter_values(data):
    """Filter values from data."""
    # Another optimization opportunity
    filtered = []
    for value in data:
        filtered.append(value.upper())
    return filtered


def get_config_value(config, key):
    """Get configuration value with default."""
    # Can use dict.get()
    if key in config:
        value = config[key]
    else:
        value = "default"
    return value


def count_items(items):
    # Missing docstring - documentation opportunity
    count = 0
    for item in items:
        if item > 0:
            count += 1
    return count


class DataProcessor:
    def process(self, data):
        # This method needs documentation
        output = []
        for d in data:
            output.append(d.strip())
        return output