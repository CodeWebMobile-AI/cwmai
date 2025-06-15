"""
Test file with known improvement opportunities
"""

def process_data(items):
    # This should be converted to list comprehension
    results = []
    for item in items:
        results.append(item.strip().upper())
    return results

def check_value(data, key):
    # This should use dict.get()
    if key in data:
        value = data[key]
    else:
        value = "default"
    return value

def iterate_with_index(items):
    # This should use enumerate
    for i in range(len(items)):
        print(f"{i}: {items[i]}")

class DataProcessor:
    def calculate(self, x, y):
        # Missing docstring
        return x + y
    
    @property
    def expensive_calculation(self):
        # Should be cached
        import time
        time.sleep(0.1)
        return sum(range(1000000))