#!/usr/bin/env python3
"""
Diagnostic script to identify where the schema complexity error is occurring.
"""

import json
import sys
from pathlib import Path
import importlib.util

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def check_schema_size(obj, path="root"):
    """Recursively calculate the size and complexity of a schema."""
    if isinstance(obj, dict):
        size = len(json.dumps(obj))
        nested_depth = 0
        total_properties = 0
        
        def count_depth(d, depth=0):
            nonlocal nested_depth, total_properties
            if isinstance(d, dict):
                if 'properties' in d:
                    total_properties += len(d.get('properties', {}))
                    nested_depth = max(nested_depth, depth)
                    for v in d.get('properties', {}).values():
                        count_depth(v, depth + 1)
                for k, v in d.items():
                    if k != 'properties':
                        count_depth(v, depth)
        
        count_depth(obj)
        return size, nested_depth, total_properties
    return 0, 0, 0

def analyze_gemini_schemas():
    """Analyze all schemas in gemini_deep_research.py."""
    print("🔍 Analyzing schemas in gemini_deep_research.py...")
    print("=" * 60)
    
    # Import the module
    module_path = Path(__file__).parent / "scripts" / "gemini_deep_research.py"
    spec = importlib.util.spec_from_file_location("gemini_deep_research", module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Read the file content
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Find all schema definitions
    import re
    import ast
    
    # Pattern to find schema definitions
    schema_pattern = r'(\w+_schema)\s*=\s*\{'
    matches = list(re.finditer(schema_pattern, content))
    
    print(f"Found {len(matches)} schema definitions\n")
    
    schemas_found = []
    
    for match in matches:
        schema_name = match.group(1)
        start = match.start()
        
        # Find the end of the schema by counting braces
        brace_count = 0
        in_string = False
        escape = False
        schema_start = None
        
        for i in range(start, len(content)):
            char = content[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == '{':
                    if schema_start is None:
                        schema_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and schema_start is not None:
                        schema_str = content[schema_start:i+1]
                        
                        # Find which method this schema is in
                        method_match = None
                        for m in re.finditer(r'async def (\w+)\(', content[:start]):
                            method_match = m
                        
                        method_name = method_match.group(1) if method_match else "unknown"
                        
                        # Try to parse the schema
                        try:
                            # Use ast.literal_eval for safer parsing
                            schema_dict = ast.literal_eval(schema_str)
                            size, depth, props = check_schema_size(schema_dict)
                            
                            schemas_found.append({
                                'name': schema_name,
                                'method': method_name,
                                'size': size,
                                'depth': depth,
                                'properties': props,
                                'line': content[:start].count('\n') + 1
                            })
                            
                            # Flag potentially problematic schemas
                            if size > 50000 or depth > 3 or props > 30:
                                print(f"⚠️  POTENTIAL ISSUE: {schema_name}")
                            else:
                                print(f"✅ {schema_name}")
                            
                            print(f"   Method: {method_name}")
                            print(f"   Line: {content[:start].count('\n') + 1}")
                            print(f"   Size: {size:,} bytes")
                            print(f"   Max depth: {depth}")
                            print(f"   Total properties: {props}")
                            print()
                            
                        except Exception as e:
                            print(f"❌ {schema_name} - Failed to parse: {e}")
                            print(f"   Method: {method_name}")
                            print(f"   Line: {content[:start].count('\n') + 1}")
                            print()
                        
                        break
    
    # Check for the specific fix
    print("\n" + "="*60)
    print("🔍 Checking for schema fix...")
    
    if '"search_evidence": {' in content and '"summary": {"type": "string"}' in content:
        print("✅ Schema fix is present in the code!")
        
        # Find the exact location
        fix_match = re.search(r'"search_evidence":\s*\{\s*"type":\s*"object",\s*"properties":\s*\{\s*"summary"', content)
        if fix_match:
            line_num = content[:fix_match.start()].count('\n') + 1
            print(f"   Found at line ~{line_num}")
    else:
        print("❌ Schema fix NOT found in the code!")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"   Total schemas analyzed: {len(schemas_found)}")
    
    large_schemas = [s for s in schemas_found if s['size'] > 50000]
    deep_schemas = [s for s in schemas_found if s['depth'] > 3]
    complex_schemas = [s for s in schemas_found if s['properties'] > 30]
    
    if large_schemas:
        print(f"\n⚠️  Large schemas (>50KB): {len(large_schemas)}")
        for s in large_schemas:
            print(f"   - {s['name']} in {s['method']} ({s['size']:,} bytes)")
    
    if deep_schemas:
        print(f"\n⚠️  Deeply nested schemas (>3 levels): {len(deep_schemas)}")
        for s in deep_schemas:
            print(f"   - {s['name']} in {s['method']} (depth: {s['depth']})")
    
    if complex_schemas:
        print(f"\n⚠️  Complex schemas (>30 properties): {len(complex_schemas)}")
        for s in complex_schemas:
            print(f"   - {s['name']} in {s['method']} ({s['properties']} properties)")
    
    if not large_schemas and not deep_schemas and not complex_schemas:
        print("\n✅ All schemas appear to be within reasonable complexity limits!")

if __name__ == "__main__":
    analyze_gemini_schemas()