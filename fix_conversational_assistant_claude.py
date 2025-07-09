#!/usr/bin/env python3
"""
Fix enhanced_conversational_assistant.py to use Claude instead of Gemini.
"""

import os
import shutil
from datetime import datetime

def fix_conversational_assistant():
    """Change the conversational assistant to use Claude instead of Gemini."""
    
    print("🔧 Fixing scripts/enhanced_conversational_assistant.py to use Claude...")
    
    assistant_file = "scripts/enhanced_conversational_assistant.py"
    
    # Backup the file
    backup_file = f"{assistant_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(assistant_file, backup_file)
    print(f"✅ Backed up to {backup_file}")
    
    # Read the file
    with open(assistant_file, 'r') as f:
        content = f.read()
    
    # Count replacements
    replacements = 0
    
    # Replace all instances of 'gemini' model with 'claude'
    # Pattern 1: 'model': 'gemini'
    if "'model': 'gemini'" in content:
        content = content.replace("'model': 'gemini'", "'model': 'claude'")
        replacements += content.count("'model': 'claude'")
    
    # Pattern 2: "model": "gemini" (if any)
    if '"model": "gemini"' in content:
        content = content.replace('"model": "gemini"', '"model": "claude"')
        replacements += content.count('"model": "claude"')
    
    # Write the fixed content
    with open(assistant_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {replacements} occurrences of Gemini model to Claude")
    print("\n🎯 The conversational assistant will now:")
    print("   - Use Claude (Anthropic) as the primary AI provider")
    print("   - Fall back to other providers if Claude is unavailable")
    print("   - Maintain all existing functionality with Claude's capabilities")
    
    # Also update the default provider in http_ai_client.py if needed
    print("\n🔧 Checking http_ai_client.py default provider...")
    
    http_client_file = "scripts/http_ai_client.py"
    
    # Backup the file
    backup_file2 = f"{http_client_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(http_client_file, backup_file2)
    print(f"✅ Backed up to {backup_file2}")
    
    # Read the file
    with open(http_client_file, 'r') as f:
        http_content = f.read()
    
    # Find and update the default provider
    if "self.default_provider = 'gemini'" in http_content:
        http_content = http_content.replace(
            "self.default_provider = 'gemini' if 'gemini' in self.available_providers else None",
            "self.default_provider = 'anthropic' if 'anthropic' in self.available_providers else 'gemini' if 'gemini' in self.available_providers else None"
        )
        print("✅ Updated default provider to prefer Claude (anthropic)")
    
    # Update the provider priority order
    if "provider_priority = ['gemini', 'openai', 'deepseek', 'anthropic']" in http_content:
        http_content = http_content.replace(
            "provider_priority = ['gemini', 'openai', 'deepseek', 'anthropic']",
            "provider_priority = ['anthropic', 'gemini', 'openai', 'deepseek']"
        )
        print("✅ Updated provider priority to prefer Claude first")
    
    # Write the fixed content
    with open(http_client_file, 'w') as f:
        f.write(http_content)
    
    print("\n✅ All changes complete!")
    print("\n📝 To run the conversational assistant with Claude:")
    print("   export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 scripts/enhanced_conversational_assistant.py")

if __name__ == "__main__":
    fix_conversational_assistant()