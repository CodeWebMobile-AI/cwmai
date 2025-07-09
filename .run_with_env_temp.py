#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('/Users/fortressarigo/PhpstormProjects/cwmai/.env.local')

# Verify at least one API key is loaded
api_keys_loaded = any([
    os.getenv('ANTHROPIC_API_KEY'),
    os.getenv('OPENAI_API_KEY'), 
    os.getenv('GOOGLE_API_KEY'),
    os.getenv('GEMINI_API_KEY'),
    os.getenv('DEEPSEEK_API_KEY')
])

if not api_keys_loaded:
    print("❌ ERROR: No API keys found in .env.local")
    print("Please check your .env.local file contains valid API keys")
    sys.exit(1)

# Run the main script
sys.argv = ['run_hybrid_continuous.py', '--workers', '20', '--log-level', 'INFO']
exec(open('/Users/fortressarigo/PhpstormProjects/cwmai/run_hybrid_continuous.py').read())
