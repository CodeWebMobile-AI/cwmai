#!/usr/bin/env python3
"""Run the hybrid continuous system in LOCAL MODE ONLY - no GitHub access."""

import os
import sys

# No need to force local mode - it's now the default!
# To use GitHub mode instead, set USE_GITHUB_MODE=true

# Import the main run script
from run_hybrid_continuous import main

if __name__ == "__main__":
    print("🏠 Starting Hybrid System in LOCAL MODE")
    print("   No GitHub API calls will be made")
    print("   All repositories will be managed locally\n")
    
    # Run the main function with local mode forced
    main()