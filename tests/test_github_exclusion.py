#!/usr/bin/env python3
"""Test script to verify .github repository exclusion."""

from scripts.repository_exclusion import RepositoryExclusion, is_excluded_repo, get_exclusion_reason

# Test various .github repository formats
test_repos = [
    ".github",
    ".GITHUB",
    "CodeWebMobile-AI/.github",
    "codewebmobile-ai/.github",
    "https://github.com/CodeWebMobile-AI/.github",
    "git@github.com:CodeWebMobile-AI/.github.git",
    ".github.git",
    "cwmai",  # Still excluded
    "ai-creative-studio",  # Not excluded
]

print("Testing .github repository exclusion:\n")

for repo in test_repos:
    excluded = is_excluded_repo(repo)
    reason = get_exclusion_reason(repo) if excluded else "Not excluded"
    status = "✗ EXCLUDED" if excluded else "✓ ALLOWED"
    print(f"{status}: {repo:45} | {reason}")

print("\n\nAll excluded repositories:")
for repo in sorted(RepositoryExclusion.get_excluded_repos_list()):
    print(f"  - {repo}")