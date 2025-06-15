#!/usr/bin/env python3
"""Batch update all architectures with detailed content."""

import os
from github import Github
import time
from update_all_architectures import REPO_CONTEXTS, generate_template_architecture

def main():
    g = Github(os.environ['GITHUB_TOKEN'])
    
    repos_to_update = [
        'ai-powered-inventory-sync',
        'community-connect-platform',
        'brand-guardian-ai',
        'reputation-ai',
        'mindleap-ai',
        'vid-gen-ai',
        'review-sentry',
        'project-analytics-dashboard',
        'business-analytics-dashboard'
    ]
    
    success_count = 0
    
    for repo_name in repos_to_update:
        print(f'\n📝 Updating {repo_name}...')
        
        try:
            # Get context
            context = REPO_CONTEXTS.get(repo_name)
            if not context:
                print(f'  ⚠️  No context found for {repo_name}')
                continue
            
            # Generate content
            content = generate_template_architecture(repo_name, context)
            
            # Update repository
            repo = g.get_repo(f'CodeWebMobile-AI/{repo_name}')
            
            try:
                arch_file = repo.get_contents('ARCHITECTURE.md')
                repo.update_file(
                    'ARCHITECTURE.md',
                    'Update architecture with detailed project-specific content',
                    content,
                    arch_file.sha,
                    branch='main'
                )
                print(f'  ✅ Successfully updated')
                success_count += 1
            except Exception as e:
                print(f'  ❌ Failed to update: {e}')
            
            # Rate limit
            time.sleep(2)
            
        except Exception as e:
            print(f'  ❌ Error: {e}')
    
    print(f'\n📊 Summary: Updated {success_count}/{len(repos_to_update)} repositories')

if __name__ == '__main__':
    main()