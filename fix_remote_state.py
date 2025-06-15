#!/usr/bin/env python3
"""Fix the remote system_state.json by removing non-existent repositories."""

import os
import json
import base64
from github import Github
from datetime import datetime, timezone

def main():
    # Get GitHub token
    github_token = os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Error: GitHub token not found in environment variables")
        return
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo('CodeWebMobile-AI/cwmai')
    
    try:
        # Get current system_state.json from repo
        file_content = repo.get_contents('system_state.json')
        if file_content.encoding == 'base64':
            content = base64.b64decode(file_content.content).decode('utf-8')
        else:
            content = file_content.content
        
        state = json.loads(content)
        
        # Remove non-existent repositories
        repos_to_remove = ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations', '.github', 'cwmai']
        
        if 'projects' in state:
            original_count = len(state['projects'])
            for repo_name in repos_to_remove:
                if repo_name in state['projects']:
                    del state['projects'][repo_name]
                    print(f"Removed {repo_name} from projects")
            
            print(f"Removed {original_count - len(state['projects'])} repositories")
        
        # Update timestamp
        state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Update the file in the repository
        updated_content = json.dumps(state, indent=2, sort_keys=True)
        
        repo.update_file(
            path='system_state.json',
            message='Remove non-existent repositories from system state',
            content=updated_content,
            sha=file_content.sha
        )
        
        print("Successfully updated remote system_state.json")
        
    except Exception as e:
        print(f"Error updating remote state: {e}")

if __name__ == "__main__":
    main()