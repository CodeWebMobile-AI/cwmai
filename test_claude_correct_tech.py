#!/usr/bin/env python3
"""Test Claude with correct Laravel/React tech stack."""

import subprocess
import os
import threading
import time

# Use actual repository
repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Correct prompt for Laravel/React project
correct_prompt = """Please implement the following task from issue #1:

Title: Initial Project Setup
Type: FEATURE

Description:
## Initial Project Setup

Based on the Laravel/React starter kit architecture for **ai-text-summarizer**, complete the following setup tasks:

1. **Configure API authentication**
   - [ ] Set up Laravel Sanctum for API authentication (run: php artisan install:api)
   - [ ] Update User model to use HasApiTokens trait
   - [ ] Create API auth controllers in app/Http/Controllers/Api/
   - [ ] Add protected API routes in routes/api.php

2. **Create initial data models**
   - [ ] Add core models: Post, Category in app/Models/
   - [ ] Create migrations with: php artisan make:migration create_posts_table
   - [ ] Set up Eloquent relationships
   - [ ] Add API resource controllers

3. **Update React dashboard component**
   - [ ] Customize the dashboard at resources/js/pages/dashboard.tsx
   - [ ] Add data display widgets using existing shadcn/ui components
   - [ ] Create API service layer in resources/js/services/
   - [ ] Update navigation in resources/js/components/app-sidebar.tsx

4. **Set up state management**
   - [ ] Install Zustand: npm install zustand
   - [ ] Create stores in resources/js/stores/
   - [ ] Create hooks for data fetching in resources/js/hooks/
   - [ ] Add axios interceptors for auth tokens

Requirements:
- Use Laravel 11+ backend features
- Use React 19 with TypeScript
- Follow the existing project structure
- Use existing Tailwind CSS and shadcn/ui components

Please make the necessary changes to implement this task.
IMPORTANT: Please go ahead and make the actual file changes now.
Do not wait for additional permission - implement the solution directly."""

print(f"Testing Claude with correct Laravel/React tech stack")
print(f"Repository: {repo_dir}")
print(f"Prompt length: {len(correct_prompt)} characters")
print("-" * 50)

# Function to monitor progress
def monitor_progress(start_time, stop_event):
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        print(f"Claude still running... ({elapsed}s elapsed)")
        stop_event.wait(30)

# Start monitoring thread
start_time = time.time()
stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_progress, args=(start_time, stop_event))
monitor_thread.daemon = True
monitor_thread.start()

# Run Claude without timeout
print("Starting Claude execution (no timeout)...")
process = subprocess.Popen(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=repo_dir
)

# Let it run as long as needed
stdout, stderr = process.communicate(input=correct_prompt)
stop_event.set()
execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(stdout) if stdout else 0} chars):")
if stdout:
    # Print first and last parts of output
    if len(stdout) > 2000:
        print(stdout[:1000])
        print("\n... [output truncated] ...\n")
        print(stdout[-1000:])
    else:
        print(stdout)
        
print(f"\nSTDERR ({len(stderr) if stderr else 0} chars):")
if stderr:
    print(stderr[:1000] + "..." if len(stderr) > 1000 else stderr)

# Check what changed
print("\nChecking changes made...")
print("\n1. User model updated:")
os.system("grep -n 'HasApiTokens' app/Models/User.php || echo 'HasApiTokens not found in User model'")

print("\n2. New models created:")
os.system("ls -la app/Models/ | grep -E '(Post|Category)' || echo 'No Post/Category models found'")

print("\n3. API controllers created:")
os.system("ls -la app/Http/Controllers/Api/ 2>/dev/null || echo 'No Api directory found'")

print("\n4. Zustand installed:")
os.system("grep zustand package.json || echo 'Zustand not in package.json'")

print("\n5. React stores created:")
os.system("ls -la resources/js/stores/ 2>/dev/null || echo 'No stores directory found'")

print("\n6. Git status summary:")
os.system("git status --porcelain | wc -l | xargs -I {} echo '{} files changed'")