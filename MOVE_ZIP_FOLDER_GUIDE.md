# Moving the Zip Folder to Its Own Project

## Step 1: Copy .env.local
```bash
cp /Users/fortressarigo/PhpstormProjects/cwmai/.env.local /Users/fortressarigo/PhpstormProjects/cwmai/zip/
```

## Step 2: Move the Folder
```bash
# Move to Desktop (or anywhere you want)
mv /Users/fortressarigo/PhpstormProjects/cwmai/zip ~/Desktop/cwmai-standalone

# Or to a new project directory
mv /Users/fortressarigo/PhpstormProjects/cwmai/zip ~/Projects/my-cwmai-instance
```

## Step 3: Run from New Location
```bash
cd ~/Desktop/cwmai-standalone  # or wherever you moved it
./run_standalone.sh --workers 10
```

## What Happens:
1. System starts with 0 projects (fresh state)
2. Only looks in its own local_repos directory
3. Creates all state files in its own directory
4. Runs completely independently

## Testing Independence:
```bash
# From the new location, test discovery
cd ~/Desktop/cwmai-standalone
export $(cat .env.local | grep -v '^#' | xargs)
export PYTHONPATH=$(pwd)
python3 -c "from scripts.state_manager import StateManager; sm = StateManager(); print(f'Working directory: {sm.local_path}')"
```

This will show it's using paths relative to the new location, not the old parent directory.