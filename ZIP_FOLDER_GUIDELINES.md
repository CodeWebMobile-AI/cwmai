# ZIP FOLDER GUIDELINES

## CRITICAL: Understanding the ZIP Folder Structure

The `zip/` folder is designed to be a **complete, independent copy** of the parent CWMAI system. This document establishes clear guidelines for maintaining and troubleshooting the zip folder.

## 🚨 GOLDEN RULE

**ANY ISSUES WITH THE ZIP FOLDER SHOULD BE FIXED IN THE ZIP FOLDER, NOT THE PARENT**

## Key Principles

### 1. The ZIP Folder is Independent
- The zip folder should contain ALL necessary files to run independently
- It should NOT rely on importing modules from the parent directory
- It maintains its own state files and configurations

### 2. Missing Files = Copy from Parent
If a file is missing in the zip folder:
```bash
# CORRECT: Copy the missing file from parent
cp scripts/missing_module.py zip/scripts/missing_module.py

# WRONG: Fix imports to use parent modules
# This creates dependencies and defeats the purpose
```

### 3. Fix Issues Where They Occur
- **Zip folder error?** → Fix it in `zip/scripts/...`
- **Parent folder error?** → Fix it in `scripts/...`
- **Never fix zip issues by modifying parent files**

## Common Issues and Solutions

### Issue 1: Module Not Found
**Symptom**: `ModuleNotFoundError: No module named 'scripts.some_module'`

**Solution**:
```bash
# Check if the module exists in parent
ls scripts/some_module.py

# If it exists, copy it to zip
cp scripts/some_module.py zip/scripts/some_module.py
```

### Issue 2: Import Errors from Parent
**Symptom**: Zip folder scripts importing from parent directory

**Wrong Approach**:
```python
# This creates unwanted dependencies
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

**Correct Approach**:
```python
# Ensure zip folder uses its own modules
zip_root = Path(__file__).parent.parent
sys.path.insert(0, str(zip_root))
```

### Issue 3: State File Conflicts
**Symptom**: Shared state files causing conflicts between parent and zip

**Solution**:
```python
# In zip folder scripts, use zip-specific paths
state_file = Path(__file__).parent.parent / "zip_state.json"
# NOT: Path.home() / ".cwmai" / "state.json"
```

## Maintenance Checklist

When setting up or updating the zip folder:

1. **Initial Setup**
   ```bash
   # Copy entire scripts directory
   cp -r scripts/ zip/scripts/
   
   # Copy configuration files
   cp .env.local zip/.env.local
   cp requirements.txt zip/requirements.txt
   ```

2. **Verify Independence**
   ```bash
   # Test that zip folder runs without parent
   cd zip
   export $(cat .env.local | grep -v '^#' | xargs)
   PYTHONPATH=/path/to/zip python3 scripts/enhanced_conversational_assistant.py
   ```

3. **Check for Parent Dependencies**
   ```bash
   # Search for parent path references
   grep -r "parent.parent.parent" zip/
   grep -r "../.." zip/
   ```

## File Synchronization

### What Should Be Copied
- All Python modules (`*.py`)
- Configuration files (`.env.local`, `*.json`)
- Custom tools (`scripts/custom_tools/`)
- Requirements (`requirements.txt`)

### What Should NOT Be Shared
- State files (`*_state.json`)
- Log files (`*.log`)
- Cache files (`__pycache__/`)
- Temporary files

## Debugging Steps

When encountering zip folder issues:

1. **Identify the Error Location**
   ```python
   # Check the traceback - is it in zip/ or parent?
   Traceback (most recent call last):
     File "/path/to/cwmai/zip/scripts/...", line X
   ```

2. **Check for Missing Dependencies**
   ```bash
   # List all imports in the failing file
   grep -E "^from|^import" zip/scripts/failing_file.py
   
   # Verify each import exists in zip folder
   ls zip/scripts/imported_module.py
   ```

3. **Copy Missing Files**
   ```bash
   # Copy any missing files from parent
   cp scripts/missing_file.py zip/scripts/
   cp -r scripts/missing_directory/ zip/scripts/
   ```

4. **Fix Import Paths**
   - Remove any `sys.path` modifications that point to parent
   - Ensure all imports use relative paths within zip folder

## Best Practices

1. **Regular Synchronization**
   - When adding new features to parent, copy to zip
   - Use a sync script if making frequent updates

2. **Isolated Testing**
   - Always test zip folder in isolation
   - Use different terminal/environment for zip testing

3. **Clear Separation**
   - Parent folder = development/canonical version
   - Zip folder = isolated runtime environment

4. **Documentation**
   - Document any zip-specific modifications
   - Keep track of files that differ between parent and zip

## Example Sync Script

```bash
#!/bin/bash
# sync_to_zip.sh - Sync parent changes to zip folder

# Sync Python scripts
rsync -av --exclude='__pycache__' scripts/ zip/scripts/

# Copy specific config files
cp .env.local zip/
cp requirements.txt zip/

# Don't copy state files
echo "Skipping state files to maintain independence"

echo "Sync complete. Test zip folder independently."
```

## Remember

The zip folder exists to provide an isolated, self-contained version of the CWMAI system. Maintaining its independence is crucial for its intended purpose. When in doubt:

1. Copy missing files from parent
2. Fix issues in place (zip issues in zip, parent issues in parent)
3. Test in isolation
4. Document any deviations

**Never compromise the zip folder's independence by creating dependencies on the parent directory.**