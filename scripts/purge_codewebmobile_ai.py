#!/usr/bin/env python3
"""
Purge all CodeWebMobile-AI repositories and remove any references
to them from main JSON state files.
"""
import json
import glob
import os

def purge_obj(obj):
    if isinstance(obj, dict):
        # drop dicts that reference the repo
        if 'repository' in obj and 'CodeWebMobile-AI' in str(obj['repository']):
            return None
        out = {}
        for k, v in obj.items():
            pv = purge_obj(v)
            if pv is not None:
                out[k] = pv
        return out
    if isinstance(obj, list):
        result = []
        for v in obj:
            pv = purge_obj(v)
            if pv is not None:
                result.append(pv)
        return result
    if isinstance(obj, str) and 'CodeWebMobile-AI' in obj:
        return None
    return obj

def scrub_json_file(path):
    try:
        data = json.load(open(path, 'r'))
        cleaned = purge_obj(data)
        with open(path, 'w') as f:
            json.dump(cleaned, f, indent=2)
        print(f"Purged references in {path}")
    except Exception as e:
        print(f"Failed to purge {path}: {e}")

if __name__ == '__main__':
    # 1) Remove the repository directory entirely
    repo_dir = 'local_repos/repositories/CodeWebMobile-AI'
    if os.path.isdir(repo_dir):
        import shutil
        shutil.rmtree(repo_dir)
        print(f"Deleted directory {repo_dir}")

    # 2) Scrub top-level JSON state files
    json_files = [
        'claude_execution_queue.json',
        'work_queue_backup.json',
        'system_state.json'
    ]
    # 3) Scrub per-run state files
    json_files += glob.glob('local_repos/state/*.json')

    for jf in json_files:
        if os.path.exists(jf):
            scrub_json_file(jf)