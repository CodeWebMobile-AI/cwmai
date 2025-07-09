
import json

def remove_duplicate_repos():
    """
    Removes duplicate repositories from the local_repos/state/repositories.json file.
    """
    repos_file = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/repositories.json"
    with open(repos_file, 'r') as f:
        data = json.load(f)

    unique_repos = {}
    for repo in data['repositories']:
        unique_repos[repo['name']] = repo

    data['repositories'] = list(unique_repos.values())

    data['repositories'] = unique_repos
    data['total_count'] = len(unique_repos)

    with open(repos_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Removed {len(data['repositories']) - len(unique_repos)} duplicate repositories.")

if __name__ == "__main__":
    remove_duplicate_repos()
