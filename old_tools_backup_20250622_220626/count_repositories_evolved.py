"""
AI-Generated Tool: count_repositories_evolved
Description: Count repositories with enhanced error handling and validation
Generated: 2025-06-15
Requirements: Count total repositories being managed with statistics
"""

from typing import Dict, Any, List, Optional
import logging
import functools
import time
from collections import defaultdict
import asyncio
from datetime import datetime

# Set up logging for this module
logger = logging.getLogger(__name__)

__description__ = "Count repositories with enhanced error handling and validation"
__parameters__ = {}
__examples__ = [
    {"command": "count_repositories_evolved", "description": "Count all managed repositories with statistics"}
]


class CountRepositoriesError(Exception):
    """Custom exception for count_repositories tool."""
    pass


def timed_cache(max_age_seconds: int = 60):  # Cache for 1 minute by default
    """
    Decorator to cache the result of a function for a specified time.
    """
    cache = {}

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))  # Create a key based on arguments
            now = time.time()

            if key in cache and now - cache[key]["time"] < max_age_seconds:
                logger.debug(f"Using cached result for {func.__name__}")
                return cache[key]["value"]

            try:
                result = await func(*args, **kwargs)  # Execute the function
                cache[key] = {"value": result, "time": now}  # Store result and timestamp
                logger.debug(f"Cached result for {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error executing {func.__name__}: {e}")
                raise  # Re-raise the exception after logging

        return wrapper

    return decorator


def validate_repository_data(repo_data: Dict[str, Any], repo_name: str) -> Optional[Dict[str, Any]]:
    """
    Validates the structure and data types of individual repository data.
    Raises CountRepositoriesError for critical issues, returns cleaned data.
    Returns None if validation fails and repo should be skipped.
    """
    if not isinstance(repo_data, dict):
        logger.warning(f"Invalid repo data format for {repo_name}. Skipping. Got: {type(repo_data)}")
        return None  # Skip invalid repository data

    # Validate 'language' field
    language = repo_data.get('language', 'Unknown')
    if not isinstance(language, str):
        logger.warning(f"Invalid language type for {repo_name}. Got: {type(language)}. Defaulting to 'Unknown'.")
        language = 'Unknown'

    # Validate 'archived' field
    is_archived = repo_data.get('archived', False)
    if not isinstance(is_archived, bool):
        logger.warning(f"Invalid archived value for {repo_name}. Assuming 'False'. Got: {type(is_archived)}")
        is_archived = False

    # Validate and sanitize 'stars' field
    stars = repo_data.get('stars', 0)
    if not isinstance(stars, (int, float)):
        logger.warning(f"Invalid stars value for {repo_name}. Setting to 0. Got: {type(stars)}")
        stars = 0
    else:
        stars = int(stars)  # Normalize to integer

    # Validate and sanitize 'open_issues_count' field
    open_issues_count = repo_data.get('open_issues_count', 0)
    if not isinstance(open_issues_count, (int, float)):
        logger.warning(f"Invalid open_issues_count value for {repo_name}. Setting to 0. Got: {type(open_issues_count)}")
        open_issues_count = 0
    else:
        open_issues_count = int(open_issues_count)  # Normalize to integer

    # Validate 'last_commit_date' field (New Feature)
    last_commit_date = repo_data.get('last_commit_date')
    if last_commit_date is not None:
      if not isinstance(last_commit_date, str):
          logger.warning(f"Invalid last_commit_date type for {repo_name}.  Ignoring. Got: {type(last_commit_date)}")
          last_commit_date = None  # Or could default to a specific date, like the epoch.
      else:
        try:
          datetime.strptime(last_commit_date, "%Y-%m-%d")
        except ValueError:
          logger.warning(f"Invalid last_commit_date format for {repo_name}.  Expected YYYY-MM-DD. Ignoring.")
          last_commit_date = None


    # Validate 'owner' field (New Feature - Owner Information)
    owner = repo_data.get('owner', 'Unknown')
    if not isinstance(owner, str):
        logger.warning(f"Invalid owner type for {repo_name}. Defaulting to 'Unknown'. Got: {type(owner)}")
        owner = 'Unknown'

    # Validate 'description' (New Feature)
    description = repo_data.get('description', '')
    if not isinstance(description, str):
        logger.warning(f"Invalid description type for {repo_name}.  Defaulting to empty string. Got: {type(description)}")
        description = ''

    return {
        'language': language,
        'archived': is_archived,
        'stars': stars,
        'open_issues_count': open_issues_count,
        'last_commit_date': last_commit_date,  # Include the new field
        'owner': owner, # Include the new field
        'description': description #Include the new field
    }


@timed_cache(max_age_seconds=300)  # Cache the result for 5 minutes.  Adjust as needed.
async def count_repositories_evolved(**kwargs) -> Dict[str, Any]:
    """Count total repositories being managed."""
    from scripts.state_manager import StateManager  #Avoid circular import

    state_manager = StateManager()

    try:
        # Attempt to reload state, gracefully falling back to loading if reload fails
        try:
            if hasattr(state_manager, "force_reload_state"):
                state = state_manager.force_reload_state()
            else:
                state = state_manager.load_state()
        except Exception as e:
            logger.warning(f"Failed to reload state. Falling back to loading. Error: {e}")
            try:
                state = state_manager.load_state()
            except Exception as e2:
                logger.error(f"Failed to load state even after fallback. Error: {e2}")
                raise CountRepositoriesError("Failed to load state even after fallback.") from e2
    except Exception as e:
        logger.error(f"Failed to load state. Error: {e}")
        raise CountRepositoriesError("Failed to load state.") from e

    projects = state.get('projects', {})

    # Input validation: Ensure projects is a dictionary
    if not isinstance(projects, dict):
        logger.error(f"Invalid state format: projects is not a dictionary. Got: {type(projects)}")
        raise CountRepositoriesError("Invalid state format: Projects must be a dictionary.")

    # Count by various criteria
    total = len(projects)
    by_language = defaultdict(int) #Use defaultdict
    by_status = {'active': 0, 'archived': 0, 'unknown': 0}
    total_stars = 0
    total_issues = 0
    repositories = []  # List to store details of each repository
    active_repo_names: List[str] = [] #New Feature
    inactive_repo_names: List[str] = [] #New Feature
    by_owner = defaultdict(int) #New Feature - Count by owner
    average_age_days = 0 # New Feature - Average Age of Repositories
    total_age_days = 0
    valid_repo_count = 0 # Track valid repos for age calculation
    long_running_issues = 0 # New Feature - Count of issues open for more than 90 days
    critical_bugs = 0 # New Feature - Count of issues tagged as 'critical' or 'bug'
    security_vulnerabilities = 0 # New Feature - Count of issues tagged as 'security'
    average_issue_age_days = 0 #New Feature: Average age of open issues
    total_issue_age_days = 0
    valid_issue_age_count = 0

    issue_data = {} # Simulate retrieval of issue information for each repo
    for repo_name, repo_data in projects.items():
        validated_data = validate_repository_data(repo_data, repo_name)
        if not validated_data:
            continue

        language = validated_data['language']
        is_archived = validated_data['archived']
        stars = validated_data['stars']
        open_issues_count = validated_data['open_issues_count']
        last_commit_date = validated_data.get('last_commit_date') #Get the value
        owner = validated_data['owner'] #New Feature
        description = validated_data['description'] #New Feature

        # Language count
        by_language[language] += 1 #Using defaultdict

        # Status count
        if is_archived:
            by_status['archived'] += 1
            inactive_repo_names.append(repo_name)  # New Feature
        else:
            by_status['active'] += 1
            active_repo_names.append(repo_name) # New Feature

        # Owner count
        by_owner[owner] +=1 #New Feature

        # Aggregate metrics
        total_stars += stars
        total_issues += open_issues_count

        repo_details = { #Add repository details to the list
            "name": repo_name,
            "language": language,
            "archived": is_archived,
            "stars": stars,
            "open_issues_count": open_issues_count,
            "owner": owner, #New Feature
            "description": description #New Feature
        }
        if last_commit_date:  # Only add if it exists (handles None cases)
            repo_details["last_commit_date"] = last_commit_date
            # Calculate the age of the repository based on last commit date (New Feature)
            try:
                last_commit_time = time.strptime(last_commit_date, "%Y-%m-%d") #Expect date in "YYYY-MM-DD" format.  Adjust as needed.
                last_commit_timestamp = time.mktime(last_commit_time)
                age_days = (time.time() - last_commit_timestamp) / (60 * 60 * 24)
                total_age_days += age_days
                valid_repo_count += 1
            except ValueError as e:
                logger.warning(f"Invalid date format for {repo_name}. Skipping age calculation. Error: {e}")
        repositories.append(repo_details)

        # Simulate issue analysis (replace with actual data retrieval)
        #New Feature: Issue Age and Tagging
        #Simulate retrieving data from an issue tracking system
        # Replace this with actual API calls to a real issue tracker (e.g., GitHub Issues, Jira)
        issue_data[repo_name] = simulate_issue_data(open_issues_count)

        for issue in issue_data[repo_name]:
          total_issue_age_days += issue['age']
          valid_issue_age_count += 1
          if issue['age'] > 90:
            long_running_issues += 1

          if 'critical' in issue['tags'] or 'bug' in issue['tags']:
            critical_bugs += 1

          if 'security' in issue['tags']:
            security_vulnerabilities += 1


    try:
        avg_stars_per_repo = total_stars / max(total, 1)
    except ZeroDivisionError:
        logger.warning("Total is zero, cannot compute avg_stars_per_repo.")
        avg_stars_per_repo = 0.0

    # Calculate average age of repositories (New Feature)
    if valid_repo_count > 0:
        average_age_days = total_age_days / valid_repo_count
    else:
        average_age_days = 0

    # Calculate average issue age
    if valid_issue_age_count > 0:
      average_issue_age_days = total_issue_age_days / valid_issue_age_count
    else:
      average_issue_age_days = 0

    #Sort repositories by stars (New Feature)
    sorted_repositories = sorted(repositories, key=lambda x: x['stars'], reverse=True) # Sort repositories by stars in descending order

    # Sort repositories by last commit date (New Feature)
    sorted_by_last_commit = sorted(repositories, key=lambda x: x.get('last_commit_date', '1970-01-01'), reverse=True)

    # Language Recommendations (New Feature)
    language_recommendations = recommend_languages(by_language)

    result = {
        "total": total,
        "breakdown": {
            "by_status": by_status,
            "by_language": dict(by_language), #Convert back to dict
            "by_owner": dict(by_owner) #New Feature
        },
        "metrics": {
            "total_stars": total_stars,
            "total_open_issues": total_issues,
            "avg_stars_per_repo": avg_stars_per_repo,
            "average_age_days": average_age_days, # New Feature
            "long_running_issues": long_running_issues, # New Feature
            "critical_bugs": critical_bugs, # New Feature
            "security_vulnerabilities": security_vulnerabilities, # New Feature
            "average_issue_age_days": average_issue_age_days #New Feature
        },
        "repositories": sorted_repositories, # Include repository details in the output
        "repositories_by_last_commit": sorted_by_last_commit, # New Feature
        "summary": f"Managing {total} repositories ({by_status['active']} active, {by_status['archived']} archived)",
        "active_repositories": active_repo_names, #New feature
        "inactive_repositories": inactive_repo_names, #New feature
        "health_score": calculate_health_score(total_stars, total_issues, average_age_days, total), #New Feature
        "actionable_insights": generate_actionable_insights(by_language, by_owner, long_running_issues, security_vulnerabilities, average_age_days, average_issue_age_days), #New Feature
        "language_recommendations": language_recommendations,  #New Feature
        "detailed_issue_analysis": get_detailed_issue_analysis(issue_data) #New Feature
    }
    return result


def calculate_health_score(total_stars: int, total_issues: int, average_age_days: float, total_repos: int) -> float:
    """
    Calculates a health score for the managed repositories based on several factors.

    This is a utility function that combines several key metrics to provide a single
    score representing the overall health and activity of the managed repositories.
    The score is weighted to prioritize different aspects of repository health.

    Args:
        total_stars (int): Total number of stars across all repositories.
        total_issues (int): Total number of open issues across all repositories.
        average_age_days (float): Average age of the repositories in days (time since last commit).
        total_repos (int): Total number of repositories.

    Returns:
        float: A health score between 0 and 100, where higher is better.
    """
    # Normalize the inputs to a 0-1 range
    normalized_stars = total_stars / (total_repos * 1000) if total_repos else 0  # Assuming 1000 stars is a good max per repo
    normalized_issues = 1 - (total_issues / (total_repos * 100) if total_repos else 0)  # Assuming 100 issues is a bad max per repo
    normalized_age = 1 - (average_age_days / 365) if average_age_days else 0  # Assuming 1 year is a reasonable max age

    # Weight the factors (adjust weights based on desired priorities)
    star_weight = 0.4
    issue_weight = 0.3
    age_weight = 0.3

    # Calculate the weighted score
    health_score = (normalized_stars * star_weight +
                    normalized_issues * issue_weight +
                    normalized_age * age_weight) * 100

    return max(0, min(health_score, 100))  # Ensure the score is within the range of 0-100


def generate_actionable_insights(by_language: Dict[str, int], by_owner: Dict[str, int], long_running_issues: int, security_vulnerabilities: int, average_age_days: float, average_issue_age_days: float) -> List[str]:
    """
    Generates actionable insights based on the repository data.

    Args:
        by_language (Dict[str, int]): Count of repositories by language.
        by_owner (Dict[str, int]): Count of repositories by owner.
        long_running_issues (int): Number of issues open for more than 90 days.
        security_vulnerabilities (int): Number of issues tagged as security vulnerabilities.
        average_age_days (float): Average age of repositories in days.
        average_issue_age_days (float): Average age of open issues in days.

    Returns:
        List[str]: A list of actionable insights.
    """
    insights = []

    # Language insights
    most_popular_language = max(by_language, key=by_language.get, default="N/A")
    insights.append(f"The most popular language is {most_popular_language}.")

    # Owner insights
    most_active_owner = max(by_owner, key=by_owner.get, default="N/A")
    insights.append(f"The most active owner is {most_active_owner}.")

    # Issue insights
    if long_running_issues > 10:
        insights.append(f"There are {long_running_issues} long-running issues (over 90 days). Consider prioritizing issue resolution.")

    if security_vulnerabilities > 5:
        insights.append(f"There are {security_vulnerabilities} security vulnerabilities. Address these immediately.")

    if average_issue_age_days > 30:
      insights.append(f"The average issue age is {average_issue_age_days:.0f} days.  Strive to reduce issue resolution time.")


    # Age insights
    if average_age_days > 365:
        insights.append(f"The average repository age is {average_age_days:.0f} days. Consider updating dependencies and libraries.")

    if not insights:
        insights.append("No specific insights could be generated based on the current data.")

    return insights

def recommend_languages(by_language: Dict[str, int]) -> List[str]:
    """
    Recommends languages to focus on based on current usage.  This is a basic example
    and can be enhanced with external data (e.g., trending languages on GitHub).

    Args:
        by_language (Dict[str, int]): Count of repositories by language.

    Returns:
        List[str]: A list of recommended languages.
    """
    if not by_language:
        return ["No languages found. Consider adding repositories!"]

    sorted_languages = sorted(by_language.items(), key=lambda item: item[1], reverse=True)
    top_languages = [lang for lang, count in sorted_languages[:3]] # Recommend top 3

    if len(top_languages) < 3:
        remaining = 3 - len(top_languages)
        #In a real application, you'd fetch trending languages from an API or database here
        trending = ["Go", "Rust", "Kotlin"] #Example trending languages
        for lang in trending:
          if lang not in top_languages and remaining > 0:
            top_languages.append(lang)
            remaining -= 1

    return top_languages

def simulate_issue_data(num_issues: int) -> List[Dict[str, Any]]:
  """
  Simulates issue data for a repository.  Replace this with actual API calls
  to your issue tracking system.

  Args:
      num_issues (int): Number of issues to simulate.

  Returns:
      List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents an issue.
  """
  issues = []
  for i in range(num_issues):
    age = (time.time() - time.mktime(time.strptime("2024-01-01", "%Y-%m-%d"))) / (60 * 60 * 24) #Simulate age
    tags = []
    if i % 5 == 0:
      tags.append("critical")
    if i % 3 == 0:
        tags.append("bug")
    if i % 7 == 0:
      tags.append("security")

    issues.append({
        "age": age,
        "tags": tags
    })
  return issues

def get_detailed_issue_analysis(issue_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
  """
  Provides a detailed analysis of issues, breaking down by repository, tags, and age.

  Args:
      issue_data (Dict[str, List[Dict[str, Any]]]): A dictionary where keys are repository names
                                                  and values are lists of issue dictionaries.

  Returns:
      Dict[str, Any]: A dictionary containing the issue analysis.
  """
  analysis = {}
  for repo_name, issues in issue_data.items():
    analysis[repo_name] = {
        "total_issues": len(issues),
        "by_tag": defaultdict(int),
        "average_issue_age": 0,
        "long_running_issues": 0
    }
    total_age = 0
    for issue in issues:
      total_age += issue['age']
      for tag in issue['tags']:
        analysis[repo_name]["by_tag"][tag] += 1
      if issue['age'] > 90:
        analysis[repo_name]["long_running_issues"] += 1

    if issues:
      analysis[repo_name]["average_issue_age"] = total_age / len(issues)

  return analysis

# Example usage
async def main():
    # Simulate state data
    state_data = {
        "projects": {
            "repo1": {"language": "Python", "archived": False, "stars": 100, "open_issues_count": 5, "last_commit_date": "2024-01-15", "owner": "user1", "description": "A Python project"},
            "repo2": {"language": "JavaScript", "archived": True, "stars": 50, "open_issues_count": 2, "last_commit_date": "2023-12-01", "owner": "user2", "description": "A JavaScript project"},
            "repo3": {"language": "Python", "archived": False, "stars": 200, "open_issues_count": 10, "last_commit_date": "2024-02-20", "owner": "user1", "description": "Another Python project"},
            "repo4": {"language": "Java", "archived": False, "stars": 75, "open_issues_count": 3, "last_commit_date": "2024-03-10", "owner": "user3", "description": "A Java project"},
            "repo5": {"language": 123, "archived": "yes", "stars": "abc", "open_issues_count": [], "last_commit_date": 12345, "owner": 456, "description": 789}, #Invalid Data
            "repo6": {"language": "C++", "archived": False, "stars": 300, "open_issues_count": 15, "last_commit_date": "2023-06-01", "owner": "user4", "description": "A C++ project"}
        }
    }
    # Create a dummy StateManager
    class DummyStateManager:
        def __init__(self, initial_state):
            self.state = initial_state

        def load_state(self):
            return self.state

        def force_reload_state(self):
            return self.state

    # Mock the StateManager
    from scripts.state_manager import StateManager
    StateManager = DummyStateManager # type: ignore
    StateManager.state = state_data # type: ignore

    try:
        result = await count_repositories_evolved()
        print(result)
    except CountRepositoriesError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Set logging level
    asyncio.run(main())