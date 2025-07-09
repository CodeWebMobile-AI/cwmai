from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.error_handler import ErrorHandler


class analyze_json_files:
    """
    A general tool for analyzing JSON files.

    This tool iterates through a list of JSON file paths, attempts to load
    and analyze each file, and returns a summary of the analysis.  Error
    handling and logging are included.
    """

    def __init__(self, error_handler: ErrorHandler, logger: logging.Logger):
        """
        Initializes the analyze_json_files tool.

        Args:
            error_handler: An instance of ErrorHandler for handling errors.
            logger: An instance of a logger for logging messages.
        """
        self.error_handler = error_handler
        self.logger = logger

    def analyze_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyzes a list of JSON files.

        Args:
            file_paths: A list of Path objects representing the paths to the JSON files.

        Returns:
            A dictionary containing the analysis results.  If any errors occur,
            the dictionary will contain an 'error' key with a description of the error.
            If successful, the dictionary will contain a 'success' key with a description of the success.
        """
        results: List[Dict[str, Any]] = []

        for file_path in file_paths:
            try:
                print(f"Analyzing file: {file_path}")
                result = self._analyze_file(file_path)  # Pass the logger here
                results.append(result)
            except Exception as e:
                error_message = f"Error analyzing file {file_path}: {e}"
                self.logger.exception(error_message)
                self.error_handler.handle_error(error_message)
                return {"error": error_message}

        success_message = "Successfully analyzed all JSON files."
        print(success_message)
        return {"success": success_message, "results": results}

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyzes a single JSON file.  This is a placeholder for the actual
        analysis logic.

        Args:
            file_path: A Path object representing the path to the JSON file.

        Returns:
            A dictionary containing the analysis results for the file.
        """
        try:
            import json  # Moved import here to avoid dependency if tool isn't used

            with open(file_path, "r") as f:
                data = json.load(f)

            # Placeholder for actual analysis logic.  Replace this with your
            # specific analysis code.
            analysis_result = {
                "file": str(file_path),
                "num_keys": len(data.keys()) if isinstance(data, dict) else 0,
                "type": type(data).__name__,
            }
            self.logger.debug(f"Analysis result for {file_path}: {analysis_result}")  # Use logger here
            return analysis_result

        except FileNotFoundError as e:
            error_message = f"File not found: {file_path}.  Error: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message)
            return {"error": error_message}

        except json.JSONDecodeError as e:
            error_message = f"Error decoding JSON in file: {file_path}. Error: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message)
            return {"error": error_message}

        except Exception as e:
            error_message = f"An unexpected error occurred while analyzing {file_path}: {e}"
            self.logger.exception(error_message)
            self.error_handler.handle_error(error_message)
            return {"error": error_message}


if __name__ == '__main__':
    # Example Usage (for testing purposes)

    # Configure logging
    logger = get_logger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the desired logging level

    # Initialize ErrorHandler
    error_handler = ErrorHandler(logger=logger)

    # Create an instance of the analyze_json_files tool
    analyzer = analyze_json_files(error_handler=error_handler, logger=logger)

    # Create some dummy JSON files for testing
    import os

    test_dir = "test_json_files"
    os.makedirs(test_dir, exist_ok=True)

    file1_path = Path(os.path.join(test_dir, "file1.json"))
    file2_path = Path(os.path.join(test_dir, "file2.json"))
    file3_path = Path(os.path.join(test_dir, "file3.json")) #doesn't exist

    with open(file1_path, "w") as f:
        f.write('{"key1": "value1", "key2": 123}')
    with open(file2_path, "w") as f:
        f.write('[1, 2, 3]')

    file_paths = [file1_path, file2_path, file3_path]  # Include a non-existent file

    # Analyze the files
    results = analyzer.analyze_files(file_paths)

    print(results) # Print the analysis result
    # Clean up the dummy files
    import shutil
    shutil.rmtree(test_dir)

