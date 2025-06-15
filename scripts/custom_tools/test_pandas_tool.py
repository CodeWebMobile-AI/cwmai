"""
AI-Generated Tool: test_pandas_tool
Description: Test tool using pandas
Generated: 2025-06-15T12:00:45.639326+00:00
Requirements: Load CSV files using pandas and return summary statistics
"""

from typing import Dict

import io
import pandas as pd

from scripts.state_manager import StateManager


"""
Module for a test tool that uses pandas to load CSV files and return summary statistics.
"""


__description__ = "Test tool using pandas to load CSV files and return summary statistics."
__parameters__ = {
    "csv_data": {"type": "string", "description": "CSV data as a string.", "required": True}
}
__examples__ = [
    {
        "description": "Load CSV data and get summary statistics.",
        "input": {"csv_data": "col1,col2\n1,2\n3,4"}
    }
]


async def test_pandas_tool(**kwargs) -> Dict:
    """
    Loads CSV data using pandas and returns summary statistics.

    Args:
        **kwargs: Keyword arguments containing the CSV data.

    Returns:
        A dictionary containing the summary statistics.

    Raises:
        ValueError: If the 'csv_data' is not provided or is empty.
        pd.errors.EmptyDataError: If the CSV data is empty or invalid.
        Exception: For other unexpected errors during processing.
    """
    state_manager = StateManager()
    state = state_manager.load_state()

    csv_data = kwargs.get("csv_data")

    if not csv_data:
        raise ValueError("The 'csv_data' parameter is required.")

    if not isinstance(csv_data, str) or not csv_data.strip():
        raise ValueError("The 'csv_data' must be a non-empty string.")

    try:
        df = pd.read_csv(io.StringIO(csv_data))
        summary = df.describe().to_dict()
        return summary
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The CSV data is empty or invalid.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the CSV data: {e}")
