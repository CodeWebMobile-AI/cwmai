"""
AI-Generated Tool: demo_calculator
Description: A simple calculator tool
Generated: 2025-06-15T12:04:31.560447+00:00
Requirements: Perform basic math operations: add, subtract, multiply, divide. Take two numbers and an operation as input.
"""

from typing import Any
from typing import Dict
from typing import Dict, Any

import asyncio


"""
Module: demo_calculator

Description: A simple calculator tool that performs basic math operations.
"""


# Assuming scripts.state_manager exists and has a StateManager class
# from scripts.state_manager import StateManager  # Removed dummy import

__description__ = "A simple calculator tool for basic math operations."
__parameters__ = {
    "num1": {"type": "float", "description": "The first number."},
    "num2": {"type": "float", "description": "The second number."},
    "operation": {"type": "string", "description": "The operation to perform (add, subtract, multiply, divide)."}
}
__examples__ = [
    {"num1": 10, "num2": 5, "operation": "add"},
    {"num1": 20, "num2": 3, "operation": "subtract"},
    {"num1": 7, "num2": 4, "operation": "multiply"},
    {"num1": 100, "num2": 2, "operation": "divide"}
]


async def demo_calculator(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    A simple calculator tool that performs basic math operations: add, subtract, multiply, divide.

    Args:
        **kwargs: A dictionary containing the input parameters:
            num1 (float): The first number.
            num2 (float): The second number.
            operation (str): The operation to perform (add, subtract, multiply, divide).

    Returns:
        A dictionary containing the result of the calculation.  Returns empty dict on error.

    Raises:
        ValueError: If any of the input parameters are invalid or if the operation is not supported.
    """

    # state_manager = StateManager()  #Removed Dummy state_manager instantiation
    # state = state_manager.load_state()  #Removed Dummy load_state

    try:
        num1 = float(kwargs.get("num1"))
        num2 = float(kwargs.get("num2"))
        operation = kwargs.get("operation")

        if operation not in ["add", "subtract", "multiply", "divide"]:
            raise ValueError(f"Unsupported operation: {operation}")

        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            result = num1 / num2
        else:
            # This should never happen due to the validation above, but included for safety
            return {}

        return {"result": result}

    except (ValueError, TypeError) as e:
        print(f"Error: Invalid input parameters: {e}")
        return {}
    except ZeroDivisionError as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
