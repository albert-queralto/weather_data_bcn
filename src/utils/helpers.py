import re
import inspect
from importlib import import_module
import pandas as pd
import functools
from typing import Callable, Any

ComposableFunction = Callable[[Any], Any]

def compose_functions(*functions: ComposableFunction) -> ComposableFunction:
    """Composes a list of functions into a single function."""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)

def get_class_methods(class_object: object) -> list:
    """Returns a list with the names of the methods of a class."""
    return [func for func in dir(class_object) if callable(getattr(class_object, func))]
        
def get_class_methods_exclude_dunder(class_object: object) -> list:
    """Returns a list with the names of the methods of a class.""" 
    return [func for func in dir(class_object) 
            if callable(getattr(class_object, func)) 
            and not func.startswith("__")]

def get_classes_by_string_name(string_name: str, module_names: list[str]) -> dict[str, Any]:
    """Gets the class associated to a string name."""
    classes = {}
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and string_name.lower() in name.lower():
                classes[name] = obj

    return classes

# def shift_date_by_window(
#         date_str: str,
#         window: int,
#         direction: str,
#         date_frequency: str
#     ) -> str:
#         """
#         Shifts the date by a window in the specified direction.
#         """
#         if direction not in ["forward", "backward"]:
#             raise ValueError("Direction must be 'forward' or 'backward'")

#         try:
#             date_freq_number = re.search(r"\d+", date_frequency)[0]
#         except TypeError as e:
#             date_freq_number = 1

#         date_freq_units = re.search(r"(\D+)$", date_frequency)[0]

#         shift_periods = int(date_freq_number) * window
#         shift_periods = pd.Timedelta(shift_periods, unit=date_freq_units) # type: ignore
#         shifted_date = pd.to_datetime(date_str) + shift_periods if direction == 'forward' else pd.to_datetime(date_str) - shift_periods
#         return shifted_date.strftime("%Y-%m-%d %H:%M:%S")

def shift_date_by_window(
        date_str: str,
        window: int,
        direction: str,
        date_frequency: str
    ) -> str:
    """
    Shifts the date by a window in the specified direction.
    """
    # Validate direction
    if direction not in ["forward", "backward"]:
        raise ValueError("Direction must be 'forward' or 'backward'")

    # Extract number from date_frequency
    number_match = re.search(r"\d+", date_frequency)
    if number_match:
        date_freq_number = int(number_match.group())
    else:
        date_freq_number = 1  # Default to 1 if no number is found

    # Extract units from date_frequency
    unit_match = re.search(r"[a-zA-Z]+", date_frequency)
    if unit_match:
        date_freq_units = unit_match.group()
    else:
        raise ValueError("Invalid date_frequency format; must include time unit (e.g., 'D', 'H')")

    # Calculate shift periods
    shift_periods = date_freq_number * window
    shift_timedelta = pd.Timedelta(shift_periods, unit=date_freq_units)

    # Parse the original date
    try:
        original_date = pd.to_datetime(date_str)
    except Exception as e:
        raise ValueError(f"Invalid date_str format: {e}") from e

    # Shift the date
    if direction == 'forward':
        shifted_date = original_date + shift_timedelta
    else:
        shifted_date = original_date - shift_timedelta

    return shifted_date.strftime("%Y-%m-%d")