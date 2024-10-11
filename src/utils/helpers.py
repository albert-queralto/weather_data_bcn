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

