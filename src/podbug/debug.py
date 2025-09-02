import inspect
import sys
from pydantic import BaseModel, Field
from typing import Any, Tuple, Union


class Result_T(BaseModel):
    data: Any = Field(default=None)
    err: str | None = Field(default=None)

    def is_ok(self):
        return not self.err

    def is_err(self):
        return self.err is not None

    def __str__(self):
        """String representation"""
        if self.err is not None:
            return f"Result(err='{self.err}')"
        return str(self.data)
    
    def __repr__(self):
        """Debug representation"""
        if self.err is not None:
            return f"Result(err='{self.err}')"
        return f"Result(data={repr(self.data)})"
    
    def __bool__(self):
        """Boolean evaluation - True if no error"""
        return self.err is None
    
    def unwrap(self):
        """Get the data - only use when you're sure there's no error"""
        return self.data
    
    def unwrap_or(self, default):
        """Get the data or return default if there's an error"""
        if self.err is not None:
            return default
        return self.data
    
    def and_then(self, func) -> "Result_T":
        """Chain operations - only applies func if no error"""
        if self.err is not None:
            return self
        try:
            return func(self.data)
        except Exception as e:
            return Result_T.error(str(e))
        
    @classmethod
    def ok(cls, data) -> "Result_T":
        """Create a successful result"""
        return cls(data=data)
    
    @classmethod
    def error(cls, err: str) -> "Result_T":
        """Create an error result"""
        return cls(err=err)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data) if self.data is not None else 0
    
def try_result(operation):
    try:
        if callable(operation):
            # call it if it's callable
            return Result_T.ok(operation())
        else:
            # if it's already a Result_T, return as-is
            return operation if isinstance(operation, Result_T) else Result_T.ok(operation)
    except Exception as e:
        return Result_T.error(str(e))

# Global debug flag - set to True to enable debug output
PO_DEBUG = True

def _format_pair(pair: Tuple[Any, Any]) -> str:
    """Format a pair/tuple of 2 elements like C++ pair output."""
    return f"({pair[0]}, {pair[1]})"

def _format_container(container: Union[list, set, tuple, dict]) -> str:
    """Format containers like C++ container output."""
    if isinstance(container, dict):
        # Format dict as {key: value, key: value}
        items = [f"{k}: {v}" for k, v in container.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(container, str):
        # Strings should be printed as-is, not as containers
        return container
    elif hasattr(container, '__iter__'):
        # Format other iterables as {item, item, item}
        items = [str(x) for x in container]
        return "{" + ", ".join(items) + "}"
    else:
        return str(container)

def format_value(value: Any) -> str:
    """Smart formatting based on type."""
    if isinstance(value, tuple) and len(value) == 2:
        return _format_pair(value)
    elif isinstance(value, (list, set, dict)) or (hasattr(value, '__iter__') and not isinstance(value, str)):
        return _format_container(value)
    else:
        return str(value)

def _tuple_out(t: tuple) -> None:
    """Output tuple elements separated by spaces."""
    output = ' '.join(str(arg) for arg in t)
    print(output, file=sys.stderr)

def deduce(arg: Any) -> None:
    """Deduce output format based on argument type."""
    if isinstance(arg, tuple):
        _tuple_out(arg)
    else:
        print(format_value(arg), end=' ', file=sys.stderr)

def println(*args: Any) -> None:
    """Print arguments with smart formatting."""
    for arg in args:
        deduce(arg)
    print(file=sys.stderr)  # Final newline

def dbug_out(*args: Any) -> None:
    """Debug output function - equivalent to the C++ macro."""
    if not PO_DEBUG:
        return
    
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('/')[-1]  # Just filename, not full path
    line_number = frame.f_lineno
    
    # Get the actual code line to extract variable names
    try:
        import linecache
        code_line = linecache.getline(filename, line_number).strip()
        
        # Extract the arguments from the dbug_out call
        if 'dbug_out(' in code_line:
            start = code_line.find('dbug_out(') + len('dbug_out(')
            end = code_line.rfind(')')
            if end > start:
                var_names_str = code_line[start:end]
            else:
                var_names_str = ', '.join(f"arg{i}" for i in range(len(args)))
        else:
            var_names_str = ', '.join(f"arg{i}" for i in range(len(args)))
    finally:
        # Fallback if we can't read the source
        var_names_str = ', '.join(f"arg{i}" for i in range(len(args)))
    
    # Format output like C++ version
    print(f"[{filename}:{line_number}] ({var_names_str}): ", end='', file=sys.stderr)
    println(*args)

def list_dimensions(self, lst):
    """
    Returns the dimensions of a nested list structure.

    Args:
        lst: The input list (can be nested)

    Returns:
        A tuple representing the dimensions of the list at each level.
        For a flat list, returns (length,).
        For a 2D list, returns (rows, columns).
    """
    if not isinstance(lst, list):
        return ()  # Not a list

    if not lst:
        return (0,)  # Empty list

    dimensions = [len(lst)]

    # Check if the first element is a list (assuming regular structure)
    if lst and isinstance(lst[0], list):
        dimensions.extend(self.list_dimensions(lst[0]))

    return tuple(dimensions)

def flatten_list(self, nested_list: list) -> list:
    # Recursive function to flatten list  
    def __flatten(a):  
        res = []  
        for x in a:  
            if isinstance(x, list):  
                res.extend(__flatten(x))  # Recursively flatten nested lists  
            else:  
                res.append(x)  # Append individual elements  
        return res 
    
    return __flatten(nested_list)




# def get_shape(obj):
#    import numpy as np
#    import torch
#     """
#     Returns the dimensions for nested Tensor/ndarry/list
#     Might be more overhead than `list_dimensions`
#     """
#     if isinstance(obj, torch.Tensor):
#         return tuple(obj.shape)
#     elif isinstance(obj, np.ndarray):
#         return obj.shape
#     elif isinstance(obj, list):
#         return np.array(obj).shape  # Convert to NumPy array to infer shape
#     else:
#         try:
#             return np.array(obj).shape
#         finally:
#             return f'{type(obj)} is not supported'


def parse_json_response(response_text: str):
    import json
    """Parse JSON from markdown"""
    import re

    # Try to find markdown code block
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # Try to find JSON structure
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
        else:
            json_text = response_text.strip()

    # Clean up
    json_text = json_text.replace("```", "").strip()

    return try_result(json.loads(json_text))
