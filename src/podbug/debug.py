import inspect
import ast
import sys
import os

# def dbug_out(*args) -> None:
#     print(_dbug_to(args))

# def dbug_out_to(*args) -> None:
#     file = args[0]

#     if not file == sys.stdin:
#         file_path = f'{os.getcwd()}/{file}'
#         if not file.startswith('/home') and len(file) > 10:
#             ask = input(f'is it a valid path? {file_path}')

#             if not ask == 'y':
#                 dbug_out(args[1:])
#                 return

#         with open(f'{os.getcwd()}/{file}', 'a') as output_file:
#             print(_dbug_to(*args), file=output_file)

# def _dbug_to(*args) -> None:
#     if not __debug__:
#         return

#     cf = inspect.currentframe().f_back              # type: ignore
#     info = inspect.getframeinfo(cf)                 # type: ignore
#     src = ''.join(info.code_context or [])

#     try:
#         tree = ast.parse(src)
#         call = next(
#             n for n in ast.walk(tree)
#             if isinstance(n, ast.Call)
#             and getattr(n.func, "id", "") == "dbug_out"
#         )
#         labels = [ast.get_source_segment(src, a).strip() for a in call.args]        # type: ignore
#         parts = [f"{lbl}={val!r}" for lbl, val in zip(labels, args[1:])]
#     except Exception:
#         parts = [f"{val!r}" for val in args[1:]]

#     # line = f"line {info.lineno}: " + " ".join(parts)
#     line = f'line {info.lineno}: {''.join(parts)}'

#     return line


# def get_shape(obj):
#     import torch
#     import numpy as np
#     if isinstance(obj, torch.Tensor):
#         return tuple(obj.shape)
#     elif isinstance(obj, np.ndarray):
#         return obj.shape
#     elif isinstance(obj, list):
#         return np.array(obj).shape  # Convert to NumPy array to infer shape
#     else:
#         try:
#             return np.array(obj).shape
#         except:
#             return f'{type(obj)} is not supported'


def parse_json_response(response_text: str) -> dict | None:
    import json
    """Parse JSON from agent response with fallback methods."""
    try:
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

        parsed_json = json.loads(json_text)
        # print("✅ Successfully parsed JSON response")
        return parsed_json

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(
            f"Raw response: {response_text[:500]}{'...' if len(response_text) > 500 else ''}"
        )
        return None

import traceback
import inspect


class ResultError:
    """Enhanced error that tracks where it occurred"""
    def __init__(self, error: Exception, location: str | None = None):
        self.error = error
        self.location = location or self._get_caller_location()
        self.traceback = traceback.format_exc() if hasattr(error, '__traceback__') else None
    
    def _get_caller_location(self):
        """Get the file and line number where this error was created"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the caller
            caller_frame = frame.f_back.f_back  # Skip __init__ and immediate caller     # type: ignore
            filename = caller_frame.f_code.co_filename.split('/')[-1]     # type: ignore
            line_number = caller_frame.f_lineno     # type: ignore
            function_name = caller_frame.f_code.co_name     # type: ignore
            return f"{filename}:{line_number} in {function_name}()"
        finally:
            del frame
    
    def __str__(self):
        return f"{self.error} [at {self.location}]"
    
    def __repr__(self):
        return f"ResultError({self.error!r}, location='{self.location}')"


class Result:
    def __init__(self, ok=None, err=None):
        self.ok = ok
        self.err = err

    def is_ok(self):
        return self.err is None

    def is_err(self):
        return self.err is not None

    def unwrap(self):
        if self.is_err():
            return self.err
        return self.ok

    def __repr__(self):
        if self.is_ok():
            return f"Ok({self.ok})"
        else:
            return f"Err({self.err})"
    
    def and_then(self, fn):
        """Monadic bind - chains operations"""
        if self.is_err():
            return self
        try:
            return fn(self.ok)
        except Exception as e:
            return Result(err=ResultError(e))


def try_result(operation):
    """
    Unified function that combines try_result + _try_result_auto:
    - If operation is a function/lambda, wrap it in try_result
    - If operation is already a Result, unwrap it with auto context
    """
    if callable(operation):
        # Act like try_result - wrap the function call
        try:
            return Result(ok=operation())
        except Exception as e:
            return Result(err=ResultError(e))
    
    elif isinstance(operation, Result):
        # Act like _q_auto - unwrap the result with context
        if operation.is_err():
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back     # type: ignore
                filename = caller_frame.f_code.co_filename.split('/')[-1]     # type: ignore
                line_number = caller_frame.f_lineno     # type: ignore
                function_name = caller_frame.f_code.co_name     # type: ignore
                
                # Try to get the actual code line for context
                try:
                    with open(caller_frame.f_code.co_filename, 'r') as f:     # type: ignore
                        lines = f.readlines()
                        if line_number <= len(lines):
                            code_line = lines[line_number - 1].strip()
                        else:
                            code_line = "unknown"
                finally:
                    code_line = "unknown"
                
                location = f"{filename}:{line_number} in {function_name}()"
                error_msg = f"Error at {location}\n  Code: {code_line}\n  Error: {operation.err}"
                raise Exception(error_msg)
            
            finally:
                del frame

        return operation.ok
    else:
        return Result(err=ResultError(Exception("this is not a function")))
    


# Keep your original functions but they can now use try_result() for both purposes
def read_file(path):
    return try_result(lambda: open(path).read())


def to_upper(s):
    return try_result(lambda: s.upper())


def safe(fn):
    """Decorator version - wraps function to always return Result"""
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, Result):
                return result
            return Result(ok=result)
        except Exception as e:
            return Result(err=ResultError(e))
    return wrapper


@safe
def read_file_upper_unified(path):
    """Now you can use try_result() for both creating and unwrapping Results"""
    # try_result() with Result objects unwraps them (like _try_result_auto)
    content = try_result(read_file(path))
    upper_content = try_result(to_upper(content))
    return upper_content
    

# Demo using try_result() instead of try/except
if __name__ == "__main__":

    result1 = read_file_upper_unified("test.txt")
    print(f'Success {result1.unwrap()}' if result1.is_ok() else f'Error {result1}')
