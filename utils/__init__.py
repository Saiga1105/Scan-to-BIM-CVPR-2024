"""
Utilities 

"""
import numpy as np
import time
from pathlib import Path

def timer(func):
    """
    Decorator that measures and prints the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the function with its arguments
        end_time = time.time()  # Capture the end time
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result  # Return the result of the function
    return wrapper

def get_list_of_files(folder: Path | str , ext: str = None) -> list:
    """
    Get a list of all filepaths in the folder and subfolders that match the given file extension.

    Args:
        folder: The path to the folder as a string or Path object
        ext: Optional. The file extension to filter by, e.g., ".txt". If None, all files are returned.

    Returns:
        A list of filepaths that match the given file extension.
    """
    folder = Path(folder)  # Ensure the folder is a Path object
    allFiles = []
    # Iterate over all the entries in the directory
    for entry in folder.iterdir():
        # Create full path
        fullPath = entry
        # If entry is a directory then get the list of files in this directory 
        if fullPath.is_dir():
            allFiles += get_list_of_files(fullPath, ext=ext)
        else:
            # Check if file matches the extension
            if ext is None or fullPath.suffix.lower() == ext.lower():
                allFiles.append(fullPath.as_posix())
    return allFiles