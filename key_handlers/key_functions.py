import re


def get_unique_well_list(directory_list: list, current_id: str) -> list:
    """
    Returns a list of directories from directory_list that contain the current_id.

    Args:
        directory_list (list): A list of directory names or paths to be searched.
        current_id (str): The identifier to search for in the directory names.

    Returns:
        list: A list of directory names containing the current_id.
    
    Raises:
        TypeError: If directory_list is not a list or current_id is not a string.
        ValueError: If directory_list is empty.
    """
    
    # Check for correct input types
    if not isinstance(directory_list, list):
        raise TypeError("directory_list must be a list")
    if not isinstance(current_id, str):
        raise TypeError("current_id must be a string")
    
    # Check if directory_list is empty
    if not directory_list:
        raise ValueError("directory_list cannot be empty")

    
    pattern = re.compile(fr"_{current_id}-\d+")  
    
    # Filter directory_list to find elements containing current_id
    return [x for x in directory_list if pattern.search(x)]
   