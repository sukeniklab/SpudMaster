# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:32:07 2025

@author: patri
"""

import os
import pandas as pd
from typing import List, Union
from constants.columns_names import col_dic
from file_handling.generic_handlers import split_filename

class Key:
    """
    The Key class processes key information from a pandas DataFrame, CSV file, or a list of CSV file paths.
    It builds a mapping to generate unique save file names based on image file information and a well identifier.
    
    Main functionality:
      - Loads and normalizes key data (forcing column names to lowercase).
      - Optionally validates that all expected columns exist.
      - Updates an internal column dictionary (col_dic) with any additional keys provided.
      - Creates a dictionary mapping each well (from the "well" column) to a generated string.
      - Generates a final filename by combining well-specific information with image-specific data.
      
    Usage:
        key_instance = Key(key_input, image_file_information=['date', 'experiment', 'well'])
        final_filename = key_instance.current_filename_save_parameters(image_file, well)
    """
    
    def __init__(self, 
                 key: Union[pd.DataFrame, str, List[str]], 
                 image_file_information: List[str], 
                 add_dic_key: Union[str, List[str]] = None,
                 skip_validation: bool = False):
        """
        Initialize the Key class.
        
        Parameters:
            key: Input key data (DataFrame, CSV file path, or list of CSV file paths).
            image_file_information: List of image file information fields to parse.
            add_dic_key: Optional additional keys to add to the internal column dictionary.
            skip_validation: If True, skip validation of column names.
        """
        # Load key data from DataFrame, CSV file, or list of CSV file paths.
        if isinstance(key, pd.DataFrame):
            self.key = key.copy()
            self.key.columns = self.column_lower()
        elif isinstance(key, str):
            try:
                self.key = pd.read_csv(key)
                self.key.columns = self.column_lower()
            except Exception as e:
                raise ValueError("Expected a valid file path for key.") from e
        elif isinstance(key, list):
            try:
                key_dfs = []
                for file_path in key:
                    tmp = pd.read_csv(file_path)
                    key_dfs.append(tmp)
                self.key = pd.concat(key_dfs, ignore_index=True)
                self.key.columns = self.column_lower()
            except Exception as e:
                raise ValueError("Expected a list of file path strings for key.") from e
        else:
            raise ValueError("Key must be a DataFrame, a file path string, or a list of file paths.")
        
        # Create a copy of the column dictionary to avoid mutating the imported dictionary.
        self.col_dic = col_dic.copy()
        if add_dic_key is not None:
            if isinstance(add_dic_key, str):
                self.col_dic[add_dic_key.lower()] = add_dic_key.lower()
            elif isinstance(add_dic_key, list):
                new_dic = {col.lower(): col.lower() for col in add_dic_key}
                self.col_dic.update(new_dic)
        
        # Optionally validate the columns.
        self.skip_validation = skip_validation
        if not self.skip_validation:
            self.validate_column_name()
        
        # Build a dictionary mapping wells to generated file name strings.
        self.save_image_dic = self._make_file_name_from_key(self.key)
        # Store the image file information used for constructing image-specific save parameters.
        self.image_file_information = image_file_information

    def column_lower(self) -> List[str]:
        """
        Convert the DataFrame's column names to lowercase.
        
        Returns:
            List[str]: List of lowercase column names.
        """
        try:
            return [col.lower() for col in self.key.columns]
        except Exception as e:
            raise RuntimeError("Failed to convert column names to lowercase.") from e

    def _check_strings(self, columns: List[str]) -> bool:
        """
        Check that every column name in the provided list exists in the internal column dictionary.
        
        Parameters:
            columns (List[str]): List of column names to check.
        
        Returns:
            bool: True if all columns are found.
        """
        return all(string in self.col_dic for string in columns)
        
    def validate_column_name(self) -> bool:
        """
        Validate that the key DataFrame's columns match the expected column names.
        If any expected column is missing, a detailed error is raised.
        
        Returns:
            bool: True if all expected columns are present.
            
        Raises:
            ValueError: If one or more expected columns are missing.
        """
        expected = list(self.col_dic.keys())
        actual = list(self.key.columns)
        missing = [col for col in expected if col not in actual]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        return True
            
    def _get_columns(self) -> List[str]:
        """
        Retrieve a list of column names from the key DataFrame.
        
        Returns:
            List[str]: List of column names.
        """
        return list(self.key.columns)
        
    def _make_file_name_from_key(self, df: pd.DataFrame) -> dict:
        """
        Create a dictionary mapping the value from the 'well' column to a generated string.
        The string is constructed by concatenating each column name with a hyphen and the
        corresponding cell value from the row.
        
        Parameters:
            df (pd.DataFrame): The key DataFrame.
            
        Returns:
            dict: Dictionary with well values as keys and generated file name strings as values.
        """
        well_col = 'well'
        # Validate columns if not skipping.
        if not self.skip_validation:
            self.validate_column_name()
        
        result = {}
        for idx, row in df.iterrows():
            try:
                key_val = row[well_col]
                # Build a string for this row using "column-value" for each column.
                value_parts = [f"{col}-{row[col]}" for col in df.columns]
                value = "_".join(value_parts)
                result[key_val] = value
            except Exception as e:
                raise RuntimeError(f"Error generating file name from key for row {idx}.") from e
        return result

    def _check_well_dic(self, diction: dict, separator: str) -> dict:
        """
        Split the well information into 'well' and 'techReplicate' parts if the separator is present.
        
        Parameters:
            diction (dict): Dictionary containing image file information.
            separator (str): Character used to separate well and technical replicate.
            
        Returns:
            dict: Updated dictionary with separated 'well' and 'techReplicate' keys.
        """
        new_diction = diction.copy()
        if "well" in new_diction and separator in new_diction["well"]:
            try:
                split_ind = new_diction["well"].split(separator, 1)
                new_diction["well"] = split_ind[0]
                new_diction["techReplicate"] = split_ind[1]
            except Exception as e:
                raise RuntimeError("Error splitting well information.") from e
        return new_diction

    def _make_image_name_dict(self, list_names: List[str]) -> dict:
        """
        Create a dictionary by pairing the pre-defined image file information with
        the components obtained by splitting an image filename.
        
        Parameters:
            list_names (List[str]): List of strings obtained from splitting an image filename.
            
        Returns:
            dict: Dictionary mapping image file info keys to their corresponding values.
        """
        try:
            new_dic = {key: value for key, value in zip(self.image_file_information, list_names) if key is not None}
            new_dic = self._check_well_dic(new_dic, '-')
            return new_dic
        except Exception as e:
            raise RuntimeError("Error generating image name dictionary.") from e
        
    def get_image_dic(self) -> dict:
        """
        Return the pre-built save image dictionary.
        """
        return self.save_image_dic

    def get_col_by_well(self, col: str, well: str):
        """
        Returns column information based on the well provided.
        
        Parameters:
            col (str): The column name to retrieve.
            well (str): The well identifier.
            
        Returns:
            pd.Series: The series of values for the given column and well.
            
        Raises:
            ValueError: If the column is not found in the DataFrame.
        """
        if col not in self.key.columns:
            raise ValueError(f"{col} not found in dataframe columns")
        return self.key.loc[self.key['well'] == well][col]
    
    def current_filename_save_parameters(self, image_file: str, well: str) -> str:
        """
        Build the final filename for saving an image by combining pre-built well-specific
        information with additional image-specific information extracted from the image filename.
        
        Parameters:
            image_file (str): Path to the image file.
            well (str): Well identifier used to retrieve pre-built save information.
            
        Returns:
            str: Final filename string ending with '.tif'.
            
        Raises:
            ValueError: If no save information is found for the provided well.
            RuntimeError: If any error occurs during filename construction.
        """
        try:
            # Parse the image file to get its components.
            image_file_names = split_filename(image_file, '_')
            # Build a dictionary mapping image file information.
            image_info_dic = self._make_image_name_dict(image_file_names)
            # Remove the well information from the dictionary.
            if 'well' in image_info_dic:
                del image_info_dic['well']
            # Convert the dictionary to a string in the format "key-value_key-value".
            image_info_str = "_".join([f"{k}-{v}" for k, v in image_info_dic.items()])
            # Retrieve prebuilt save information for the given well.
            current_well_information = self.save_image_dic.get(well, "")
            if not current_well_information:
                raise ValueError(f"No save information found for well: {well}")
            # Construct the final filename.
            final_filename = f"{image_info_str}_{current_well_information}.tif"
            return final_filename
        except Exception as e:
            raise RuntimeError("Error constructing current filename save parameters.") from e
