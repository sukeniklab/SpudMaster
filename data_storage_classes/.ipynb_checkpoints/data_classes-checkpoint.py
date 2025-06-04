import os
from typing import Callable, Dict, List, Optional
import xarray
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from file_handling.generic_handlers import split_filename


@dataclass
class DirectoryStorageInformation:
    directories: List[str] = None
    current_directory: str = None
    experiment_type: str = None
    save_dir: List[str] = None
    save_dir_dict: Dict[str, str] = field(default_factory=dict)
    total_images: List[str] = None
    """
    Save directory and image directories need to be the same length.
    Possibly handle single inputs by returning a single directory if none other is given.
    """

@dataclass
class ImageInfoStorage:
    """Class for keeping track of images, masks, and save names."""
    image_file: str =None
    image_name: Optional[List[str]] = None
    image_dict: Dict[int, xarray.DataArray] = None
    construct: str = None
    data_columns: Optional[Dict[str, str]] = None
    image_size_dict: Dict[str, xarray.DataArray] = field(default_factory=dict)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    mask_save_information: Optional[str] = None

    def get_image_name(self) -> List[str]:
        # Assumes split_filename returns a list of strings based on the delimiter.
        return split_filename(self.image_file, '_')

    def get_image_column_variables(self) -> Dict[str, str]:
        if self.image_name is not None:
            # Create a dictionary by splitting each element of image_name by '-'
            # and using the first part as key and the second as value.
            return {item.split('-')[0]: item.split('-')[1] for item in self.image_name if '-' in item}
        else:
            return {}

    def get_construct(self):
        if self.data_columns != None:
            return self.data_columns['construct']
        else:
            raise ValueError('construct not found.')






@dataclass
class CutoffsParameters:
    tmp_item: list


@dataclass
class OrderingParameters:
    constr_order: list = field(default_factory=list)
    experiment_order: list = field(default_factory=list)
    experiment_condition: list = field(default_factory=list)
    experiment_parameter: list = field(default_factory=list)
    sort_dict: dict = field(default_factory=dict)

@dataclass
class DataFrameHolder:
    storage_file: str = "dataframe.pkl"  # File to store the DataFrame
    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # Load existing DataFrame if the file exists; otherwise, create an empty one
        if os.path.exists(self.storage_file):
            try:
                self.df = pd.read_pickle(self.storage_file)
                print(f"Loaded existing DataFrame from {self.storage_file}")
            except Exception as e:
                print(f"Error loading DataFrame: {e}. Initializing new DataFrame.")
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()
            print("Initialized new empty DataFrame.")

    def concat_data(self, new_data: pd.DataFrame):
        """Concatenates a new sub-dataframe and saves the result."""
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self._save_dataframe()

    def _save_dataframe(self):
        """Saves the DataFrame to the storage file."""
        try:
            self.df.to_pickle(self.storage_file)
            print(f"DataFrame saved to {self.storage_file}")
        except Exception as e:
            print(f"Failed to save DataFrame: {e}")

@dataclass
class PlottingStorage:
    plotting_medians: pd.Series
    plotting_location: Dict = field(default_factory=dict)