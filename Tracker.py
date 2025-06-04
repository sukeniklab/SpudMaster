import os
import json
import shutil
import logging
from datetime import datetime

class Tracker:
    """
    Tracker class for an imaging pipeline.
    
    This class tracks progress, stores the list of all files and the subset of files
    that remain to be processed, logs progress and errors, and supports persistence so
    that processing can resume from previous runs.
    """

    def __new__(cls, tracking_file='tracker_data.json', fail_dir = None,  *args, **kwargs):
        instance = super().__new__(cls)
        instance.tracking_file = tracking_file
        instance.fail_dir = fail_dir
        # Initialize logger here so that it is available in __new__
        instance.logger = logging.getLogger('Tracker')
        if not instance.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            instance.logger.addHandler(handler)
        instance.logger.setLevel(logging.INFO)

        # Load existing tracking data if available.
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r') as f:
                    instance.tracker_data = json.load(f)
                instance._log_info(f"Loaded tracker data from {tracking_file}")
            except Exception as e:
                instance._log_error(f"Failed to load tracker file: {e}")
                instance.tracker_data = {}
        else:
            # Initialize the tracker data if file doesn't exist.
            instance.tracker_data = {
                'all_files': [],
                'current_files': [],
                'failed_files': [],
                'pipeline_failed': False,
                'last_run': str(datetime.now())
            }
        return instance

    def __init__(self, tracking_file='tracker_data.json'):
        # __init__ can be used to further set up the instance if needed.
        pass

    def _log_info(self, message: str):
        self.logger.info(message)

    def _log_error(self, message: str):
        self.logger.error(message)

    def pipeline_failed(self) -> bool:
        """
        Determines whether the last iteration of images was successful.
        Returns True if there were errors flagged, False otherwise.
        """
        return self.tracker_data.get('pipeline_failed', False)
    
    def load_files(self, file_list: list['str']):
        self.tracker_data['all_files'] =file_list
        self.tracker_data['current_files'] = file_list.copy()
        self._log_info(f"Loaded {len(file_list)} files")
        self.save_tracker()
    
    def list_all_files(self, directory: str, file_extension: str = None) -> list:
        """
        Scans the given directory (and subdirectories) for files.
        Optionally filter by file extension.
        
        Updates internal 'all_files' and 'current_files' lists.
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if file_extension is None or filename.lower().endswith(file_extension.lower()):
                    files.append(os.path.join(root, filename))
        self.tracker_data['all_files'] = files
        # Start with all files as the current processing list.
        self.tracker_data['current_files'] = files.copy()
        self._log_info(f"Found {len(files)} files in {directory}")
        self.save_tracker()
        return files

    def current_file_list(self) -> list:
        """
        Returns the list of files that have yet to be processed.
        """
        return self.tracker_data.get('current_files', [])

    def display_progress(self):
        """
        Logs a summary of progress: total files, processed files, and remaining files.
        """
        total = len(self.tracker_data.get('all_files', []))
        remaining = len(self.tracker_data.get('current_files', []))
        failed_files = len(self.tracker_data.get('failed_files', []))
        processed = total - remaining
        self._log_info(f"Progress: {processed}/{total} files processed. {remaining} remaining. files failed: {failed_files}.")

    
    def get_failed_files(self):
       return self.tracker_data['failed_files']
       
    def file_failed(self, file_path):
        # Check if the file is listed in current_files.
        if file_path in self.tracker_data.get('current_files', []):
            # Use append() instead of extend() to add the full file_path.
            self.tracker_data['failed_files'].append(file_path)
            # Remove the file from the correct key; assuming the key should be 'current_files'
            self.tracker_data['current_files'].remove(file_path)
            
            if self.fail_dir is not None:
                self.__log_info(f"File failed, moved {file_path} to {self.fail_dir}")
                self.move_file_to_directory(file_path, self.fail_dir)
                self.save_tracker()
            else:
                self.__log_info(f"{file_path} failed processing")
    
    def remove_file_from_list(self, file_path: str):
        """
        Removes a file from the current file list (e.g. after it has been processed).
        """
        if file_path in self.tracker_data.get('current_files', []):
            self.tracker_data['current_files'].remove(file_path)
            self._log_info(f"Removed file from list: {file_path}")
            self.save_tracker()
        else:
            self._log_error(f"File not found in current list: {file_path}")

    def move_file_to_directory(self, file_path: str, destination_directory: str):
        """
        Moves the specified file to the destination directory (for example, a 'failed' folder).
        """
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
            self._log_info(f"Created destination directory: {destination_directory}")
        try:
            shutil.move(file_path, destination_directory)
            self._log_info(f"Moved {file_path} to {destination_directory}")
        except Exception as e:
            self._log_error(f"Error moving {file_path} to {destination_directory}: {e}")

    def evaluate_error(self, file_path: str, error: Exception):
        """
        Logs an error for a specific file and adds it to the failed_files list.
        Also flags the pipeline as failed.
        """
        self._log_error(f"Error processing {file_path}: {error}")
        # Add to the list of failed files.
        failed = self.tracker_data.get('failed_files', [])
        if file_path not in failed:
            failed.append(file_path)
        self.tracker_data['failed_files'] = failed
        self.tracker_data['pipeline_failed'] = True
        self.save_tracker()

    def save_tracker(self):
        """
        Saves the current tracker data to a JSON file.
        """
        try:
            with open(self.tracker_file_path(), 'w') as f:
                json.dump(self.tracker_data, f, indent=4)
            self._log_info("Tracker data saved.")
        except Exception as e:
            self._log_error(f"Failed to save tracker data: {e}")

    def tracker_file_path(self) -> str:
        """
        Returns the path to the tracker file.
        """
        return self.tracking_file

    def reset_tracker(self):
        """
        Resets the tracker data to start a new run.
        """
        self.tracker_data = {
            'all_files': [],
            'current_files': [],
            'failed_files': [],
            'pipeline_failed': False,
            'last_run': str(datetime.now())
        }
        self.save_tracker()
        self._log_info("Tracker has been reset.")

    # Helper functions that can be used outside the class:
    @staticmethod
    def helper_format_file_name(file_path: str) -> str:
        """
        Returns a formatted file name (for logging or saving).
        """
        return os.path.basename(file_path)

    @staticmethod
    def helper_create_directory(directory: str):
        """
        Ensures that a directory exists.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory