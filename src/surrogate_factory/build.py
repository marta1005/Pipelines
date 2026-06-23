"""
 Copyright (c) 2025 Airbus Operations S. L. This file is part of project Surrogate Factory released under the Airbus Inner Source shared-maintenance
 """

import os
from pathlib import Path
import argparse
import json
import yaml
import shutil
from shutil import ignore_patterns # Still needed for the ignore function



from .version import version
import fsspec

from typing import Callable, Optional, Set, List

class FileSystemManager:
    """
    Abstracts filesystem operations (local, S3, GCS, etc.)
    using fsspec.
    """
    def __init__(self, base_path_str: str):
        """
        Initializes the manager.
        
        Args:
            base_path_str (str): The root path to operate on,
                                 e.g., './local/folder' or 's3://my-bucket'.
        """
        # url_to_fs returns the filesystem (fs) object and the "clean" path
        self.fs, self.root_path = fsspec.core.url_to_fs(base_path_str)
        self.sep = self.fs.sep

    def join(self, *parts) -> str:
        """
        Joins path components using the correct separator for the filesystem.
        
        Args:
            *parts: Path components to join.
            
        Returns:
            str: The fully joined path.
        """
        return self.sep.join(parts)

    def mkdir(self, path_str: str, exist_ok=True):
        """
        Creates a directory in the target filesystem.
        
        Args:
            path_str (str): The directory path to create.
            exist_ok (bool): If True, does not raise an error if the
                             directory already exists.
        """
        self.fs.mkdir(path_str, parents=True,exist_ok=exist_ok)

    def exists(self, path_str: str) -> bool:
        """
        Checks if a path exists in the target filesystem.
        
        Args:
            path_str (str): The path to check.
            
        Returns:
            bool: True if the path exists, False otherwise.
        """
        return self.fs.exists(path_str)

    def write_yaml(self, path_str: str, data: dict):
        """
        Writes a dictionary as a YAML file to the specified path.
        
        Args:
            path_str (str): The destination file path.
            data (dict): The dictionary to serialize and write.
        """
        with self.fs.open(path_str, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

    def copy_file(self, src_path_str: str, dest_path_str: str):
        """
        Copies a single file from ANY source (local, S3, http) to
        a destination within THIS filesystem.
        
        Args:
            src_path_str (str): The source file path (any protocol).
            dest_path_str (str): The destination file path (on this fs).
        """
        # FIX: Open the source file using fsspec (auto-detects protocol)
        with fsspec.open(src_path_str, "rb") as f_src:
            # Open the destination file using this manager's filesystem
            with self.fs.open(dest_path_str, "wb") as f_dest:
                # Stream the data from source to destination
                shutil.copyfileobj(f_src, f_dest)


    def copy_tree(self, 
                  src_local_dir: str, 
                  dest_remote_dir: str, 
                  ignore: Optional[Callable[[str, List[str]], Set[str]]] = None, 
                  dirs_exist_ok=True):
        """
        Replicates shutil.copytree, copying from a LOCAL directory
        to a directory in the fsspec filesystem (potentially remote),
        while respecting the 'ignore' function logic.
        
        Args:
            src_local_dir (str): The path to the LOCAL source directory.
            dest_remote_dir (str): The path to the destination directory
                                   on the fsspec filesystem.
            ignore (Callable, optional): A function (like one returned by
                                       shutil.ignore_patterns) that takes
                                       (src_dir, names) and returns a set
                                       of names to ignore.
            dirs_exist_ok (bool): If False, raises FileExistsError if the
                                  destination directory already exists.
        """
        src_local_dir = str(src_local_dir) # Ensure it's a string
        
        if not dirs_exist_ok and self.exists(dest_remote_dir):
            raise FileExistsError(f"Destination '{dest_remote_dir}' already exists.")

        # self.mkdir(dest_remote_dir, exist_ok=True)

        # The 'ignore' function should be the one returned by ignore_patterns
        ignore_func = ignore if ignore else lambda src, names: set()

        for src_dir, dirs, files in os.walk(src_local_dir, topdown=True):
            # Apply the exclusion logic from shutil.copytree
            ignored_dirs = ignore_func(src_dir, dirs)
            ignored_files = ignore_func(src_dir, files)

            # Filter directories (modifying 'dirs' in-place so os.walk ignores them)
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            # Calculate the relative path to create in the destination
            rel_dir = os.path.relpath(src_dir, src_local_dir)
            
            if rel_dir == ".":
                current_dest_dir = dest_remote_dir
            else:
                current_dest_dir = self.join(dest_remote_dir, rel_dir)

            # Create directories in the destination
            for d in dirs:
                self.mkdir(self.join(current_dest_dir, d), exist_ok=True)
                
            # Copy (upload) files
            for f in files:
                if f not in ignored_files:
                    local_file_path = os.path.join(src_dir, f)
                    remote_file_path = self.join(current_dest_dir, f)
                    
                    # 'put' uploads a local file to a remote system
                    self.fs.put(local_file_path, remote_file_path)



class Builder:
    """
    Pipeline class to prepare the environment.
    
    This class is filesystem-agnostic and uses a FileSystemManager
    to handle all file operations, allowing it to work with
    local paths, S3, GCS, etc.
    
    Main methods:
     - build : generate folders and copy templates with the job_name
    """
    def __init__(self, job_name: str, pipelinepath_str: str = os.getcwd(), data: Optional[str] = None):
        """
        Initializes the Builder.
        
        Args:
            job_name (str): The name for the job/pipeline.
            pipelinepath_str (str): The root path for the new pipeline.
                                    Can be local ('./my-project') or
                                    remote ('s3://my-bucket/my-project').
            data (str, optional): The path to the input data. Can be
                                  local or remote.
        """
        self.job_name = job_name
        
        # 1. The templates path is ALWAYS LOCAL (where this code runs)
        self.templatespath = Path(__file__).parent / "templates"
        
        # 2. The input data path can be anything (local, S3, etc.)
        self.input_data = data
        
        # 3. The destination is managed by our FileSystemManager
        self.fs_manager = FileSystemManager(pipelinepath_str)

        # 4. All destination paths are created using the manager
        self.pipelinepath = self.fs_manager.join(self.fs_manager.root_path, self.job_name)
        self.fs_manager.mkdir(self.pipelinepath, exist_ok=True)
        
        self.metadatadir = self.fs_manager.join(self.pipelinepath, "metadata")
        self.datadir = self.fs_manager.join(self.pipelinepath, "data")
        self.artifacts = self.fs_manager.join(self.datadir, "artifacts")
        self.pythondir = self.fs_manager.join(self.pipelinepath, "python_nodes_library")
        

    def build(self):
        """
        Creates the folder structure and copies the template files
        to the target filesystem (local, S3, etc.).
        """
        
        print("###  Creating Folder Structure  ###")
        # pipelinepath.mkdir() is already done in __init__
        self.fs_manager.mkdir(self.datadir, exist_ok=True)
        self.fs_manager.mkdir(self.artifacts, exist_ok=True)

        
        
        if self.input_data:
            # If data is provided, ignore the file for data GENERATION
            ignore_pattern = ignore_patterns("SF_2_DataGeneration_Acquisition.*", "*-doe.pipeline")
        else:
            # If no data is provided, ignore the file for data ACQUISITION
            ignore_pattern = ignore_patterns("SF_2_Data_Acquisition.*", "*-input_data.pipeline")
            
        # 2. Replace shutil.copytree with our manager's method
        print(f"Copying templates from {self.templatespath} to {self.pipelinepath}...")
        self.fs_manager.copy_tree(
            self.templatespath, 
            self.pipelinepath,
            ignore=ignore_pattern,
            dirs_exist_ok=True
        )

        ## Copy input data
        if self.input_data:
            print(f"Copying input data from {self.input_data}...")
            try:
                # Check if the input data exists (local or remote)
                input_fs, input_path = fsspec.core.url_to_fs(self.input_data)
                
                if input_fs.exists(input_path):
                    # Use Pathlib only to get the suffix (it's safe)
                    suffix = Path(self.input_data).suffix 
                    dest_data_path = self.fs_manager.join(
                        self.datadir, 
                        f"{self.job_name}_raw_dataset{suffix}"
                    )
                    
                    # 3. Replace shutil.copy2 with our manager's copy_file
                    # This copies from ANY source (src) to OUR filesystem (dest)
                    self.fs_manager.copy_file(self.input_data, dest_data_path)
                    print("Input data copied.")
                else:
                    print(f"Warning: Input data path not found: {self.input_data}")
            except Exception as e:
                print(f"Warning: Could not copy input data. Error: {e}")


        print("###  Initialising metadata yaml file  ###")
        metadata = self.__gen_metadata()
        
        config_path = self.fs_manager.join(self.pipelinepath, "pipeline_config.yaml")
        
        # 4. Replace open() with our manager's method
        self.fs_manager.write_yaml(config_path, metadata)
        
        print("###  Pipeline successfully built  ###")
        
    
    def __gen_metadata(self) -> dict:
        """
        Generates the metadata dictionary for the config file.
        All paths are already correctly formatted (e.g., 's3://...').
        
        Returns:
            dict: The metadata dictionary.
        """
        
        metadata = dict()
        
        metadata['job_name'] = self.job_name
        metadata['Surrogate Factory'] = str(version)
        metadata['pipeline.folder'] = str(self.pipelinepath)
        metadata['python_libs.folder']= str(self.pythondir)
        metadata['metadata.folder'] = str(self.metadatadir)
        metadata['data.folder'] =  str(self.datadir)
        metadata['# metadata_schema'] =  "Uncomment and add path to metadata schema"
        metadata['artifacts.folder'] =  str(self.artifacts)
        metadata['debug'] = "False #True to get all execution logs"
        metadata['verbose'] = "True #False to silent execution"

        if self.input_data:
             metadata['input_data'] =  str(self.input_data) # The original input path
        
        metadata['tracker'] = {
                             "tool": "mlflow",
                             "options":{
                                 # The tracking_uri must also use the fsspec path
                                 "tracking_uri": "file:///" & str(self.fs_manager.join(self.datadir, "tracking")),
                                 # "apikey_path": ""
                                 },
                             }
        print("Please, edit the config file with the path to your artifactory apikey")
        
        return metadata
    
    def run(self, steps: list = []) -> str:
        """_summary_
        This function will execute all notebooks of the pipeline.
        It will check if the libraries are in the python folder
        Args:
            steps (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        return "Not Implemented"

#### Main program
def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j', '--jobname', 
        help='Job Name of the Pipeline', 
        required=True
    )
    parser.add_argument(
        '-p', '--path', 
        help="Path of the Pipeline to build (e.g., ./local-proj or s3://my-bucket/remote-proj)"
    )
    parser.add_argument(
        '--data', 
        default=None, 
        help="Input data path (local or remote, e.g., s3://...)"
    )

    args = parser.parse_args()
    
    if args.path is not None:
        pipelinepath = args.path
    else:
        pipelinepath = os.getcwd()
        print(f"No path specified, using current working directory: {pipelinepath}")
    
    # 'pipelinepath' can now be any string that fsspec understands
    pipeline = Builder(args.jobname, pipelinepath, args.data)
    
    pipeline.build()
    
if __name__ == '__main__':
    main()