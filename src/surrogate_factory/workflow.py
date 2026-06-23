"""
 Copyright (c) 2025 Airbus Operations S. L. This file is part of project Surrogate Factory released under the Airbus Inner Source shared-maintenance
 """

from functools import partial
from collections import OrderedDict
from typing import Dict, Union, Any
from toolz import curry
import importlib
import pkgutil

import pandas as pd
import json
import yaml

import os, sys
from pathlib import Path
import functools
import inspect
import logging

# ==========================================
# Logger Configuration
# ==========================================
class LogsFormatter(logging.Formatter):
    """Custom logger formatter to add colors based on the log level."""
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Global handlers
console_handler = logging.StreamHandler(sys.stdout)

# Initialize the global logger for the workflow module
logger = logging.getLogger("SurrogateFactoryLogs")
logger.setLevel(logging.DEBUG) # Catch all levels, output filtered by handler

if not logger.handlers:
    # Console Handler (Colored)
    console_handler.setLevel(logging.INFO) # Change to DEBUG to see more granular logs
    console_handler.setFormatter(LogsFormatter())
    logger.addHandler(console_handler)

# ==========================================

def set_paths(config: Dict[str, Any]):
    """Insert the python node library into sys.path and change cwd to data folder."""
    p = Path(config['python_libs.folder'])
    if not p.exists():
        logger.error(f"Python node library directory ({p}) does not exist.")
        raise RuntimeError(f"The python node library directory for the current job ({p}) doesn't exist.")

    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
        logger.debug(f"Python library path inserted: {p}")

    data_folder = config['data.folder']
    cwd = Path(data_folder)
    cwd.mkdir(exist_ok=True)
    if Path(os.getcwd()) != cwd.resolve():
        logger.info(f"Changing working directory to Data folder: {data_folder}")
        os.chdir(str(cwd))


def setup_tracker(config):

    from .plugins import trackers

    tracker_name = config["tracker"]['tool']
    logger.info(f"Tracker requested: '{tracker_name}'")

    tracker = None
    for finder, name, ispkg in iter_namespace(trackers):
        if tracker_name in name:
            tracker = importlib.import_module(name)
            logger.debug(f"Successfully imported tracker module: {name}")
            if tracker_name == 'mlflow':
                logger.info("Setting up MLflow tracking environment...")
                tracker.setup_mlflow(config)

    if tracker is None:
        logger.warning(f"Tracker '{tracker_name}' is not available in the current environment.")
        available_trackers = {
            name: importlib.import_module(name)
            for finder, name, ispkg in iter_namespace(trackers)
        }
        logger.info(f"Available trackers are: {list(available_trackers.keys())}")

    return tracker


def set_catalog(config):

    from .catalog import Catalog
    catalog = Catalog(logger=logger)
    if "catalog" in config.keys():
        logger.info("Adding methods to Catalog from configuration.")
        catalog.load_config(config['catalog'])
    else:
        logger.debug("No catalog configuration found. Proceeding with empty catalog.")

    return catalog



class Workflow:
    
    def __init__(self, config_file):
        
        from .metadata import MetadataManager

        # Instanciar el logger como atributo de la clase para que el usuario pueda usarlo
        self.logger = logging.getLogger("SurrogateFactoryLogs")

        self.config_file = config_file
        self._load_config()
        
        # Configurar salida de terminal según los parámetros del YAML/JSON
        verbose = self.config.get('verbose', True)
        debug = self.config.get('debug', False)

        if not verbose:
            console_handler.setLevel(logging.CRITICAL + 1) # Silencia la terminal por completo
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        elif debug:
            console_handler.setLevel(logging.DEBUG) # Muestra absolutamente todo en terminal
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            console_handler.setLevel(logging.INFO) # Nivel estándar (sin debug)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Configurar FileHandler para guardar todos los logs en data.folder
        data_folder = self.config.get('data.folder', '.')
        self.log_file_path = os.path.join(data_folder, 'workflow_logs.txt')

        # Asegurar que el FileHandler solo se añade una vez por sesión
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(self.log_file_path, mode='a')
            file_handler.setLevel(logging.DEBUG) # Guardar SIEMPRE todo en el fichero
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)

        self.logger.info("Initializing Workflow...")
        self.logger.info("Setting up Workflow folders and paths...")
        set_paths(self.config)

        # Build or load metadata
        if "metadata_schema" in self.config.keys():
            self.logger.info("Loading custom metadata schema...")
            with open(self.config['metadata_schema'], "r") as f:
                schema = dict(yaml.safe_load(f))
            self.metadata = MetadataManager(schema, logger=self.logger)
        else: 
            self.logger.info("Loading default metadata schema...")
            self.metadata = MetadataManager(logger=self.logger)

        # Setup trackers
        self.tracker = setup_tracker(self.config)
        self.catalog = set_catalog(self.config)
        
        self.logger.info("Workflow initialization completed successfully.")

    def _load_config(self):
        
        if '.json' in self.config_file:
            with open (Path(self.config_file), "r") as file:
                self.config = json.load(file)
        elif '.yaml' in self.config_file or '.yml' in self.config_file:
            with open(Path(self.config_file), 'r') as file:
                self.config = dict(yaml.safe_load(file))
        else:
            self.logger.error(f"Unsupported configuration file format: {self.config_file}")

    def resume(self):

        job_name = self.config['job_name']
        self.logger.info(f"Resuming workflow job: '{job_name}'")
        
        if "artifacts.folder" not in self.config.keys():
            self.config['artifacts.folder'] = str(Path(self.config['data.folder']) / "artifacts")
            
        metadata_file_path = self.config['artifacts.folder'] + f"/metadata_{job_name}.json"
        self.logger.debug(f"Reading metadata from: {metadata_file_path}")

        try:
            with open(metadata_file_path, "r") as file:
                metadata = json.loads(file.read(), object_pairs_hook=OrderedDict)
            # Update the Pipeline Config variables with the Flow Variables
            self.metadata.load_workflow_data(metadata)
            self.logger.info("Workflow metadata successfully loaded for resume.")
        except FileNotFoundError:
            self.logger.error(f"Metadata file not found: {metadata_file_path}. Cannot resume job.")
            raise

    def get_step(self, path=None):
        if path:
            return self.metadata.get_step_data(path)
        else:
            return self.metadata.get_step_data()
            
    def update_step(self, data, path=None):
        if path:
            return self.metadata.update_step_data(data, path)
        else:
            return self.metadata.update_step_data(data)

    def get_logs(self):
        """Reads and returns all logs from the configured log file."""
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'r') as f:
                return f.read()
        return "No logs found or file does not exist yet."

    def clear_logs(self):
        """Clears the contents of the log file."""
        if os.path.exists(self.log_file_path):
            open(self.log_file_path, 'w').close()
            self.logger.info("Log file has been cleared.")

    def import_metadata(self, stage_name):

        from .schemas import default, schema_from_yaml_file
        from .widgets.update_form import MetadataForm
        from .widgets.create_form import WorkflowMetadataEditor
        
        self.logger.info(f"Importing metadata for stage: '{stage_name}'")
        
        if "metadata.folder" in self.config:
            base_folder = Path(self.config["metadata.folder"])
            base_folder.mkdir(parents=True, exist_ok=True)
            stage_file_path = base_folder / stage_name
        else:
            stage_file_path = Path(stage_name)

        yaml_path = f"{stage_file_path}.yaml"
        if not os.path.exists(yaml_path) and os.path.exists(f"{stage_file_path}.yml"):
            yaml_path = f"{stage_file_path}.yml"
        
        file_exists = os.path.exists(yaml_path)
        stage = stage_name.split("_", 2)[-1] 

        # 2. INTENTO DE USAR METADATAFORM (SI EXISTE FICHERO Y SCHEMA)
        if file_exists:
            self.logger.debug(f"Found existing YAML metadata file at {yaml_path}")
            with open(yaml_path, 'r') as file:
                new_data = dict(yaml.safe_load(file))

            self.logger.info(f"Updating metadata for stage '{stage}' in memory.")
            self.metadata.update_step_data(new_data, ["metadata"])

            # Intentar obtener el schema de la metadata del workflow
            current_schema = None
            if self.metadata.schema:
                current_schema = self.metadata._get_schema_for_path(["metadata", stage])
            else:
                if os.path.exists(f"{stage_file_path}.yml") or os.path.exists(f"{stage_file_path}.yaml"):
                    self.logger.debug("Building schema dynamically from YAML file.")
                    current_schema = schema_from_yaml_file(stage, f"{stage_file_path}.yaml")
                else:
                    self.logger.debug("Falling back to default metadata schema.")
                    current_schema = default.schema['properties']['metadata']['properties'][stage] 
            
            def handle_schema_submit(data):
                if data:
                    self.logger.info(f"Syncing updated schema data to memory for stage '{stage}'...")
                    self.metadata.update_step_data({stage: data}, ["metadata"])
            
            form = MetadataForm(current_schema, str(yaml_path), on_submit_callback=handle_schema_submit)
            form.display()
            return

        # 3. FALLBACK: USAR WORKFLOW METADATA EDITOR (Nuevo / Catálogo)
        self.logger.info(f"✨ Using Catalog-Based Editor for stage '{stage}'.")
        mapping = self.metadata.func_process_path_mapping
        
        def handle_catalog_submit(new_data):
            if new_data:
                self.metadata.update_step_data(new_data, ["metadata"])
                self.logger.info(f"Catalog metadata successfully updated for '{stage_name}'.")

        w = WorkflowMetadataEditor(stage_name, str(yaml_path), mapping, self.catalog, on_save_callback=handle_catalog_submit)
        display(w) # Assuming 'display' is defined globally (e.g. in a Jupyter context)

    def save_metadata(self):
        file_name = f"metadata_{self.config['job_name']}.json"
        save_path = os.path.join(self.config["artifacts.folder"], file_name)
        
        with open(save_path, "w") as file:
            json.dump(self.metadata.to_json(), file, indent=4)
            
        self.logger.info(f"Successfully saved workflow metadata to: {save_path}")

    def save_data(self, data, outputFilename, format="csv"):
        """
        data: pd.DataFrame to save
        outputFilename: name of the file
        """
        file_path = os.path.join(self.config["data.folder"], outputFilename)
        
        if Path(file_path).exists():
            self.logger.info(f"File '{file_path}' already exists — skipping save (delete data/ folder to rerun from scratch)")
        
        else:
            if format == "csv":
                data.to_csv(file_path, index=False)
                self.logger.info(f"Data saved successfully as CSV to: {file_path}")
            elif format == "h5":
                for key in data: 
                    pd.DataFrame(data[key]).to_hdf(file_path, key=key)
                self.logger.info(f"Data saved successfully as HDF5 to: {file_path}")
            else:
                self.logger.error(f"Unsupported save format: '{format}'")

    def load_data(self, outputFilename, format="csv", **kwargs):
        file_path = os.path.join(self.config["data.folder"], outputFilename)
        self.logger.info(f"Loading data from '{file_path}' (Format: {format})")
        
        data = pd.DataFrame()
        try:
            if format == "csv":
                data = pd.read_csv(file_path)
            elif format.lower() == "h5":
                key = kwargs.get("key")
                if key is not None:
                    store = pd.HDFStore(file_path, mode='r')
                    data = store.select(key=key)
                    store.close()
            self.logger.info(f"Successfully loaded data shape: {data.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}. Error: {str(e)}")
            
        return data
    
def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

def run(function, workflow: Workflow, *args, **kwargs):
    """ Set up environment and convert flow_variables to a FlowVariables instance """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        filename = inspect.getfile(function)      
        function_name = function.__name__
        
        wf.metadata.set_current_step(function_name)
        wf.logger.info(f"▶ Executing Workflow Step: '{function_name}'")

        wf.metadata.update_step_data({
            "path_to_script": inspect.getabsfile(function),
        })

        try:
            result = function(*args, **kwargs)
            wf.logger.debug(f"Step '{function_name}' executed successfully.")
            return result
        except Exception as e:
            wf.logger.error(f"Error during execution of step '{function_name}': {str(e)}")
            raise

    if isinstance(workflow, Workflow):
        wf = workflow
    else:
        logger.critical(f"Passed argument is not a Workflow instance. Type: {type(workflow)}")
        raise TypeError(f"workflow is not a Workflow Class\n{workflow}")

    return wrapper(wf, *args, **kwargs)

def node(func):
    """ Node decorator """
    return partial(run, func)