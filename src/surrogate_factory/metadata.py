"""
 Copyright (c) 2025 Airbus Operations S. L. This file is part of project Surrogate Factory released under the Airbus Inner Source shared-maintenance
 """

import json
import jsonschema
import copy
import logging
from functools import reduce # For deep dictionary navigation
import operator # For deep dictionary navigation
from collections import OrderedDict


class MetadataManager:
    """
    Manages workflow data, optionally based on a provided JSON schema.

    Allows retrieving schema information for specific steps, validating data
    for those steps (if a schema is provided), updating the workflow data,
    and retrieving current data. Steps are identified by their path within
    the data structure (e.g., ['metadata', 'Requirements']).
    """

    # Mapping from function/process names to their corresponding path in the workflow data
    func_process_path_mapping = {
        "define_doe": ["metadata","Data_Acquisition_Generation", "Data_Generation", "Create_Design_of_Experiments"],
        "launch_sim": ["metadata","Data_Acquisition_Generation", "Data_Generation", "Generate_Data"],
        "batch_extract":['metadata', 'Data_Acquisition_Generation','Data_Acquisition','Extract'],
        "batch_transform":['metadata', 'Data_Acquisition_Generation','Data_Acquisition','Transform'],
        "batch_load":['metadata', 'Data_Acquisition_Generation','Data_Acquisition','Load'],
        "replace_missing_values":['metadata', 'Data_Cleansing', 'Managing_Missing_Values'],
        "filter_outliers":['metadata', 'Data_Cleansing', 'Managing_Outliers'],
        "data_split": ["metadata", "Data_Partition"],
        "explore_data_analysis": ['metadata', 'Feature_Selection', 'Add_Remove_Features'],
        "add_features": ['metadata', 'Feature_Selection', 'Add_Remove_Features'],
        "preprocessor": ['metadata', 'Feature_Selection', 'Preprocess'],
        "postprocessor": ['metadata', 'Feature_Selection', 'Postprocess'],
        "define_model": ['metadata','Model_Selection'],
        "train": ['metadata', 'Model_Training'],
        "model_deployment": ['metadata', 'Model_Deployment'],
        "predict":['metadata', 'Model_Validation'],
        "calculate_metrics":['metadata', 'Model_Validation'],
        "validate":['metadata', 'Model_Validation'],
    }


    def __init__(self, schema=None, logger=None):
        """
        Initializes the WorkflowManager.

        Args:
            schema (dict, optional): The JSON schema defining the workflow structure
                                     and validation rules. Defaults to None, in which
                                     case no validation is performed.
            logger (logging.Logger, optional): Global logger instance to use.
        """
        self.logger = logger or logging.getLogger(__name__)

        if schema is not None and not isinstance(schema, dict):
            self.logger.error("Schema must be a dictionary or None.")
            raise TypeError("Schema must be a dictionary or None.")
            
        self.schema = schema
        self.workflow_data = OrderedDict()  # Holds the actual workflow instance data
 
        self.current_step = []   # Path to the current step in the workflow data
        # Pre-resolve references for easier navigation (optional but can improve performance)
        # self.resolver = jsonschema.RefResolver.from_schema(schema) if schema else None

    
    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def set_current_step(self, step_func):
        """
        Sets the current step based on a predefined mapping.

        Args:
            step_func (str): A string key representing the function or process.
        """
        if step_func in self.func_process_path_mapping:
            self.current_step = self.func_process_path_mapping[step_func]
        else:
            self.logger.warning(
                f"'{step_func}' is not in the process mapping. It must be one of these: {list(self.func_process_path_mapping.keys())}"
            )
            self.current_step = [] # Reset or handle as an error as appropriate

    def _resolve_ref(self, ref, current_schema_node):
        """
        Resolves a $ref pointer within the schema.
        Assumes self.schema is not None when called.

        Args:
            ref (str): The reference string (e.g., "#/definitions/context").
            current_schema_node (dict): The schema node where the ref was found.

        Returns:
            dict: The resolved schema definition.

        Raises:
            ValueError: If the reference format is unsupported or resolution fails.
        """
        if not self.schema: # Should not happen if called correctly
            self.logger.error("Cannot resolve reference: No schema loaded.")
            raise ValueError("Cannot resolve reference: No schema loaded.")
        if not ref.startswith('#/'):
            self.logger.error(f"Unsupported reference format: {ref}. Only '#/...' is supported.")
            raise ValueError(f"Unsupported reference format: {ref}. Only '#/...' is supported.")

        path_parts = ref[2:].split('/')
        target = self.schema # Start resolution from the root of the main schema
        try:
            for part in path_parts:
                target = target[part]
            return target
        except (KeyError, TypeError) as e:
            self.logger.error(f"Could not resolve reference '{ref}': {e}")
            raise ValueError(f"Could not resolve reference '{ref}': {e}")

    def _get_schema_for_path(self, data_path):
        """
        Navigates the schema to find the sub-schema for a given data path.
        Handles paths that may include integer indices for arrays.

        Args:
            data_path (list): A list of keys/indices representing the path.

        Returns:
            dict or bool: The JSON sub-schema. True if no schema loaded (allow any).
                          None if path is invalid in an existing schema.
        """
        if self.schema is None:
            return True

        current_schema = self.schema
        if not data_path:
            return current_schema

        try:
            for i, segment in enumerate(data_path):
                if '$ref' in current_schema:
                    current_schema = self._resolve_ref(current_schema['$ref'], current_schema)

                if not isinstance(current_schema, dict):
                    raise KeyError(f"Schema path resolves to a non-dictionary type ('{type(current_schema)}') before reaching segment '{segment}' at path {data_path[:i+1]}")

                if isinstance(segment, int): # Path segment is an array index
                    if current_schema.get("type") == "array":
                        if "items" in current_schema:
                            # If items is a single schema, use it.
                            # If items is an array of schemas (tuple validation), this basic navigation
                            # assumes the index refers to the general item schema or the first one.
                            # For simplicity, we take current_schema["items"]. A more complex schema
                            # might have different schemas per index.
                            current_schema = current_schema["items"]
                            if isinstance(current_schema, list): # Tuple validation: schema per index
                                if segment < len(current_schema):
                                    current_schema = current_schema[segment]
                                elif "additionalItems" in current_schema: # Check additionalItems if index is out of tuple bounds
                                    if isinstance(current_schema["additionalItems"], dict):
                                        current_schema = current_schema["additionalItems"]
                                    elif current_schema["additionalItems"] is False:
                                        raise KeyError(f"Index {segment} out of bounds for tuple-defined array items and additionalItems disallowed at path {data_path[:i+1]}")
                                    elif current_schema["additionalItems"] is True:
                                         return True # Allows any schema for additional items
                                else: # No additionalItems, index out of bounds
                                     raise KeyError(f"Index {segment} out of bounds for tuple-defined array items at path {data_path[:i+1]}")

                        else: # No "items" defined for array type
                            raise KeyError(f"Schema for array at path {data_path[:i]} does not define 'items'.")
                    else: # Path segment is an index, but schema isn't an array
                        raise KeyError(f"Path segment '{segment}' is an index, but schema at {data_path[:i]} is not an array (type: {current_schema.get('type')}).")
                
                else: # Path segment is a dictionary key (string)
                    if current_schema.get("type") == "object":
                        if "properties" in current_schema and segment in current_schema["properties"]:
                            current_schema = current_schema["properties"][segment]
                        elif "patternProperties" in current_schema:
                            found_pattern = False
                            import re
                            for pattern, subschema in current_schema["patternProperties"].items():
                                if re.match(pattern, str(segment)): # Ensure segment is string for re.match
                                    current_schema = subschema
                                    found_pattern = True
                                    break
                            if not found_pattern: # Not in properties, not in patternProperties
                                if "additionalProperties" in current_schema:
                                    if isinstance(current_schema["additionalProperties"], dict):
                                        current_schema = current_schema["additionalProperties"]
                                    elif current_schema["additionalProperties"] is False:
                                        raise KeyError(f"Schema disallows additional properties; '{segment}' not found at path {data_path[:i+1]}")
                                    elif current_schema["additionalProperties"] is True:
                                        return True # Allows any schema
                                else: # No additionalProperties defined
                                    raise KeyError(f"Key '{segment}' not found and no additionalProperties schema at path {data_path[:i+1]}.")
                        elif "additionalProperties" in current_schema:
                             if isinstance(current_schema["additionalProperties"], dict):
                                current_schema = current_schema["additionalProperties"]
                             elif current_schema["additionalProperties"] is False:
                                raise KeyError(f"Schema disallows additional properties; '{segment}' not found at path {data_path[:i+1]}")
                             elif current_schema["additionalProperties"] is True:
                                return True # Allows any schema
                        else: # Not in properties, no patternProperties, no additionalProperties
                            raise KeyError(f"Key '{segment}' not found in object schema properties at path {data_path[:i+1]}.")
                    else: # Path segment is a key, but schema isn't an object
                        raise KeyError(f"Path segment '{segment}' is a key, but schema at {data_path[:i]} is not an object (type: {current_schema.get('type')}).")

            if '$ref' in current_schema:
                current_schema = self._resolve_ref(current_schema['$ref'], current_schema)
            
            return current_schema

        except (KeyError, ValueError) as e:
            self.logger.warning(f"Could not retrieve sub-schema for path {data_path}: {e}")
            return None

    def get_required_info(self, data_path):
        """
        Gets information about required fields and properties for a specific step.

        Args:
            data_path (list): The path to the step in the data structure.

        Returns:
            dict or None: A dictionary containing 'required_fields' and 'properties',
                          or None if schema info cannot be retrieved (e.g., no schema loaded).
        """
        if self.schema is None:
            self.logger.warning("Cannot get_required_info: No schema loaded.")
            return {"required_fields": [], "properties": {}} # Default empty info

        sub_schema = self._get_schema_for_path(data_path)

        if sub_schema is None: # Path was invalid within the schema
            return None
        if sub_schema is True: # Schema allows anything for this path
             self.logger.warning(f"Schema for path {data_path} allows any data; no specific properties or requirements.")
             return {"required_fields": [], "properties": {}}


        if not isinstance(sub_schema, dict): # Should be a schema object (dict)
            self.logger.warning(f"Sub-schema for path {data_path} is not a dictionary: {sub_schema}")
            return {"required_fields": [], "properties": {}}


        required_fields = sub_schema.get("required", [])
        properties_info = {}
        
        # Handle properties for object type
        if sub_schema.get("type") == "object" and "properties" in sub_schema:
            for prop, details in sub_schema["properties"].items():
                if isinstance(details, dict): 
                    properties_info[prop] = {
                        "type": details.get("type", "any"),
                        "description": details.get("description", "No description"),
                        "required": prop in required_fields
                    }
                elif details is True: 
                     properties_info[prop] = {
                        "type": "any",
                        "description": "Allows any value.",
                        "required": prop in required_fields
                    }
        # For array types, info might be about 'items' schema
        elif sub_schema.get("type") == "array" and "items" in sub_schema:
            items_schema = sub_schema["items"]
            # Provide info based on the items schema, could be complex if items is a list of schemas
            # For simplicity, showing general item properties if items_schema is a dict
            if isinstance(items_schema, dict) and items_schema.get("type") == "object" and "properties" in items_schema:
                 for prop, details in items_schema["properties"].items():
                    if isinstance(details, dict):
                        properties_info[f"item.{prop}"] = { # Prefix to indicate it's an item property
                            "type": details.get("type", "any"),
                            "description": details.get("description", "No description for item property"),
                            "required": prop in items_schema.get("required", [])
                        }
            else:
                properties_info["items"] = {"type": items_schema.get("type", "any") if isinstance(items_schema, dict) else "array_items", "description": "Schema for array items."}


        return {
            "required_fields": required_fields, # Required fields of the current sub_schema object/array itself
            "properties": properties_info
        }


    def validate_step_data(self, data_path, data):
        """
        Validates provided data against the schema for a specific step.
        If no schema was provided to the manager, this always returns True.

        Args:
            data_path (list): The path to the step in the data structure.
            data (any): The data to validate (can be dict, list, primitive, etc.).

        Returns:
            bool: True if the data is valid or no schema is loaded, False otherwise.
        """
        if self.schema is None:
            return True 

        sub_schema = self._get_schema_for_path(data_path)

        if sub_schema is None:
            self.logger.warning(f"Validation skipped for path {data_path} as sub-schema could not be determined.")
            return False 
        # If sub_schema is True, it means "allow anything", so jsonschema validation will pass.

        try:
            validator = jsonschema.Draft7Validator(sub_schema)
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                for error in errors:
                    # error.path is relative to 'data', data_path is the absolute path to 'data'
                    full_error_path_parts = list(data_path) + list(error.path)
                    self.logger.warning(f"Validation Error at path {'/'.join(map(str, full_error_path_parts))}: {error.message}")
                return False
            return True
        except jsonschema.SchemaError as e: # Problem with the sub_schema itself
            self.logger.warning(f"Schema Error for path {data_path}: {e}. Check schema definition.")
            return False
        except Exception as e: # Other unexpected errors during validation
            self.logger.warning(f"An unexpected error occurred during validation for {data_path} (data: {type(data)}, schema: {type(sub_schema)}): {e}")
            return False

    def _navigate_and_get(self, data_dict, path):
        """Helper to get a value from a nested dictionary or list using a path list."""
        current = data_dict
        try:
            for segment in path:
                if isinstance(segment, int): # Array index
                    if not isinstance(current, list) or not (0 <= segment < len(current)):
                        # Path segment is index, but current is not list or index out of bounds
                        return None 
                    current = current[segment]
                else: # Dictionary key
                    if not isinstance(current, dict) or segment not in current:
                        # Path segment is key, but current is not dict or key not found
                        return None
                    current = current[segment]
            return current
        except (TypeError): # e.g. trying to index a non-list/dict or use non-int on list
            return None

    def _navigate_and_set(self, data_dict, path, value):
        """
        Helper to set a value in a nested dictionary or list, creating 
        the necessary structure (dictionaries or lists) along the path if it doesn't exist.
        """
        current_container = data_dict

        # Iterate through the path, stopping at the parent of the target
        for i in range(len(path) - 1):
            current_segment = path[i]  # e.g., k1, then k2, then k3, then arraypos
            # Determine if the *next* segment implies the current one should hold a list or dict
            next_segment_is_index = isinstance(path[i+1], int)

            if isinstance(current_segment, int):  # current_segment is an index
                # current_container must be a list to be indexed by an integer
                if not isinstance(current_container, list):
                    self.logger.warning(f"Path error during set: expected a list to access index '{current_segment}' at path segment {path[:i+1]}, but found {type(current_container)}.")
                    return False
                
                # Ensure list is long enough, pad with None if necessary
                while len(current_container) <= current_segment:
                    current_container.append(None)
                
                # If the element at this index is None, or not the correct container type for the *next* step, create/replace it
                if current_container[current_segment] is None or \
                   (next_segment_is_index and not isinstance(current_container[current_segment], list)) or \
                   (not next_segment_is_index and not isinstance(current_container[current_segment], dict)):
                    current_container[current_segment] = [] if next_segment_is_index else {}
                
                current_container = current_container[current_segment]  # Descend into the (potentially newly created) container

            else:  # current_segment is a dictionary key (string)
                # current_container must be a dictionary to be keyed by a string
                if not isinstance(current_container, dict):
                    self.logger.warning(f"Path error during set: expected a dict to access key '{current_segment}' at path segment {path[:i+1]}, but found {type(current_container)}.")
                    return False

                # If the key doesn't exist, or if it exists but its value is not the correct container type for the *next* step
                if current_segment not in current_container or \
                   (next_segment_is_index and not isinstance(current_container[current_segment], list)) or \
                   (not next_segment_is_index and not isinstance(current_container[current_segment], dict)):
                    current_container[current_segment] = [] if next_segment_is_index else {}
                
                current_container = current_container[current_segment]  # Descend

        # After the loop, current_container is the direct parent where the value needs to be set.
        # path[-1] is the final key or index within this current_container.
        final_target_segment = path[-1]

        if isinstance(final_target_segment, int):  # Final target is an element in a list
            if not isinstance(current_container, list):
                self.logger.warning(f"Path error during set: expected parent (at {path[:-1]}) to be a list for final index '{final_target_segment}', but found {type(current_container)}.")
                return False
            # Ensure list is long enough for the final index
            while len(current_container) <= final_target_segment:
                current_container.append(None)
            current_container[final_target_segment] = value
        else:  # Final target is a property in a dictionary
            if not isinstance(current_container, dict):
                self.logger.warning(f"Path error during set: expected parent (at {path[:-1]}) to be a dict for final key '{final_target_segment}', but found {type(current_container)}.")
                return False
            current_container[final_target_segment] = value
        
        return True


    def update_step_data(self, data, path=None):
        """
        Validates (if schema is present) and updates the workflow data for the current_step.

        Args:
            data (dict): The data dictionary to insert/update at self.current_step.
            path (list): By default None, if provided, it will update the path provided

        Returns:
            bool: True if the update was successful, False otherwise.
        """

        if path == None:
            path = self.current_step
        
        current_data = self.get_step_data(path)
        if current_data:
 
            for key in current_data.keys():
                if key not in data:
                    data[key]=current_data[key]

            # data.update(current_data)

    
        # data is the actual data for the current_step, not the whole workflow_data
        if self.validate_step_data(path, data): # Validation now respects optional schema
            data_to_insert = copy.deepcopy(data)
            if not path: # Updating the root object
                self.workflow_data = data_to_insert
                return True
            else:
                if self._navigate_and_set(self.workflow_data, path, data_to_insert):
                    return True
                else:
                    self.logger.warning(f"Failed to set data at path {path}.")
                    return False
        else:
            # validate_step_data would have issued warnings
            self.logger.warning(f"Validation failed for data at path {path}. Data not updated.")
            return False

    def get_step_data(self, path =None):
        """
        Retrieves the current data associated with self.current_step.

        Returns:
            dict or None: The data dictionary found at the path, or None if not found.
        """

        if path == None:
            path = self.current_step
    
        if not path:
            return copy.deepcopy(self.workflow_data) # Return all data for root path
        
        data = self._navigate_and_get(self.workflow_data, path)
        return copy.deepcopy(data) if data is not None else None

    def to_json(self):
        """Returns a deep copy of the entire current workflow data dictionary."""
        return copy.deepcopy(self.workflow_data)

    def load_workflow_data(self, data_to_load):
        """
        Loads an existing workflow data dictionary.
        If a schema and current_step are provided, it validates only the data
        at current_step before loading the entire data_to_load.
        If no schema is provided, it loads data without validation.
        If a schema is provided but no current_step, it validates the entire data_to_load.

        Args:
            data_to_load (dict): The workflow data to load.

        Returns:
            bool: True if data is loaded (and validated if applicable), False otherwise.
        """
        if not isinstance(data_to_load, dict):
            self.logger.warning("Loading failed. Provided data is not a dictionary.")
            return False

        validation_passed = False
        if self.schema is None:
            self.logger.warning("Loading data without validation as no schema was provided.")
            validation_passed = True
        elif not self.current_step: # Schema provided, no current_step, so validate all
            self.logger.warning("Validating entire provided data against the root schema.")
            # Temporarily set current_step to empty for validate_step_data to use full schema
            # Or, more directly:
            validator = jsonschema.Draft7Validator(self.schema)
            errors = sorted(validator.iter_errors(data_to_load), key=lambda e: e.path)
            if errors:
                self.logger.warning("Loading failed. Provided data does not conform to the root schema:")
                for error in errors:
                    self.logger.warning(f"  Error at path {'/'.join(map(str, error.path))}: {error.message}")
                return False # Explicit return on failure
            validation_passed = True
        else: # Schema and current_step are provided, validate only that part
            self.logger.warning(f"Validating provided data at current step: {self.current_step}")
            sub_data_to_validate = self._navigate_and_get(data_to_load, self.current_step)
            
            # It's possible sub_data_to_validate is None if path doesn't exist in data_to_load.
            # validate_step_data will handle this based on the sub-schema (e.g. if None is allowed or field is not required)
            if self.validate_step_data(self.current_step, sub_data_to_validate):
                validation_passed = True
            else:
                self.logger.warning(f"Loading failed. Data at step {self.current_step} is not valid.")
                return False # Explicit return on failure

        if validation_passed:
            self.workflow_data = copy.deepcopy(data_to_load)
            self.logger.info("Workflow data loaded successfully.")
            return True
        else:
            # This path should ideally not be reached if logic above returns False on failure
            self.logger.warning("Loading failed due to validation errors (unexpected state).")
            return False