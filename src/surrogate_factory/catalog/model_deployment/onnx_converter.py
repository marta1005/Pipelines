import os
import joblib
import numpy as np
import pandas as pd

# Scikit-learn imports for creating dummy models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# skl2onnx imports for conversion
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType



def convert_sklearn_to_onnx(
    preprocessor_path: str,
    model_path: str,
    onnx_output_path: str,
    integer_categorical_features: list = None
):
    """
    Loads a scikit-learn ColumnTransformer and a model using joblib, combines
    them, infers feature types (handling integer categoricals), and converts
    the pipeline to an ONNX file.

    Args:
        preprocessor_path (str): Path to the joblib file for the ColumnTransformer.
        model_path (str): Path to the joblib file for the MLPClassifier model.
        onnx_output_path (str): The path where the final .onnx file will be saved.
        integer_categorical_features (list, optional): A list of column names or
            indices that are categorical but have an integer dtype. Defaults to None.
    """
    print("\nStarting ONNX conversion process...")

    # --- 1. Load the Models using joblib ---
    print(f"Loading preprocessor from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # --- 2. Create a Full Pipeline ---
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', model)])
    print("Combined preprocessor and model into a single scikit-learn pipeline.")

    # --- 3. Define Initial Types for the ONNX Model ---
    print("Defining initial types for ONNX conversion...")
    initial_types = []
    if integer_categorical_features is None:
        integer_categorical_features = []
    
    first_transformer_features = preprocessor.transformers_[0][2]
    if first_transformer_features is None:
         raise ValueError("The first transformer in the ColumnTransformer has no features.")
    using_named_columns = isinstance(first_transformer_features[0], str)
    print(f"Detected feature type: {'Named Columns (string)' if using_named_columns else 'Column Indices (int)'}")

    if using_named_columns:
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                for feature_name in features:
                    initial_types.append((feature_name, FloatTensorType([None, 1])))
            elif name == 'cat':
                for feature_name in features:
                    # MODIFIED: Check if the categorical feature is integer-based
                    if feature_name in integer_categorical_features:
                        print(f"  - Assigning INT64 type to categorical feature: {feature_name}")
                        initial_types.append((feature_name, Int64TensorType([None, 1])))
                    else:
                        print(f"  - Assigning STRING type to categorical feature: {feature_name}")
                        initial_types.append((feature_name, StringTensorType([None, 1])))
            elif name == 'remainder' and transformer != 'drop':
                print(f"Handling remainder columns: {features}. Assuming they are floats.")
                for feature_name in features:
                     initial_types.append((feature_name, FloatTensorType([None, 1])))
    else: # Fallback for index-based columns
        feature_map = {}
        for name, _, features in preprocessor.transformers_:
            if name == 'num':
                for i in features: feature_map[i] = FloatTensorType([None, 1])
            elif name == 'cat':
                for i in features:
                    # MODIFIED: Check if the categorical feature is integer-based
                    if i in integer_categorical_features:
                        feature_map[i] = Int64TensorType([None, 1])
                    else:
                        feature_map[i] = StringTensorType([None, 1])
        
        total_features = max(feature_map.keys()) + 1 if feature_map else 0
        for i in range(total_features):
            input_type = feature_map.get(i, FloatTensorType([None, 1]))
            initial_types.append((f'input_{i}', input_type))

    if not initial_types:
        raise RuntimeError("Failed to generate initial_types for the ONNX model.")

    print("\nFinal initial types for ONNX conversion:")
    for name, type in initial_types:
        print(f"  - {name}: {type}")

    # --- 4. Convert to ONNX ---
    print("Converting the pipeline to ONNX format...")
    try:
        onnx_model = convert_sklearn(
            full_pipeline,
            initial_types=initial_types,
            target_opset=12
        )

        # --- 5. Save the ONNX Model ---
        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"\nSuccessfully converted and saved the model to: {onnx_output_path}")

    except Exception as e:
        print(f"\nAn error occurred during ONNX conversion: {e}")
        print("Please ensure your scikit-learn, skl2onnx, and onnxruntime versions are compatible.")




import onnx
from onnx import helper, checker, compose

def merge_onnx_models(model_paths, output_path):
    """
    Merges multiple ONNX models into a single graph.
    
    Automatically detects the union of all inputs:
    - If multiple models share an input (same name), a single global input is created
      that feeds all of them (assuming type/dimension compatibility).
    - If a model has unique inputs, they are also added to the global inputs.
    
    Args:
        model_paths (list): List of strings with paths to .onnx files.
        output_path (str): Path where the merged model will be saved.
    """
    print(f"--- Starting merger of {len(model_paths)} models ---")
    
    loaded_models = []
    
    # Dictionary to track unique global inputs: {name: ValueInfoProto}
    # Using a dict to maintain insertion order and avoid duplicates.
    global_inputs_map = {}

    # 1. Load models and collect all possible inputs
    for path in model_paths:
        model = onnx.load(path)
        checker.check_model(model)
        loaded_models.append(model)
        
        # Identify inputs that are NOT initializers (fixed weights).
        # In ONNX, graph.input sometimes includes initializers.
        # We only want actual data inputs.
        initializer_names = set(i.name for i in model.graph.initializer)
        
        for inp in model.graph.input:
            if inp.name not in initializer_names:
                # If this is the first time we see this input, save it as global
                if inp.name not in global_inputs_map:
                    global_inputs_map[inp.name] = inp
                else:
                    # Optional: Type compatibility check could be done here
                    pass

    global_inputs_list = list(global_inputs_map.values())
    print(f"Global inputs detected ({len(global_inputs_list)}): {list(global_inputs_map.keys())}")

    # Containers for the merger
    all_nodes = []
    all_initializers = []
    all_value_infos = []
    all_outputs = []
    
    # 2. Process and merge each model
    for i, model in enumerate(loaded_models):
        prefix = f"net{i}_"
        print(f"Processing model {i+1}/{len(loaded_models)} with prefix '{prefix}'...")
        
        # Save the ORIGINAL input names of this model before renaming.
        # We are only interested in those identified as global inputs (not weights).
        original_input_names = [inp.name for inp in model.graph.input if inp.name in global_inputs_map]
        
        # Rename EVERYTHING inside the model to avoid collisions.
        # This prevents name clashes if two models have internal nodes with the same name.
        model_with_prefix = compose.add_prefix(
            model, 
            prefix=prefix, 
            rename_inputs=True, 
            rename_outputs=True, 
            rename_initializers=True
        )
        
        graph = model_with_prefix.graph
        
        # --- IMPORTANT: Generate Bridge Nodes FIRST ---
        # We must add Identity nodes before the rest of the graph
        # to comply with topological order (Producer before Consumer).
        
        bridge_nodes = []
        for original_name in original_input_names:
            prefixed_name = prefix + original_name
            
            # Create an Identity node that connects:
            # Global Input (original_name) -> Model Local Input (prefixed_name)
            bridge_node = helper.make_node(
                'Identity',
                inputs=[original_name], 
                outputs=[prefixed_name],
                name=f"Bridge_{prefix}_{original_name}"
            )
            bridge_nodes.append(bridge_node)
            
        # Append bridge nodes first
        all_nodes.extend(bridge_nodes)
        
        # AND THEN the model nodes (which consume the output of the bridges)
        all_nodes.extend(graph.node)
        
        # Append the rest of the components
        all_initializers.extend(graph.initializer)
        all_value_infos.extend(graph.value_info)
        all_outputs.extend(graph.output)

    # 4. Create the Merged Graph
    # Use opset imports from the first model (assuming compatibility)
    opset_imports = loaded_models[0].opset_import
    
    merged_graph = helper.make_graph(
        nodes=all_nodes,
        name="Merged_Universal_Model",
        inputs=global_inputs_list,    # Union of all detected inputs
        outputs=all_outputs,          # All outputs from all models
        initializer=all_initializers,
        value_info=all_value_infos
    )

    # 5. Create and save the final model
    merged_model = helper.make_model(
        merged_graph, 
        producer_name="ONNX_Merger_Universal", 
        opset_imports=opset_imports
    )
    
    try:
        checker.check_model(merged_model)
        onnx.save(merged_model, output_path)
        print(f"Success! Model saved at: {output_path}")
        print(f"Total outputs: {len(merged_model.graph.output)}")
    except onnx.checker.ValidationError as e:
        print("The merged model has validation errors:")
        print(e)