

task = {
      "type": "object",
      "description": "A standard structure for a task with defined inputs, outputs, and parameters.",
      "properties": {
        "inputs": {
          "type": "array",
          "title": "Inputs",
          "description": "Names for the inputs this task consumes (e.g., 'raw_data.csv'). Enter each on a new line.",
          "items": { "type": "string" }
        },
        "output": {
          "type": "array",
          "title": "Outputs",
          "description": "Names for the outputs this task produces (e.g., 'cleaned_data.csv'). Enter each on a new line.",
          "items": { "type": "string" }
        },
        "params": {
          "type": "array",
          "title": "Parameters",
          "description": "Key-value parameters for configuring this task.",
          "items": {
            "type": "object",
            "title": "Parameter",
            "required": ["name", "value"],
            "properties": {
              "name": {
                "type": "string",
                "title": "Parameter Name",
                "description": "Name of the parameter (e.g., 'learning_rate')."
              },
              "value": {
                "type": "string",
                "title": "Parameter Value",
                "description": "Value of the parameter (e.g., '0.01'). Use JSON for complex values."
              }
            }
          }
        }
      },
      "required": ["inputs", "output", "params"]
    }
  
step_tasks= {
      "type": "array",
      "title": "Associated Tasks",
      "description": "A list of custom tasks performed in this step, each with defined inputs and outputs.",
      "items": {
        "type": "object",
        "title": "Custom Task",
        "required": ["inputs", "outputs"],
        "properties": {
          "task_name": {
            "type": "string",
            "title": "Task Name",
            "description": "A descriptive name for this specific task."
          },
          "inputs": {
            "type": "array",
            "title": "Inputs",
            "description": "Enter each input on a new line.",
            "items": { "type": "string" }
          },
          "outputs": {
            "type": "array",
            "title": "Outputs",
            "description": "Enter each output on a new line.",
            "items": { "type": "string" }
          },
          "notes": {
            "type": "string",
            "title": "Notes",
            "description": "Additional notes or comments about this task."
          }
        }
      }
    }


schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Process Pipeline Metadata",
  "description": "A form to input metadata for each stage of the data science pipeline.",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "title": "Metadata",
      "properties": {
        "Requirements_and_Planning": {
          "type": "object",
          "title": "Requirements_and_Planning",
          "description": "Initial phase for defining project requirements and planning.",
          "properties": {
            "Requirements": {
                  "description": "Metadata schema of a Surrogate Model - Requirements",
                  "type": "object",
                  "properties": {
                      "usage": {"type": "string"},
                      "customer": {"type": "string"},
                      "center": {"type": "string"},
                      "domain": {"type": "string"},
                      "use_case": {"type": "string"},
                      "project": {"type": "string"},
                      "aircraft": {
                          "type": "array",
                          "items":{
                              "enum":["A220", "A320", "A330", "A340", "A350", "A380", "R&T", "Others", "All"]},
                          "minItems":1,
                      },
                      "life_cycle_step": {"type": "string"},
                      "accuracy": {
                          "description": "Model accuracy metrics",
                          "type": "array",
                          "minItems": 1,
                          "uniqueItems": True,
                          "items": {
                              "type": "object",
                              "properties": {
                                  "metric": {"type": "string"},
                                  "value": {"type": "number"}
                              },
                              "required": ["metric", "value"]
                          }
                      },
                      # "other_requirements": task,
                  },
                  "required":["usage","customer", "center", "domain", "use_case", "project", "aircraft", "accuracy"],
              },
              
          
            "Planning": {
              "type": "object",
              "title": "Project Planning",
              "properties": {
                "Start date": {"type": "string"},
                "End data": {"type": "string"},
                "Workload": {"type": "string"},
                "Data Source": {"type": "string", "description": "Data generation or else (e.g. TSAS)"},
                "Data availability": {"type": "string"},
                "project": {"type": "string"},
                "Link to GitHub repository": {
                  "type": "string",
                  # "title": "Link to GitHub repository",
                  "description": "Location of the code"
                }
              }
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Data_Acquisition_Generation": {
          "type": "object",
          "title": "2. Data Acquisition and Generation",
          "description": "Phase for acquiring and generating the necessary data.",
          "properties": {
            "Data_Acquisition": {
              "type": "object",
              "title": "Data Acquisition Tasks",
              "properties": {
                "batch_extract": task|{
                  
                  "title": "Batch Extract",
                  "description": "Task for extracting data from a source."
                }
              }
            },
            "Data_Generation": {
              "type": "object",
              "title": "Data Generation Tasks",
              "properties": {
                "define_doe": task|{
                  
                  "title": "Design of Experiments (DOE)",
                  "description": "Task for defining the design of experiments (DOE)."
                },
                "launch_sim": task|{
                  
                  "title": "Create/Generate Inputs (Simulation)",
                  "description": "Task for creating or generating input datasets (e.g., via simulation)."
                }
              }
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Data_Cleansing": {
          "type": "object",
          "title": "Data_Cleansing",
          "description": "Phase for cleaning and preprocessing data.",
          "properties": {
            "replace_missing_values": task|{
              
              "title": "Manage Missing Values",
              "description": "Task for handling missing data points."
            },
            "filter_outliers": task|{
              
              "title": "Manage Outliers",
              "description": "Task for identifying and managing outliers."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Data_Split": {
          "type": "object",
          "title": "4. Data Split",
          "description": "Phase for partitioning data into sets (e.g., training, testing).",
          "properties": {
            "data_split": {
              "$ref": "#/$defs/task",
              "title": "Data Partitioning",
              "description": "Task for splitting the dataset into training, validation, and test sets."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Feature_Selection": {
          "type": "object",
          "title": "5. Feature Engineering & Selection",
          "description": "Phase for selecting, creating, and transforming features.",
          "properties": {
            "Analyze_Correlations": {
              "type": "object",
              "title": "Analyze Correlations",
              "description": "Record findings from feature correlation analysis.",
              "properties": {
                "notes": {
                  "type": "string",
                  "title": "Analysis Notes",
                  "description": "Summary of findings from the correlation analysis."
                }
              }
            },
            "add_features": task|{
              
              "title": "Add/Remove Features",
              "description": "Task for adding new features (feature engineering)."
            },
            "preprocessor": task|{
              
              "title": "Preprocessor",
              "description": "Preprocessing task for feature transformation (e.g., scaling, encoding)."
            },
            "postprocessor": task|{
              
              "title": "Postprocessor",
              "description": "Post-processing task for features."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Model_Selection": {
          "type": "object",
          "title": "6. Model Selection",
          "description": "Phase for defining the model architecture.",
          "properties": {
            "define_model": task|{
              
              "title": "Define Model",
              "description": "Task for defining the model to be trained."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Model_Training": {
          "type": "object",
          "title": "7. Model Training",
          "description": "Phase for training the selected model.",
          "properties": {
            "train": task|{
              
              "title": "Train Model",
              "description": "Task for executing the model training process."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Model_Deployment": {
          "type": "object",
          "title": "8. Model Deployment",
          "description": "Phase for deploying the trained model.",
          "properties": {
            "model_deployment": task|{
              
              "title": "Model Deployment Wrapper",
              "description": "Task for wrapping the model for deployment."
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Model_Validation": {
          "type": "object",
          "title": "9. Model Validation",
          "description": "Phase for validating the deployed model's performance.",
          "properties": {
            "Model_Predictions": task|{
              
              "title": "Generate Predictions",
              "description": "Task to generate predictions using the trained model."
            },
            "Model_Wrapper_Integrity_test": {
              "type": "object",
              "title": "Wrapper Integrity Test",
              "description": "Record the results of the model wrapper integrity test.",
              "properties": {
                "test_passed": { "type": "boolean", "title": "Test Passed" },
                "notes": { "type": "string", "title": "Test Notes" }
              }
            },
            "calculate_metrics": task|{
              
              "title": "Calculate Model Metrics",
              "description": "Task to calculate performance metrics (e.g., accuracy, MSE)."
            },
            "validate": {
              "type": "object",
              "title": "Final Model Validation Sign-off",
              "properties": {
                "is_validated": { "type": "boolean", "title": "Model Approved for Deployment" },
                "validation_summary": { "type": "string", "title": "Validation Summary" }
              }
            },
            # "Associated_Tasks": step_tasks
          }
        },
        "Model_Storage": {
          "type": "object",
          "title": "10. Model Storage",
          "description": "Phase for storing the final model and associated artifacts.",
          "properties": {
            "Store": {
              "type": "object",
              "title": "Store Model Artifact",
              "properties": {
                "storage_path": {
                  "type": "string",
                  "title": "Storage Path/URI",
                  "description": "The location where the final model artifact is stored."
                }
              }
            },
            # "Associated_Tasks": step_tasks
          }
        }
      }
    }
  },
  
}
