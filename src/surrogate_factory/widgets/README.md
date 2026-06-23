# Workflow Metadata Editor Widget
This repository contains an interactive Jupyter Notebook widget designed to facilitate the configuration of metadata for various stages of a machine learning workflow.

The system automatically switches between two editing modes depending on the context:
1. Catalog-Based Editor (Create Form): Used when configuring a stage for the first time or when no strict JSON schema is enforced. It allows selecting methods from a catalog and defining parameters flexibly.
2. Schema-Based Editor (Update Form): Used when an existing configuration file (.yaml) and a formal JSON Schema are detected. It enforces structure and validation rules defined in the schema.


## Usage
The primary entry point is the workflow.import_metadata(stage_name) method. This function handles the logic to determine which editor to display by importing the appropriate widget.
### Example usage within a Jupyter Notebook cell
workflow.import_metadata(stage_name="SF_6_Model_Selection")



## Internal Logic
When import_metadata is called, it checks for the existence of a configuration file and executes one of the following imports:
from .widgets.update_form import MetadataForm
from .widgets.create_form import WorkflowMetadataEditor


## Case 1: Catalog-Based Editor (create_form)
This editor (instantiated via WorkflowMetadataEditor) is displayed when:
* No YAML file exists for the stage.
* Or no formal JSON Schema is associated with the stage in workflow.metadata.

### Features
* Dynamic Structure: Sections are generated based on your workflow mapping configuration.
* Catalog Integration:
    * Catalog Mode: Select pre-defined methods from your system's catalog. The widget automatically loads templates (if available) or generates form fields based on the method's arguments.
    * Show All: A checkbox allows you to access methods from any category in the catalog, not just the current stage's category.
    * Custom Mode: Manually define a method name and its parameters if it's not in the catalog.
* Smart Types:
    * Lists: Automatically rendered as multi-line text areas. Enter one item per line or use JSON format for complex lists.
    * Dictionaries: Rendered as recursive, collapsible sections.
* Persistence: Saves data to the YAML file and immediately updates the in-memory workflow.metadata object.

### Example
![Create Form widget](../../docs/images/create_form.png)


## Case 2: Schema-Based Editor (update_form)
This editor (instantiated via MetadataForm) is strictly enforcing validation and is displayed when:
* A YAML file already exists for the stage.
* AND a valid JSON Schema can be retrieved from workflow.metadata for that stage.

### Features
* Strict Validation: Input fields are generated exactly as defined in the JSON Schema (e.g., specific enums, data types, required fields).
* Complex Types:
    * Arrays of Objects: Provides "Add Item" and "Remove" buttons to manage lists of complex configurations dynamically.
    * Enums: Renders as dropdowns or multi-select widgets.
* Validation: Prevents saving if the data does not strictly conform to the schema.

### Example
![Update Form Widget](../../docs/images/update_form.png)

