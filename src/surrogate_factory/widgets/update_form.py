import ipywidgets as widgets
from IPython.display import display, clear_output
import yaml
import jsonschema
import os
import uuid

class MetadataForm:
    """
    A class to generate, display, and manage an ipywidgets form based on a JSON schema.

    This class encapsulates the logic for:
    - Building a widget form recursively from a schema.
    - Populating the form with existing data from a YAML file.
    - Handling dynamic lists (arrays of objects) with add/remove functionality.
    - Collecting data from the widgets.
    - Validating the collected data against the schema.
    - Saving the validated data back to the YAML file.
    """

    def __init__(self, schema, yaml_filepath, on_submit_callback=None):
        """
        Initializes the SchemaForm instance.

        Args:
            schema (dict): The JSON schema defining the form structure.
            yaml_filepath (str): The path to the YAML file for loading and saving data.
        """
        self.schema = schema
        self.yaml_filepath = yaml_filepath
        self.on_submit_callback = on_submit_callback
        self.widget_refs = {}  # Holds references to all generated widgets by their data path.
        
        # The main container widget for the entire form UI.
        self.form_container = self._create_form_ui()
        self.collected_data = {}

    def display(self):
        """
        Displays the generated form widget in the notebook.
        """
        display(self.form_container)

    # --- Private Helper Methods ---

    @staticmethod
    def _get_widget_value(widget):
        """
        Centralizes getting data from different widget types.
        This is a static method as it doesn't rely on instance state.
        """
        if isinstance(widget, (widgets.Text, widgets.Textarea, widgets.Dropdown, widgets.Password)):
            return widget.value
        elif isinstance(widget, (widgets.IntText, widgets.FloatText, widgets.BoundedIntText, widgets.BoundedFloatText)):
            return widget.value
        elif isinstance(widget, widgets.Checkbox):
            return widget.value
        elif isinstance(widget, widgets.SelectMultiple):
            return list(widget.value)
        elif isinstance(widget, (widgets.VBox, widgets.HBox, widgets.Accordion)):
            # Containers don't have a single value; data comes from children.
            return None
        else:
            # Fallback for unknown types
            try:
                return widget.value
            except AttributeError:
                return None

    def _build_widget_from_schema(self, prop_name, prop_schema, current_data, path_prefix, required_fields):
        """
        Recursively builds ipywidgets based on a schema fragment. This is the core
        widget-generating engine.
        """
        widget = None
        prop_type = prop_schema.get("type")
        description = prop_schema.get("description", "")
        title = prop_schema.get("title", prop_name.replace('_', ' ').title())
        is_required = prop_name in required_fields
        label = f"{title}{' *' if is_required else ''}"
        full_path = f"{path_prefix}.{prop_name}" if path_prefix else prop_name

        # --- Object Type ---
        if prop_type == "object" and "properties" in prop_schema:
            content_box = widgets.VBox(layout=widgets.Layout(padding='10px'))
            obj_widgets = []
            nested_required = prop_schema.get("required", [])

            for sub_prop_name, sub_prop_schema in prop_schema["properties"].items():
                sub_data = current_data.get(sub_prop_name) if isinstance(current_data, dict) else None
                child_widget = self._build_widget_from_schema(
                    sub_prop_name, sub_prop_schema, sub_data, full_path, nested_required
                )
                if child_widget:
                    obj_widgets.append(child_widget)

            if obj_widgets:
                content_box.children = obj_widgets
                accordion = widgets.Accordion(
                    children=[content_box],
                    layout=widgets.Layout(border='solid 1px #e0e0e0', margin='10px 0')
                )
                accordion.set_title(0, label)
                accordion.selected_index = None  # Start collapsed
                widget = accordion
                self.widget_refs[full_path] = {'widget': content_box, 'schema': prop_schema, 'type': 'object'}

        # --- Array Type ---
        elif prop_type == "array" and "items" in prop_schema:
            items_schema = prop_schema.get("items")

            # Case 1: Array of simple enums -> MultiSelect
            if items_schema.get("type") == "string" and "enum" in items_schema:
                options = sorted(list(set(items_schema.get("enum", []))))
                default_value = list(current_data) if isinstance(current_data, list) else []
                enum_widget = widgets.SelectMultiple(
                    options=options, value=default_value, description=label, tooltip=description,
                    layout=widgets.Layout(width='auto', margin='5px 0'), style={'description_width': 'initial'}
                )
                widget = enum_widget
                self.widget_refs[full_path] = {'widget': widget, 'schema': prop_schema, 'type': 'array_enum'}

            # Case 2: Array of objects -> Dynamic Add/Remove List
            elif items_schema.get("type") == "object":
                array_items_box = widgets.VBox([])
                add_button = widgets.Button(description=f"Add {items_schema.get('title', 'Item')}", button_style='info', icon='plus')
                array_section = widgets.VBox(
                    [array_items_box, add_button],
                    layout=widgets.Layout(border='solid 1px #e0e0e0', padding='10px', margin='10px 0')
                )

                def _add_array_item_row(item_data, index):
                    item_widgets_box = widgets.VBox([])
                    item_required = items_schema.get("required", [])
                    item_path_prefix = f"{full_path}.{index}"
                    for sub_prop_name, sub_prop_schema in items_schema.get("properties", {}).items():
                        sub_data = item_data.get(sub_prop_name) if isinstance(item_data, dict) else None
                        child_widget = self._build_widget_from_schema(
                            sub_prop_name, sub_prop_schema, sub_data, item_path_prefix, item_required
                        )
                        if child_widget:
                            item_widgets_box.children += (child_widget,)
                    
                    remove_button = widgets.Button(description="Remove", button_style='danger', icon='trash', layout=widgets.Layout(width='100px', margin='5px 0 5px auto'))
                    item_row = widgets.HBox([item_widgets_box, remove_button], layout=widgets.Layout(border='solid 1px #f0f0f0', margin='5px 0', padding='5px'))
                    item_widgets_box.layout.flex = '1'

                    def _on_remove_click(b):
                        paths_to_remove = [p for p in self.widget_refs if p.startswith(item_path_prefix + '.')]
                        for p in paths_to_remove:
                            del self.widget_refs[p]
                        current_children = list(array_items_box.children)
                        current_children.remove(item_row)
                        array_items_box.children = tuple(current_children)
                    
                    remove_button.on_click(_on_remove_click)
                    return item_row

                # Populate initial items
                min_items = prop_schema.get("minItems", 0)
                data_source = current_data if isinstance(current_data, list) else []
                num_initial_items = max(min_items, len(data_source))
                for i in range(num_initial_items):
                    item_data = data_source[i] if i < len(data_source) else None
                    new_row = _add_array_item_row(item_data, i)
                    array_items_box.children += (new_row,)

                def _on_add_click(b):
                    new_index = len(array_items_box.children)
                    new_row = _add_array_item_row(None, new_index)
                    array_items_box.children += (new_row,)
                
                add_button.on_click(_on_add_click)

                accordion = widgets.Accordion(children=[array_section], layout=widgets.Layout(margin='10px 0'))
                accordion.set_title(0, label)
                accordion.selected_index = None
                widget = accordion
                self.widget_refs[full_path] = {'widget': array_items_box, 'schema': prop_schema, 'type': 'array_object'}

            # Case 3: Fallback for other simple arrays (e.g., array of strings)
            else:
                default_val = '\n'.join(map(str, current_data)) if isinstance(current_data, list) else ""
                widget = widgets.Textarea(value=default_val, description=label, tooltip=f"{description} (Enter each item on a new line)", layout=widgets.Layout(margin='5px 0'))
                self.widget_refs[full_path] = {'widget': widget, 'schema': prop_schema, 'type': 'array_simple_text'}

        # --- String Type ---
        elif prop_type == "string":
            if "enum" in prop_schema:
                options = sorted(list(set(prop_schema.get("enum", []))))
                value = current_data if current_data in options else None
                if value is None and options and is_required: value = options[0]
                widget = widgets.Dropdown(
                    options=[(str(o), o) for o in options], value=value, description=label,
                    tooltip=description, style={'description_width': 'initial'}, layout=widgets.Layout(margin='5px 0')
                )
            else:
                widget_cls = widgets.Textarea if len(str(current_data or "")) > 80 else widgets.Text
                widget = widget_cls(
                    value=str(current_data or ""), description=label, tooltip=description,
                    style={'description_width': 'initial'}, layout=widgets.Layout(margin='5px 0')
                )
            self.widget_refs[full_path] = {'widget': widget, 'schema': prop_schema, 'type': 'string'}

        # --- Number / Integer Type ---
        elif prop_type in ["number", "integer"]:
            num_widget_type = widgets.IntText if prop_type == "integer" else widgets.FloatText
            default_value = current_data
            if current_data is None and is_required:
                default_value = 0 if prop_type == "integer" else 0.0

            widget = num_widget_type(
                value=default_value, description=label, step=(1 if prop_type == "integer" else None),
                tooltip=description, style={'description_width': 'initial'}, layout=widgets.Layout(margin='5px 0')
            )
            self.widget_refs[full_path] = {'widget': widget, 'schema': prop_schema, 'type': prop_type}

        # --- Boolean Type ---
        elif prop_type == "boolean":
            widget = widgets.Checkbox(
                value=bool(current_data) if current_data is not None else False,
                description=label, indent=False, tooltip=description, layout=widgets.Layout(margin='5px 0')
            )
            self.widget_refs[full_path] = {'widget': widget, 'schema': prop_schema, 'type': 'boolean'}

        return widget

    def _collect_data_from_widgets(self, widget_info):
        """
        Recursively collects data by traversing the widget structure.
        """
        widget = widget_info['widget']
        schema = widget_info['schema']
        widget_type = widget_info['type']

        if widget_type in ['string', 'number', 'integer', 'boolean', 'array_enum']:
            return self._get_widget_value(widget)
        
        elif widget_type == 'array_simple_text':
            value = self._get_widget_value(widget)
            if not value: return []
            items = [item.strip() for item in value.split('\n') if item.strip()]
            item_type = schema.get('items', {}).get('type')
            try:
                if item_type == 'number': return [float(i) for i in items]
                if item_type == 'integer': return [int(i) for i in items]
                return items
            except ValueError:
                print(f"Warning: Could not convert all items in '{schema.get('title')}' to {item_type}.")
                return items

        elif widget_type == 'object':
            obj_data = {}
            parent_path = next((p for p, i in self.widget_refs.items() if i['widget'] is widget and i['type'] == 'object'), None)
            if parent_path:
                for prop_name, prop_schema in schema.get("properties", {}).items():
                    child_path = f"{parent_path}.{prop_name}"
                    if child_path in self.widget_refs:
                        child_info = self.widget_refs[child_path]
                        prop_value = self._collect_data_from_widgets(child_info)
                        if prop_value is not None:
                            obj_data[prop_name] = prop_value
            return obj_data if obj_data else None

        elif widget_type == 'array_object':
            array_data = []
            item_schema = schema.get("items", {})
            array_items_box = widget
            array_path = next((p for p, i in self.widget_refs.items() if i['widget'] is array_items_box and i['type'] == 'array_object'), None)

            if array_path:
                for index, item_row_hbox in enumerate(array_items_box.children):
                    item_data = {}
                    item_path_prefix = f"{array_path}.{index}"
                    for prop_name, prop_schema in item_schema.get("properties", {}).items():
                        prop_path = f"{item_path_prefix}.{prop_name}"
                        if prop_path in self.widget_refs:
                            prop_info = self.widget_refs[prop_path]
                            prop_value = self._collect_data_from_widgets(prop_info)
                            if prop_value is not None:
                                item_data[prop_name] = prop_value
                    if item_data:
                        array_data.append(item_data)
            return array_data if array_data else None

        return None

    def _on_submit(self, b):
        """
        Handles the submit button click event. Collects, validates, and saves data.
        """
        print("click")

        with self.output_area:
            clear_output(wait=True)
            print("Collecting and validating data...")
            try:
                # self.collected_data.clear() # Good practice to clear old data

                for prop_name in self.schema.get("properties", {}).keys():
                    if prop_name in self.widget_refs:
                        prop_info = self.widget_refs[prop_name]
                        prop_value = self._collect_data_from_widgets(prop_info)
                        if prop_value is not None:
                            self.collected_data[prop_name] = prop_value

                jsonschema.validate(instance=self.collected_data, schema=self.schema)

                print(f"Saving validated data to {self.yaml_filepath}...")
                
                print(f"title", self.schema.get('title'))
                with open(self.yaml_filepath, 'w') as f:
                    # yaml.dump({self.schema.get('title'):self.collected_data}, f, default_flow_style=False, 
                    yaml.dump(self.collected_data, f, default_flow_style=False, 
                    sort_keys=False, indent=2)
                    
                print("Success! Data saved.")

                # *** EXECUTE THE CALLBACK FUNCTION HERE ***
                if self.on_submit_callback:
                    # Pass the collected data to the callback
                    self.on_submit_callback(self.collected_data)

            except jsonschema.ValidationError as e:
                path_str = " -> ".join(map(str, e.path)) if e.path else "root"
                print(f"Validation Error: {e.message}\nOn instance path: {path_str}")
            except Exception as e:
                import traceback
                print("An unexpected error occurred during submission:")
                traceback.print_exc()

    def _create_form_ui(self):
        """
        Builds the entire form UI, including title, widgets, and buttons.
        """
        self.widget_refs.clear()
        existing_data = None
        initial_load_error = None

        # 1. Load Existing Data
        if os.path.exists(self.yaml_filepath):
            try:
                with open(self.yaml_filepath, 'r') as f:
                    existing_data = yaml.safe_load(f)
                if existing_data:
                    jsonschema.validate(instance=existing_data, schema=self.schema)
            except (yaml.YAMLError, jsonschema.ValidationError, Exception) as e:
                initial_load_error = f"Could not load or validate existing YAML: {e}"

        # 2. Build Form Widgets
        title_widget = widgets.HTML(f"<h1>{self.schema.get('title', 'Form').replace('_',' ')}</h1>")
        desc_widget = widgets.HTML(f"<p>{self.schema.get('description', '')}</p>") if self.schema.get('description') else None

        if self.schema.get("type") == "object":
            form_content_box = widgets.VBox([])
            root_required = self.schema.get("required", [])
            root_widgets = []
            for prop_name, prop_schema in self.schema.get("properties", {}).items():
                prop_data = existing_data.get(prop_name) if isinstance(existing_data, dict) else None
                w = self._build_widget_from_schema(prop_name, prop_schema, prop_data, "", root_required)
                if w:
                    root_widgets.append(w)
            form_content_box.children = root_widgets
            form_content_widget = form_content_box
        else:
            form_content_widget = widgets.Label("Error: Schema root must be of type 'object'.")

        # 3. Create Buttons and Output Area
        submit_button = widgets.Button(description="Save to YAML", button_style='success', icon='save')
        self.output_area = widgets.Output() # Stored on self to be accessible by _on_submit
        
        if initial_load_error:
            with self.output_area:
                print(f"Warning: {initial_load_error}")

        submit_button.on_click(self._on_submit)

        # 4. Assemble Final Form Layout
        form_elements = [title_widget]
        if desc_widget:
            form_elements.append(desc_widget)
        form_elements.extend([form_content_widget, submit_button, self.output_area])

        full_form_layout = widgets.VBox(
            form_elements,
            layout=widgets.Layout(
                border='solid 2px #4CAF50',
                padding='15px',
                margin='10px',
                width='auto'
            )
        )
        return full_form_layout


# if __name__ == '__main__':
#     # This block demonstrates how to use the SchemaForm class.
#     # It will only run when the script is executed directly, not when imported.
    
#     # 1. Define a complex JSON schema
#     sample_schema = {
#         "title": "User Configuration",
#         "description": "A form to configure user settings and preferences.",
#         "type": "object",
#         "required": ["username", "user_level"],
#         "properties": {
#             "username": {
#                 "type": "string",
#                 "title": "Username",
#                 "description": "Your unique username."
#             },
#             "is_active": {
#                 "type": "boolean",
#                 "title": "Active Status",
#                 "description": "Check if the user account is active."
#             },
#             "user_level": {
#                 "type": "string",
#                 "title": "User Level",
#                 "enum": ["Guest", "Member", "Moderator", "Admin"],
#                 "description": "The permission level of the user."
#             },
#             "profile": {
#                 "type": "object",
#                 "title": "User Profile",
#                 "required": ["full_name"],
#                 "properties": {
#                     "full_name": {
#                         "type": "string",
#                         "title": "Full Name"
#                     },
#                     "age": {
#                         "type": "integer",
#                         "title": "Age"
#                     }
#                 }
#             },
#             "tags": {
#                 "type": "array",
#                 "title": "User Tags",
#                 "description": "Select tags that apply to this user.",
#                 "items": {
#                     "type": "string",
#                     "enum": ["developer", "designer", "tester", "manager"]
#                 }
#             },
#             "contacts": {
#                 "type": "array",
#                 "title": "Contact List",
#                 "description": "A list of user contacts.",
#                 "items": {
#                     "type": "object",
#                     "title": "Contact",
#                     "required": ["type", "value"],
#                     "properties": {
#                         "type": {
#                             "type": "string",
#                             "title": "Contact Type",
#                             "enum": ["Email", "Phone", "Skype"]
#                         },
#                         "value": {
#                             "type": "string",
#                             "title": "Contact Info"
#                         }
#                     }
#                 }
#             }
#         }
#     }

#     # 2. Specify the path for the output YAML file
#     output_yaml_path = 'user_config_output.yaml'

#     # --- How to use the class in a Jupyter Notebook ---
#     #
#     # try:
#     #     # 3. Create an instance of the form
#     #     my_form = SchemaForm(schema=sample_schema, yaml_filepath=output_yaml_path)
#     #
#     #     # 4. Display the form
#     #     my_form.display()
#     #
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")
#     #
#     # ---------------------------------------------------
    
#     print("SchemaForm class defined. To use it in a Jupyter Notebook, create an instance and call the .display() method.")
#     print(f"Example: form = SchemaForm(sample_schema, '{output_yaml_path}')")
#     print("           form.display()")

