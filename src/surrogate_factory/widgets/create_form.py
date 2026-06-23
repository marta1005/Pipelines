import ipywidgets as widgets
from IPython.display import display
import yaml
import os
import json
import copy
import ast
import traceback
from pathlib import Path

# Intentamos importar el formulario basado en Schema del usuario
try:
    from metadata_form import MetadataForm
except ImportError:
    MetadataForm = None
    # print("⚠️ 'metadata_form.py' not found. Schema-based editing will be disabled.")

# ==============================================================================
# 1. WIDGETS DE EDICIÓN (RECURSIVOS Y LISTAS INTELIGENTES)
# ==============================================================================

class RecursiveValueEditor(widgets.VBox):
    def __init__(self, key=None, value=None, depth=0, type_hint=None):
        super().__init__()
        self.key = key
        self.depth = depth
        self.type_hint = type_hint
        
        indent = 15 if depth > 0 else 0
        bg_color = "#f8f9fa" if depth % 2 == 0 else "#ffffff"
        border = "1px solid #e9ecef" if depth > 0 else "none"
        
        self.container_layout = widgets.Layout(
            border=border, padding='8px', width='98%', 
            margin=f'2px 0 2px {indent}px', background_color=bg_color
        )
        self.input_container = widgets.VBox(layout=self.container_layout)
        self.children = [self.input_container]
        
        self.is_complex = isinstance(value, dict)
        self._render_interface(value)

    def _render_interface(self, value):
        self.input_container.children = []
        row_widgets = []
        
        if self.key is not None:
             label_style = "font-weight:bold; color:#34495e;"
             row_widgets.append(widgets.HTML(f"<span style='{label_style}'>{self.key}:</span>", layout=widgets.Layout(width='150px')))
        
        if isinstance(value, dict):
            toggle_btn = widgets.ToggleButton(value=True, description='Collapse', icon='folder-open', layout=widgets.Layout(width='100px', height='25px'))
            row_widgets.append(toggle_btn)
            children_box = widgets.VBox()
            
            toggle_btn.observe(lambda c: setattr(children_box.layout, 'display', 'block' if c['new'] else 'none'), names='value')
            
            for k, v in value.items():
                children_box.children += (RecursiveValueEditor(key=k, value=v, depth=self.depth + 1),)
            
            self.input_container.children = [widgets.HBox(row_widgets), children_box]
            self.value_holder = children_box
            self.is_complex = True
        else:
            widget_to_add = None
            if isinstance(value, bool) or str(value).lower() in ['true', 'false']:
                val_bool = value if isinstance(value, bool) else str(value).lower() == 'true'
                widget_to_add = widgets.Checkbox(value=val_bool, indent=False, layout=widgets.Layout(width='auto'))
            elif isinstance(value, list) or self.type_hint in ['list', 'array']:
                val_str = ""
                if isinstance(value, list):
                    if len(value) > 0 and (isinstance(value[0], (list, dict))):
                        val_str = json.dumps(value)
                    else:
                        val_str = '\n'.join([str(x) for x in value])
                else:
                    val_str = str(value)
                widget_to_add = widgets.Textarea(value=val_str, placeholder='One item per line', layout=widgets.Layout(width='60%', height='80px'))
            else:
                val_str = str(value) if value is not None else ""
                widget_to_add = widgets.Text(value=val_str, layout=widgets.Layout(width='60%'))

            row_widgets.append(widget_to_add)
            self.input_container.children = [widgets.HBox(row_widgets)]
            self.value_holder = widget_to_add
            self.is_complex = False

    def get_data(self):
        if self.is_complex:
            return {child.key: child.get_data() for child in self.value_holder.children}
        else:
            widget = self.value_holder
            val = widget.value
            if isinstance(widget, widgets.Checkbox): return val
            if isinstance(widget, widgets.Textarea):
                try: return json.loads(val)
                except:
                    items = [x.strip() for x in val.split('\n') if x.strip()]
                    parsed = []
                    for item in items:
                        try: parsed.append(ast.literal_eval(item))
                        except: parsed.append(item)
                    return parsed
            if isinstance(widget, widgets.Text):
                try: return ast.literal_eval(val)
                except:
                    if val.lower() == 'true': return True
                    if val.lower() == 'false': return False
                    return val

class KeyValueListEditor(widgets.VBox):
    def __init__(self, initial_data=None):
        super().__init__()
        self.items_container = widgets.VBox()
        self.add_btn = widgets.Button(description="Add Param", icon='plus', button_style='info')
        self.add_btn.on_click(self.add_row)
        self.children = [self.items_container, self.add_btn]
        if initial_data and isinstance(initial_data, dict):
            for k, v in initial_data.items(): self.add_row_recursive(k, v)

    def add_row_recursive(self, key, value, type_hint=None):
        key_w = widgets.Text(value=str(key), placeholder='Name', layout=widgets.Layout(width='150px'))
        val_editor = RecursiveValueEditor(key=None, value=value, type_hint=type_hint)
        del_btn = widgets.Button(icon='trash', layout=widgets.Layout(width='40px'), button_style='danger')
        header = widgets.HBox([key_w, del_btn])
        container = widgets.VBox([header, val_editor], layout=widgets.Layout(border='1px solid #ddd', margin='5px', padding='5px'))
        
        def delete_row(_):
            kids = list(self.items_container.children)
            if container in kids:
                kids.remove(container)
                self.items_container.children = kids
        
        del_btn.on_click(delete_row)
        self.items_container.children = list(self.items_container.children) + [container]

    def add_row(self, _=None): self.add_row_recursive("", "")

    def get_data(self):
        data = {}
        for row in self.items_container.children:
            key_w = row.children[0].children[0]
            val_editor = row.children[1]
            if key_w.value: data[key_w.value] = val_editor.get_data()
        return data

# ==============================================================================
# 2. TASK ENTRY WIDGET
# ==============================================================================

class TaskEntryWidget(widgets.VBox):
    def __init__(self, category_key, initial_data=None, catalog=None):
        super().__init__()
        self.category_key = category_key
        self.catalog = catalog
        self.layout = widgets.Layout(border='1px solid #bdc3c7', padding='15px', margin='10px 0', width='98%', border_radius='5px')
        
        data = initial_data or {}
        
        raw_inputs = data.get('input', []) or data.get('inputs', [])
        raw_outputs = data.get('output', []) or data.get('outputs', [])
        init_inputs = '\n'.join([str(x) for x in raw_inputs]) if isinstance(raw_inputs, list) else str(raw_inputs)
        
        try:
            if len(raw_outputs) > 0 and isinstance(raw_outputs[0], dict):
                init_outputs = json.dumps(raw_outputs, indent=2)
            else:
                init_outputs = '\n'.join([str(x) for x in raw_outputs]) if isinstance(raw_outputs, list) else str(raw_outputs)
        except:
            init_outputs = str(raw_outputs)

        self.w_inputs = widgets.Textarea(description='Inputs:', value=init_inputs, placeholder='Line separated', layout=widgets.Layout(height='60px', width='95%'))
        self.w_outputs = widgets.Textarea(description='Outputs:', value=init_outputs, placeholder='Line separated or JSON', layout=widgets.Layout(height='60px', width='95%'))
        
        init_method = data.get('method', '')
        self.current_settings = data.get('settings', {})
        for k, v in data.items():
            if k not in ['input', 'inputs', 'output', 'outputs', 'method', 'settings']:
                if k not in self.current_settings: self.current_settings[k] = v

        local_methods = self.catalog.get_method_names(category_key)
        all_methods = self.catalog.get_all_method_names()
        
        mode_start = 'Catalog'
        should_show_all = False

        if init_method:
            if init_method in local_methods: pass
            elif init_method in all_methods: should_show_all = True
            else: mode_start = 'New Custom'
        elif not local_methods:
             if all_methods: should_show_all = True
             else: mode_start = 'New Custom'
        
        self.chk_show_all = widgets.Checkbox(value=should_show_all, description="Show all catalog methods", indent=False, layout=widgets.Layout(width='auto'))
        self.mode_selector = widgets.ToggleButtons(options=['Catalog', 'New Custom'], value=mode_start, description='Source:', style={'button_width': '100px'})
        self.w_catalog_picker = widgets.Dropdown(description='Method:', layout=widgets.Layout(width='50%'))
        self.w_custom_name = widgets.Text(description='Name:', value=init_method if mode_start == 'New Custom' else '')
        self.settings_container = widgets.VBox()

        self._update_catalog_options()
        if init_method and init_method in self.w_catalog_picker.options:
            self.w_catalog_picker.value = init_method

        self.chk_show_all.observe(self._on_show_all_change, names='value')
        self.mode_selector.observe(self._on_mode_change, names='value')
        self.w_catalog_picker.observe(self._on_catalog_method_change, names='value')

        self.children = [
            widgets.HBox([self.w_inputs, self.w_outputs]),
            widgets.HTML("<hr>"),
            widgets.HBox([self.mode_selector, self.chk_show_all]),
            self.settings_container
        ]
        self._on_mode_change()

    def _update_catalog_options(self):
        current = self.w_catalog_picker.value
        opts = self.catalog.get_all_method_names() if self.chk_show_all.value else self.catalog.get_method_names(self.category_key)
        self.w_catalog_picker.options = opts
        if current in opts: self.w_catalog_picker.value = current

    def _on_show_all_change(self, _):
        if self.mode_selector.value == 'Catalog': self._update_catalog_options()

    def _on_mode_change(self, _=None):
        mode = self.mode_selector.value
        self.chk_show_all.layout.display = 'block' if mode == 'Catalog' else 'none'
        self.settings_container.children = []
        if mode == 'Catalog':
            if not self.w_catalog_picker.options:
                self.settings_container.children = [widgets.Label("No methods available.")]
            else:
                self.settings_container.children = [self.w_catalog_picker]
                self._on_catalog_method_change() 
        else:
            self.editor = KeyValueListEditor(initial_data=self.current_settings)
            self.settings_container.children = [self.w_custom_name, widgets.HTML("<b>Custom Settings:</b>"), self.editor]

    def _on_catalog_method_change(self, _=None):
        if self.mode_selector.value != 'Catalog': return
        method_name = self.w_catalog_picker.value
        if not method_name: return
        
        template = self.catalog.get_method_template(self.category_key, method_name)
        arguments = self.catalog.get_method_arguments(self.category_key, method_name)
        
        self.editor = KeyValueListEditor()
        info_html = ""
        
        if self.current_settings:
            info_html = "<span style='color:blue'>Loaded from existing configuration</span>"
            for k, v in self.current_settings.items(): self.editor.add_row_recursive(k, v)
        elif template:
            info_html = "<span style='color:green'>Loaded from Catalog Template</span>"
            for k, v in template.items(): self.editor.add_row_recursive(k, v)
        elif arguments:
            info_html = "<span style='color:orange'>Generated from Catalog Arguments</span>"
            for k, v in arguments.items():
                default = v.get('default', '')
                if default == 'REQUIRED': default = ''
                self.editor.add_row_recursive(k, default, type_hint=v.get('type', 'string'))
        
        self.settings_container.children = [self.w_catalog_picker, widgets.HTML(f"<div style='margin:5px 0 10px 0; font-size:0.9em;'>{info_html}</div>"), self.editor]

    def get_data(self):
        in_list = [x.strip() for x in self.w_inputs.value.split('\n') if x.strip()]
        out_val = self.w_outputs.value
        try: out_list = json.loads(out_val)
        except: out_list = [x.strip() for x in out_val.split('\n') if x.strip()]
        
        method = self.w_catalog_picker.value if self.mode_selector.value == 'Catalog' else self.w_custom_name.value
        settings = self.editor.get_data() if hasattr(self, 'editor') else {}
        
        # --- CORRECCIÓN: Usar plural 'inputs' y 'outputs' en el retorno ---
        return {"inputs": in_list, "outputs": out_list, "method": method, "settings": settings}

# ==============================================================================
# 3. GENERIC SECTION WIDGET (Para Estructura Plana)
# ==============================================================================

class GenericSectionWidget(widgets.VBox):
    def __init__(self, title, data):
        super().__init__()
        self.title = title
        display_title = title.replace('_', ' ')
        self.header = widgets.HTML(f"<h3 style='color:#2c3e50; border-bottom: 3px solid #3498db; padding-bottom:5px'>{display_title}</h3>")
        self.editor = KeyValueListEditor(initial_data=data)
        self.children = [self.header, self.editor]

    def get_data(self):
        return self.editor.get_data()

# ==============================================================================
# 4. FUNCTION SECTION WIDGET (Para Listas de Tareas)
# ==============================================================================

class FunctionSectionWidget(widgets.VBox):
    def __init__(self, title, initial_data_list=None, catalog=None, catalog_key=None, sub_sections=None):
        super().__init__()
        self.title = title
        self.catalog = catalog
        self.catalog_lookup_key = catalog_key if catalog_key else title
        self.wrapper_key = 'params' # Default wrapper key per requirement
        self.final_data_list = []
        self.sub_sections = sub_sections or [] 
        self.sub_section_widgets = {} 
        
        display_title = title.replace('_', ' ')
        
        self.header = widgets.HTML(f"<h3 style='color:#2c3e50; border-bottom: 3px solid #3498db; padding-bottom:5px'>{display_title}</h3>")
        self.tasks_container = widgets.VBox()
        self.add_btn = widgets.Button(description="Add Task", icon='plus', button_style='success')
        self.add_btn.on_click(self.add_task)
        
        # --- LOGICA DE DETECCION Y WRAPPING ---
        if self.sub_sections:
            # Sub-secciones anidadas
            data_dict = initial_data_list if isinstance(initial_data_list, dict) else {}
            sub_widgets = []
            for sub_name in self.sub_sections:
                sub_data = data_dict.get(sub_name, [])
                w = FunctionSectionWidget(sub_name, initial_data_list=sub_data, catalog=self.catalog, catalog_key=sub_name)
                self.sub_section_widgets[sub_name] = w
                sub_widgets.append(w)
            
            self.children = [self.header, widgets.VBox(sub_widgets, layout=widgets.Layout(margin='0 0 0 20px'))]
            
        else:
            # Sección Hoja (Lista de Tareas)
            
            # 1. Normalizar entrada a lista
            if isinstance(initial_data_list, list):
                self.final_data_list = initial_data_list
            
            elif isinstance(initial_data_list, dict):
                # Caso: Diccionario
                keys = list(initial_data_list.keys())
                
                # A. Detectar si es una Tarea Única (tiene method, input, etc.)
                # Si es tarea, la metemos en una lista. wrapper_key será params.
                if any(k in keys for k in ['method', 'input', 'inputs', 'output', 'outputs']):
                    self.final_data_list = [initial_data_list]
                    self.wrapper_key = 'params'
                
                # B. Detectar si es un Wrapper existente (una sola key que contiene lista/tarea)
                elif len(keys) == 1:
                    content = initial_data_list[keys[0]]
                    if isinstance(content, list):
                        self.wrapper_key = keys[0]
                        self.final_data_list = content
                        display_title = f"{display_title} > {self.wrapper_key.replace('_', ' ')}"
                        self.header.value = f"<h3 style='color:#2c3e50; border-bottom: 3px solid #3498db; padding-bottom:5px'>{display_title}</h3>"
                    else:
                        # Si no es lista, asumimos que es una tarea compleja dentro de un wrapper
                        self.wrapper_key = keys[0]
                        self.final_data_list = [content]
                else:
                    # Fallback
                    self.final_data_list = [initial_data_list]

            self.children = [self.header, self.tasks_container, self.add_btn]
            
            if self.final_data_list:
                for d in self.final_data_list: self.add_task(data=d)
            else:
                self.add_task()

    def add_task(self, _=None, data=None):
        task_widget = TaskEntryWidget(self.catalog_lookup_key, initial_data=data, catalog=self.catalog)
        del_btn = widgets.Button(icon='times', button_style='warning', layout=widgets.Layout(width='30px', margin='10px'))
        container = widgets.HBox([task_widget, del_btn], layout=widgets.Layout(align_items='flex-start'))
        
        def delete_task(_):
            kids = list(self.tasks_container.children)
            if container in kids:
                kids.remove(container)
                self.tasks_container.children = kids
        
        del_btn.on_click(delete_task)
        self.tasks_container.children = list(self.tasks_container.children) + [container]

    def get_data(self):
        if self.sub_sections:
            data = {}
            for name, w in self.sub_section_widgets.items():
                data[name] = w.get_data()
            return data
        else:
            tasks = [child.children[0].get_data() for child in self.tasks_container.children]
            # --- CORRECCIÓN: Siempre retornar encapsulado en el wrapper (default: params) ---
            return {self.wrapper_key: tasks}

# ==============================================================================
# 5. WORKFLOW METADATA EDITOR (PRINCIPAL)
# ==============================================================================

class WorkflowMetadataEditor(widgets.VBox):
    def __init__(self, stage_name, file_path, mapping, catalog, on_save_callback=None):
        super().__init__()
        self.stage_name = stage_name
        self.chapter_name = self._clean_name(stage_name)
        self.file_path = file_path
        self.mapping = mapping
        self.catalog = catalog
        self.on_save_callback = on_save_callback 
        self.has_root_wrapper = True
        
        display_chapter = self.chapter_name.replace('_', ' ')
        self.title_w = widgets.HTML(f"<h2>Metadata Editor: <span style='color:#e67e22'>{display_chapter}</span></h2>")
        self.content_area = widgets.VBox()
        self.debug_area = widgets.Output()
        
        self.save_btn = widgets.Button(description="Save YAML", icon="save", button_style='primary', layout=widgets.Layout(width='200px', margin='20px 0'))
        self.save_btn.on_click(self.save_yaml)
        self.output_msg = widgets.Output()
        
        self.sections = {} 
        
        self.children = [self.title_w, self.content_area, self.debug_area, self.save_btn, self.output_msg]
        self._load_interface()

    def _clean_name(self, name):
        parts = name.split('_')
        clean = '_'.join(parts[2:]) if len(parts) >= 3 and parts[0] == 'SF' else name
        return clean.strip()

    def _load_interface(self):
        try:
            yaml_data = {}
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, 'r') as f:
                        full = yaml.safe_load(f) or {}
                        self.has_root_wrapper = False 
                        
                        target_key_match = None
                        for k in full.keys():
                            if k.lower() == self.chapter_name.lower():
                                target_key_match = k
                                break
                        
                        if len(full) == 1 and target_key_match:
                            content = full[target_key_match]
                            if isinstance(content, dict):
                                is_wrapper_likely = True
                                for sub_v in content.values():
                                    if not isinstance(sub_v, (dict, list)):
                                        is_wrapper_likely = False
                                        break
                                if is_wrapper_likely:
                                    self.has_root_wrapper = True
                                    yaml_data = content
                                else:
                                    yaml_data = full
                            else:
                                self.has_root_wrapper = True
                                yaml_data = content
                        else:
                            self.has_root_wrapper = False
                            yaml_data = full
                except Exception as e:
                    with self.debug_area: print(f"⚠️ Error reading YAML file: {e}")

            sections_to_render = [] 
            default_cat_key = self.chapter_name
            target_chapter = self.chapter_name.lower()

            mapping_structure = {} 
            found_in_mapping = False

            for path in self.mapping.values():
                if len(path) > 1 and str(path[1]).lower() == target_chapter:
                    found_in_mapping = True
                    section_name = path[2] if len(path) > 2 else "Configuration"
                    if section_name not in mapping_structure:
                        mapping_structure[section_name] = set()
                    if len(path) > 3:
                        mapping_structure[section_name].add(path[3])

            if not found_in_mapping and not yaml_data:
                self.content_area.children = [widgets.HTML("<i>No data found. Creating new generic structure...</i>")]
                return

            if yaml_data:
                yaml_keys_lower = {k.lower(): k for k in yaml_data.keys()}
                
                for sec_map, sub_secs in mapping_structure.items():
                    real_key = yaml_keys_lower.get(sec_map.lower())
                    data = yaml_data.get(real_key, {}) if real_key else {}
                    
                    widget_type = 'generic'
                    sub_sec_list = sorted(list(sub_secs)) if sub_secs else None
                    
                    if sub_sec_list:
                        widget_type = 'task'
                    elif isinstance(data, list):
                        widget_type = 'task'
                    elif isinstance(data, dict):
                        if 'method' in data or 'inputs' in data or 'input' in data:
                            widget_type = 'task'
                        elif sec_map == "Configuration": 
                             widget_type = 'task'
                        elif 'params' in data: # Detect explicit params wrapper
                             widget_type = 'task'
                        else:
                             widget_type = 'generic'
                    
                    sections_to_render.append((sec_map, data, widget_type, sub_sec_list))

                for k, v in yaml_data.items():
                    is_processed = False
                    for existing_sec in mapping_structure.keys():
                        if k.lower() == existing_sec.lower():
                            is_processed = True
                            break
                    
                    if not is_processed and k != "Configuration":
                        widget_type = 'generic'
                        if isinstance(v, list): widget_type = 'task'
                        elif isinstance(v, dict) and ('method' in v or 'inputs' in v or 'params' in v): widget_type = 'task'
                        sections_to_render.append((k, v, widget_type, None))
            else:
                for sec_map, sub_secs in mapping_structure.items():
                    sub_sec_list = sorted(list(sub_secs)) if sub_secs else None
                    sections_to_render.append((sec_map, {}, 'task', sub_sec_list))

            widget_list = []
            for name, data, w_type, subs in sections_to_render:
                if w_type == 'task':
                    w = FunctionSectionWidget(name, initial_data_list=data, catalog=self.catalog, catalog_key=default_cat_key, sub_sections=subs)
                else:
                    w = GenericSectionWidget(name, data=data)
                
                self.sections[name] = w
                widget_list.append(w)
                
            self.content_area.children = widget_list
            
        except Exception as e:
            with self.debug_area:
                print("❌ CRITICAL ERROR IN INTERFACE LOADING:")
                print(traceback.format_exc())

    def save_yaml(self, _):
        self.output_msg.clear_output()
        chapter_data = {}
        
        for name, w in self.sections.items():
            if name == "Configuration":
                data = w.get_data()
                if isinstance(data, dict): chapter_data.update(data)
                else: chapter_data[name] = data
            else:
                chapter_data[name] = w.get_data()
        
        if self.has_root_wrapper:
            full_data = {self.chapter_name: chapter_data}
        else:
            full_data = chapter_data
            
        try:
            with open(self.file_path, 'w') as f: yaml.dump(full_data, f, sort_keys=False, default_flow_style=False)
            
            msg_extra = ""
            if self.on_save_callback:
                try:
                    self.on_save_callback(full_data)
                    msg_extra = "<br><span style='color:green'>✅ Memory Updated (self.metadata)</span>"
                except Exception as e:
                    msg_extra = f"<br><span style='color:red'>⚠️ Memory Update Failed: {e}</span>"

            with self.output_msg: 
                print(f"✅ Saved to {self.file_path}")
                display(widgets.HTML(msg_extra))

        except Exception as e:
            with self.output_msg: print(f"❌ Error saving: {e}")