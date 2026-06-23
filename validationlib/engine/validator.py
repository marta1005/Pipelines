'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
"""
This module contains the `Validator` class, which is responsible for executing a given notebook template and saving the results.

Classes:
- Validator: A notebook template class in charge of the proper execution of a given notebook template.

Functions:
- config_checker: A function that checks for discrepancies between the `json_template` and `user_config`.
- build: A function that compares the definition of the validator and user config. It will also save the path of where is the notebook to be executed and general configuration.
- nbconvertToHTML: A function that converts the executed notebook to HTML format.

Dependencies:
- typing.Dict
- pandas
- pathlib.Path
- json
- ast
- os
- shutil
- nbformat
- tqdm
- nbclient.NotebookClient
- traitlets.config.Config
- nbconvert.exporters.HTMLExporter
- nbconvert.preprocessors.ExecutePreprocessor
- nbconvert.preprocessors.CellExecutionError
- jupyter_client.manager.start_new_kernel
"""

from pathlib import Path
import json
import ast
import os

# Jupyter tools
import nbformat
from tqdm import tqdm
from nbclient import NotebookClient
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


## de la lista de chequeos poner si son warning o errors


class Validator:
    '''
    Notebook template class in charge of the proper execution of a given notebook template.

    Methods:
    - __init__: Initialize the Validator with template parameters, configuration, and other settings.
    
    - config_checker: Check and validate the user configuration against the template definition.
    
    - build: Build the Validator by checking for discrepancies between the JSON template and user configuration.
    
    - nbconvertToHTML: Convert the notebook to an HTML format and save it.
    
    - nbpreprocess: Preprocess the notebook, executing code cells.
    
    - nbpreprocessWithProgress: Preprocess the notebook with progress tracking.
    
    - createNotebook: Process and execute the notebook, creating a new notebook.
    
    - writeTempfile: Write a configuration tempfile.
    
    - run: Execute the template, saving the results and exporting to HTML.
    '''

    def __init__(self,
            templateParams,
            surrogateConfig,
            datasetConfig,
            name,
            outputPath,
            valiLibPath,
            ml_deployPath,
            debugmode,
            colorblind
        ):
        '''
        Initialize the Validator with the given parameters.

        Parameters:
        - templateParams: Dictionary containing template parameters.
        - surrogateConfig: Configuration for surrogate models.
        - datasetConfig: Configuration for the dataset.
        - name: Name of the Validator.
        - outputPath: Path to the output directory.
        - valiLibPath: Path to the valiLib library.
        - ml_deployPath: Path to the ml_deploy library.
        - debugmode: Debug mode setting.
        - colorblind: Colorblind mode setting.
        '''
        # 3-composed name: model-dataset-template
        # The file will be called dataset-template, and stored in output/model/
        self.folder = name.split('--')[0]
        self.name = '--'.join(name.split('--')[1:])
        # Notebook path
        self.templateParams = templateParams
        self.notebookPath = templateParams['notebook_path']
        # Output path for this specific model-dataset-template
        self.outputPath = outputPath
        self.runOutputPath = Path(self.outputPath) / rf'{self.folder}/{self.name.split("--")[0]}/{self.name.split("--")[1]}'
        # Template specific configuration file. It contains mandatory inputs, outputs and requirements
        with open(templateParams['template_config'], 'r') as file:
            self.templateConfig = json.load(file)
        # User configuration regarding this specific template. It contains modifications
        # over the inputs, outputs and requirements defined in the template configuration.
        self.userConfig = templateParams['user_config']
        # Surrogate specific configuration file. It contains the orders (if any)
        # some present categories may have, the location of the model file and
        # anything related to the model itself
        self.surrogateConfig = surrogateConfig
        # Dataset specific information, such as the file type, the location,
        # name of relevant key in the case of hdf5 files...
        self.datasetConfig = datasetConfig
        # Location of valiLib and ml_deploy libraries. If they are used as
        # installed they default to None
        self.valiLibPath = valiLibPath
        self.ml_deployPath = ml_deployPath
        # Debugmode
        self.debugmode = debugmode
        # Colorblind mode
        self.colorblind = colorblind

        self.built_ = False

        self.checks_passed = {
            "passed":True,
            "logs":[],
        }

    def config_checker(self):
        '''
        Check and validate the user configuration against the template definition.
        '''
        if "inputs" not in self.userConfig:
            self.userConfig["inputs"] = {}
        if "outputs" not in self.userConfig:
            self.userConfig["outputs"] = {}
        if "requirements" not in self.userConfig:
            self.userConfig["requirements"] = {}

        for input_ in self.userConfig['inputs']:
            if not input_ in self.templateConfig['inputs'].keys():
                self.checks_passed["passed"] = False
                self.checks_passed["logs"].append(f"input {input_} not available")

        for req in self.userConfig['requirements']:
            if not req in self.templateConfig['requirements'].keys():
                self.checks_passed["passed"] = False
                self.checks_passed["logs"].append(f"requirement {req} does not exist")

        if self.debugmode: print(json.dumps(self.checks_passed, indent=4))

        return self.checks_passed

    def build(self):
        '''
        Build the Validator by checking for discrepancies between the JSON template and user configuration.
        this function checks for discrepancies from json_template and user_config
        '''
        if self.debugmode: print(f'Building {self.name} of model {self.folder}')
        self.config_checker()

        if self.checks_passed["passed"]:
            # Default requirements overwriten by user defined requirements
            for req in self.userConfig['requirements']:
                for req_field in self.userConfig['requirements'][req]:
                    self.templateConfig['requirements'][req][req_field] = self.userConfig['requirements'][req][req_field]

            # Default inputs overwriten by user defined inputs
            for input_name in self.userConfig['inputs']:
                for input_field in self.userConfig['inputs'][input_name]:
                    self.templateConfig['inputs'][input_name][input_field] = self.userConfig['inputs'][input_name][input_field]

            self.built_ = True
        else:
            raise ValueError(self.checks_passed["logs"])


    def nbconvertToHTML(self):
        '''
        Convert the notebook to an HTML format and save it.
        '''
        if self.debugmode: print(f"Converting {self.name}.ipynb to HTML")

        htmlExporter = HTMLExporter()
        htmlExporter.exclude_input = True
        htmlData, resources = htmlExporter.from_notebook_node(self.nb, resources={"metadata":{"name":self.name}})

        with open(self.htmlOutputPath, 'w', encoding="utf-8") as file:
            file.write(htmlData)
        # with open('C:/Users/noeli/Desktop/results/hi.html', 'w', encoding="utf") as file:
        #     file.write(htmlData)
            # Path(('C:/Users/noeli/Desktop/results'/{self.name.split("--")[0]}/{self.name.split("--")[1]}.html'))
        with open(self.htmlOutputPathResults, 'w', encoding="utf-8") as file:
            file.write(htmlData)


    def nbpreprocess(self, notebook, resources = {'metadata': {'path': os.getcwd()}}, kernel_name='python3'):
        '''
        Preprocess the notebook, executing code cells.

        Parameters:
        - notebook: The notebook to preprocess.
        - resources: Additional resources for preprocessing.
        - kernel_name: Name of the kernel to use for execution.
        '''
        ## Activate Kernel
        ep = ExecutePreprocessor(timeout=None, kernel_name=kernel_name)
        
        # Run in one go
        try:
            ep.preprocess(notebook, resources)
        except CellExecutionError:
            #https://nbconvert.readthedocs.io/en/latest/execute_api.html#handling-errors
            msg = 'Error executing the notebook "%s".\n\n' % self.notebookPath
            msg += 'See notebook "%s" for the traceback.' % self.notebookPath
            print(msg)

        return notebook

    def nbpreprocessWithProgress(self, notebook, resources = {'metadata': {'path': os.getcwd()}}, kernel_name='python3'):
        '''
        Preprocess the notebook with progress tracking.

        Parameters:
        - notebook: The notebook to preprocess.
        - resources: Additional resources for preprocessing.
        - kernel_name: Name of the kernel to use for execution.
        '''
        ## Activate Kernel
        ep = ExecutePreprocessor(timeout=None, kernel_name=kernel_name)

        ncells = len(notebook.cells)

        NotebookClient.__init__(ep, notebook, None)
        ep._check_assign_resources(resources)

        with ep.setup_kernel():
            #assert ep.kc  # noqa
            #info_msg = ep.wait_for_reply(ep.kc.kernel_info())
            #assert info_msg  # noqa
            #ep.nb.metadata["language_info"] = info_msg["content"]["language_info"]
            try:
                for index, cell in tqdm(enumerate(ep.nb.cells), total=ncells):
                    ep.preprocess_cell(cell, resources, index)

                if self.templateConfig['requirements']:
                    self.reqs_results_ = ast.literal_eval(self.nb['cells'][-1]['outputs'][0]['data']['text/plain'])
                    self.nb['cells'][-1]['outputs'] = []
            except CellExecutionError:
                #https://nbconvert.readthedocs.io/en/latest/execute_api.html#handling-errors
                msg = 'Error executing the notebook "%s".\n\n' % self.notebookPath
                msg += 'See notebook "%s" for the traceback.' % self.notebookPath
                print(msg)

                if self.templateConfig['requirements']:
                    print("Attempting to read requirements' outputs results...")

                    try:
                        ep.preprocess_cell(ep.nb.cells[-1], resources, index+1)
                        self.reqs_results_ = ast.literal_eval(self.nb['cells'][-1]['outputs'][0]['data']['text/plain'])
                        self.nb['cells'][-1]['outputs'] = []
                    except CellExecutionError:
                        print("Evaluation of requirements' outputs failed")

        ep.set_widgets_metadata()

        return ep.nb

        # # Run Jupyter notebook cell by cell
        # ncells = len(notebook.cells)
        # with ep.setup_kernel():
        # #with ep.setup_preprocessor(notebook, resources):
            # for index, cell in tqdm(enumerate(notebook.cells), total=ncells):
                # notebook.cells[index], resources = ep.preprocess_cell(cell, resources, index)
        # return notebook

    def createNotebook(self, tempfile_name:str = "tempfile.json"):
        '''
        Process and execute the notebook, creating a new notebook.

        Parameters:
        - tempfile_name: Name of the tempfile to create.
        '''
        if self.debugmode: print(f"Processing {self.name} notebook")
        ## Read template
        with open(self.notebookPath) as file:
            self.nb = nbformat.read(file, as_version=4)#nbformat.NO_CONVERT

        location = str(self.runOutputPath / self.tempfile_name).replace('\\','/')
        self.nb['cells'][0]['source'] = "configLocation = '" + location + "'"

        # Execute the notebook
        self.reqs_results_ = {}
        self.nb = self.nbpreprocessWithProgress(self.nb, kernel_name='python3')

        # Save it
        with self.notebookOutputPath.open("wt") as file:
            nbformat.write(self.nb, file)

    def writeTempfile(self, tempfile_name=None):
        '''
        Write a configuration tempfile.

        Parameters:
        - tempfile_name: Name of the tempfile to create.
        '''
        if tempfile_name is None:
            self.tempfile_name = self.folder+'--'+self.name+'.json'

        # Build tempfile
        tempfileContents = {
            'NAME': self.folder+'--'+self.name,
            'ML_DEPLOY_PATH': str(self.ml_deployPath),
            'VALILIB_PATH': str(self.valiLibPath),
            'OUTPUT_PATH': str(self.outputPath),
            **self.colorblind,
            **self.templateParams,
            **self.templateConfig,
            **self.surrogateConfig,
            **self.datasetConfig
        }
        # Save it
        with open(self.runOutputPath / self.tempfile_name, "w") as file:
            json.dump(tempfileContents, file, indent=4, default=lambda x: str(x))

        # Always have a copy in the output folder so that templates can be run from templates/ folder.
        tempfile = self.outputPath / "tempfile.json"
        with open(tempfile, "w") as file:
            json.dump(tempfileContents, file, indent=4, default=lambda x: str(x))

    def run(self, resume=False):
        '''
        Execute the template, saving the results and exporting to HTML.

        Parameters:
        - resume: If True, resume execution if the HTML output file already exists.
        '''
        print('Running',self.name)

        self.writeTempfile()

        # Run the noteook
        if self.built_:
            self.notebookOutputPath = self.runOutputPath / f'{self.name.split("--")[1]}.ipynb'
            self.htmlOutputPath = self.runOutputPath / f'{self.name.split("--")[1]}.html'

            # Save a copy in Last_Results folder.
            self.htmlOutputPathResults = Path(self.outputPath) / rf'{"Last_Results"}'
            self.htmlOutputPathResults.mkdir(parents=True, exist_ok=True)
            self.htmlOutputPathResults = self.htmlOutputPathResults / f'{self.name.split("--")[1]}.html'

            if resume and self.htmlOutputPath.is_file():
                print(self.name, "already exists. Resuming...")
            else:
                self.createNotebook()
                self.nbconvertToHTML()
        else:
            raise ValueError(f"Validator {self.name} not built")

