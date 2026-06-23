'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List
import json
import importlib
import subprocess
#
import numpy as np
from scipy.optimize import nnls

#
current_file_path = Path(__file__).parent.resolve()
templates_path = (current_file_path / ".." / "template_notebooks").resolve()
cwd_path = Path().cwd()

def color_blind(color: bool = True, debug: bool = True) -> Dict[str, List[str]]:
    #### COLORBLINDNESS
    # Matplotlib's default C0, C1, C2... colors are colorblind-friendly and look good, there is no need to change those.
    if color:
        colorblind = {
            'colorblindPair': ['#009E73','#D55E00'],    #Positive-TEAL vs Negative-ORANGE
            'colorblindNeutral': '#F0E442',             #Neutral-YELLOW
            'colorblindDivergingCmap': 'seismic_r',     #Diverging Negative-RED-WHITE-BLUE-Positive
            'colorblindDivergingCmap_r': 'seismic',
            'colorblindSequentialCmap': 'cividis',      #Uniform Sequential BLUE-GRAY-YELLOW
            'colorblindSequentialCmap_r': 'cividis_r'
        }
        if debug: print('Colorblind mode selected')
    else:
        colorblind = {
            'colorblindPair': ['#006400','#8B0000'],    #Positive-GREEN vs Negative-RED
            'colorblindNeutral': '#003887',             #Neutral-BLUE
            'colorblindDivergingCmap': 'seismic_r',     #Diverging Negative-RED-WHITE-BLUE-Positive
            'colorblindDivergingCmap_r': 'seismic',
            'colorblindSequentialCmap': 'viridis',      #Uniform Sequential PURPLE-BLUE-GREEN-YELLOW
            'colorblindSequentialCmap_r': 'viridis_r'
        }
        if debug: print('Non-colorblind mode selected')
    return colorblind

def check_packages(debug: bool = True) -> None:
    """
    Check if the required packages are available
    """
    requirements_path = (current_file_path / Path("../requirements.yml")).resolve()

    with requirements_path.open(mode="r") as file:
        required_packages = [line.partition('- ')[2].partition('=')[0].partition('>')[0].partition('<')[0].partition('\n')[0].partition(' ')[0] for line in file.readlines()[5:]]
        
    list_c = subprocess.check_output(['conda', 'list']).decode('utf-8')
    installed_packages = [line.split(' ')[0] for line in list_c.split('\n')[2:-1]]
    check = all(package in installed_packages for package in required_packages)
    missing_packages = list(set(required_packages) - set(installed_packages))

    if  not check:
        print(f'The following packages are missing:\n{", ".join(str(elem) for elem in missing_packages)}\n')
    elif debug:
        print('All packages are installed\n')
    
    return

def insert_path(iPath: str, lib_name: str, debug: bool = True) -> str:
    """
    Insert some library on the python paths (sys.path)
    """
    spec = importlib.util.find_spec(lib_name)
    if spec is not None:
        if debug:
            print(f"{lib_name} is installed")
        return spec.submodule_search_locations[0]
    if iPath is None:
        raise ValueError(f"{lib_name} wasn't found in installed packages or specified in specified path")
    else:
        if debug: 
            print(f'Add {lib_name} from specified local folder {iPath}')
        sys.path.insert(1, iPath)
        return iPath

def find_path(filename: str, path_list: List[str], end_pattern: str = "*.ipynb") -> str:
    """
    Find filename in path in the list order.
    Return the first path that match the filename path
    """

    if Path(filename).is_absolute():
        return filename
    for p in path_list:
        list_files = Path(p).glob(p + end_pattern)
        if len(list_files) > 0:
            return str(list_files[0].resolve())
    return ""


if __name__ == '__main__':

    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', help='Full path to the user config file', required=True)
    
    # Additional execution options, will be prompted if missing
    parser.add_argument('-t', '--templates', nargs='+', help='Desired templates (names as in the user config file) separated by spaces')
    parser.add_argument('-s', '--surrogates', nargs='+', help='Desired surrogates (names as in the user config file) separated by spaces')
    parser.add_argument('--run', action="store_true", help="Execute the pipeline")
    parser.add_argument('--resume', action="store_true", help="Leave as false if it's desired to recompute already finished templates (recommended unless time constrained)")

    # Additional options
    parser.add_argument('--debugmode', action="store_true", help="Activates debug mode")
    parser.add_argument('--colorblind', action="store_true", help="Activates colorblind mode")
    parser.add_argument('--benchmark', help="Benchmark this machine for time prediction purposes. Must provide an ID")
    
    args = parser.parse_args()

    colorblind = color_blind(args.colorblind, args.debugmode)
    #### LOAD JSON
    with open(args.json, "r") as file:
        user_options = json.loads(file.read())
    if args.debugmode: print('Config file:',json.dumps(user_options, indent=4))
    

    #### CHECK INSTALLED PACKAGES
    check_packages()
    #### CONFIG CHECKS AND RELATIVE PATH FIXING

    # Get absolute path
    rootPath = Path(args.json).parents[0]

    # CHECK if valiLib not specified in the config file, otherwise add to python
    valiLibPath = insert_path(user_options.get("VALILIB-PATH", None), 'validationlib', args.debugmode)
    from validationlib import engine
    if args.debugmode: print('validationlib successfully imported')

    valiLibPath = engine.common.absRelativePath(valiLibPath, rootPath)

    ml_deployPath = insert_path(user_options.get('ML_DEPLOY_PATH',None), "ml_deploy", args.debugmode)
    ml_deployPath = engine.common.absRelativePath(ml_deployPath, rootPath)

    # CHECK if the output location is specified
    outputPath = user_options.get('OUTPUT_PATH',None)
    if outputPath is None: 
        # if  path is not specified take the one from config.json
        outputPath = rootPath
    # Create output / appendix / model_outputs folders if needed
    outputPah = engine.common.absRelativePath(outputPath, rootPath)
    
    if args.debugmode: print('Output folder defined as: ',outputPath)
    engine.common.createFolders(outputPath / 'intermediate')
    
    # CHECK if templates are correctly defined
    templates = user_options.get('TEMPLATES', None)
    if templates is None:
        raise ValueError('TEMPLATES not specified. Please read the provided default config file to understand the required fields')
    else:
        for x in templates:
            # Template itself
            templates[x]['notebook_path'] = engine.common.absRelativePath(templates[x]['notebook_path'], rootPath)
            
            if templates[x]['template_config'] == 'default': templates[x]['template_config'] = Path(str(templates[x]['notebook_path']).split('.')[0] + '.json')
            else: templates[x]['template_config'] = engine.common.absRelativePath(templates[x]['template_config'], rootPath)
            
            if not Path(templates[x]['notebook_path']).exists():
                raise ValueError(x + f"'s template not found in {templates[x]['notebook_path']}")
            # Template specific config
            elif not Path(templates[x]['template_config']).exists():
                raise ValueError(x + f"'s JSON file for template configuration not found in {templates[x]['template_config']}")

            if args.debugmode: print(f'Template "{x}" preliminary checks passed')
                
    # CHECK if surrogates are correctly defined
    surrogates = user_options.get('SURROGATES', None)
    if templates is None:
        raise ValueError('SURROGATES not specified. Please read the provided default config file to understand the required fields')
    else:
        for x in surrogates:
            # Model validation file
            surrogates[x]['validation_config_file'] = engine.common.absRelativePath(surrogates[x]['validation_config_file'], rootPath)
            if not Path(surrogates[x]['validation_config_file']).exists():
                raise ValueError(x + f"'s validation_config_file not found in {surrogates[x]['validation_config_file']}")
            # Load validation file model_definition_filename
            with open(surrogates[x]['validation_config_file'], 'r') as file:
                surrogates[x]['surrogate_config'] = json.load(file)
            # Make sure model definition is ok
            surrogates[x]['surrogate_config']['model_definition_filename'] = engine.common.absRelativePath(
                surrogates[x]['surrogate_config']['model_definition_filename'], 
                surrogates[x]['validation_config_file'].parents[0]
            )
            if not Path(surrogates[x]['surrogate_config']['model_definition_filename']).exists():
                raise ValueError(x + f"'s model definition not found in {surrogates[x]['surrogate_config']['model_definition_filename']}")
            
            for dataset in surrogates[x]['datasets']:
                surrogates[x]['datasets'][dataset]['data_filename'] = engine.common.absRelativePath(surrogates[x]['datasets'][dataset]['data_filename'], rootPath)
                if not Path(surrogates[x]['datasets'][dataset]['data_filename']).exists():
                    raise ValueError(f"Model {x}'s dataset {dataset}'s data file not found in {surrogates[x]['datasets'][dataset]['data_filename']}")
            if args.debugmode: print(f'Model "{x}" and its datasets {list(surrogates[x]["datasets"].keys())} preliminary checks passed')
    
    #### BENCHMARKING

    if args.benchmark is not None:
        with open('../benchmarking.json', 'r') as file:
            benchArchive = json.load(file)
        
        bench = engine.common.CPUbenchmark()

    #### BUILD REQUEST LIST TEMPLATE-SURROGATE+DATASET
    chosenTemplates = engine.common.inputTemplates(args.templates, templates, args.debugmode, args.run)
    chosenSurrogates = engine.common.inputSurrogates(args.surrogates, surrogates, args.debugmode, args.run)
    
    #### START REQUESTS, PERFORM CHECKS
    names = {}
    fileWeights = {}
    etcs = {}
    validatorsDict = {}
    
    for template in chosenTemplates:

        benchFlag = True if args.benchmark is not None and template in benchArchive['notebook_ID-MB_ST_MT_minutes'] else False
        if benchFlag:
            ## Assemble A and y
            A = []
            y = []
            for entry in benchArchive['notebook_ID-MB_ST_MT_minutes'][template]:
                weight = float(entry.split('---')[1])
                vector = [float(elem) for elem in benchArchive['notebook_ID-MB_ST_MT_minutes'][template][entry].split('[')[1].split(']')[0].split(',')]
                A.append([weight, vector[0], vector[1]])
                y.append(vector[2])

            A = np.array(A)
            y = np.array(y)

            C = nnls(A, y)[0]


        for surrogate in chosenSurrogates:
            
            for dataset in surrogates[surrogate]['datasets']:
                name = f'{surrogate}--{dataset}--{template}'
                
                engine.common.createFolders(outputPath / surrogate / dataset / template)
                
                # Triplet definition
                validatorsDict[name] = engine.Validator(
                    templates[template],
                    surrogates[surrogate]['surrogate_config'],
                    surrogates[surrogate]['datasets'][dataset],
                    name,
                    outputPath,
                    valiLibPath,
                    ml_deployPath,
                    args.debugmode,
                    colorblind
                )
                
                # Triplet build
                validatorsDict[name].build()
                names[name] = name

                if args.benchmark is not None:
                    # Time estimation
                    fileWeights[name] = round(Path(surrogates[surrogate]['datasets'][dataset]['data_filename']).stat().st_size / 1024**2,2)

                    if benchFlag:
                        x = np.array([[fileWeights[name], bench[0], bench[1]]])
                        etcs[name] = np.dot(x,C)*60
                    else: etcs[name] = -1

    if args.debugmode:
        print('-------- List of runs')
        for x in names: print(x)
        print('--------')
    
    #### COMPLETE REQUESTS
    if args.run:
        reqs_results_ = {}

        for validator in validatorsDict:
            if args.benchmark is not None:
                if etcs[validator]!=-1: print(f'\n -- Estimated time to complete: {engine.common.secondsTohhmmss(etcs[validator])} --')
                else: print(f'\n -- Estimated time to complete: ? --')
                start = time.time()
            validatorsDict[validator].run(resume=args.resume)
            reqs_results_[validator] = validatorsDict[validator].reqs_results_

            for key, result in reqs_results_[validator].items():
                if result["raise_type"] == "error" and result["pass"] == False:
                    warnings.warn(f"Requirement '{key}, specified as mandatory, has not been fulfilled. Aborting execution...")
                    exit

            if args.benchmark is not None:
                duration = time.time() - start
                print(f' -- Real time to complete: {engine.common.secondsTohhmmss(duration)} --\n')

                if names[validator].split('--')[2] not in benchArchive['notebook_ID-MB_ST_MT_minutes']:
                    benchArchive['notebook_ID-MB_ST_MT_minutes'][names[validator].split('--')[2]] = {}
                entryName = args.benchmark+'---'+str(fileWeights[validator])
                entry = f'[{bench[0]}, {bench[1]}, {round(duration/60,2)}]'
                benchArchive['notebook_ID-MB_ST_MT_minutes'][names[validator].split('--')[2]][entryName] = entry

        if args.benchmark:
            with open('../benchmarking.json', 'w') as file:
                json.dump(benchArchive, file, indent=4)

        with open(outputPath / "output.json", 'w') as file:
            json.dump(reqs_results_, file)
    else:
        for validator in validatorsDict:
            validatorsDict[validator].writeTempfile()
        print('"run" argument not specified, stopping.')
