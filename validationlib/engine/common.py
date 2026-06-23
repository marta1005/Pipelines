'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
import os
import sys
import time
import warnings
import subprocess
from pathlib import Path
from typing import List

import json
import importlib
from tqdm import tqdm
from joblib import Parallel, delayed

from .validator import Validator

libraryPath = Path(__file__).parent.resolve()

def createFolders(
    absoluteFolderPath: os.PathLike
) -> os.PathLike:
    """
    Create folders at the specified absolute path.

    This function creates folders at the specified absolute path.
    If the folders do not exist, it creates them.

    :param absoluteFolderPath: The absolute path where folders will be created.
    :type absoluteFolderPath: os.PathLike

    :return: The path to the created folders.
    :rtype: os.PathLike
    """
    Path(absoluteFolderPath).mkdir(parents=True, exist_ok=True)

def absRelativePath(
    newPath: os.PathLike,
    rootPath: os.PathLike
) -> os.PathLike:
    """
    Return an absolute or relative path based on the provided inputs.

    This function returns an absolute path if the `newPath` is already absolute.
    Otherwise, it combines `newPath` with `rootPath` to create a new relative path 
    and returns the absolute path.

    :param newPath: The new path to be resolved, which may be absolute or relative.
    :type newPath: os.PathLike

    :param rootPath: The root path used for relative path resolution.
    :type rootPath: os.PathLike

    :return: The resolved absolute path.
    :rtype: os.PathLike
    """
    if Path(newPath).is_absolute():
        return Path(newPath)
    return (rootPath / Path(newPath)).resolve()

def colorblindPalette(
    colorblindtype: bool,
    debugmode: bool
) -> dict:
    """
    Create a color palette for colorblind or non-colorblind mode.

    This function generates a color palette based on the specified colorblind mode and debug mode settings.

    :param colorblindtype: A boolean value indicating whether colorblind mode is enabled.
    :type colorblindtype: bool

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: A dictionary containing color palette information.
    :rtype: dict
    """
    # Matplotlib's default C0, C1, C2... colors are colorblind-friendly and look good, there is no need to change those.
    if colorblindtype:
        colorblind = {
            'colorblindPair': ['#009E73','#D55E00'],    #Positive-TEAL vs Negative-ORANGE
            'colorblindNeutral': '#F0E442',             #Neutral-YELLOW
            'colorblindDivergingCmap': 'seismic_r',     #Diverging Negative-RED-WHITE-BLUE-Positive
            'colorblindDivergingCmap_r': 'seismic',
            'colorblindSequentialCmap': 'cividis',      #Uniform Sequential BLUE-GRAY-YELLOW
            'colorblindSequentialCmap_r': 'cividis_r'
        }
        if debugmode: print('Colorblind mode selected')
    else:
        colorblind = {
            'colorblindPair': ['#006400','#8B0000'],    #Positive-GREEN vs Negative-RED
            'colorblindNeutral': '#003887',             #Neutral-BLUE
            'colorblindDivergingCmap': 'seismic_r',     #Diverging Negative-RED-WHITE-BLUE-Positive
            'colorblindDivergingCmap_r': 'seismic',
            'colorblindSequentialCmap': 'viridis',      #Uniform Sequential PURPLE-BLUE-GREEN-YELLOW
            'colorblindSequentialCmap_r': 'viridis_r'
        }
        if debugmode: print('Non-colorblind mode selected')

    return colorblind

def checkPackages(
    currentFilePath: os.PathLike,
    debugmode: bool
) -> None:
    """
    Check for required packages and display missing ones.

    This function checks for the presence of required packages listed in a file 
    and compares them with the installed packages. It prints a message 
    about any missing packages.

    :param currentFilePath: The current file's path used to locate the requirements file.
    :type currentFilePath: os.PathLike

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: None
    :rtype: None
    """

    requirements_path = (currentFilePath / Path("../docs/requirements/requirements.yml")).resolve()

    with requirements_path.open(mode="r") as file:
        required_packages = [line.partition('- ')[2].partition('=')[0]
            .partition('>')[0].partition('<')[0].partition('\n')[0]
            .partition(' ')[0] for line in file.readlines()[5:]]

    list_c = subprocess.check_output(['conda', 'list']).decode('utf-8')
    installed_packages = [line.split(' ')[0] for line in list_c.split('\n')[2:-1]]
    check = all(package in installed_packages for package in required_packages)
    missing_packages = list(set(required_packages) - set(installed_packages))

    if check == False: print(f'The following packages are missing:\n{", ".join(str(elem) for elem in missing_packages)}\n')
    elif debugmode: print('All packages are installed\n')

def checkMLdeployPath(
    ml_deployPath: os.PathLike,
    rootPath: os.PathLike,
    debugmode: bool
) -> None:
    """
    Check and resolve the ml_deploy library path.

    This function checks the ml_deploy library path. If it is not specified, it looks for the library in the installed packages. 
    It also resolves the path based on the provided `ml_deployPath` and `rootPath`.

    :param ml_deployPath: The path to the ml_deploy library or 'default' to use a default path.
    :type ml_deployPath: os.PathLike

    :param rootPath: The root path used for path resolution.
    :type rootPath: os.PathLike

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: The resolved ml_deploy library path.
    :rtype: os.PathLike
    """
    spec = importlib.util.find_spec('ml_deploy')
    if ml_deployPath is None:
        if spec is None:
            raise ValueError('ml_deploy wasnt found in installed packages or specified in ML_DEPLOY_PATH')

        if debugmode: print('Selected ml_deploy from installed packages')
    else:
        if ml_deployPath == 'default': ml_deployPath = '../../ml_deploy'

        ml_deployPath = absRelativePath(ml_deployPath, rootPath)
        if debugmode: print(f'Selected ml_deploy from from {ml_deployPath}')

    return ml_deployPath

def checkOutputPath(
    outputPath: os.PathLike,
    rootPath: os.PathLike,
    debugmode: bool
) -> os.PathLike:
    """
    Check and resolve the output folder path.

    This function checks the output folder path. If it is not specified, it defaults to 'validation' in the working directory. 
    It also resolves the path based on the provided `outputPath` and `rootPath`.

    :param outputPath: The specified output folder path or None to use the default.
    :type outputPath: os.PathLike

    :param rootPath: The root path used for path resolution.
    :type rootPath: os.PathLike

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: The resolved output folder path.
    :rtype: os.PathLike
    """
    if outputPath is None:
        outputPath = 'validation'
        if debugmode: print('OUTPUT_PATH not specified, defaulting to working directory')

    outputPath = absRelativePath(outputPath, rootPath)

    if debugmode: print('Output folder defined as: ',outputPath)
    # Create output / appendix / model_outputs folders if needed
    createFolders(outputPath / 'intermediate')

    return outputPath

def checkTemplates(
    templates: dict,
    rootPath: os.PathLike,
    debugmode: bool
) -> dict:
    """
    Check and validate template configurations.

    This function checks and validates template configurations. It ensures that the required files exist and returns a dictionary of templates with their configurations.

    :param templates: A dictionary of template configurations.
    :type templates: dict

    :param rootPath: The root path used for file resolution.
    :type rootPath: os.PathLike

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: A dictionary of validated template configurations.
    :rtype: dict
    """

    if templates is None:
        raise ValueError('TEMPLATES not specified. Please read the provided default config file to understand the required fields')

    for x in templates:
        # Template itself
        if templates[x]['notebook_path'] == 'default':
            templates[x]['notebook_path'] = f'../../templates/{templates[x]["template_ID"]}.ipynb'
            templates[x]['notebook_path'] = absRelativePath(templates[x]['notebook_path'], libraryPath)
        else:
            templates[x]['notebook_path'] = absRelativePath(templates[x]['notebook_path'], rootPath)

        if not Path(templates[x]['notebook_path']).exists():
            raise ValueError(x + f"'s template not found in {templates[x]['notebook_path']}")

        # Template specific config
        if templates[x]['template_config'] == 'default':
            templates[x]['template_config'] = f'../../templates/templateConfig/{templates[x]["template_ID"]}.json'
            templates[x]['template_config'] = absRelativePath(templates[x]['template_config'], libraryPath)
        else:
            templates[x]['template_config'] = absRelativePath(templates[x]['template_config'], rootPath)

        if not Path(templates[x]['template_config']).exists():
            raise ValueError(x + f"'s JSON file for template configuration not found in {templates[x]['template_config']}")

        # User config
        if "inputs" in templates[x]["user_config"].keys():
            for input_ in templates[x]["user_config"]["inputs"]:
                if "from_file" in templates[x]["user_config"]["inputs"][input_].keys():
                    templates[x]["user_config"]["inputs"][input_]["from_file"] = absRelativePath(
                        templates[x]["user_config"]["inputs"][input_]["from_file"],
                        rootPath
                    )
                    if not Path(templates[x]["user_config"]["inputs"][input_]["from_file"]).exists():
                        raise ValueError(f"Input file {templates[x]['user_config']['inputs'][input]['from_file']} not found")

        if debugmode: print(f'Template "{x}" preliminary checks passed')

    return templates

def checkSurrogates(
    surrogates: dict,
    rootPath: os.PathLike,
    debugmode: bool
) -> dict:
    """
    Check and validate surrogate configurations.

    This function checks and validates surrogate configurations. 
    It ensures that the required files exist, loads validation files, and 
    returns a dictionary of surrogates with their configurations.

    :param surrogates: A dictionary of surrogate configurations.
    :type surrogates: dict

    :param rootPath: The root path used for file resolution.
    :type rootPath: os.PathLike

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: A dictionary of validated surrogate configurations.
    :rtype: dict
    """

    if surrogates is None:
        raise ValueError('SURROGATES not specified. Please read the provided "\
                         "default config file to understand the required fields')

    for x in surrogates:
        # Model validation file
        surrogates[x]['validation_config_file'] = absRelativePath(surrogates[x]['validation_config_file'], rootPath)
        if not Path(surrogates[x]['validation_config_file']).exists():
            raise ValueError(x + f"'s validation_config_file not found in {surrogates[x]['validation_config_file']}")
        # Load validation file model_definition_filename
        with open(surrogates[x]['validation_config_file'], 'r') as file:
            surrogates[x]['surrogate_config'] = json.load(file)
        # Make sure model definition is ok
        surrogates[x]['surrogate_config']['model_definition_filename'] = absRelativePath(
            surrogates[x]['surrogate_config']['model_definition_filename'],
            surrogates[x]['validation_config_file'].parents[0]
        )
        if not Path(surrogates[x]['surrogate_config']['model_definition_filename']).exists():
            raise ValueError(x + f"'s model definition not found in {surrogates[x]['surrogate_config']['model_definition_filename']}")

        for dataset in surrogates[x]['datasets']:
            surrogates[x]['datasets'][dataset]['data_filename'] = absRelativePath(surrogates[x]['datasets'][dataset]['data_filename'], rootPath)
            if not Path(surrogates[x]['datasets'][dataset]['data_filename']).exists():
                raise ValueError(f"Model {x}'s dataset {dataset}'s data file not found in {surrogates[x]['datasets'][dataset]['data_filename']}")
        if debugmode: print(f'Model "{x}" and its datasets {list(surrogates[x]["datasets"].keys())} preliminary checks passed')

    return surrogates

def CPUbenchFunction(iters):
    """
    Simulate CPU benchmarking with a specified number of iterations.

    This function simulates CPU benchmarking by performing various calculations for a specified number of iterations.

    :param iters: The number of iterations for benchmarking.
    :type iters: int

    :return: None
    :rtype: None
    """
    for _ in range(0, iters):
        for x in range(1,1000):
            _ = 3.141592 * 2**x
        for x in range(1,10000):
            _ = float(x) / 3.141592
        for x in range(1,10000):
            _ = float(3.141592) / x

def CPUbenchmark(
        STiters: int=200,
        STtrials: int=10,
        MTiters: int=50,
        MTtrials: int=200
    ) -> List[float]:
    """
    Perform CPU benchmarking for single-core and multi-core performance.

    This function performs CPU benchmarking to measure single-core and multi-core performance.

    :param STiters: The number of iterations for single-core benchmarking.
    :type STiters: int

    :param STtrials: The number of trials for single-core benchmarking.
    :type STtrials: int

    :param MTiters: The number of iterations for multi-core benchmarking.
    :type MTiters: int

    :param MTtrials: The number of trials for multi-core benchmarking.
    :type MTtrials: int

    :return: A list containing single-core and multi-core performance scores.
    :rtype: list
    """
    print('------ Starting CPU benchmark\n')

    average_benchmark = 0
    for _ in tqdm(range(STtrials)):
        start = time.time()
        CPUbenchFunction(STiters)
        duration = time.time() - start

        average_benchmark += duration
    singleScore = round((average_benchmark / STtrials)/0.33*100, 2)


    # Pre-execution
    Parallel(n_jobs=-1)(delayed(CPUbenchFunction)(MTiters) for _ in range(os.cpu_count()))

    start = time.time()
    Parallel(n_jobs=-1)(delayed(CPUbenchFunction)(MTiters) for _ in tqdm(range(MTtrials)))
    multiScore = round(((time.time() - start) / (MTtrials))/0.014*100, 2)

    print(f'\nResults (Lower is better)\n  {singleScore}% - Single-core performance average ({STtrials} trials)')
    print(f'  {multiScore}% - Multi-core ({os.cpu_count()} threads) performance average ({MTtrials} trials)')
    print('\n------ CPU benchmark finished')
    return singleScore, multiScore

def secondsTohhmmss(
    seconds: float
    ) -> str:
    """
    Convert seconds to a formatted time string.

    This function converts a duration in seconds to a formatted time string in the format "hh:mm:ss".

    :param seconds: The duration in seconds.
    :type seconds: float

    :return: The formatted time string.
    :rtype: str
    """

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f'{int(hours)}h {int(minutes)}min {int(seconds)}s'

def inputTemplates(
    chosenTemplates: list,
    availableTemplates: dict,
    debugmode: bool,
    run: bool
) -> list:
    """
    Select and configure template configurations.

    This function allows the user to select and configure template configurations, either by choosing from available templates or specifying custom settings.

    :param chosenTemplates: A list of selected template names or None for interactive selection.
    :type chosenTemplates: list

    :param availableTemplates: A dictionary of available template configurations.
    :type availableTemplates: dict

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :param run: A boolean value indicating whether to run all selected templates.
    :type run: bool

    :return: A list of chosen template names.
    :rtype: list
    """

    # Choose templates
    if chosenTemplates is None:
        print('\n---No chosen templates were specified:---')
        names = []
        #print('-2 --> Save temporal file and exit')
        if run: print('-1 --> Run all notebooks')
        for i, elem in enumerate(availableTemplates):
            print(f'{i} --> Select {elem} template')
            names.append(elem)

        answer = int(input())
        #if answer==-2:
        #    chosenTemplates = []
        if answer == -1 and run:
            chosenTemplates = list(availableTemplates.keys())
        else:
            chosenTemplates = [names[answer]]
    elif availableTemplates is None:
        raise ValueError('No templates were defined in the configuration file.')
    else:
        for x in chosenTemplates:
            if x not in availableTemplates:
                raise ValueError(x + ' template name not found')

    print('Templates selected:\n'+str(chosenTemplates)+'\n-')

    # Debug template configuration
    if debugmode:
        for x in chosenTemplates:
            with open(availableTemplates[x]['template_config'], "r") as file:
                templateConfig = json.loads(file.read())
            print(x,'config file:',json.dumps(templateConfig, indent=2))
    return chosenTemplates

def inputSurrogates(
    chosenSurrogates: list,
    availableSurrogates: dict,
    debugmode: bool,
    run: bool
) -> list:
    """
    Select and configure surrogate configurations.

    This function allows the user to select and configure surrogate configurations, either by choosing from available surrogates or specifying custom settings.

    :param chosenSurrogates: A list of selected surrogate names or None for interactive selection.
    :type chosenSurrogates: list

    :param availableSurrogates: A dictionary of available surrogate configurations.
    :type availableSurrogates: dict

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :param run: A boolean value indicating whether to run all selected surrogates.
    :type run: bool

    :return: A list of chosen surrogate names.
    :rtype: list
    """

    # Choose surrogates
    if chosenSurrogates is None:
        print('\n---No chosen surrogates were specified:---')
        names = []
        if run: print('-1 --> Run all surrogates')
        for i, elem in enumerate(availableSurrogates):
            print(f'{i} --> Select {elem} surrogate (model+dataset)')
            names.append(elem)

        answer = int(input())
        if answer==-1 and run:
            chosenSurrogates = list(availableSurrogates.keys())
        else:
            chosenSurrogates = [names[answer]]
    elif availableSurrogates is None:
        raise ValueError('No surrogates were defined in the configuration file.')
    else:
        unfound: bool = False
        unfound_error: str = ""
        for x in chosenSurrogates:
            if x not in availableSurrogates:
                unfound = True
                unfound_error += x + ' surrogate name not found\n'
        if unfound:
            raise ValueError(unfound_error)

    print('Surrogates selected:\n'+str(chosenSurrogates)+'\n-')

    # Debug template configuration
    if debugmode:
        for x in chosenSurrogates:
            with open(availableSurrogates[x]['validation_config_file'], "r") as file:
                surrogatesConfig = json.loads(file.read())
            print(x,'validation config file:',json.dumps(surrogatesConfig, indent=2))
            for dataset in availableSurrogates[x]['datasets']:
                print(f'Loaded dataset "{dataset}" for "{x}" model')
    return chosenSurrogates

def buildTriplets(
    chosenTemplates: list,
    chosenSurrogates: list,
    templates: dict,
    surrogates: dict,
    outputPath: os.PathLike,
    valiLibPath: os.PathLike,
    ml_deployPath: os.PathLike,
    colorblind: dict,
    debugmode: bool
) -> dict:
    """
    Build and validate triplets for the selected templates and surrogates.

    This function builds and validates triplets for the selected templates and surrogates. Triplets consist of templates, surrogates, and datasets.

    :param chosenTemplates: A list of selected template names.
    :type chosenTemplates: list

    :param chosenSurrogates: A list of selected surrogate names.
    :type chosenSurrogates: list

    :param templates: A dictionary of template configurations.
    :type templates: dict

    :param surrogates: A dictionary of surrogate configurations.
    :type surrogates: dict

    :param outputPath: The output folder path.
    :type outputPath: os.PathLike

    :param valiLibPath: The Validator library path.
    :type valiLibPath: os.PathLike

    :param ml_deployPath: The path to the ml_deploy library.
    :type ml_deployPath: os.PathLike

    :param colorblind: Colorblind mode configuration.
    :type colorblind: dict

    :param debugmode: A boolean value indicating whether debug mode is enabled.
    :type debugmode: bool

    :return: A dictionary of Validator objects for each triplet.
    :rtype: dict
    """

    names = {}
    #fileWeights = {}
    #etcs = {}
    validatorsDict = {}

    for template in chosenTemplates:

        # benchFlag = True if args.benchmark is not None and template in benchArchive['notebook_ID-MB_ST_MT_minutes'] else False
        # if benchFlag:
        #     ## Assemble A and y
        #     A = []
        #     y = []
        #     for entry in benchArchive['notebook_ID-MB_ST_MT_minutes'][template]:
        #         weight = float(entry.split('---')[1])
        #         vector = [float(elem) for elem in benchArchive['notebook_ID-MB_ST_MT_minutes'][template][entry].split('[')[1].split(']')[0].split(',')]
        #         A.append([weight, vector[0], vector[1]])
        #         y.append(vector[2])

        #     A = np.array(A)
        #     y = np.array(y)

        #     C = nnls(A, y)[0]


        for surrogate in chosenSurrogates:

            for dataset in surrogates[surrogate]['datasets']:
                name = f'{surrogate}--{dataset}--{template}'

                createFolders(outputPath / surrogate / dataset / template)

                # Triplet definition
                validatorsDict[name] = Validator(
                    templates[template],
                    surrogates[surrogate]['surrogate_config'],
                    surrogates[surrogate]['datasets'][dataset],
                    name,
                    outputPath,
                    valiLibPath,
                    ml_deployPath,
                    debugmode,
                    colorblind
                )

                # Triplet build
                validatorsDict[name].build()
                names[name] = name

                # if args.benchmark is not None:
                #     # Time estimation
                #     fileWeights[name] = round(Path(surrogates[surrogate]['datasets'][dataset]['data_filename']).stat().st_size / 1024**2,2)

                #     if benchFlag:
                #         x = np.array([[fileWeights[name], bench[0], bench[1]]])
                #         etcs[name] = np.dot(x,C)*60
                #     else: etcs[name] = -1

    if debugmode:
        print('-------- List of triplets')
        for x in names: print(x)
        print('--------')

    return validatorsDict

def runTriplets(
    validatorsDict: dict,
    outputPath,
    run: bool,
    resume: bool
) -> None:
    """
    Run the selected triplets and generate validation reports.

    This function runs the selected triplets, generates validation reports, and saves the results to the specified output folder.

    :param validatorsDict: A dictionary of Validator objects representing triplets.
    :type validatorsDict: dict

    :param outputPath: The output folder path.
    :type outputPath: os.PathLike

    :param run: A boolean value indicating whether to run the selected triplets.
    :type run: bool

    :param resume: A boolean value indicating whether to resume from previous runs.
    :type resume: bool

    :return: None
    :rtype: None
    """

    if run:
        reqs_results_ = {}
        for validator in validatorsDict:
            # if args.benchmark is not None:
            #     if etcs[validator]!=-1: print(f'\n -- Estimated time to complete: {engine.common.secondsTohhmmss(etcs[validator])} --')
            #     else: print(f'\n -- Estimated time to complete: ? --')
            #     start = time.time()

            validatorsDict[validator].run(resume=resume)
            reqs_results_[validator] = validatorsDict[validator].reqs_results_

            for key, result in reqs_results_[validator].items():
                if result["raise_type"] == "error" and result["pass"] == False:
                    warnings.warn(f"Requirement '{key}, specified as mandatory, has not been fulfilled. Aborting execution...")
                    sys.exit()

            # if args.benchmark is not None:
            #     duration = time.time() - start
            #     print(f' -- Real time to complete: {engine.common.secondsTohhmmss(duration)} --\n')

            #     if names[validator].split('--')[2] not in benchArchive['notebook_ID-MB_ST_MT_minutes']:
            #         benchArchive['notebook_ID-MB_ST_MT_minutes'][names[validator].split('--')[2]] = {}
            #     entryName = args.benchmark+'---'+str(fileWeights[validator])
            #     entry = f'[{bench[0]}, {bench[1]}, {round(duration/60,2)}]'
            #     benchArchive['notebook_ID-MB_ST_MT_minutes'][names[validator].split('--')[2]][entryName] = entry

        # if args.benchmark:
        #     with open('../benchmarking.json', 'w') as file:
        #         json.dump(benchArchive, file, indent=4)

        with open(outputPath / "output.json", 'w') as file:
            json.dump(reqs_results_, file)
    else:
        for validator in validatorsDict:
            validatorsDict[validator].writeTempfile()
        print('"run" argument not specified, stopping.')

# WIP
#### BENCHMARKING. Every function related to benchmarking has been commented out
# if args.benchmark is not None:
#     with open('../benchmarking.json', 'r') as file:
#         benchArchive = json.load(file)

#     bench = engine.common.CPUbenchmark()
