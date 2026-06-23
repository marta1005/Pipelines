'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
"""
Module devoted to the tools for communicating with the notebooks and pass
inputs, requirements and desired outputs.

"""
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings
import subprocess
import shutil
from os import remove
from os.path import relpath

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import h5py
from IPython.display import HTML, Markdown, display

from ..misc.metrics import DistanceMetrics
from ..misc.subsampling import maskApply, maskFilter, mask_condense

def loadDataframe(filename: Path, file_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load data from a file and return it as a Pandas DataFrame.

    :param filename: The path to the data file.
    :param file_type: Optional. The type of the data file (e.g., 'hdf5', 'csv').
     If not specified, the function will try to determine it from the file extension.
    :param kwargs: Additional options for reading the data file.
    :return: A Pandas DataFrame containing the loaded data.
    :raises ValueError: If the file_type is undefined or if there is an issue
                        with loading the data.
    """

    if file_type is None:
        # Try to determine by extension
        if filename.name.lower().endswith([".h5", ".hdf5"]):
            file_type = "hdf5"
        elif filename.name.lower().endswith([".csv"]):
            file_type = "csv"
        else:
            raise ValueError("Undefined file_type")
    df = pd.DataFrame()
    if file_type.lower() == "hdf5":
        key = kwargs.get("key")
        if key is not None:
            store = pd.HDFStore(filename, mode="r")
            df = store.select(key=key)
            store = pd.HDFStore(filename, mode="r")
            store.close()
            return df.reset_index(drop=True)
        raise ValueError(f"Missing key: {key}, for HDF5")
    if file_type.lower() == "csv":
        df = pd.read_csv(filename, **kwargs)
        return df.reset_index(drop=True)

    raise ValueError(f"Unknown file_type: {file_type}")

def data_compatibility(df1:pd.DataFrame, df2:pd.DataFrame):
    """
    Check the compatibility of two Pandas DataFrames with the same columns.

    :param df1: The first Pandas DataFrame for comparison.
    :param df2: The second Pandas DataFrame for comparison.
    :return: True if the data in df2 is compatible with df1, False otherwise.
    """
    check = True

    not_same_type = df2.columns[~(df2.dtypes == df1[df2.columns].dtypes)].to_list()

    for col in not_same_type:
        try:
            df1[col].astype(df2[col].dtype)
        except Exception:
            check = False

    return check

def hypercube_applicability_mask(data):
    """
    Generate a mask for data based on hypercube applicability criteria.

    :param data: The input data for generating the mask.
    :return: A mask for data based on hypercube applicability criteria.
    """
    ranges = []
    if data.numFlag:
        ranges += [data.inputNumRange[col].tolist()
                   for col in data.inputNumRange.columns]
    if data.catFlag:
        ranges += [data.inputCatRange[col].tolist()[0]
                   for col in data.inputCatRange.columns]
    criteriaCat = [False] * len(data.inputsNum) + [True] * len(data.inputsCat)

    hypercube_mask = mask_condense(maskFilter(data.x[data.inputsNum+data.inputsCat],
                                              criteria=ranges,
                                              fullperRow=True,
                                              criteriaCat=criteriaCat))

    return hypercube_mask

def custom_applicability_mask(data, template_tools, verbose=True):
    """
    Generate a custom applicability mask for data.

    :param data: The input data for generating the mask.
    :param template_tools: An instance of the TemplateTools class.
    :param verbose: Optional. If True, display the number of points fulfilling
                    the applicability criteria.
    :return: A custom applicability mask for data.
    """
    criteriaCat = [col in data.outputsCat for col in data.yh.columns]
    criteria = [list(data.yh[col].unique()) if col in data.outputsCat else template_tools.inputs_dict["custom_output_applicability"] for col in data.yh.columns]
    customapp_mask = maskFilter(data.yh, criteria=criteria, criteriaCat=criteriaCat, fullperRow=False)
    if verbose:
        n_samples = (customapp_mask.sum(axis=1) == 0).sum()
        display(Markdown("A total of %i points fulfill $y_h > y_{max}$ (%.2f%% of the dataset)"%(n_samples, n_samples/data.x.shape[0]*100)))

    return customapp_mask

def custom_error_mask(data, metric, template_tools, verbose=True):
    """
    Generate a custom error mask for data based on a given metric.

    :param data: The input data for generating the error mask.
    :param metric: The error metric function for evaluating data.
    :param template_tools: An instance of the TemplateTools class.
    :param verbose: Optional. If True, display the number of points fulfilling the error criteria.
    :return: A custom error mask for data.
    """
    criteriaCat = [col in data.outputsCat for col in data.yh.columns]
    criteria = [list(data.yh[col].unique()) if col in data.outputsCat else [template_tools.inputs_dict["custom_error_filtering"], np.inf] for col in data.yh.columns]
    customerr_mask = maskFilter(np.abs(metric(data.y, data.yh)), criteria=criteria, criteriaCat=criteriaCat, fullperRow=True)
    if verbose:
        n_samples = (customerr_mask.sum(axis=1) == 0).sum()
        display(Markdown("A total of %i points fulfils $E < E_{min}$ (%.2f%% of the dataset)"%(n_samples, n_samples/data.x.shape[0]*100)))
    return customerr_mask

def mldeploy_applicability_mask(data):
    """
    Generate a mask for data based on mldeploy applicability criteria.

    :param data: The input data for generating the mask.
    :return: A mask for data based on mldeploy applicability criteria.
    """
    output_ranges = [[]]*len(data.yh.columns)
    if data.numOutFlag:
        for col in data.outputNumRange.columns:
            if col in data.yh.columns:
                i = data.yh.columns.get_loc(col)
                output_ranges[i] = data.outputNumRange[col].tolist()
    if data.catOutFlag:
        for col in data.outputCatRange.columns:
            if col in data.yh.columns:
                i = data.yh.columns.get_loc(col)
                output_ranges[i] = data.outputCatRange[col].tolist()[0]

    criteriaCat = [col in data.outputsCat for col in data.yh.columns]
    mldepapp_mask = maskFilter(data.yh, criteria=output_ranges, criteriaCat=criteriaCat)

    return mldepapp_mask

def check_preserve_ranges(data, mask, verbose=True, tol=1e-4):
    """
    Check if the variable ranges are preserved after applying a mask to the data.

    :param data: The input data.
    :param mask: The mask to be applied to the data.
    :param verbose: Optional. If True, display the variable ranges before and after masking.
    :param tol: Optional. The tolerance for range preservation.
    :return: True if variable ranges are preserved, False otherwise.
    """
    if data.numFlag:
        aux_series = np.abs(maskApply(data.x, mask, 'all', to_numpy=False)[data.inputsNum].min() - data.inputNumRange.loc["min", :]) / np.abs(data.inputNumRange.loc["min", :]) <= tol
        num_check = aux_series & (np.abs((maskApply(data.x, mask, 'all', to_numpy=False)[data.inputsNum].max() - data.inputNumRange.loc["max", :])) / np.abs(data.inputNumRange.loc["max", :]) <= tol)
        num_check = num_check.tolist()

        if not all(num_check) and verbose:
            multi_index = pd.MultiIndex(levels=[[], []], codes=[[], []])
            num_table = pd.DataFrame(index=multi_index, dtype='object')
            for i, chk in enumerate(num_check):
                if not chk:
                    col = data.inputsNum[i]
                    num_table.loc[("Min", "Original"), col] = data.inputNumRange.loc["min", col]
                    num_table.loc[("Min", "Masked"), col] = maskApply(data.x, mask, 'all', to_numpy=False)[col].min()
                    num_table.loc[("Max", "Original"), col] = data.inputNumRange.loc["max", col]
                    num_table.loc[("Max", "Masked"), col] = maskApply(data.x, mask, 'all', to_numpy=False)[col].max()

            if not num_table.empty:
                num_table = num_table.style.set_caption("Variable ranges before and after masking (numerical)")
                display(num_table)
            else:
                display("No differences on numerical variables before and after masking.")
    else:
        num_check = [True]

    if data.catFlag:
        cat_check = []
        for col in data.inputCatRange:
            cat_check.append(pd.Series(data.inputCatRange.loc['discrete', col]).isin(maskApply(data.x, mask, 'all', to_numpy=False)[col]).all())

        if not all(cat_check) and verbose:
            cat_to_check = [data.inputsCat[k] for k, v in enumerate(cat_check) if v]
            cat_table = pd.DataFrame(pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []]),
                                                  dtype='object'))
            for i, chk in enumerate(cat_to_check):
                if not chk:
                    col = data.inputsCat[i]
                    cat_table.at["Missing values", col] = pd.Series(data.inputCatRange.loc['discrete', col])[~pd.Series(data.inputCatRange.loc['discrete', col]).isin(maskApply(data.x, mask, 'all', to_numpy=False)[col])].unique().tolist()

            if not cat_table.empty:
                cat_table = cat_table.style.set_caption("Variable ranges before and after masking (categorical)")
                display(cat_table)
            else:
                display("No differences on categorical variables before and after masking")
    else:
        cat_check = [True]


    return all(cat_check) and all(num_check)


class TemplateTools:
    """
    Helper class for working with templates and requirements.

    :param options: The template options dictionary.
    :param print_flag: Optional. If True, enable printing messages. Default is False.
    """
    def __init__(self, options: object, print_flag: bool = False):
        self.template_name = options["NAME"].split("--")[2]
        self.model_name = options["NAME"].split("--")[0]
        self.dataset_name = options["NAME"].split("--")[1]

        # Colorblind dictionary
        self.colorblind = {
            'pair': options['colorblindPair'],
            'neutral': options['colorblindNeutral'],
            'divergingCmap': plt.cm.__getattribute__(options['colorblindDivergingCmap']),
            'divergingCmap_r': plt.cm.__getattribute__(options['colorblindDivergingCmap_r']),
            'sequentialCmap': plt.cm.__getattribute__(options['colorblindSequentialCmap']),
            'sequentialCmap_r': plt.cm.__getattribute__(options['colorblindSequentialCmap_r'])
        }

        self.template_options = options

        self.print_flag = print_flag
        self.reqs_results = {}

        self.figNumber_ = 0
        self.tableNumber_ = 0

    def nb_startup(self):
        """
        Perform necessary startup operations for the notebook.

        :return: An instance of the TemplateTools class.
        """
        def _preprocess_dataset(dataset_key):
            if self.print_flag:
                display(Markdown(f"#### Loading surrogate output and input dataframes \
                                 from template input {dataset_key}"))
            for key in ["inputNumRange", "outputNumRange", "inputCatRange", "outputCatRange"]:

                if pd.api.types.is_string_dtype(inputs_dict[dataset_key][key].columns):
                    inputs_dict[dataset_key][key].columns = inputs_dict[dataset_key][key].columns.str.replace('_', ' ')

            inputs_dict[dataset_key]["inputsNum"] = [string.replace('_', ' ') for string in inputs_dict[dataset_key]["inputNumRange"].keys()]
            inputs_dict[dataset_key]["inputsCat"] = [string.replace('_', ' ') for string in inputs_dict[dataset_key]["inputCatRange"].keys()]
            inputs_dict[dataset_key]["inputs"] = inputs_dict[dataset_key]["inputsNum"] + inputs_dict[dataset_key]["inputsCat"]

            inputs_dict[dataset_key]["outputsNum"] = [string.replace('_', ' ') for string in inputs_dict[dataset_key]["outputNumRange"].keys()]
            inputs_dict[dataset_key]["outputsCat"] = [string.replace('_', ' ') for string in inputs_dict[dataset_key]["outputCatRange"].keys()]
            inputs_dict[dataset_key]["outputs"] = inputs_dict[dataset_key]["outputsNum"] + inputs_dict[dataset_key]["outputsCat"]

            inputs_dict[dataset_key]["catFlag"] = len(inputs_dict[dataset_key]["inputsCat"]) > 0
            inputs_dict[dataset_key]["numFlag"] = len(inputs_dict[dataset_key]["inputsNum"]) > 0

            inputs_dict[dataset_key]["catOutFlag"] = len(inputs_dict[dataset_key]["outputsCat"]) > 0
            inputs_dict[dataset_key]["numOutFlag"] = len(inputs_dict[dataset_key]["outputsNum"]) > 0

            dataset = loadDataframe(self.template_options["data_filename"], **self.template_options["dataframe_options"])
            dataset.columns = dataset.columns.str.replace('_', ' ')

            inputs_dict[dataset_key]["x"] = dataset[inputs_dict[dataset_key]["inputs"]]
            inputs_dict[dataset_key]["y"] = dataset[inputs_dict[dataset_key]["outputs"]].fillna(self.template_options.get("na", False))

            inputs_dict[dataset_key] = SurrogateIO(inputs_dict[dataset_key])
            inputs_dict[dataset_key].__getattribute__('yh').columns = inputs_dict[dataset_key].__getattribute__('yh').columns.str.replace('_', ' ')

            assert len(inputs_dict[dataset_key].__getattribute__('y').columns.intersection(inputs_dict[dataset_key].__getattribute__('yh').columns)) == len(inputs_dict[dataset_key].__getattribute__('y').columns), "y and yh must have the same columns"

            # Make sure that the order of the columns is the same
            setattr(inputs_dict[dataset_key], 'yh', inputs_dict[dataset_key].__getattribute__('yh')[inputs_dict[dataset_key].__getattribute__("outputs")])
            setattr(inputs_dict[dataset_key], 'y', inputs_dict[dataset_key].__getattribute__('y')[inputs_dict[dataset_key].__getattribute__("outputs")])


        if self.print_flag:
            display(HTML("<style>.container { width:100% !important; }</style>"))
            display(Markdown("#### Loaded project libraries"))
            display(Markdown("#### Parameters of the validation"))
            self.options_dump()

        inputs_dict = self.load_inputs()

        if "surrogate_io" in inputs_dict:
            _preprocess_dataset("surrogate_io")

        if "surrogate_io_train" in inputs_dict:
            _preprocess_dataset("surrogate_io_train")

        metric_sign: float = inputs_dict.get("distance_metric_sign", 1.0)
        metric_neutral: float = inputs_dict.get("distance_metric_neutral", "NaN")
        if "distance_metric_expr" in inputs_dict:
            if self.print_flag:
                display(Markdown("#### Loading metric"))

            if "distance_metric_name" not in inputs_dict:
                inputs_dict["distance_metric_name"] = inputs_dict["distance_metric_expr"]

            self.metric = DistanceMetrics(inputs_dict["distance_metric_name"]).\
                          define_metric(inputs_dict["distance_metric_expr"])
        else:
            self.metric = DistanceMetrics("residual")

        self.metric.sign = metric_sign
        if metric_neutral != "NaN":
            self.metric.neutral = metric_neutral

        self.inputs_dict = inputs_dict
        return self

    def options_dump(self):
        """
        Display the template options in a structured format.
        """
        print(json.dumps(self.template_options, indent=2))

    def load_inputs(self):
        """
        Load input data specified in the template options.

        :return: A dictionary containing the loaded input data.
        """
        input_dict = {}
        for input_key in self.template_options["inputs"]:
            raw_input = self.template_options["inputs"][input_key]

            if all(field in raw_input for field in ["from", "value"]):
                if raw_input["from"] is not None and raw_input["value"] is not None:
                    raise ValueError(
                        f"Fields 'from' and 'value' are specified for input '{input_key}'"
                        "Please, use only one.\nStopping execution..."
                    )

            if "from" in raw_input:
                if raw_input["from"] is not None:
                    input_dict[input_key] = self._load_input_from_output(raw_input)

            if "value" in raw_input:
                if raw_input["value"] is not None:
                    input_dict[input_key] = self._load_input_from_value(raw_input)

            if "from_file" in raw_input:
                if raw_input["from_file"] is not None:
                    input_dict[input_key] = loadDataframe(raw_input["from_file"], **raw_input["file_options"])

        return input_dict

    def _load_input_from_output(self, raw_input):
        data_from = dict(zip(["model", "dataset", "template", "variable"], raw_input["from"].split(":")))

        if data_from["model"] == "":
            data_from["model"] = self.model_name
        if data_from["dataset"] == "":
            data_from["dataset"] = self.dataset_name

        path_to_file = Path(self.template_options["OUTPUT_PATH"]) / ("intermediate/" + data_from["model"] + ".h5")
        group_name = "/".join([str(value) for value in list(data_from.values())[1:-1]])

        if path_to_file.is_file():
            if data_from["variable"] == "":
                with h5py.File(str(path_to_file), "r") as file:
                    vars_to_read = list(file[group_name].keys())
            else:
                vars_to_read = data_from["variable"]

            if not isinstance(vars_to_read, list):
                return self.__read_hdf(str(path_to_file), group_name)

            read_vars = {}
            for var in vars_to_read:
                path_to_var = group_name + "/" + var
                read_vars[var] = self.__read_hdf(str(path_to_file), path_to_var)
            return read_vars
        raise FileExistsError(f"The file {path_to_file} does not exist")

    def _load_input_from_file(self, raw_input, filetype: str = None):
        if filetype is None:
            raise ValueError("Undefined file_type")

        file_data = raw_input["file"].split(":")
        filename = file_data[0]

        if not Path(filename).is_file():
            raise FileExistsError(f"The file {filename} does not exist")

        if filetype == "hdf5":
            if len(file_data) > 1:
                group_name = file_data[1:]
            else:
                group_name = "/"
            return self.__read_hdf(filename, group_name)
        if filetype == "csv":
            return pd.read_csv(filename).reset_index(drop=True)

        raise ValueError(f"Unknown file_type: {filetype}")

    def _load_input_from_value(self, raw_input):
        return raw_input["value"]

    def save_outputs(self, output_dict: dict, print_flag: bool = None):
        """
        Save output data to an HDF5 file.

        :param output_dict: A dictionary containing output data to be saved.
        :param print_flag: Optional. If True, enable printing messages. Default is False.
        """
        _print_flag = print_flag if print_flag is not None else self.print_flag

        path_to_file = Path(self.template_options["OUTPUT_PATH"]) / ("intermediate/" + self.model_name + ".h5")

        for key, val in output_dict.items():
            group_name = "/" + self.dataset_name + "/" + self.template_name + "/" + str(key)
            if isinstance(val,(pd.DataFrame, pd.Series)):
                with pd.HDFStore(str(path_to_file), mode="a") as store:
                    store.put(group_name, val)
                    store.get_storer(group_name).attrs.is_pandas = True
            else:
                with h5py.File(str(path_to_file), "a") as file:
                    if group_name in file:
                        del file[group_name]
                    file.create_dataset(group_name, data=val)
            if _print_flag:
                print("Saved ", key)

        try:
            # Repack HDF5 file to avoid increase in its size with each run
            path_temp_file = Path(self.template_options["OUTPUT_PATH"]) / ("intermediate/" + self.model_name + "_temp.h5")
            shutil.copy(str(path_to_file), str(path_temp_file))
            repack_cmd = ["ptrepack", "-o", "--chunkshape=auto", "--propindexes", relpath(path_temp_file), relpath(path_to_file)]
            if subprocess.call(repack_cmd) != 0:
                print("Error repacking HDF5 file")
            remove(path_temp_file)
        except Exception as e:
            print(f"Error while repacking HDF5 file: {str(e)}")
            print("Continuing without repacking. This might increase the HDF5 in size with every run")

    def check_requirement(self, req_key: str, pass_condition: bool, print_flag: bool = False):
        """
        Check if a specific requirement is fulfilled and handle it according to its type.

        :param req_key: The key of the requirement in the template options.
        :param pass_condition: True if the requirement is fulfilled, False otherwise.
        :param print_flag: Optional. If True, enable printing messages. Default is False.
        """
        _print_flag = print_flag if print_flag is not None else self.print_flag

        requirement = self.template_options["requirements"][req_key]
        req_name = requirement["name"]
        self.reqs_results[req_key] = {}
        self.reqs_results[req_key]["raise_type"] = requirement["raise_type"]

        if not requirement.get("active", True):
            self.reqs_results[req_key]["pass"] = "Inactive"
            return

        self.reqs_results[req_key]["pass"] = pass_condition

        if pass_condition and _print_flag:
            self.__display_requirement_msg(requirement, pass_condition)

        elif not pass_condition:
            if requirement["raise_type"] == "error":
                warnings.warn(
                    f"Requirement '{req_name}', specified as mandatory, has not been fulfilled. Stopping execution...", RuntimeWarning
                )
                raise ValueError(f"Requirement '{req_name}', specified as mandatory, has not been fulfilled. Stopping execution...")

            if requirement["raise_type"] == "warning":
                warnings.warn(f"Requirement '{req_name}' has not been fulfilled.", RuntimeWarning)
            elif requirement["raise_type"] == "info":
                self.__display_requirement_msg(requirement, pass_condition)

    def __display_requirement_msg(self, req, passed):
        if passed:
            display(Markdown(f'<p style="background-color:{self.colorblind["pair"][0]}; color:white; padding:1em">{req["name"]} passed.</p>'))
        else:
            display(Markdown(f'<p style="background-color:{self.colorblind["pair"][1]}; color:white; padding:1em">Requirement "{req["name"]}" has not been fulfilled.<br/>Requirement description: {req["description"]}</p>'))

    def reqs_results_to_table(self):
        """
        Display a table summarizing the results of requirement checks.
        """

        dtypes_tables = {"Type": pd.Series(dtype="str"),
                         "Check": pd.Series(dtype="str")}
        df = pd.DataFrame(dtypes_tables)
        req_type: str = ""
        for req_key, req in self.template_options["requirements"].items():
            if self.reqs_results[req_key]["raise_type"] == "error":
                req_type = "Mandatory"
            elif self.reqs_results[req_key]["raise_type"] in ["warning", "info"]:
                req_type = "Optional"

            df.loc[req["name"], "Type"] = req_type
            df.loc[req["name"], "Check"] = self.reqs_results[req_key]["pass"]

        table = df.style.set_caption("Requirement check")
        table.map(lambda x: 'color: white', subset=pd.IndexSlice[:, "Check"])
        table.map(lambda x: 'color: grey' if x == "Inactive" else (f'background-color: {self.colorblind["pair"][0]}' if x else f'background-color: {self.colorblind["pair"][1]}'), subset=pd.IndexSlice[:, "Check"])

        display(table)

    def __read_hdf(self, filename, group_name):
        try:
            # Check if variable was saved with pandas
            with h5py.File(str(filename), "r") as file:
                if not group_name in file:
                    raise KeyError(f"The file {filename} does not contain the group '{group_name}'")

                is_pandas = "is_pandas" in file[group_name].attrs.keys()

            # Read with Pandas
            if is_pandas:
                input_value = pd.read_hdf(str(filename), key=group_name)
                return input_value

            # Read with h5py
            with h5py.File(str(filename), "r") as file:
                input_value = file[group_name][:]
                return input_value
        except Exception as error:
            raise ValueError(f"Data could not be read from file: {error}")\
                from error

    def exportAppendix(self, data: Union[Styler, Figure, pd.DataFrame]):
        """
        Export and save figures or tables to an appropriate directory.

        :param data: A Pandas Styler or Matplotlib Figure to be saved.
        """
        designatedPath = fr'{self.template_options["OUTPUT_PATH"]}/{self.model_name}/{self.dataset_name}/{self.template_name}/{"media"}'
        Path(designatedPath).mkdir(parents=True, exist_ok=True)

        if isinstance(data, Styler):
            data.to_html(Path(designatedPath + fr'/table{self.tableNumber_}.html'))
            self.tableNumber_ += 1
        elif isinstance(data, Figure):
            data.savefig(Path(designatedPath + fr'/figure{self.figNumber_}.jpg'))
            self.figNumber_ += 1
        elif isinstance(data, pd.DataFrame):
            data.style.to_html(Path(designatedPath + fr'/table{self.tableNumber_}.html'))
            self.tableNumber_ += 1
        else:
            print(f'No figure or table was saved, check format: {type(data)}')


class SurrogateIO:
    """
    Class for handling surrogate input and output data.

    :param io_dict: A dictionary containing input and output data for the surrogate.
    """
    def __init__(self, io_dict:dict):
        for key, val in io_dict.items():
            self.__setattr__(key, val)

    def __eq__(self, other: Dict[Any, Any]):
        """
        Check if two SurrogateIO objects are equal.

        :param other: Another SurrogateIO object for comparison.
        :return: True if the two objects are equal, False otherwise.
        """
        self_set = set(list(self.keys()))
        other_set = set(list(other.keys()))
        sym_dif_set = self_set ^ other_set
        if len(sym_dif_set) == 0:
            check = []
            for key in self.keys():
                if hasattr(getattr(self, key), "shape") and\
                   hasattr(getattr(self, key), "shape"):
                    if getattr(self, key).shape != getattr(other, key).shape:
                        return False

                try:
                    comparison = getattr(self, key) == getattr(other, key)
                    if hasattr(comparison, '__iter__'):
                        comparison = all(comparison)
                    check.append(comparison)
                except Exception as error:
                    warnings.warn(f"Could not perform equal comparison between\
                                  SurrogateIO objects. Returning false. {error}")
                    return False

            return all(check)
        return False

    def add_fields(self, new_fields:dict):
        """
        Add new fields to the SurrogateIO object.

        :param new_fields: A dictionary containing new fields to be added.
        """
        for key, val in new_fields.items():
            if hasattr(self, key):
                raise ValueError(f"Field {key} already exists!")
            self.__setattr__(key, val)

    def keys(self):
        """
        Get a list of keys (field names) in the SurrogateIO object.

        :return: A list of field names in the SurrogateIO object.
        """
        return [attr for attr in dir(self) if not (attr.startswith('__') or attr=='keys' or attr=='add_fields')]
