from doe import (BoundedVariable, Constant, Constraint, CatalogVariable, DependentVariable, DoE,
                 concat_inputs, split, UniqueName, Function)
import numpy as np
import json
from toolz import compose

import surrogate_factory as sf
import sympy as sp





@sf.node
def define_doe(workflow):
    """_summary_
    Update the flow_variables with the DoE definition. Dependant variables, constants, ...
    It uses the "Create_Design_of_Experiments" excel file + some other variables to declare the DoE.
    The outcome has to be added in the Flow_variables and it will be used in the next steps to build the data.
    Args:
        flow_variables (_type_): flow_variables

    Returns:
        sf.FlowVariables: Nothing

    """
   
    pass