import os
import logging
import sys
from pandas import DataFrame
import random
import warnings
import numpy as np
import logging

from pathlib import Path
import surrogate_factory as sf

import mlflow
import shutil
import sklearn




@sf.node
def upload_model(workflow):
    """Upload model wrapper
    Store the model inference as defined in the archiving solution/Artifactory
    Model wrapper ID must be added in the Metadata of the file before upload
    Args:
        workflow (_type_): Workflow class
    """


    ### not implemented
    ## do mlflow onnx...


    
    pass



