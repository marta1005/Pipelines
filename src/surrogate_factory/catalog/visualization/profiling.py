import pandas as pd
import numpy as np

from pathlib import Path
from toolz.curried import get, first, compose
from toolz.curried.operator import eq
from collections import UserDict
# import surrogate_factory as sf



def profiling_report(data, title ="Profiling Report" ):

    from ydata_profiling import ProfileReport
    profile = ProfileReport(data, title=title)
    profile.to_notebook_iframe()

    profile.to_file(f"EDA_{title}")

    return profile

