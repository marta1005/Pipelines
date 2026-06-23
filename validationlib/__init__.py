'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''

"""
# Validation library
<hr style="border:3px solid"> </hr>
Welcome to the validation library used to make the notebooks and HTMLs.
### Introduction
The library is composed of 5 main subpackages; misc, models, plots, tables & tests
### Acknowledgements
Airbus & UPM joint collaboration

<br />
## -
<hr style="border:2px solid"> </hr>
"""
__version__ = '2025.03.31'

# MISC - Subpackage containing miscellanous modules
from . import misc

# PLOTS - Plotting modules
from . import plots

# TABLES - Tables and other graphing modules
from . import tables

# TESTS - Statistical and other performance evaluation tests
from . import tests
