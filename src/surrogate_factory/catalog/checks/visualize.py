import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pair_plot(input_table, sample = True, density = False):
    if sample:
        input_table = input_table.sample(frac = 0.1)
    if density:
        diag_kind = 'kde'
    else:
        diag_kind = 'auto'
    return sns.pairplot(input_table, diag_kind = diag_kind)

def box_plot(input_table, sample = True):
    if sample:
        input_table = input_table.sample(frac = 0.1)
    f, ax = plt.subplots()
    ax.boxplot(input_table)
    return ax