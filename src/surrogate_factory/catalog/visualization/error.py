## model visualization metrics

import numpy as np
import pandas as pd
import base64
import io
import json
import pathlib
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path

# from .utils import Path, redirect_stdout, redirect_stderr


def create_plot(y_pred, output_test, outfile, outfile_png):
    with PdfPages(str(outfile)) as pp:
        colname = outfile.stem
        fig = plt.figure(figsize=(15,10))
        fig.suptitle(colname)
        ax=fig.add_subplot(1,2,1)
        ax.scatter(output_test,y_pred)
        ax.set_title('y_pred as a function of y_true')

        # plot histogram of errors
        ax=fig.add_subplot(1, 2, 2)
        create_hist(np.abs(y_pred - output_test) / np.abs(output_test+0.01))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        pp.savefig(fig)
        fig.savefig(outfile_png)
        return fig


def create_hist(x):
    hist, bin_edges = np.histogram(x, bins=50, range=(0.,0.5))
    cumulative = np.cumsum(hist)
    plt.plot(bin_edges[:-1], cumulative/float(len(x)), c='blue')
    plt.plot(np.linspace(0,0.5,100),[0.90]*100,'r')
    plt.axvline(0.1,color='r')
    plt.grid()
    plt.title('Cumulative distribution of error')
    plt.xlim(0,0.5)
    plt.show()



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          outfile_png=None
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig.show()

    if outfile_png !=None:
        fig.savefig(outfile_png)

    # return plt


def plot_ratio(df: pd.DataFrame, inputs, outputs, outfile_png=None):
    """
    Plots the error ratio for multiple inputs against a SINGLE output (passed in list).
    Designed to be called inside an external loop.
    """
    # Since your external loop passes outputs=[output], we take the first one
    target_name = outputs[0] 
    
    n_inputs = len(inputs)
    
    # --- 1. DYNAMIC SIZE ADJUSTMENT ---
    # Define a fixed height per plot (input). 
    # For example, 3.5 inches of height per input.
    height_per_plot = 3.5
    total_height = n_inputs * height_per_plot
    
    # Create the figure. constrained_layout=True prevents overlapping text/titles.
    # squeeze=False ensures 'axs' is always an iterable 2D array, even with 1 input.
    fig, axs = plt.subplots(nrows=n_inputs, ncols=1, 
                            figsize=(10, total_height), 
                            constrained_layout=True,
                            squeeze=False)

    # --- 2. DATA CALCULATION (Vectorized for speed) ---
    # Use the columns already prepared in the input_table dataframe
    y_pred = df['y_pred'].values
    y_test = df['y_test'].values
    
    # Avoid division by zero warnings (numpy handles inf/nan automatically)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = y_pred / y_test

    # Calculate colors using vectorization (much faster than row-by-row lambda apply)
    diff = np.abs(1 - ratio)
    colors = np.where(diff <= 0.1, 'green', 
             np.where(diff <= 0.5, 'orange', 'red'))

    # --- 3. PLOTTING ---
    # axs has shape (n_inputs, 1) because ncols=1
    for i, input_col in enumerate(inputs):
        ax = axs[i, 0] # Access row i, column 0
        
        # Scatter plot
        # Use alpha to visualize overlapping points if there is high density
        ax.scatter(x=df[input_col], y=ratio, s=10, c=colors, alpha=0.6)
        
        # Reference line (ratio = 1 represents a perfect prediction)
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Titles and Labels
        ax.set_title(f"Input: {input_col} vs Ratio ({target_name})", fontsize=10, fontweight='bold')
        ax.set_xlabel(input_col)
        ax.set_ylabel(f"Ratio {target_name}")
        ax.grid(True, linestyle=':', alpha=0.5)

    # --- 4. SAVING ---
    if outfile_png is not None:
        # bbox_inches='tight' ensures no labels/titles are cut off when saving
        fig.savefig(outfile_png, dpi=100, bbox_inches='tight')
        print(f"Figure saved to: {outfile_png}")
    
    # Return the figure object so the external loop can display and then close it
    return fig