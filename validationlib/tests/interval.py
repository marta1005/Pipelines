import warnings
from packaging.version import Version
from random import random, sample
from typing import List, Union, Callable, Optional, Tuple
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from matplotlib import colors
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.stats import binomtest

from .dist import fit_dist
from ..misc.subsampling import maskApply
from ..plots.common import multiplePlots
from ..plots import histogram

def KDF(data: np.ndarray, q: float = 95, conf: float = 0.95) -> List[float]:
    """
    Computes the confidence interval using the Wilson-Score proportion method.
    # REMINDER
    # KDF-A =: Wilson-Score proportion @95% confidence on the 99% percentile
    # KDF-B =: Wilson-Score proportion @95% confidence on the 90% percentile

    :param data: Input data as a NumPy array.
    :param q: Desired percentile (default is 95).
    :param conf: Confidence level (default is 0.95).
    :return: Lower and upper bounds of the confidence interval.
    """
    
    n = len(data)

    try:
        n_min = int(5 / (1 - q / 100) + 1)
        if n < n_min:
            raise ValueError(f"sample size 'n': {n} is lower than the minimum required sample size: {n_min}")
    except ValueError as e:
        print('ValueError: '+str(e))

    LL, UL = proportion_confint(count=n * q / 100, nobs=n, alpha=1 - conf, method="wilson")

    sortdata = np.sort(data)  # For Load Ascending for RF Descending
    LL, UL = sortdata[int(LL * n)], sortdata[int(UL * n)]

    return LL, UL


class BinnedUncertaintyModel:
    """
    Represents a model for binning and uncertainty estimation.
    """
    def __init__(
        self,
        percentile_range: Union[list[float], float],
        ci_type: str = "two_tailed",
        model_type: str = "eqspaced_bins",
        method_kwargs: dict = {},
        matched_analysis: bool = False
    ):
        """
        Initialize the BinnedUncertaintyModel.

        :param percentile_range: Range of percentiles or a single percentile value.
        :param ci_type: Type of confidence interval ("two_tailed", "left_tailed", "right_tailed").
        :param model_type: Type of model ("eqspaced_bins", "categorical").
        :param method_kwargs: Additional keyword arguments for the method.
        :param matched_analysis: Flag for matched analysis.
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError("Type of model not recognized. Please use one of the following: " + str(list(self.MODEL_TYPES.keys())))

        if ci_type not in ["two_tailed", "left_tailed", "right_tailed"]:
            raise ValueError("'ci_type' must be one of the following: 'two_tailed', 'right_tailed', 'left_tailed'")

        if ci_type == "two_tailed" and not isinstance(percentile_range, list):
            raise ValueError("If 'ci_type' is 'two_tailed', 'percentile_range' must be a list of two values")

        if ci_type == "left_tailed" and isinstance(percentile_range, list):
            warnings.warn("'ci_type' is 'left_tailed' but 'percentile_range' is a list of values. Using only the first value")
            percentile_range = percentile_range[0]

        if ci_type == "right_tailed" and isinstance(percentile_range, list):
            warnings.warn("'ci_type' is 'right_tailed' but 'percentile_range' is a list of values. Using only the last value")
            percentile_range = percentile_range[-1]

        self.percentile_range = percentile_range
        self.ci_type = ci_type
        self.model_type = model_type
        self.matched_analysis = matched_analysis
        self.method_kwargs = method_kwargs

        self.model_method = self.MODEL_TYPES[self.model_type]


    def train(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        calset_mask: csr_matrix = None,
        x_variables: list[str] = None,
        y_variables: list[str] = None
    ):
        """
        Train the model.

        :param x: Input data as a Pandas DataFrame.
        :param y: Output data as a Pandas DataFrame.
        :param calset_mask: Calibration set mask.
        :param x_variables: List of input variable names.
        :param y_variables: List of output variable names.
        """
        if x_variables is None:
            x_variables = x.columns
        if y_variables is None:
            y_variables = y.columns

        self.x_variables_ = x_variables
        self.y_variables_ = y_variables

        self.y_mean_df_ = pd.DataFrame(columns=y_variables, index=x_variables, dtype=object)
        self.y_ci_df_ = pd.DataFrame(columns=y_variables, index=x_variables, dtype=object)
        self.bin_edges_df_ = pd.DataFrame(columns=y_variables, index=x_variables, dtype=object)
        for i, y_var in enumerate(y_variables):
            for j, x_var in enumerate(x_variables):
                if (i != j) and self.matched_analysis: continue

                x_cal = maskApply(x, calset_mask if calset_mask is None else calset_mask[:, i], columnIndex='all', row_wise=True, to_numpy=False)[x_var].to_numpy()
                y_cal = maskApply(y, calset_mask, columnIndex=i, np_dtype=float)

                results_dict = self.model_method(self, x_cal, y_cal, **self.method_kwargs)
                self.y_mean_df_.loc[x_var, y_var] = results_dict["y_mean"]
                self.y_ci_df_.loc[x_var, y_var] = results_dict["y_ci"]
                self.bin_edges_df_.loc[x_var, y_var] = results_dict["bin_edges"]


    def predict(
        self,
        x: pd.DataFrame,
        testset_mask: csr_matrix = None,
        x_variables: list[str] = None,
        y_variables: list[str] = None
    ):
        if x_variables is None:
            x_variables = x.columns
        if y_variables is None:
            y_variables = self.y_mean_df_.columns 

        output_df = pd.DataFrame(columns=y_variables, index=x_variables, dtype=object)

        for i, y_var in enumerate(y_variables):
            if y_var not in self.y_mean_df_.columns:
                raise ValueError("The model has not been trained on variable %s"%(y_var))

            for j, x_var in enumerate(x_variables):
                if x_var not in self.y_mean_df_.index:
                    raise ValueError("The model has not been trained on variable %s"%(x_var))

                if (i != j) and self.matched_analysis: continue

                x_test = maskApply(x, testset_mask if testset_mask is None else testset_mask[:, i], columnIndex='all', row_wise=True, to_numpy=False)[x_var].to_numpy()

                y_ci = self.y_ci_df_.loc[x_var, y_var]
                bin_edges = self.bin_edges_df_.loc[x_var, y_var]

                y_ci_bin = np.empty([x_test.shape[0], 2])

                # Filter out samples outside the bin range
                if self.model_type == "categorical":
                    na_mask = np.isin(x_test, bin_edges, invert=True)
                    indices_dict = dict(zip(bin_edges, range(len(bin_edges))))
                    bin_indices = np.vectorize(indices_dict.get)(x_test[~na_mask])
                else:
                    na_mask = (x_test > bin_edges[-1]) | (x_test < bin_edges[0])
                    bin_indices = np.argmax(x_test[~na_mask][:, np.newaxis] <= bin_edges[1:], axis=1)

                y_ci_bin[na_mask, :] = [np.nan, np.nan]
                y_ci_bin[~na_mask, :] = y_ci.take(bin_indices, axis=0)

                output_df.loc[x_var, y_var] = y_ci_bin

        return output_df


    def __eqspacedbin_model(
        self,
        x_cal:np.ndarray,
        y_cal:np.ndarray,
        bins: int = 10,
        nsim: int = 1000,
        min_elems: int = 0,
        conf: float = 0.95
    ):
        hist, bin_edges = np.histogram(x_cal, bins=bins)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        nwindows = len(hist)

        y_mean = np.empty(nwindows)
        y_ci = np.empty([nwindows, 2])

        for i in range(nwindows):
            x_window = x_cal[(x_cal >= bin_edges[i]) & (x_cal <= bin_edges[i+1])]
            y_window = y_cal[(x_cal >= bin_edges[i]) & (x_cal <= bin_edges[i+1])]

            mean_y, mean_pinf, mean_psup = self.__get_quantiles(x_window, y_window, nsim, min_elems, conf)

            y_mean[i] = mean_y
            y_ci[i, 0] = mean_pinf
            y_ci[i, 1] = mean_psup

        return {
            "y_mean" : y_mean,
            "y_ci" : y_ci,
            "bin_edges" : bin_edges
        }


    def __categorical_model(
        self,
        x_cal: np.ndarray,
        y_cal: np.ndarray,
        nsim: int = 1000,
        min_elems: int = 0,
        conf: float = 0.95
    ):
        x_unique = np.unique(x_cal)
        nwindows = x_unique.shape[0]

        y_mean = np.empty(nwindows)
        y_ci = np.empty([nwindows, 2])

        for i in range(nwindows):
            x_window = x_cal[x_cal == x_unique[i]]
            y_window = y_cal[x_cal == x_unique[i]]

            mean_y, mean_pinf, mean_psup = self.__get_quantiles(x_window, y_window, nsim, min_elems, conf)

            y_mean[i] = mean_y
            y_ci[i, :] = mean_pinf, mean_psup

        return {
            "y_mean": y_mean,
            "y_ci": y_ci,
            "bin_edges": x_unique
        }


    def __get_quantiles(
            self,
            x_window: list,
            y_window: list,
            nsim: int,
            min_elems: int,
            conf: float = 0.95
    ):
        if len(x_window) > min_elems:
            bootstrapper = percentileBootstrap(y_window, nsim=nsim, conf=conf)

            mean_y, ci_mean = bootstrapper.compute(np.mean)

            if self.ci_type == "two_tailed":
                mean_pinf, ci_pinf = bootstrapper.compute(np.percentile, **{"q": self.percentile_range[0]})
                mean_psup, ci_psup = bootstrapper.compute(np.percentile, **{"q": self.percentile_range[1]})
            elif self.ci_type == "left_tailed":
                mean_pinf, ci_pinf = bootstrapper.compute(np.percentile, **{"q": self.percentile_range})
                mean_psup = np.inf
            elif self.ci_type == "right_tailed":
                mean_pinf = -np.inf
                mean_psup, ci_psup = bootstrapper.compute(np.percentile, **{"q": self.percentile_range})
        else:
            mean_y, ci_mean = np.nan, np.nan
            mean_pinf, ci_pinf = np.nan, np.nan
            mean_psup, ci_psup = np.nan, np.nan

        return mean_y, mean_pinf, mean_psup


    def get_tables(self): 
        tables = []
        for j, y_var in enumerate(self.y_mean_df_.columns):
            for i, x_var in enumerate(self.y_mean_df_.index):
                if (i != j) and self.matched_analysis: continue

                bin_edges = self.bin_edges_df_.loc[x_var, y_var]
                if self.model_type == "categorical":
                    bins = list(bin_edges)
                else:
                    bins = ["[%.4g, %.4g]"%(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

                table = pd.DataFrame(columns=bins, index=["Mean", "CI"], dtype=object)
                n_values = self.y_mean_df_.loc[x_var, y_var]
                if  n_values.size > 1:
                    table.loc["Mean", bins] = ["%.4g"%(mean) for mean in self.y_mean_df_.loc[x_var, y_var]]
                else:
                    table.loc["Mean", bins] = "%.4g"%(n_values)
                table.loc["CI", bins] = ["[%.4g, %.4g]"%(ci[0], ci[1]) for ci in self.y_ci_df_.loc[x_var, y_var][:].tolist()]

                table = table.style.set_caption("Input: %s; Output: %s"%(x_var, y_var))
                tables.append(table)

        return tables


    def scatterplot(self, *args, **kwargs):
        """ Create a coverage_plot (backward compatibility)"""
        warnings.warn("scatterplot is deprecated. Use coverage_plot instead.", DeprecationWarning)
        return self.coverage_plot(*args, **kwargs)

    def coverage_plot(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        calset_mask: csr_matrix = None, 
        testset_mask: csr_matrix = None,
        model_coverage : "ModelCoverage" = None,
        margin: float = 0.1,
        focus_on_ci: bool = True,
        x_variables: list[str] = None,
        y_variables: list[str] = None,
        x_label: Union[list[str], str] = None,
        y_label: Union[list[str], str] = None,
        label_significant_figures: int = 4,
        alpha: float = 0.5,
        plot_type: str = "scatter",
        hist_data: str = "test",
        bins: int = 10,
        logscale: bool = False,
        cov_kwargs: Optional[dict] = {}
    ):
        """
        Create a coverage_plot.
        
        :param x: Input data as a Pandas DataFrame.
        :param y: Output data as a Pandas DataFrame.
        :param calset_mask: Calibration set mask.
        :param testset_mask: Test set mask.
        :param model_coverage: ModelCoverage object.
        :param margin: Margin for the plot.
        :param x_variables: List of input variable names.
        :param y_variables: List of output variable names.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        :param label_significant_figures: Number of significant figures for axis tick-labels.
        :param plot_type: Type of plot ("scatter", "histogram").
        :param cov_kwargs: Additional keyword arguments for coverage computation.
        :return: List of figures.
        """

        if model_coverage is not None:
            if testset_mask is None or calset_mask is None:
                raise ValueError("If 'model_coverage' is specified, 'testset_mask' and 'calset_mask' must be specified")

        if calset_mask is not None and testset_mask is None or calset_mask is None and testset_mask is not None:
            raise ValueError("Either both or none of 'testset_mask' and 'calset_mask' must be specified")

        def categorical_plot(axes, xbins_plot, y_ci, y_mean):
            bar_width = 0.5
            x_bars = np.arange(len(xbins_plot))

            xbins_plot = [f"{cat:.{max(label_significant_figures,np.ceil(np.log10(cat)).astype(int))}g}" if isinstance(cat, np.number) and not isinstance(cat, int) else cat for cat in xbins_plot]

            if self.ci_type != "right_tailed": 
                percentile_inf = axes.bar(x_bars, y_ci[:, 0]-y_mean, tick_label=xbins_plot, width=bar_width, bottom=y_mean, color='k', alpha=0.2)
            else:
                ylim = axes.get_ylim()
                ylim_array = np.ones(y_ci.shape[0])*ylim[0]
                ylim_array[np.isnan(y_ci[:, 1])] = np.nan
                percentile_inf = axes.bar(x_bars, ylim_array-y_mean, tick_label=xbins_plot, width=bar_width, bottom=y_mean, color='k', alpha=0.2)

            if self.ci_type != "left_tailed":
                percentile_sup = axes.bar(x_bars, y_ci[:, 1]-y_mean, tick_label=xbins_plot, width=bar_width, bottom=y_mean, color='k', alpha=0.2)
            else:
                ylim = axes.get_ylim()
                ylim_array = np.ones(y_ci.shape[0])*ylim[1]
                ylim_array[np.isnan(y_ci[:, 1])] = np.nan
                percentile_sup = axes.bar(x_bars, ylim_array-y_mean, tick_label=xbins_plot, width=bar_width, bottom=y_mean, color='k', alpha=0.2)

            percs_to_iterate = []
            if self.ci_type != "right_tailed": percs_to_iterate += percentile_inf
            if self.ci_type != "left_tailed": percs_to_iterate += percentile_sup

            for idx, bar in enumerate(percs_to_iterate):
                label = "CI" if idx == 0 else None

                x_coord, y_coord = bar.get_xy()
                width, height = bar.get_width(), bar.get_height()
                axes.plot([x_coord, x_coord + width], [y_coord + height, y_coord + height], "--", c="k", label=label)

            #axes.scatter(xbins_plot, y_mean, c='k', marker='x', label="Mean")

        def continuous_plot(axes, xbins_plot, y_ci, y_mean):
            xbins_plot =  [xbins_plot[0]] + sum([[xbins_plot[i]]*2 for i in range(1, len(xbins_plot)-1)], []) + [xbins_plot[-1]]

            y_ci = np.repeat(y_ci, repeats=2, axis=0)
            if self.ci_type != "right_tailed": axes.plot(xbins_plot, y_ci[:, 0], "--", c='k', label="CI")
            if self.ci_type != "left_tailed": axes.plot(xbins_plot, y_ci[:, 1], "--", c='k', label=("CI" if self.ci_type != "two_tailed" else None))

            # y_mean = np.repeat(y_mean, repeats=2)
            # axes.plot(xbins_plot, y_mean, "-", c='k', label="Mean")

            if self.ci_type == "two_tailed":
                axes.fill_between(xbins_plot, y_ci[:, 0], y_ci[:, 1], color="k", alpha=0.15)
            elif self.ci_type == "left_tailed":
                ylim = axes.get_ylim()
                axes.fill_between(xbins_plot, y_ci[:, 0], ylim[1], color="k", alpha=0.15)
            elif self.ci_type == "right_tailed":
                ylim = axes.get_ylim()
                axes.fill_between(xbins_plot, ylim[0], y_ci[:, 1], color="k", alpha=0.15)

        def func(
            i,
            axes: plt.Axes,
            j: int = None
        ):
            if self.matched_analysis: j = i
            x_var = x_variables[i]

            x_cal = maskApply(x, calset_mask if calset_mask is None else calset_mask[:, j], columnIndex='all', row_wise=True, to_numpy=False)[x_var].to_numpy()
            y_cal = maskApply(y, calset_mask, columnIndex=j, np_dtype=float)

            if testset_mask is not None:
                x_test = maskApply(x, testset_mask[:, j], columnIndex='all', row_wise=True, to_numpy=False)[x_var].to_numpy()
                y_test = maskApply(y, testset_mask, columnIndex=j, np_dtype=float)

            # Scatter
            if self.model_type == "categorical":
                x_cal = np.argwhere(self.bin_edges_df_.iloc[i,j] == np.array(x_cal)[:,None])[:,1]
                if testset_mask is not None:
                    x_test = np.argwhere(self.bin_edges_df_.iloc[i,j] == np.array(x_test)[:,None])[:,1]

            if plot_type == "scatter":
                axes.scatter(x_cal, y_cal, s=1, label="Calibration", alpha=alpha)
                if testset_mask is not None:
                    axes.scatter(x_test, y_test, s=1, label="Test", alpha=alpha)
            elif plot_type == "histogram" and testset_mask is not None:
                if logscale:
                    norm = mcolors.LogNorm()
                else:
                    norm = None

                if hist_data == "cal" or testset_mask is None:
                    hist_h = axes.hist2d(x_cal, y_cal, bins=bins, cmap="rainbow", norm=norm)
                    if testset_mask is not None:
                        cbar_suffix = "\n(calibration set)"
                    else:
                        cbar_suffix = ""
                elif hist_data == "test" and testset_mask is not None:
                    hist_h = axes.hist2d(x_test, y_test, bins=bins, cmap="rainbow", norm=norm)
                    cbar_suffix = "\n(test set)"
                plt.colorbar(hist_h[3], ax=axes, label="Frequency"+cbar_suffix)

            if self.model_type == "categorical":
                axes.set_xticklabels(self.bin_edges_df_.iloc[i,j])

            xbins_plot = self.bin_edges_df_.iloc[i, j]
            y_ci = self.y_ci_df_.iloc[i, j]
            y_mean = self.y_mean_df_.iloc[i, j]

            if self.ci_type != "right_tailed" and focus_on_ci:
                y_lim_min = np.nanmin(y_ci[y_ci != -np.inf])
            else:
                y_lim_min = np.nanmin([np.nanmin(y_test), np.nanmin(y_cal)])

            if self.ci_type != "left_tailed" and focus_on_ci:
                y_lim_max = np.nanmax(y_ci[y_ci != np.inf])
            else:
                y_lim_max = np.nanmax([np.nanmax(y_test), np.nanmax(y_cal)])

            y_lim_range = y_lim_max - y_lim_min

            y_lim_min_plot = y_lim_min - y_lim_range * margin
            y_lim_max_plot = y_lim_max + y_lim_range * margin
            axes.margins(x=margin / 2)
            axes.set_ylim([y_lim_min_plot, y_lim_max_plot])

            if self.model_type == "categorical":
                categorical_plot(axes, xbins_plot, y_ci, y_mean)
            else:
                xlims = axes.get_xlim()
                xbins_plot[0] = xlims[0]
                xbins_plot[-1] = xlims[1]
                continuous_plot(axes, xbins_plot, y_ci, y_mean)
                axes.set_xlim(xlims)

            if model_coverage is not None:
                mean_cov = cov_df.iloc[i, j]
                ci_cov = covci_df.iloc[i, j]

            if self.ci_type == "two_tailed":
                ci_p = self.percentile_range[1] - self.percentile_range[0]
            else:
                ci_p = self.percentile_range

            coverage_suffix = "; Coverage=%.2f%% \n CI%i=(%.2f, %.2f)%%" % (mean_cov, ci_p, ci_cov[0], ci_cov[1]) if model_coverage is not None else ""
            axes.set_title(self.y_mean_df_.index[i] + coverage_suffix)

            if x_label is None:
                axes.set_xlabel(x_var)
            elif isinstance(x_label, str):
                axes.set_xlabel(x_label)
            else:
                axes.set_xlabel(x_label[i])

            if y_label is None:
                axes.set_ylabel(y_variables[j])
            elif isinstance(y_label, str):
                pass
            else:
                axes.set_ylabel(y_label[j])

            if i==0 and calset_mask is not None and plot_type == "scatter": 
                legend = axes.legend()
                if Version(mpl.__version__) >= Version("3.7"):
                    legend.legend_handles[0]._sizes = [20]
                    legend.legend_handles[1]._sizes = [20]
                else:
                    legend.legendHandles[0]._sizes = [20]
                    legend.legendHandles[1]._sizes = [20]

            return axes, None

        if x_variables is None:
            x_variables = x.columns
        if y_variables is None:
            y_variables = self.y_mean_df_.columns 

        if model_coverage is not None:
            cov_df, covci_df = model_coverage.compute_coverage(
                x,
                y,
                testset_mask,
                model=self,
                x_variables=x_variables,
                y_variables=y_variables,
                **cov_kwargs
            )

        nPlots = len(x_variables)

        if self.matched_analysis:
            fig = multiplePlots(
                nPlots,
                func,
                tight_layout=True,
                ylabel=y_label if isinstance(y_label, str) else None
            )

            return [fig]

        else:
            figs = []
            for j, y_var in enumerate(y_variables):                
                if self.ci_type == "two_tailed":
                    ci_p = self.percentile_range[1] - self.percentile_range[0]
                else:
                    ci_p = self.percentile_range
                # cov_suffix = "" if model_coverage is None else "\nCoverage=%.2f%%; CI%i=(%.2f, %.2f)%%" % (cov_df.loc["TOTAL COV", y_var], ci_p, covci_df.loc["TOTAL COV", y_var][0], covci_df.loc["TOTAL COV", y_var][1])
                cov_suffix = ""
                fig = multiplePlots(
                    nPlots,
                    func,
                    func_kwargs = {"j":j},
                    tight_layout=True,
                    figTitle=str(y_var) + cov_suffix,
                    ylabel=y_label if isinstance(y_label, str) else None
                )

                figs.append(fig)

            return figs


    @classmethod
    def combine_model_predictions(
        cls,
        model_predictions: list[pd.DataFrame],
        testset_mask: csr_matrix,
        x_variables: list[list[str]],
        y_variables: list[str]
    ):
        """
        Deprecated. Use CombinedUncertaintyModel instead.
        """
        combined_out = pd.DataFrame(columns=y_variables, index=y_variables, dtype=object)

        for j, y_var in enumerate(y_variables):
            n_samples = np.asarray(testset_mask.sum(axis=0)).flatten()[j]
            combined_out.loc[y_var, y_var] = np.empty([n_samples, 2])
            for i in range(n_samples):
                uncs_list = []
                for k, pred in enumerate(model_predictions):
                    if all(x_var in pred.index for x_var in x_variables[k]):
                        uncs_list += [pred.loc[x_var, y_var][i] if isinstance(pred.loc[x_var, y_var], Iterable) else pred.loc[x_var, y_var] for x_var in x_variables[k]]
                    else:
                        raise("Variables of prediction %i do not match with specified variables")

                int_sizes = [unc_int[-1] - unc_int[0] if isinstance(unc_int, Iterable) else unc_int for unc_int in uncs_list]
                combined_out.loc[y_var, y_var][i, :] = uncs_list[np.nanargmax(int_sizes)]


        return combined_out


    MODEL_TYPES = {
        "eqspaced_bins" : __eqspacedbin_model,
        "categorical" : __categorical_model
    }


class CombinedUncertaintyModel:
    def __init__(self, *models: "BinnedUncertaintyModel"):
        self.models = models

    def predict(
        self,
        x: pd.DataFrame,
        testset_mask: csr_matrix = None,
        x_variables: list[str] = None,
        y_variables: list[str] = None
    ):


        if x_variables is None:
            x_variables = x.columns
        if y_variables is None:
            y_variables = list(set().union(*(model.y_variables_ for model in self.models)))
            # y_variables = set((tuple(model.y_variables_.tolist()) for model in self.models))

        output_df = pd.DataFrame(columns=y_variables, index=y_variables, dtype=object)

        for i, y_var in enumerate(y_variables):
            if testset_mask is None:
                n_samples = x.shape[0]
            else:
                n_samples = np.asarray(testset_mask.sum(axis=0)).flatten()[i]

            model_predictions = []
            for model in self.models:
                model_predictions.append(model.predict(
                    x,
                    testset_mask,
                    model.x_variables_,
                    model.y_variables_
                ))

            output_df.loc[y_var, y_var] = np.empty([n_samples, 2])
            for j in range(n_samples):
                uncs_list = []
                unc_type_list = []
                for k, pred in enumerate(model_predictions):
                    uncs_list += [pred.loc[x_var, y_var][j] if isinstance(pred.loc[x_var, y_var], Iterable) else pred.loc[x_var, y_var] for x_var in self.models[k].x_variables_]
                    unc_type_list += [self.models[k].ci_type for x_var in self.models[k].x_variables_]

                int_min_list = [uncs_list[k][0] if isinstance(uncs_list[k], Iterable) else uncs_list[k] for k in range(len(uncs_list)) if unc_type_list[k] != "right_tailed"]
                int_max_list = [uncs_list[k][-1] if isinstance(uncs_list[k], Iterable) else uncs_list[k] for k in range(len(uncs_list)) if unc_type_list[k] != "left_tailed"]
                # int_sizes = [unc_int[-1] - unc_int[0] if isinstance(unc_int, Iterable) else unc_int for unc_int in uncs_list]

                # output_df.loc[y_var, y_var][j, :] = uncs_list[np.nanargmax(int_sizes)]
                if all(unc_type_list[k] == "right_tailed" for k in range(len(unc_type_list))):
                    output_df.loc[y_var, y_var][j, 0] = -np.inf
                else:
                    output_df.loc[y_var, y_var][j, 0] = np.nanmin(int_min_list)

                if all(unc_type_list[k] == "left_tailed" for k in range(len(unc_type_list))):
                    output_df.loc[y_var, y_var][j, 1] = np.inf
                else:
                    output_df.loc[y_var, y_var][j, 1] = np.nanmax(int_max_list)

        return output_df


class ModelCoverage:
    """
    Handles computing coverage for a model.
    """
    def __init__(
        self,
        matched_analysis: bool = False,
        conf: float = 0.95
    ):
        """
        Initialize the ModelCoverage.

        :param matched_analysis: Flag for matched analysis.
        :param conf: Confidence level (default is 0.95).
        """
        self.matched_analysis = matched_analysis
        self.conf = conf


    def compute_coverage(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        mask: csr_matrix,
        model: object = None,
        model_predictions: pd.DataFrame = None,
        x_variables: list[str] = None,
        y_variables: list[str] = None,
        method: str = 'exact',
        nsim: int = 100
    ):
        """
        Compute coverage for the model.

        :param x: Input data as a Pandas DataFrame.
        :param y: Output data as a Pandas DataFrame.
        :param mask: Mask for data.
        :param model: Model object.
        :param model_predictions: Predicted values from the model.
        :param x_variables: List of input variable names.
        :param y_variables: List of output variable names.
        :param method: Method for computing coverage ('exact', 'bootstrap').
        :param nsim: Number of simulations (default is 100).
        :return: DataFrames with coverage and confidence interval values.
        """

        def __compute_coverage(results):
            if method in ['exact', 'wilson', 'wilsoncc']:
                test_results = binomtest(k=results.sum(), n=results.shape[0], p=results.sum()/results.shape[0])
                coverage = test_results.k / test_results.n * 100
                coverage_ci = np.array(test_results.proportion_ci(method=method, confidence_level=self.conf)) * 100

            elif method == 'bootstrap':
                bootstrapper = percentileBootstrap(results.astype(float), nsim=nsim, conf=self.conf)
                coverage, coverage_ci = bootstrapper.compute(np.mean)

                coverage *= 100
                coverage_ci = np.array(coverage_ci) * 100

            else:
                raise ValueError("Method %s not recognized"%(method))

            return coverage, coverage_ci

        if x_variables is None:
            x_variables = x.columns
        if y_variables is None:
            y_variables = y.columns

        if model is not None:
            model_predictions = model.predict(x, mask, x_variables, y_variables)
        elif model_predictions is None:
            raise ValueError("Either 'model' or 'model_predictions' must be specified")

        self.coverage_df = pd.DataFrame(columns=y_variables, index=x_variables, dtype=float)
        self.coverageci_df = pd.DataFrame(columns=y_variables, index=x_variables, dtype=object)

        results_list = []

        for i, y_var in enumerate(y_variables):
            for j, x_var in enumerate(x_variables):
                if (i != j) and self.matched_analysis: continue

                y_test = maskApply(y, mask, columnIndex=i, np_dtype=float)

                yci_pred = model_predictions.loc[x_var, y_var]
                results = (y_test >= yci_pred[:, 0]) & (y_test <= yci_pred[:, 1])
                results_list.append(results)

                coverage, coverage_ci = __compute_coverage(results)
                self.coverage_df.loc[x_var, y_var] = coverage
                self.coverageci_df.loc[x_var, y_var] = coverage_ci

        if len(results_list) == 1:
            total_results = results_list[0]
        else:
            total_results = np.logical_or.reduce(results_list)
        coverage, coverage_ci = __compute_coverage(total_results)
        self.coverage_df.loc["TOTAL COV", y_var] = coverage
        self.coverageci_df.loc["TOTAL COV", y_var] = coverage_ci

        return self.coverage_df, self.coverageci_df


    def get_tables(self):
        tables = []

        if self.matched_analysis: tables.append(pd.DataFrame(columns=self.coverage_df.index, index=["Coverage", "CI"], dtype=object))

        for j, y_var in enumerate(self.coverage_df.columns):
            if not self.matched_analysis: tables.append(pd.DataFrame(columns=self.coverage_df.index, index=["Coverage", "CI"], dtype=object))

            for i, x_var in enumerate(self.coverage_df.index):
                if (i != j) and self.matched_analysis: continue

                tables[-1].loc["Coverage", x_var] = "%.4g"%self.coverage_df.loc[x_var, y_var]
                ci = self.coverageci_df.loc[x_var, y_var]
                tables[-1].loc["CI", x_var] = "[%.4g, %.4g]"%(ci[0], ci[1])

            if not self.matched_analysis: 
                tables[-1] = tables[-1].style.set_caption("Coverage table. Output: %s"%(y_var))

        if self.matched_analysis:
            tables[-1] = tables[-1].style.set_caption("Coverage table")

        return tables


class equaltailedCrInterval:
    """
    Computes confidence intervals using a given distribution.
    """
    def __init__(self, data: np.ndarray, stdist: Callable, conf: float = 0.95,
                 nsim: int = 10000, simsize: int = 1000):
        """
        Initialize equaltailedCrInterval.

        :param data: Input data as a NumPy array.
        :param stdist: Standard distribution function.
        :param conf: Confidence level (default is 0.95).
        :param nsim: Number of simulations (default is 10000).
        :param simsize: Size of simulation (default is 1000).
        """
        self.conf, self.conf2 = conf, (1 - conf) * 50
        # Assume our sample is drawn from a certain known distribution, generate nsim samples, calculate the statistic "pdf"
        [params], _, _ = fit_dist([data], stdist)
        arg, loc, scale = params[:-2], params[-2], params[-1]

        simsize = min(len(data), simsize)
        self.X, self.nsim = [None] * nsim, nsim
        for i in range(nsim):
            self.X[i] = stdist.rvs(size=simsize, loc=loc, scale=scale, *arg)

    def compute(self, func: Callable, title: str = "", 
                visualize: bool = False, **kwargs) -> List[Union[float, list]]:
        """
        Compute confidence intervals.

        :param func: Function to compute intervals for.
        :param title: Title for the computation.
        :param visualize: Flag to visualize the result.
        :param kwargs: Additional keyword arguments.
        :return: Computed statistic and confidence intervals.
        """
        title = [title + " @" + str(self.conf * 100) + "% confidence"]
        statistic_label = func.__name__ + " distribution"
        statistic_vector = np.empty(self.nsim)
        for i in range(self.nsim):
            statistic_vector[i] = func(self.X[i], **kwargs)

        statistic = np.mean(statistic_vector)
        CrI = [np.percentile(statistic_vector, self.conf2), np.percentile(statistic_vector, 100 - self.conf2)]
        if visualize == True:
            kernel = histogram([statistic_vector], labels=title, xlabel=statistic_label, kde=True)
            plt.plot((statistic, statistic), (0, 0.99 * kernel(statistic)), color="C3", linewidth=2)
            plt.plot((CrI[0], CrI[0]), (0, 0.99 * kernel(CrI[0])), color="C3", linewidth=2)
            plt.plot((CrI[1], CrI[1]), (0, 0.99 * kernel(CrI[1])), color="C3", linewidth=2)
        return statistic, CrI


class percentileBootstrap:
    """
    ## **Bootstrap confidence intervals**
    Calculates confidence intervals using bootstrapping

    #### Inputs
    ###### data : list
    ###### func : function to bootstrap
    ###### title : str
        Specify the title of the histogram

    #### Additional inputs
    ###### conf: float
        Confidence level from 0 to 1
    ###### nsim: int
        Number of bootstraps
    ###### visualize: bool

    #### Prints
        If visualize==True, plots the mentioned histogram

    #### Outputs
    ###### Statistic, CI: float, float
    <hr style="border:4px solid blue"> </hr> <br />
    """

    def __init__(self, data: np.ndarray, conf: float = 0.95, nsim: int = 10000, 
                 fraction: float = 1, random_state: int = None):
        """
        Initialize percentileBootstrap.

        :param data: Input data as a NumPy array.
        :param conf: Confidence level (default is 0.95).
        :param nsim: Number of simulations (default is 10000).
        :param fraction: Fraction of data to use (default is 1).
        :param random_state: Seed for random state.
        """
        self.conf, self.conf2 = conf, (1 - conf) * 50
        # Bootstrap
        self.X, self.nsim = [None] * nsim, nsim
        n = len(data)
        simsize = int(n * fraction)

        rs = np.random.RandomState(seed=random_state)

        for i in range(nsim):
            index = rs.randint(0, n, simsize)
            self.X[i] = data[index]

    def compute(self, func: Callable, title: str = "",
                visualize: bool = False, **kwargs) -> List[Union[float, list]]:
        """
        Compute confidence intervals using percentile bootstrapping.

        :param func: Function to compute intervals for.
        :param title: Title for the computation.
        :param visualize: Flag to visualize the result.
        :param kwargs: Additional keyword arguments.
        :return: Computed statistic and confidence intervals.
        """
        title = [title + " @" + str(self.conf * 100) + "% confidence"]
        statistic_label = "Bootstrapped " + func.__name__
        statistic_vector = np.empty(self.nsim)
        for i in range(self.nsim):
            statistic_vector[i] = func(self.X[i], **kwargs)

        statistic = np.mean(statistic_vector)
        CI = [np.percentile(statistic_vector, self.conf2), np.percentile(statistic_vector, 100 - self.conf2)]
        if visualize == True:
            kernel = histogram([statistic_vector], labels=title, xlabel=statistic_label, kde=True)
            plt.plot((statistic, statistic), (0, 0.99 * kernel(statistic)), color="C3", linewidth=2)
            plt.plot((CI[0], CI[0]), (0, 0.99 * kernel(CI[0])), color="C3", linewidth=2)
            plt.plot((CI[1], CI[1]), (0, 0.99 * kernel(CI[1])), color="C3", linewidth=2)
        return statistic, CI
