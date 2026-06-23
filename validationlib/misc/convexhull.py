'''
Copyright (c) 2025 Airbus Operations S. L.
This file is part of project ISAMI+ released under ther Airbus Inner Source shared-maintenance
'''
import numpy as np

from scipy.optimize import linprog

def chmp_lp(points, x):
    """
    Solve a linear programming problem to check feasibility.

    This function uses a linear programming solver to check whether the given point 'x'
    is feasible with respect to the convex hull defined by a set of points.

    :param points: An array of shape (n_points, n_dim) representing the points defining the convex hull.
    :param x: A point of interest to check for feasibility.
    :return: True if 'x' is feasible within the convex hull, False otherwise.
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b, method='highs-ds')
    return lp.success