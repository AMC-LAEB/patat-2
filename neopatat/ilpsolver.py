from ortools.linear_solver import pywraplp
import numpy as np
from numba import jit
from math import comb

def solve_max_cov(area_coverage, max_machine_n):
    ## -- create solver -- ##
    solver = pywraplp.Solver('Maximize', pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

    ## -- create variables -- ##
    # units at location
    nloc = area_coverage.shape[0]
    units = [solver.IntVar(0, 1, "unit_%i"%(loc)) for loc in range(nloc)]

    ## -- add constraints -- ##
    # sum of all units must not be more than max_machine_n
    solver.Add( sum(units[loc] for loc in range(nloc)) <= max_machine_n )

    ## -- maximize obj. function -- ##
    solver.Maximize( sum(area_coverage[loc,:].sum() * units[loc] for loc in range(nloc)) )

    # solve
    status = solver.Solve()
    print (status)
