import sys
import time
import pandas as pd
from pyomo import environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from pyomo.opt import SolverFactory


def get_purities(dfall):
    """
    if purities are specified by users, we fix the purities to the provided values.
    otherwise, we estimate the lowerbound of the purity and use the range LB_i<=r_<=1 for
    sample i.
    """
    # estimate the lowerbound using the cluster with the smallest BAF.
    # The mininum purity is when this cluster has (6,0) copy number state 
    # (assuming that (x,0) cannot be present in the data if x>6)
    lbs = []
    ubs = []
    for sample, df in dfall.groupby("SAMPLE"):
        total_bins = np.sum(df["#BINS"])
        df = df[df["#BINS"] > 0.003 * total_bins]
        df = df[df["RD"] < 5]
        minbaf = np.min(df["BAF"])
        if minbaf > 0.4:
            minpur = 0
        else:
            minpur = (1 - minbaf) / (5 * minbaf + 1)
        lbs.append(minpur)
        ubs.append(1)
        sys.stderr.write(f"Purity lower bound for sample {sample} is {minpur}\n")
    return [lbs, ubs]


def solve_ab_diff_model(df):
    model = pe.ConcreteModel(name="Diff-infer")
    y = {}
    z = {}

    v = pe.Var(bounds=(1, np.inf), domain=pe.Reals)
    model.add_component("v", v)
    for i in df["#ID"]:
        y[i] = pe.Var(domain=pe.NonNegativeReals)
        model.add_component(f"y_{i}", y[i])
        z[i] = pe.Var(bounds=(0, 8), domain=pe.Integers)
        model.add_component(f"z_{i}", z[i])

    # CONSTRAINTS
    model.constraints = pe.ConstraintList()
    for index, row in df.iterrows():
        i = row["#ID"]
        model.constraints.add(y[i] >= (z[i] - row["d"]*v))
        model.constraints.add(y[i] >= -(z[i] - row["d"]*v))

    obj = 0
    for i in df["#ID"]:
        obj += y[i]
    model.obj = pe.Objective(expr=obj, sense=pe.minimize)

    # SOLVER
    solver = SolverFactory('gurobi')
    start_time = time.time()
    solver.solve(model)
    end_time = time.time()

    runtime = end_time - start_time
    sys.stderr.write(f"The runtime of the solver at stage {str(model)} is {runtime} seconds.\n")

    # print(model)
    # model.display()
    df["abdelta"] = np.array([int(z[i].value) for i in df["#ID"]])
    df["d"] = df["d"] * v.value
    df["v"] = v.value
    # print(df)
    df.to_csv('data/diffed.seg', sep='\t', index=False)
    return df


def solve_cn(df):
    lbs, ubs = get_purities(df)
    lb = lbs[0]
    ub = ubs[0]
    model = pe.ConcreteModel(name="CN-call")
    y = {}
    b = {}

    if lb == 0:
        t = pe.Var(bounds=(2/ub, np.inf), domain=pe.Reals)
    else:
        t = pe.Var(bounds=(2/ub, 2/lb), domain=pe.Reals)
    model.add_component("t", t)
    for index, row in df.iterrows():
        i = row["#ID"]
        y[i] = pe.Var(domain=pe.NonNegativeReals)
        model.add_component(f"y_{i}", y[i])
        b[i] = pe.Var(bounds=(0, 8-row["abdelta"]), domain=pe.Integers)
        model.add_component(f"b_{i}", b[i])
    
    # CONSTRAINTS
    model.constraints = pe.ConstraintList()
    for index, row in df.iterrows():
        i = row["#ID"]
        model.constraints.add(y[i] >= (row["RD"] * row["v"] - 2 * b[i] + 2 - row["abdelta"] - t))
        model.constraints.add(y[i] >= -(row["RD"] * row["v"] - 2 * b[i] + 2 - row["abdelta"] - t))
    #model.constraints.pprint()

    obj = 0
    for i in df["#ID"]:
        obj += y[i]
    model.obj = pe.Objective(expr=obj, sense=pe.minimize)
        
    # SOLVER
    solver = SolverFactory('gurobi')
    start_time = time.time()
    solver.solve(model)
    end_time = time.time()

    runtime = end_time - start_time
    sys.stderr.write(f"The runtime of the solver at stage {str(model)} is {runtime} seconds.\n")
    # print(model)
    #model.display()
    cns = []
    for index, row in df.iterrows():
        i = row["#ID"]
        aa = int(b[i].value + row["abdelta"])
        bb = int(b[i].value)
        cns.append(f"{aa}|{bb}")
    df["cn"] = np.array(cns)
    df["purity"] = 2 / t.value
    df["gamma"] = df["v"] * df["purity"]
    df["FCN"] = df["gamma"] * df["RD"]
    df.to_csv('data/cn.seg', sep='\t', index=False)

    return df


# start main function
def main():
    df = pd.read_table(sys.argv[1], header=0)
    r = df["RD"]
    f = df["BAF"]
    df["d"] = r - 2 * r * f
    df = solve_ab_diff_model(df)
    solve_cn(df)


if __name__ == '__main__':
    main()
