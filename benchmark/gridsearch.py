# How to run this: python gridsearch.py > gridsearch.out 2>&1 & 
# or : nohup python gridsearch.py >> gridsearch.out 2>&1 &
import os
from subprocess import PIPE, run
import numpy as np
from io import StringIO
import pickle

def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def format_opt_method(opt_method, gridsize):
    source = opt_method
    if opt_method == "BlocksizeAdjusting":
        source += "=\""+",".join([str(i) for i in gridsize]) + "\""
    else: 
        source += "=True"
    return source

def format_command(backend, config, path, ni, outdict):
    gridsize = [ni, 8, 1]
    parameter_str = f"env GT4PY_BACKEND={backend} "
    parameter_str += " ".join([format_opt_method(method, gridsize) for method in config])
    parameter_str += " python " + os.path.join(path,"benchmark_stencils.py") + " 100"
    out_str = out(parameter_str)
    out_arr = np.genfromtxt(StringIO(out_str),delimiter=",",skip_header=True)
    print(f"{backend}, {config}, {ni}")
    outdict[config+(ni,backend)] = out_arr
    titles = out_str.split("\n")[0].split(",")
    for i in range(len(titles)):
        arr_i = out_arr[:,i]
        print(f"{titles[i]}: 5% Quantile: {np.quantile(arr_i, 0.05)} Median: {np.median(arr_i)} Mean: {np.average(arr_i)}")

def main():
    OPTIMIZATION_METHODS = ["IJKLoop", "Prefetching", "ReadonlyCaching", "LoopUnrolling", "KCaching", "BlocksizeAdjusting"]
    path = os.path.dirname(os.path.abspath(__file__))
    outdict = dict()
    format_command("gtcuda", tuple(), path, 32, outdict)
    for config in powerset(OPTIMIZATION_METHODS):
        if "IJKLoop" not in config and ("KCaching" in config or "LoopUnrolling" in config):
            continue
        if "BlocksizeAdjusting" in config:
            for ni in [512, 256, 128, 64, 32]:
                format_command("cuda", config, path, ni, outdict)
        else:
            format_command("cuda", config, path, 32, outdict)
    with open(os.path.join(path, "gridsearch_out.pickle"), "wb") as f:
        pickle.dump(outdict, f)

if __name__ == "__main__":
    main()
