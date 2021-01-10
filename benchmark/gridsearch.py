# How to run this: python gridsearch.py > gridsearch.out 2>&1 & 
# or : nohup gridsearch.py &
import os

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

def format_command(config, path, ni):
    gridsize = [ni, 8, 1]
    parameter_str = "env GT4PY_BACKEND=cuda "
    parameter_str += " ".join([format_opt_method(method, gridsize) for method in config])
    parameter_str += " python " + path + " 100"
    print(f"{config}, {ni}")
    os.system(parameter_str)

def main():
    OPTIMIZATION_METHODS = ["IJKLoop", "Prefetching", "ReadonlyCaching", "LoopUnrolling", "KCaching", "BlocksizeAdjusting"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"benchmark_fv3core_stencils.py")
    for config in powerset(OPTIMIZATION_METHODS):
        if "IJKLoop" not in config and ("KCaching" in config or "LoopUnrolling" in config):
            continue
        if "BlocksizeAdjusting" in config:
            for ni in [512, 256, 128, 64, 32]:
                format_command(config, path, ni)
        else:
            format_command(config, path, 32)

if __name__ == "__main__":
    main()
