import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

OPTIMIZATION_METHODS = ["IJKLoop", "Prefetching", "ReadonlyCaching", "LoopUnrolling", "KCaching", "BlocksizeAdjusting"]
COLUMN_NAMES = ["Fillz","Riem_Solver3","Riem_Solver_C","SatAdjust3d", "Thomas_inplace"]

def set_to_bool_tuple(config):
    return tuple(True if i in config else False for i in OPTIMIZATION_METHODS) + (config[-2], config[-1])

def summarize_data(mat, method = "median"):
    assert(isinstance(mat, np.ndarray))
    if method == "median":
        res = np.nanmedian(mat, axis = 0)
    elif method == "mean":
        res = np.nanaverage(mat, axis = 0)
    elif method == "5%quantile":
        res = np.nanquantile(mat, 0.05, axis = 0)
    elif method == "95%quantile":
        res = np.nanquantile(mat, 0.95, axis = 0)
    return res

def mark_pareto_front(outdict, method="5%quantile"):
    on_pareto = {key: True for key in outdict}
    sum_dict = {key: summarize_data(outdict[key], method=method) for key in outdict}
    for key1 in outdict:
        sum1 = sum_dict[key1]
        for key2 in outdict:
            if not key1 == key2:
                sum2 = sum_dict[key2]
                tot = (sum1 < sum2).sum()
                if tot == 5:
                    on_pareto[key2] = False
                elif tot == 0:
                    on_pareto[key1] = False
    return sum_dict, on_pareto

def plot_parallel(outdict, method="median"):
    gt_out = outdict.pop((32,"gtcuda"))
    base_out = outdict[(32,"cuda")] #baseline
    base_med = summarize_data(base_out, method=method)
    med_percent_dict = {key:(summarize_data(outdict[key], method=method)-base_med)/base_med for key in outdict}
    _, on_pareto_dict = mark_pareto_front(outdict, method=method)
    plt.figure()
    plt.xticks(np.arange(5), COLUMN_NAMES)
    colordict = dict()
    count = 0
    for key in med_percent_dict:
        line = med_percent_dict[key]
        if on_pareto_dict[key] and (line < 0.01).sum() == 5:
            if key[:-2] not in colordict:
                colordict[key[:-2]] = count
                plt.plot(line, label=",".join(list(key[:-2])), color=f"C{count}")
                count += 1
            plt.plot(line, color=f"C{colordict[key[:-2]]}")
        else:
            plt.plot(line, "k", alpha = 0.05)
    #plt.plot((summarize_data(gt_out, method=method)-base_med)/base_med, "k", label="gtcuda")
    plt.ylim((-0.05,0.05))
    plt.xlim((0,4))
    plt.yticks(ticks=np.array([0.025, 0, -0.025, -0.05, -0.075, -0.1]), #-0.1,-0.15]),
               labels=np.array(["+2.5%","0","-2.5%","-5%", "-7.5%", "-10%"]))#,"-10%","-15%"]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',borderaxespad=0)
    plt.grid(axis='x', color='k', linestyle='-')
    plt.ylabel("Percentage of time reduced (compared with original cuda backend)")
    plt.xlabel("Stencil name")
    plt.savefig("Gridsearch_2.pdf", bbox_inches="tight")
    plt.show()

path = os.path.dirname(os.path.abspath(__file__))
outdict = pickle.load(open(os.path.join(path,"gridsearch_out.pickle"),"rb"))
#gt_out = outdict.pop((32,"gtcuda"))
plot_parallel(outdict)
#for key in outdict:
#    print(set_to_bool_tuple(key))
