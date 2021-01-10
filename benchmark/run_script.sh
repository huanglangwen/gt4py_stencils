python benchmark/benchmark_fv3core_stencils.py 100
env IJKLoop=True Prefetching=True ReadonlyCaching=True LoopUnrolling=True KCaching=True BlocksizeAdjusting="256,8,1" GT4PY_BACKEND=cuda python benchmark/benchmark_fv3core_stencils.py 100
env IJKLoop=True Prefetching=True ReadonlyCaching=True LoopUnrolling=True KCaching=True BlocksizeAdjusting="256,8,1" GT4PY_BACKEND=cuda /home/huanglangwen/anaconda3/envs/gt4py/bin/python benchmark/benchmark_fv3core_stencils.py 100
