python benchmark/benchmark_fv3core_stencils.py 100
env IJKLoop=True Prefetching=True ReadonlyCaching=True LoopUnrolling=True KCaching=True BlocksizeAdjusting="256,8,1" GT4PY_BACKEND=cuda python benchmark/benchmark_fv3core_stencils.py 100
GT4PY_BACKEND=gtcuda python benchmark/benchmark_fv3core_stencils.py 100
env IJKLoop=True Prefetching=True ReadonlyCaching=True LoopUnrolling=True KCaching=True BlocksizeAdjusting="256,8,1" GT4PY_BACKEND=cuda /home/huanglangwen/anaconda3/envs/gt4py/bin/python benchmark/benchmark_fv3core_stencils.py 100


env IJKLoop=True Prefetching=True ReadonlyCaching=True KCaching=True BlocksizeAdjusting="32,8,1" GT4PY_BACKEND=cuda python benchmark_fv3core_stencils.py 100
#'IJKLoop', 'Prefetching', 'ReadonlyCaching', 'KCaching', 'BlocksizeAdjusting'), 32
#Repeat times: 100, backend: cuda
#Stencil: Fillz, time: 108.73573803901672
#Stencil: Riem_Solver3, time: 29.819245100021362
#Stencil: Riem_Solver_C, time: 27.37993335723877
#Stencil: SatAdjust3d, time: 42.720603466033936

#(venv) huanglangwen10@langwen-vm:~/gt4py_stencils/benchmark$ env IJKLoop=True Prefetching=True ReadonlyCaching=True KCaching=True BlocksizeAdjusting="32,8,1" GT4PY_BACKEND=cuda python benchmark_fv3core_stencils.py 100
#Repeat times: 100, backend: cuda
#Stencil: Fillz, time: 110.19049835205078
#Stencil: Riem_Solver3, time: 29.80409049987793
#Stencil: Riem_Solver_C, time: 27.678062200546265
#Stencil: SatAdjust3d, time: 43.10806226730347


env IJKLoop=True BlocksizeAdjusting="32,8,1" GT4PY_BACKEND=cuda python benchmark_fv3core_stencils.py 100
#('IJKLoop', 'BlocksizeAdjusting'), 32
#Repeat times: 100, backend: cuda
#Stencil: Fillz, time: 106.99809384346008
#Stencil: Riem_Solver3, time: 29.561962366104126
#Stencil: Riem_Solver_C, time: 27.295790195465088
#Stencil: SatAdjust3d, time: 42.49861764907837

#(venv) huanglangwen10@langwen-vm:~/gt4py_stencils/benchmark$ env IJKLoop=True BlocksizeAdjusting="32,8,1" GT4PY_BACKEND=cuda python benchmark_fv3core_stencils.py 100
#Repeat times: 100, backend: cuda
#Stencil: Fillz, time: 109.03531670570374
#Stencil: Riem_Solver3, time: 30.171139240264893
#Stencil: Riem_Solver_C, time: 27.917303562164307
#Stencil: SatAdjust3d, time: 43.37281632423401

