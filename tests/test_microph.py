try:
    import gtstencil_example
    import tests
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import os
import numpy as np
from tests.serialization import read_data, numpy_dict_to_gt4py_dict, view_gt4py_storage, compare_data
from gtstencil_example.microph_driver import gfdl_cloud_microphys_driver_split as gfdl_cloud_microphys_driver
from gtstencil_example.microph_driver import gfdl_cloud_microphys_init

in_data = read_data(os.path.join(os.path.dirname(__file__), "data_microph"), 0, 0, True)
out_data = read_data(os.path.join(os.path.dirname(__file__), "data_microph"), 0, 0, False)
n_timings = 1
timings   = { "warm_rain_1_call"  : np.zeros(n_timings), 
                "warm_rain_1_run"   : np.zeros(n_timings), 
                "sedimentation_call": np.zeros(n_timings), 
                "sedimentation_run" : np.zeros(n_timings), 
                "warm_rain_2_call"  : np.zeros(n_timings), 
                "warm_rain_2_run"   : np.zeros(n_timings), 
                "icloud_call"       : np.zeros(n_timings), 
                "icloud_run"        : np.zeros(n_timings), 
                "main_loop_call"    : np.zeros(n_timings), 
                "main_loop_run"     : np.zeros(n_timings), 
                "driver"            : np.zeros(n_timings), 
                "tot"               : 0. }

in_data = numpy_dict_to_gt4py_dict(in_data)
gfdl_cloud_microphys_init()
qi, qs, \
qv_dt, ql_dt, qr_dt, qi_dt, qs_dt, qg_dt, qa_dt, \
pt_dt, w, udt, vdt, \
rain, snow, ice, graupel, \
refl_10cm \
= gfdl_cloud_microphys_driver(in_data, False, True, 0, 0, timings, 0, False)

data = view_gt4py_storage( { "qi":    qi   [:, :, :], "qs":      qs     [:, :, :], "qv_dt":     qv_dt    [:, :, :], 
                                "ql_dt": ql_dt[:, :, :], "qr_dt":   qr_dt  [:, :, :], "qi_dt":     qi_dt    [:, :, :], 
                                "qs_dt": qs_dt[:, :, :], "qg_dt":   qg_dt  [:, :, :], "qa_dt":     qa_dt    [:, :, :],   
                                "pt_dt": pt_dt[:, :, :], "w":       w      [:, :, :], "udt":       udt      [:, :, :], 
                                "vdt":   vdt  [:, :, :], "rain":    rain   [:, :, 0], "snow":      snow     [:, :, 0], 
                                "ice":   ice  [:, :, 0], "graupel": graupel[:, :, 0], "refl_10cm": refl_10cm[:, :, :] } )
compare_data(data, out_data, explicit=False, blocking=True)