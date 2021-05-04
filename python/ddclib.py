import numpy as np
import pandas as pd
import ctypes

lib = ctypes.cdll.LoadLibrary("../cmake-build-debug/libddclib.so")


def cor_wrap(x):
    n = x.shape[1]
    res = np.zeros((n, n), dtype=np.float64)
    lib.cor_wrap_c(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.shape[0], x.shape[1],
                   res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return res


# Robust parameter estimator
def estimate_params(x):
    n = x.shape[1]
    locs = np.zeros(n, dtype=np.float64)
    scales = np.zeros(n, dtype=np.float64)
    lib.estimate_params_c(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.shape[0], x.shape[1],
                          locs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          scales.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return locs, scales


def standardize(x):
    res = np.zeros(x.shape, dtype=np.float64)
    data = x if type(x) is np.ndarray else x.to_numpy()
    lib.standardise_c(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.shape[0], data.shape[1],
                   res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if type(x) is pd.DataFrame:
        res = pd.DataFrame(res, index=x.index, columns=x.columns)
    return res


def predict_univariate(x):
    res = np.zeros(x.shape, dtype=np.float64)
    data = x if type(x) is np.ndarray else x.to_numpy()
    lib.predict_univariate_c(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.shape[0], data.shape[1],
                   res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if type(x) is pd.DataFrame:
        res = pd.DataFrame(res, index=x.index, columns=x.columns)
    return res


def ddc(x, n_cor = 100, p = 0.99, min_cor = 0.5):
    res = np.zeros(x.shape, dtype=np.float64)
    data = x if type(x) is np.ndarray else x.to_numpy()
    
    in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    n_cor = ctypes.c_double(n_cor)
    p = ctypes.c_double(p)
    min_cor = ctypes.c_double(min_cor)
    out_ptr = res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    lib.ddc_c(in_ptr, data.shape[0], data.shape[1], n_cor, p, min_cor, out_ptr)

    if type(x) is pd.DataFrame:
        res = pd.DataFrame(res, index=x.index, columns=x.columns)
    return res