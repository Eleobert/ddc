#!/usr/bin/python3

import pandas as pd
import numpy as np
import ctypes
from scipy import stats

lib = ctypes.cdll.LoadLibrary("../cmake-build-debug/libddclib.so")

df = pd.read_csv("trans_top_gear.csv")

df = df.set_index("Unnamed: 0")


df = df[["Price", "Displacement", "BHP", "Torque", "Acceleration", "TopSpeed", "MPG", "Weight", "Length", "Width",
         "Height"]]
df = df.drop(['Citroen C5 Tourer'])


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


def ddc(x, p = 0.99, min_cor = 0.5):
    res = np.zeros(x.shape, dtype=np.float64)
    data = x if type(x) is np.ndarray else x.to_numpy()

    lib.ddc_c(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.shape[0], data.shape[1], ctypes.c_double(p), 
              ctypes.c_double(min_cor), res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    if type(x) is pd.DataFrame:
        res = pd.DataFrame(res, index=x.index, columns=x.columns)
    return res

sample = ["Audi A4", "BMW i3", "Chevrolet Cruze", "Corvette C6", "Fiat 500 Abarth", "Ford Kuga", "Honda CR-V",
                "Land Rover Defender", "Mazda CX-5", "Mercedes-Benz G", "Mini Coupe", "Peugeot 107", "Porsche Boxster",
                "Renault Clio", "Ssangyong Rodius", "Suzuki Jimny", "Volkswagen Golf"]

print("\noriginal data:\n", df.loc[sample, :])
res = ddc(df)

print("\ndata after prediction:\n", res.loc[sample, :])

res = res.loc[sample, :]
df  = df.loc[sample, :]

mae = (res - df).abs().to_numpy().flatten()
mae = mae[~np.isnan(mae)]
mae = mae.mean()

print("\values that were fixed:\n", res != df)

from sklearn.metrics import mean_absolute_error

print("\nmae: ", mae)

