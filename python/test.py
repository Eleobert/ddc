#!/usr/bin/python3

import pandas as pd
import numpy as np
import ctypes
from scipy import stats

lib = ctypes.cdll.LoadLibrary("cmake-build-debug/libddc.so")

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



std = predict_univariate(df)

std = std.loc[["Audi A4", "BMW i3", "Chevrolet Cruze", "Corvette C6", "Fiat 500 Abarth", "Ford Kuga", "Honda CR-V",
                "Land Rover Defender", "Mazda CX-5", "Mercedes-Benz G", "Mini Coupe", "Peugeot 107", "Porsche Boxster",
                "Renault Clio", "Ssangyong Rodius", "Suzuki Jimny", "Volkswagen Golf"], :]

print(std)
print("\vouliers:\n", std.reset_index().melt(id_vars='Unnamed: 0').query('value > 0.99'))
std[std <= 0.99] = 0
std[std > 0.99] = 1


import numpy as np 
from pandas import DataFrame
import matplotlib.pyplot as plt

plt.pcolor(std)
plt.yticks(np.arange(0.5, len(std.index), 1), std.index)
plt.xticks(np.arange(0.5, len(std.columns), 1), std.columns)
plt.show()

