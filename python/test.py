#!/usr/bin/python3

import pandas as pd
import numpy as np
from scipy import stats
from ddclib import *

df = pd.read_csv("trans_top_gear.csv")

df = df.set_index("Unnamed: 0")


df = df[["Price", "Displacement", "BHP", "Torque", "Acceleration", "TopSpeed", "MPG", "Weight", "Length", "Width",
         "Height"]]
df = df.drop(['Citroen C5 Tourer'])


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

