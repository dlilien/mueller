# coding: utf-8

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob


modern_acc = 0.25

t0 = datetime.datetime(2021, 5, 4)
now = datetime.datetime(2023, 5, 5)

queens_temp_name = "Temperature_21049794_deg_C"
queens_data = pd.read_csv("../climate/queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
queens_data["Date_Time"] = pd.to_datetime(queens_data["Date_Time"])

snow_name = "SnowDepth   _20908992_cm   "
plt.figure()
plt.plot(queens_data["Date_Time"].values, queens_data[snow_name].values)
queens_data[snow_name].loc[queens_data[snow_name] < 1100] = np.nan
queens_data[snow_name].loc[queens_data[snow_name] > 1220] = np.nan
queens_data[snow_name][queens_data["Date_Time"] > datetime.datetime.strptime("04/24/22 07:00", "%m/%d/%y %H:%M")] = (
    queens_data[snow_name][queens_data["Date_Time"] > datetime.datetime.strptime("04/24/22 07:00", "%m/%d/%y %H:%M")]
    + 47.0
)  # moved from 47 to 94 cm to snow, says Laura

plt.figure()
plt.plot(queens_data["Date_Time"].values, queens_data[snow_name].values)
plt.show()
