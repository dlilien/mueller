#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from constants import Hmax, Hfinal, time_of_temp_measurement

acc_df = pd.read_csv("../../climate/other_ice_cores/combined/combined_acc.csv")
temp_df = pd.read_csv("../../climate/other_ice_cores/combined/combined_temp.csv")
temp_df = temp_df.loc[~np.isnan(temp_df["Temp (C)"])]
smooth_temp_df = pd.read_csv("../../climate/other_ice_cores/combined/combined_temp_smooth.csv")

run_list = ["Constant thickness", "Holocene thinning", "406 m during HCO", "200 m during HCO"]
# color_dict = {name: "C{:d}".format(i) for i, name in enumerate(run_list)}
okabe_ito = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (0, 114, 178)]
okabe_ito = [(a[0] / 255.0, a[1] / 255.0, a[2] / 255.0) for a in okabe_ito]
color_dict = {name: tup for name, tup in zip(run_list, okabe_ito)}

thickness_interpers = {"Constant thickness": lambda x: np.ones_like(x) * Hfinal}

times_thinning = [-1.0e6, -11700, -4000, 1.0e6]
thicks_thinning = [Hmax, Hmax, Hfinal, Hfinal]
thickness_interpers["Holocene thinning"] = interp1d(times_thinning, thicks_thinning)


times_rethick250 = [-1.0e6, -11700, -8000, -4000, -2000, 1.0e6]
thicks_rethick250 = [Hmax, Hmax, 200, 200, Hfinal, Hfinal]
thickness_interpers["200 m during HCO"] = interp1d(times_rethick250, thicks_rethick250)

times_rethick100 = [-1.0e6, -11700, -8000, -4000, -2000, 1.0e6]
thicks_rethick100 = [Hmax, Hmax, 406, 406, Hfinal, Hfinal]
thickness_interpers["406 m during HCO"] = interp1d(times_rethick100, thicks_rethick100)

marker_dict = {"0.18": "o", "0.28": "s", "0.30": "^", "0.41": "d", "0.34": "<"}
acc_dict = {
    "0.18": "Koerner, 1979",
    "0.28": "Snow pit",
    "0.30": "Met station",
    "0.34": "Firn core",
    "0.41": "Müller, 1962",
}
linestyle = {"0.18": "dotted", "0.28": "dashed", "0.30": "solid", "0.41": "dashdot", "0.34": (0, (3, 1, 1, 1, 1, 1))}
plot_dict = {
    "0.18": "Koerner",
    "0.28": "Snow pit",
    "0.30": "Met stat.",
    "0.34": "Firn core",
    "0.41": "Müller",
}


def get_ages(timespan, timestep=1.0 / 365.25, end=time_of_temp_measurement):
    return np.arange(end - timespan, end + timestep, timestep)


def get_temp_series(ages, smooth=True):
    if smooth:
        data = smooth_temp_df
    else:
        data = temp_df
    interper = interp1d(data["Age (yrs a2k)"], data["Temp (C)"])
    return interper(ages)


def get_acc_series(ages, modern_acc=0.25):
    interper = interp1d(acc_df["Age (yrs a2k)"], acc_df["Acc (m/yr)"])
    return interper(ages) * modern_acc / 0.25


def get_thick_series(ages, name=None):
    if name is None:
        return {name: interper(ages) for name, interper in thickness_interpers.items()}
    else:
        return thickness_interpers[name](ages)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ages = get_ages(20000, timestep=20.0 / 365.0)
    thicks = get_thick_series(ages)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    for name, thick in thicks.items():
        ax1.plot(ages, thick, label="Name")
    acc = get_acc_series(ages)
    ax2.plot(ages, acc)
    temp_smooth = get_temp_series(ages, smooth=True)
    temp_rough = get_temp_series(ages, smooth=False)
    ax3.plot(ages, temp_smooth, color="C7")
    ax3.plot(ages, temp_rough, color="C8")
    ax1.set_xlim(-20000, 24)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.set_xlim(218500, 219500)
    ax1.set_ylim(199, 210)
    for name, thick in thicks.items():
        ax1.plot(thick, label="Name")
    ax2.plot(acc)
    ax3.plot(temp_smooth, color="C7")
    ax3.plot(temp_rough, color="C8")
    plt.show()
