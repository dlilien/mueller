#!/usr/bin/env python
# coding: utf-8

import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import interp1d
import h5py
from common_inputs import get_ages, get_acc_series, get_temp_series, get_thick_series
from constants import Hmax, Hfinal, time_of_temp_measurement
from model_mueller_full_temp import model_temperature_variableH


nz = 150
dt_dict = {
    "200 m during HCO": 1.0 / 365.0,
    "406 m during HCO": 5.0 / 365.0,
    "Constant thickness": 10.0 / 365.0,
    "Holocene thinning": 10.0 / 365.0,
}
dt_dict = {name: item * (100.0 / nz) ** 2.0 for name, item in dt_dict.items()}
times_dict = {
    name: (
        np.hstack(
            (
                get_ages(11500.0, timestep=dt, end=time_of_temp_measurement - 8500.0)[:-1],
                get_ages(250.0, timestep=dt / 10.0, end=time_of_temp_measurement - 8250.0)[:-1],
                get_ages(500.0, timestep=dt / 100.0, end=time_of_temp_measurement - 7750.0)[:-1],
                get_ages(250.0, timestep=dt / 10.0, end=time_of_temp_measurement - 7500.0)[:-1],
                get_ages(7500.0, timestep=dt),
            )
        )
        if "HCO" in name
        else get_ages(20000.0, timestep=dt)
    )
    for name, dt in dt_dict.items()
}
acc_dict = {name: get_acc_series(times) for name, times in times_dict.items()}
temp_dict = {name: get_temp_series(times, smooth=True) for name, times in times_dict.items()}
thick_dict = {name: get_thick_series(times, name=name) for name, times in times_dict.items()}

z_modern = np.linspace(0, Hfinal, 500)
z_full = np.linspace(0, Hmax, 1000)
with h5py.File("full_temp_n{:d}.h5".format(nz), "a") as fout:
    if "HOB_final" not in fout:
        fout.create_dataset("HOB_final", data=z_modern)
    if "HOB_full" not in fout:
        fout.create_dataset("HOB_full", data=z_full)


output_dict = {}
for name, thicks in thick_dict.items():
    with h5py.File("full_temp_n{:d}.h5".format(nz), "a") as fout:
        if name not in fout:
            print("Running", name)
            output_dict[name] = model_temperature_variableH(
                times_dict[name], thicks, acc_dict[name], temp_dict[name], fd.Constant(dt_dict[name]), nz=nz
            )
            out_Ts, out_times = output_dict[name][2], output_dict[name][0]

            group = fout.create_group(name)
            Tmat = np.empty((z_full.shape[0], len(output_dict[name][0])))
            for i, (z, T) in enumerate(output_dict[name][2]):
                Tmat[:, i] = interp1d(z, T, fill_value=np.nan, bounds_error=False)(z_full)
            group.create_dataset("times", data=times_dict[name])
            group.create_dataset("T_final", data=output_dict[name][1][-1].at(z_modern))
            group.create_dataset("T_full", data=Tmat)
            group.create_dataset("output_times", data=output_dict[name][0])
            group.create_dataset("Tb", data=output_dict[name][3])

            plt.figure()
            for i in range(len(out_Ts)):
                plt.plot(
                    out_Ts[i][1],
                    out_Ts[i][0],
                    color=cm.viridis(
                        (out_times[i] - times_dict[name][0]) / (times_dict[name][-1] - times_dict[name][0])
                    ),
                )
            plt.xlabel("Temp")
            plt.ylabel("Depth")


fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

for i, (name, (_, Ts, _, Tb)) in enumerate(output_dict.items()):
    fd.plot(Ts[-1], axes=ax, color="C{:d}".format(i), label=name)
    ax1.plot(times_dict[name], Tb, color="C{:d}".format(i), label=name)

ax1.set_xlabel("Years after 2000")
ax1.set_ylabel("Basal temp (C)")
ax1.legend(loc="best")
ax.legend(loc="best")
ax.set_xlabel("Height above bedrock (m)")
ax.set_ylabel("Temp (C)")
plt.show()
