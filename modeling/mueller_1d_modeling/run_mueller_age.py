#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import h5py
from scipy.optimize import minimize

from model_mueller_age import run_age

from common_inputs import get_ages, get_acc_series, get_thick_series
from constants import Hfinal, Hmax

# dt = H / nz / ndiffusivity / 5.0
# dt = 1.0 / 365.0 * 12.0
n = 9600
dt = 0.125 / 4.0
times = get_ages(250000.0, timestep=dt)

accs = [0.176, 0.275, 0.298, 0.407]  # Koerner 1979, camp pit, Queens met, Mueller 1962
acc_dict = {acc: get_acc_series(times, acc) for acc in accs}
input_dict = get_thick_series(times)
Zs = np.linspace(0, Hfinal, int(Hfinal) + 1)

i = 0
for name, H_np in input_dict.items():
    i += 1
    if i != 4:
        continue
    for acc, acc_np in acc_dict.items():
        print(name, acc)
        with h5py.File("modeled_ages_n{:d}.h5".format(n), "a") as fout:
            if name in fout and "{:1.2f}".format(acc) in fout[name]:
                continue
        out_times, Zs_full, age_mat, age, age_at_dist2bed, tenk_age, dage = run_age(times, H_np, acc_np, n)
        cmap = cm.viridis
        # norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
        norm = colors.Normalize(0, age_mat.shape[1])
        mycm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(figsize=(14, 8), num=name + " {:1.2f} m/yr".format(acc))
        for i in range(age_mat.shape[1]):
            ax.plot(age_mat[:, i], Zs_full, color=mycm.to_rgba(i))
            try:
                plt.plot(0, Hmax - H_np[5000 * i], marker="o", markerfacecolor=mycm.to_rgba(i), markeredgecolor="k")
            except IndexError:
                pass
            # print(T.dat.data[:])
        ax.set_ylim(Hmax, 0)
        ax.set_xlim(0, 50000)
        fig.canvas.manager.set_window_title(name)

        tenk_depth = minimize(tenk_age(10000), 100.0, method="nelder-mead", bounds=[(0, H_np[-1])]).x[0]
        pleis_depth = minimize(tenk_age(11700), 100.0, method="nelder-mead", bounds=[(0, H_np[-1])]).x[0]

        print("For {:s}, {:1.2f} m/yr run:".format(name, acc))
        for dist in [100, 50, 10, pleis_depth, tenk_depth]:
            dageval = dage.at(Hmax - dist)
            ageval = age.at(Hmax - dist)
            print("{:f} m above bedrock, age is {:f} with {:f} yr / m resolution".format(dist, ageval, dageval))

        with h5py.File("modeled_ages_n{:d}.h5".format(n), "a") as fout:
            if name not in fout:
                biggroup = fout.create_group(name)
            else:
                biggroup = fout[name]
            group = biggroup.create_group("{:1.2f}".format(acc))
            group.create_dataset("time", data=times)
            group.create_dataset("acc", data=acc_dict[acc])
            group.create_dataset("H", data=input_dict[name])
            group.create_dataset("depth", data=Zs)
            group.create_dataset("age", data=age.at(Zs + (Hmax - Hfinal)))
            group.create_dataset("ages", data=age_mat)
            group.create_dataset("out_times", data=out_times)
            group.create_dataset("full_depth", data=Zs_full)
plt.show()
