#! /usr/bin/env python3

# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from common_inputs import get_temp_series, get_thick_series, get_acc_series, color_dict, acc_dict, linestyle, plot_dict

wx_df = pd.read_csv("../../climate/queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
wx_df["Date_Time"] = pd.to_datetime(wx_df["Date_Time"], format="%m/%d/%y %H:%M:%S")

with h5py.File("firn_temp.h5", "r") as fin:
    by_depth = fin["by_depth"]
    by_time = fin["by_time"]
    times = by_depth["time"][:]
    T_662cm = by_depth["662cm"][:]
    days = by_time["days_after_20230501"][:]
    depth = by_time["depths"][:]
    T = by_time["temps"][:]

accs = []
with h5py.File("full_temp.h5", "r") as fin:
    z_full = fin["HOB_full"][:]
    z_final = fin["HOB_final"][:]
    T_bytime_dict = {}
    T_final_dict = {}
    T_basal_dict = {}
    times_full_dict = {}
    basal_times_dict = {}
    for name in fin.keys():
        if name == "times" or name[:3] == "HOB":
            continue
        ogroup = fin[name]
        for aname in ogroup.keys():
            group = ogroup[aname]
            T_final_dict[name + "_" + aname] = group["T_final"][:]
            T_bytime_dict[name + "_" + aname] = group["T_full"][:]
            times_full_dict[name + "_" + aname] = group["output_times"][:]
            T_basal_dict[name + "_" + aname] = group["Tb"][:]
            basal_times_dict[name + "_" + aname] = group["times"][:]
            if aname not in accs:
                accs.append(aname)


dummy_times = np.linspace(-50000, 24, 10000)
dummy_ages = (dummy_times[-1] - dummy_times) / 1000.0

surf_temp = get_temp_series(dummy_times)
accs_dict = {acc: get_acc_series(dummy_times, modern_acc=float(acc)) for acc in acc_dict}
thick_dict = {
    name.split("_")[0]: get_thick_series(dummy_times, name=name.split("_")[0])
    for name, times in basal_times_dict.items()
}


def mod_date(date, delta):
    deltayear = int(delta)
    frac = delta - deltayear

    newdate = date.replace(year=date.year + deltayear)
    year_in_seconds = (newdate.replace(year=newdate.year + 1) - newdate).total_seconds()

    return newdate + timedelta(seconds=year_in_seconds * frac)


basedate = datetime(2023, 5, 26, 0, 0, 0)
model_dates = np.array([mod_date(basedate, time - times[-1]) for time in times], dtype=np.datetime64)

fig, ax = plt.subplots()
plt.plot(wx_df["Date_Time"], wx_df["Temperature_21049794_deg_C"], color="k")
plt.plot(model_dates, T_662cm)
ax.set_xlim(datetime(2021, 5, 1), datetime(2023, 5, 26))


cmap = cm.viridis
# bounds = days
# norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
norm = colors.Normalize(0, 390)
mycm = cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_3p():
    gs = gridspec.GridSpec(
        2,
        2,
        left=0.0944,
        right=0.992,
        bottom=0.1,
        top=0.98,
        width_ratios=[1.15, 0.85],
        wspace=0.20,
        hspace=0.25,
        height_ratios=(1, 0.4),
    )
    fig = plt.figure(figsize=(7.05, 5.0))

    ax_full_temp = fig.add_subplot(gs[0, 0])
    ax_firntemp = fig.add_subplot(gs[0, 1])
    ax_basal_temp = fig.add_subplot(gs[1, :])

    in_H = []
    in_A = []
    for name, T_final in T_final_dict.items():
        if name.split("_")[0] not in in_H:
            ax_full_temp.plot([], [], label=name.split("_")[0], color=color_dict[name.split("_")[0]])
            in_H.append(name.split("_")[0])
        ax_full_temp.plot(
            T_final,
            z_final[-1] - z_final,
            color=color_dict[name.split("_")[0]],
            linestyle=linestyle[name.split("_")[1]],
        )

    for name, T_final in T_final_dict.items():
        aname = name.split("_")[1]
        if aname == "0.18":
            anamep = "0.1758"
        else:
            anamep = aname
        if aname not in in_A:
            ax_full_temp.plot(
                [],
                [],
                linestyle=linestyle[aname],
                color="k",
                label="{:d} kg m$^{{-2}}$ yr$^{{-1}}$, {:s}".format(int(float(anamep) * 911), plot_dict[aname]),
            )
            in_A.append(aname)

    ax_full_temp.legend(loc="upper right", frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.03))
    ax_full_temp.set_ylabel("Depth (m)")
    ax_full_temp.set_xlabel("Temperature (°C)")
    ax_full_temp.set_xlim(-24, -9)
    ax_full_temp.set_xticks([-24, -19, -14, -9])
    ax_full_temp.set_ylim(z_final[-1], 0)

    ax_firntemp.axhline(6.62, color="k", linewidth=0.5)
    ax_firntemp.axvline(-19.5, color="k", linewidth=0.5)
    for i in range(14):
        if i == 13:
            lw = 2
        else:
            lw = 1
        ax_firntemp.plot(T[:, i], depth, color=mycm.to_rgba(days[i]), lw=lw)

    ax_firntemp.plot(
        -21.1,
        6.62,
        marker="o",
        markerfacecolor=mycm.to_rgba(days[i]),
        markeredgecolor="k",
    )

    ax_firntemp.set_ylim(15, 0)
    ax_firntemp.set_yticks([0, 5, 10, 15])
    ax_firntemp.set_xlim(-45, 5)
    ax_firntemp.set_xlabel("Temperature (°C)")
    ax_firntemp.set_ylabel("Depth (m)")

    ax_firntemp.annotate(
        "At 6.62m,\nMay 26, 2023:\n-21.1 °C obs.\n-21.2 °C sim.",
        (-21.2, 6.62),
        xytext=(-1, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-|>", color="k"),
        ha="right",
        va="top",
        color="k",
        fontsize=9,
    )

    cax_firn = fig.add_axes([0.87, 0.475, 0.015, 0.22])
    cbr = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax_firn,
        ticks=[0, days[3], days[6], days[9], days[12]],
    )
    cbr.set_ticklabels(["May 1\n2022", "", "Nov 1\n2023", "", "May 1\n2023"])
    cbr.ax.tick_params(labelsize=9)

    for name, Tb in T_basal_dict.items():
        ax_basal_temp.plot(
            np.abs(basal_times_dict[name] - basal_times_dict[name][-1]) / 1000.0,
            Tb,
            color=color_dict[name.split("_")[0]],
            linestyle=linestyle[name.split("_")[1]],
        )
    ax_basal_temp.set_ylabel("Basal temp. (°C)")
    ax_basal_temp.set_xlim(20, 0)
    ax_basal_temp.set_ylim(-25, -5)
    ax_basal_temp.set_xticks([20, 15, 10, 5, 0])
    ax_basal_temp.set_xlabel("Time (kyr b.p.)")

    ax_full_temp.text(
        0.01,
        0.99,
        "a",
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_full_temp.transAxes,
    )
    ax_firntemp.text(
        0.02,
        0.99,
        "b",
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_firntemp.transAxes,
    )
    ax_basal_temp.text(
        0.02,
        0.96,
        "c",
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_basal_temp.transAxes,
    )
    fig.savefig("../../plots/JOG-2024-0020.Figure9.pdf")


def plot_forcings():
    gs = gridspec.GridSpec(3, 1, left=0.115, right=0.992, bottom=0.108, top=0.995, wspace=0.21)
    fig = plt.figure(figsize=(7.05, 4.0))

    ax_acc = fig.add_subplot(gs[1, 0])
    ax_surf_temp = fig.add_subplot(gs[0, 0])
    ax_H = fig.add_subplot(gs[2, 0])

    in_H = []
    in_A = []
    for name, H in thick_dict.items():
        if name not in in_H:
            ax_H.plot(dummy_ages, H, label=name, color=color_dict[name])
            in_H.append(name)
    ax_H.legend(loc="lower center", frameon=False, fontsize=9, ncol=2)
    ax_H.set_ylabel("Ice thick.\n(m)")
    ax_H.set_ylim(0, 1250)
    ax_H.set_xlim(50, 0)
    ax_H.set_xlabel("Time (kyr b.p.)")
    ax_acc.axes.xaxis.set_ticklabels([])
    ax_surf_temp.axes.xaxis.set_ticklabels([])

    name = "Constant thickness_0.18"
    ax_surf_temp.plot(dummy_ages, surf_temp, color="k")

    ax_surf_temp.set_ylabel("Surface\ntemp. (°C)")
    ax_surf_temp.set_xlim(50, 0)

    for name, Tb in T_basal_dict.items():
        A = name.split("_")[1]
        if name.split("_")[0] == "Constant thickness":
            ax_acc.plot(
                dummy_ages,
                accs_dict[A] * 910.0,
                linestyle=linestyle[A],
                color="C{:d}".format(4 + len(in_A)),
                label=plot_dict[A],
            )
            in_A.append(A)
    ax_acc.legend(loc="upper center", frameon=False, fontsize=9, ncol=2)
    ax_acc.set_ylabel("SMB\n(kg m$^{-2}$ yr$^{-1}$)")
    ax_acc.set_xlim(50, 0)

    ax_H.text(0.01, 0.99, "c", ha="left", va="top", fontsize=14, transform=ax_H.transAxes)
    ax_surf_temp.text(
        0.01,
        0.99,
        "a",
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_surf_temp.transAxes,
    )
    ax_acc.text(0.01, 0.99, "b", ha="left", va="top", fontsize=14, transform=ax_acc.transAxes)
    fig.savefig("../../plots/JOG-2024-0020.Figure8.pdf")


plot_3p()
plot_forcings()
