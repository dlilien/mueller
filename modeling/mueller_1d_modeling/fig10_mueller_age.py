#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from common_inputs import color_dict, linestyle, acc_dict, marker_dict

figdum, axdum = plt.subplots()
fig40, ax40 = plt.subplots()

out_dict = {}
with h5py.File("modeled_ages.h5", "r") as fin:
    for name in fin.keys():
        bgroup = fin[name]
        out_dict[name] = {}
        for accn in bgroup.keys():
            group = bgroup[accn]
            times = group["time"][:]
            z = group["depth"][:]
            out_dict[name][accn] = (group["H"][:], group["acc"][:], group["age"][:])

gs = gridspec.GridSpec(1, 2, top=0.98, left=0.085, right=0.98, wspace=0.2, bottom=0.12)
fig = plt.figure(figsize=(7.05, 4.0))

ax_age = fig.add_subplot(gs[0, 0])
ax_comp = fig.add_subplot(gs[0, 1])
ax_comp.set_xscale("log")

ax_comp.fill_between([-100, 100], [-1, -1], [20, 20], color="0.6", alpha=0.5)
ax_comp.fill_betweenx([-1000, 1000], [10, 10], [200, 200], color="0.6", alpha=0.5)

targ_age = 11700
in_H = []
in_A = []
at_20 = []
at_56 = []
depths_of_250ka = []
for name, odict in out_dict.items():
    for aname, (H, acc, age) in odict.items():

        if aname == "0.18":
            anamep = "0.1758"
        else:
            anamep = aname
        ax_age.plot(age / 1000, z, linestyle=linestyle[aname], color=color_dict[name])

        interper = interp1d(age, z)
        age_interper = interp1d(z, age)
        print("For {:s}, {:s} m/yr, 20 m above bedrock, age is {:4.0f}".format(name, aname, age_interper(z[-1] - 20)))
        print("For {:s}, {:s} m/yr, 56 m above bedrock, age is {:4.0f}".format(name, aname, age_interper(z[-1] - 56)))
        at_20.append(age_interper(z[-1] - 20))
        at_56.append(age_interper(z[-1] - 56))
        z_holocene = interper(targ_age)
        dage = np.diff(age) / np.diff(z)
        dage_interper = interp1d((z[1:] + z[:-1]) / 2.0, dage)
        dage_holocene = dage_interper(z_holocene)
        rev_interper = interp1d(dage, (z[1:] + z[:-1]) / 2.0)
        depths_of_250ka.append(z[-1] - interper(250000))
        print(
            "For {:s}, {:s} m/yr, 11.7 ka is {:4.1f} m above bedrock, with res {:4.0f}".format(
                name, aname, z[-1] - interper(11700), dage_interper(interper(11700))
            )
        )

        depth_40 = rev_interper(40)
        depth_100 = rev_interper(100)
        depth_1000 = rev_interper(1000)

        print(
            "For 40, 100 and 1000 yr per meter, {:4.1f}, {:4.1f} and {:4.1f} m".format(depth_40, depth_100, depth_1000)
        )

        ax40.plot(
            age_interper(depth_40),
            606 - depth_40,
            marker=marker_dict[aname],
            markerfacecolor=color_dict[name],
            markeredgecolor="k",
            linestyle="none",
        )

        axdum.plot(
            40,
            depth_40,
            marker=marker_dict[aname],
            markerfacecolor=color_dict[name],
            markeredgecolor="k",
            linestyle="none",
        )

        axdum.plot(
            100,
            depth_100,
            marker=marker_dict[aname],
            markerfacecolor=color_dict[name],
            markeredgecolor="k",
            linestyle="none",
        )

        axdum.plot(
            1000,
            depth_1000,
            marker=marker_dict[aname],
            markerfacecolor=color_dict[name],
            markeredgecolor="k",
            linestyle="none",
        )

        ax_comp.plot(
            dage_holocene / 1000,
            z[-1] - z_holocene,
            marker=marker_dict[aname],
            markerfacecolor=color_dict[name],
            markeredgecolor="k",
            linestyle="none",
        )

        if name not in in_H:
            ax_age.plot([], [], label=name, color=color_dict[name])
            in_H.append(name)

for name, odict in out_dict.items():
    for aname, (H, acc, age) in odict.items():
        if aname == "0.18":
            anamep = "0.1758"
        else:
            anamep = aname
        if aname not in in_A:
            ax_age.plot(
                [],
                [],
                linestyle=linestyle[aname],
                color="k",
                label="{:d} kg m$^{{-2}}$ yr$^{{-1}}$, {:s}".format(int(float(anamep) * 911), acc_dict[aname]),
            )
            ax_comp.plot(
                [],
                [],
                label="{:d} kg m$^{{-2}}$ yr$^{{-1}}$, {:s}".format(int(float(anamep) * 911), acc_dict[aname]),
                linestyle="none",
                marker=marker_dict[aname],
                color="k",
            )
            in_A.append(aname)

print("20 m up, min max", np.min(at_20), np.max(at_20))
print("56 m up, min max", np.min(at_56), np.max(at_56))
print("250 ka, min max", np.min(depths_of_250ka), np.max(depths_of_250ka))

ax_age.set_ylabel("Depth (m)")
ax_age.set_xlabel("Age (ka)")
ax_age.set_xlim(0, 20)
ax_age.set_ylim(z[-1], 0)

ax_age.legend(loc="upper right", frameon=False, fontsize=9)
ax_comp.legend(loc="upper right", frameon=False, fontsize=9)
ax_comp.set_ylim(0, 175)
ax_comp.set_xlim(1.0e-1, 1.0e2)
ax_comp.set_xlabel("Age resolution (kyr m$^{{-1}}$)")
ax_comp.set_ylabel("Height above bedrock of 11.7 ka")

ax_age.text(0.01, 0.98, "a", fontsize=14, transform=ax_age.transAxes, ha="left", va="top")
ax_comp.text(0.01, 0.98, "b", fontsize=14, transform=ax_comp.transAxes, ha="left", va="top")

fig.savefig("../../plots/JOG-2024-0020.Figure10.pdf")
