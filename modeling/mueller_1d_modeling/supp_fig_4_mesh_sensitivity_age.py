# coding: utf-8
import matplotlib.pyplot as plt
import h5py
import numpy as np
from common_inputs import acc_dict

fn_template = "modeled_ages_n{:d}.h5"
nxs = [150, 300, 600, 1200, 2400, 4800, 9600]
target_run = "200 m during HCO"
target_acc = "0.28"

with h5py.File(fn_template.format(nxs[-1]), "r") as f:
    runs = [name for name in f[target_run].keys()]
    H = f[target_run][target_acc]["depth"][:]
    truth = {name: f[target_run][name]["age"][:] for name in runs}

errs = np.zeros((len(nxs) - 1, len(runs)))

fig1, (ax1, ax1a, ax1b) = plt.subplots(1, 3, figsize=(12, 6))
fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
ax2.set_yscale("log")
for i, n in enumerate(nxs):
    with h5py.File(fn_template.format(n), "r") as f:
        ax1.plot(
            f[target_run][target_acc]["age"][:] / 1000,
            f[target_run][target_acc]["depth"][:],
            color="C{:d}".format(i + 4),
            label=str(n),
        )
        if i < (len(nxs) - 1):
            ax1a.plot(
                truth[target_acc] / 1000.0 - f[target_run][target_acc]["age"][:] / 1000,
                f[target_run][target_acc]["depth"][:],
                color="C{:d}".format(i + 4),
                label=str(n),
            )
            ax1b.plot(
                (truth[target_acc] - f[target_run][target_acc]["age"][:]) / truth[target_acc],
                f[target_run][target_acc]["depth"][:],
                color="C{:d}".format(i + 4),
                label=str(n),
            )
        for j, run in enumerate(runs):
            if i < (len(nxs) - 1):
                errs[i, j] = np.std((truth[run][2:] - f[target_run][run]["age"][:][2:]) / truth[run][2:])

for j, run in enumerate(runs):
    if run == "0.18":
        anamep = "0.1758"
    else:
        anamep = run
    ax2.plot(
        np.array(nxs[:-1]) / 606.0,
        errs[:, j],
        marker="o",
        color="C{:d}".format(j + 4),
        label="{:d} kg m$^{{-2}}$ yr$^{{-1}}$, {:s}".format(int(float(anamep) * 911), acc_dict[run]),
    )

ax1.set_xlim(0, 20)
ax1.set_ylim(610, 0)
ax1.legend(loc="best")
ax2.legend(loc="best", frameon=False)
ax2.set_ylim(1e-3, 100)
ax2.set_xlim(0, 8)
ax1a.set_xlim(-5, 5)
ax1a.set_ylim(610, 0)
ax1b.set_xlim(-5, 5)
ax1b.set_ylim(610, 0)

ax2.set_xlabel("Mesh resolution (nodes / m)")
ax2.set_ylabel(r"$\sigma_{age}$ / age")

fig2.tight_layout(pad=0.1)
fig2.savefig("../../plots/JOG-2024-0020.SuppFigure4.pdf")
