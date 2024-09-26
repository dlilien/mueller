# coding: utf-8
import matplotlib.pyplot as plt
import h5py
import numpy as np
from common_inputs import color_dict

fn_template = "full_temp_n{:d}.h5"

nxs = [6, 12, 25, 50, 100, 150, 200, 250, 300, 350]

with h5py.File(fn_template.format(100), "r") as f:
    runs = [name for name in f.keys() if "HOB" not in name]
    truth = {name: f[name]["T_final"][:] for name in runs}
    z = f["HOB_final"][:]

with h5py.File(fn_template.format(nxs[-1]), "r") as f:
    truns = [name for name in f.keys() if "HOB" not in name]
    for name in truns:
        truth[name] = f[name]["T_final"][:]

ls_dict = {name: ls for name, ls in zip(runs, ["solid", "dashed", "dotted", "dashdot"])}
errs = np.zeros((len(nxs) - 1, len(runs)))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(figsize=(6.5, 5.0))
ax2.set_yscale("log")
for i, n in enumerate(nxs):
    with h5py.File(fn_template.format(n), "r") as f:
        for j, (run, ls) in enumerate(ls_dict.items()):
            if run in f:
                ax1.plot(
                    f[run]["T_final"][:], f["HOB_final"][:], color="C{:d}".format(i + 4), label=str(n), linestyle=ls
                )
                if i < len(nxs) - 1:
                    errs[i, j] = np.std((truth[run] - f[run]["T_final"][:]) / truth[run])
            elif i < len(nxs) - 1:
                errs[i, j] = np.nan
errs[errs == 0.0] = np.nan

for j, run in enumerate(runs):
    ax2.plot(nxs[:-1], errs[:, j], label=run, marker="o", color=color_dict[run])

ax1.legend(loc="best")
ax2.legend(loc="best", frameon=False)

ax2.set_xlim(0, 300)
ax2.set_ylim(1.0e-5, 1.0e-1)
ax2.set_ylabel(r"$\sigma_{T}$ / T")
ax2.set_xlabel("Mesh resolution (total nodes)")

fig2.tight_layout(pad=0.1)
# fig2.savefig('mesh_sensitivity_temp.pdf')
fig2.savefig("../../plots/JOG-2024-0020.SuppFigure3.pdf")

plt.show()
