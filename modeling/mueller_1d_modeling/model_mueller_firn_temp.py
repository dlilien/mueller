#!/usr/bin/env python
# coding: utf-8

from firedrake import (
    IntervalMesh,
    FunctionSpace,
    Function,
    TestFunction,
    Constant,
    dx,
    SpatialCoordinate,
    ds,
    DirichletBC,
    project,
    conditional,
    solve,
)
import tqdm
import numpy as np
from math import floor
import h5py
import pandas as pd
from scipy.interpolate import interp1d


H = 50
Hfinal = 606.0
n = 500
mesh = IntervalMesh(n, H)

V = FunctionSpace(mesh, "CG", 2)
V_out = FunctionSpace(mesh, "CG", 1)


u_ = Function(V, name="Temperature")
u = Function(V, name="TemperatureNext")

v = TestFunction(V)

(x,) = SpatialCoordinate(mesh)

acc = 0.25
bottom_gradient = 0.025


def vert_vel(a, z, W):
    # Essentially assume snow falls at 450 kg m3 and exits at 900 at the accumulation rate
    return Function(W).interpolate(Constant(a))


def vert_vel_lliboutry(a, z, H, W, p=3.0):
    return Function(W).interpolate(
        a * (1 - (p + 2.0) / (p + 1.0) * (1.0 - (H - z) / H) + 1.0 / (p + 1.0) * (1.0 - (H - z) / H) ** (p + 2))
    )


w = vert_vel_lliboutry(acc, x, Hfinal, V, p=3.0)

year = 365.25 * 24 * 60 * 60
# conductivity = 2.35 # W / m / K
capacity = 2090  # 2.108 * year ** 2.0  # kJ / kg / K
# diffusivity = fd.Constant(2.3e-3 / (917 * 2.0) * year)
diffusivity = 1.02e-6 * year
cdiffusivity = Constant(diffusivity)
density = 910.0
timestep = min(H / n / diffusivity, 1.0 / 365.0)

F = ((u - u_) / timestep * v + cdiffusivity * u.dx(0) * v.dx(0) + u * (v * w).dx(0)) * dx + (
    u.dx(0) - bottom_gradient
) * v / timestep * 1.0e4 * conditional(x > H / 2, Constant(1.0), Constant(0.0)) * ds

nyears = 200.0
t = 0.0
end = (
    nyears + 23.0 / 365.0
)  # There is a 23-day offset from the start of record (May 3) to the date we measured (May 26)
times = np.arange(t, end + timestep, timestep)


wx_df = pd.read_csv("../../queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
wx_df["Date_Time"] = pd.to_datetime(wx_df["Date_Time"])
daily_df = wx_df.groupby(pd.Grouper(key="Date_Time", freq="1D")).mean()
surf_T_in = daily_df["Temperature_21049794_deg_C"].values
td = daily_df.index - daily_df.index[0]
times_in = (td.seconds / year + td.days / 365).values
surf_T = np.hstack(
    [
        interp1d(times_in, surf_T_in)(times[np.logical_and(times < 2.0 + i * 2.0, times >= i * 2.0)] - i * 2.0)
        for i in range(int(end) // 2)
    ]
    + [interp1d(times_in, surf_T_in)(times[times >= floor(end)] - floor(end))]
)


# In[5]:


ic = project(Constant(np.mean(surf_T)), V)

u_.assign(ic)
u.assign(ic)


# In[6]:


# surf_T = np.sin(2 * times * np.pi / end)
surf_T_i = Constant(surf_T[0])
bc = DirichletBC(V, surf_T_i, 1)


# In[7]:


# We now create an object for output visualisation::
Ts = [project(u, V_out)]

T_1cm = [u.at(0.0)]
T_160cm = [u.at(1.6)]
T_662cm = [u.at(6.62)]

for i in tqdm.trange(len(times[1:])):
    surf_T_i.assign(surf_T[i])
    solve(F == 0, u, bcs=[bc])
    u_.assign(u)
    Ts.append(project(u, V_out))
    T_1cm.append(u.at(0.0))
    T_160cm.append(u.at(1.6))
    T_662cm.append(u.at(6.62))

T_662cm = np.array(T_662cm)
print("Final temperature at 6.62 m is:", T_662cm[-1])

num_offsets = int(1.0 / timestep)
corrs = np.zeros((num_offsets,))
subset = (int((nyears - 2.0) / timestep), int((nyears - 1.0) / timestep))

for i in range(num_offsets):
    corrs[i] = np.corrcoef(surf_T[subset[0] : subset[1]], T_662cm[subset[0] + i : subset[1] + i])[0, 1]

ind_max = np.argmax(corrs)

print("Offset in days is:", timestep * ind_max * 365.25)
print("Offset in degrees is:", timestep * ind_max * 360)
print("Max correlation is:", corrs[ind_max])

x_out = np.linspace(0, H, 500)
time_interval = 365.25 / 12.0
inds = np.array([-390, -359, -329, -298, -267, -237, -206, -176, -145, -114, -86, -56, -26, -1])
out_ts = np.empty((x_out.shape[0], inds.shape[0]))
for i, ind in enumerate(inds):
    out_ts[:, i] = Ts[ind].at(x_out)

with h5py.File("firn_temp.h5", "w") as fout:
    by_depth = fout.create_group("by_depth")
    by_time = fout.create_group("by_time")
    by_corr = fout.create_group("corr")
    by_depth.create_dataset("time", data=times)
    by_depth.create_dataset("662cm", data=T_662cm)
    by_depth.create_dataset("160cm", data=T_160cm)
    by_depth.create_dataset("1cm", data=T_1cm)
    by_time.create_dataset("days_after_20230501", data=390 + inds)
    by_time.create_dataset("depths", data=x_out)
    by_time.create_dataset("temps", data=out_ts)
    by_corr.create_dataset("corr", data=corrs)
    by_corr.create_dataset("offset", data=times[:num_offsets])
