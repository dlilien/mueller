# coding: utf-8
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

f = h5py.File("full_temp_n50.h5")
plt.figure(num="Fine")
for i in range(len(f["200 m during HCO"]["output_times"][:])):
    plt.plot(
        f["200 m during HCO"]["T_full"][:, i],
        f["HOB_full"][:],
        color=cm.viridis(
            (f["200 m during HCO"]["output_times"][i] - f["200 m during HCO"]["output_times"][0])
            / (f["200 m during HCO"]["output_times"][-1] - f["200 m during HCO"]["output_times"][0])
        ),
    )

f2 = h5py.File("full_temp_n25.h5")
plt.figure(num="Coarse")
for i in range(len(f2["200 m during HCO"]["output_times"][:])):
    plt.plot(
        f2["200 m during HCO"]["T_full"][:, i],
        f2["HOB_full"][:],
        color=cm.viridis(
            (f2["200 m during HCO"]["output_times"][i] - f2["200 m during HCO"]["output_times"][0])
            / (f2["200 m during HCO"]["output_times"][-1] - f2["200 m during HCO"]["output_times"][0])
        ),
    )
plt.show()
