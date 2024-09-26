#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.gridspec as mgridspec
import rasterio
from rasterio.plot import plotting_extent
from scalebar import scalebar

from cartopolar.canadian_maps import CAN, mueller_map
from cartopolar.canadian_maps import MUELLER_CAN_EXTENT as ME

FS = 10

crs = "EPSG:3348"

muller_asp = (ME[3] - ME[2]) / (ME[1] - ME[0])

raster = rasterio.open("../imagery/sentinel/sentinel_2_cloudless.tif")

# Convert to numpy arrays
red = raster.read(3)
green = raster.read(2)
blue = raster.read(1)
nrg = np.dstack((red, green, blue))

best_loc_fn = "../RES/UWB/best_loc.gpkg"
best_loc = gpd.read_file(best_loc_fn).to_crs(crs)

v = rasterio.open("../velocity/insar/sentinel_1/TC/S1_v_can.tif")
vel = v.read(1, masked=True)

widthm = 7.05
heightm = widthm * 0.5 * muller_asp * 0.95
fig = plt.figure(figsize=(widthm, heightm))
gs = mgridspec.GridSpec(1, 2, left=0.064, right=0.995, bottom=0.05, top=0.995, wspace=0.137, hspace=0.0)
axsc = fig.add_subplot(gs[0, 0], projection=CAN())
axsc_itslive = fig.add_subplot(gs[0, 1], projection=CAN())

for ax in [axsc, axsc_itslive]:
    mueller_map(ax)
    ax.set_ylim(ME[2], ME[3])
    ax.set_xlim(ME[0], ME[1])
    ax.imshow(blue, cmap="gray", extent=plotting_extent(raster))
    gl = ax.gridlines(color="k", linewidth=0.5, draw_labels=True, dms=True)
    gl.right_labels = False
    gl.top_labels = False
    ax.plot(best_loc.geometry.x[0], best_loc.geometry.y[0], marker="*", color="k", markersize=5, zorder=99999)

v = rasterio.open("../velocity/insar/sentinel_1/TC/S1_v_can.tif")
vel = v.read(1, masked=True)
# vel[vel == -9999.0] = np.nan

cm = axsc.imshow(vel, extent=plotting_extent(v), vmin=0, vmax=20, cmap="Reds", alpha=0.75)

v2 = rasterio.open("../velocity/its_live/CAN_G0120_0000_v_can.tif")
vel2 = v2.read(1, masked=True)
axsc_itslive.imshow(vel2, extent=plotting_extent(v2), vmin=0, vmax=20, cmap="Reds", alpha=0.75)

ll, bb = 0.52, 0.83
bb -= 0.15
axsc.fill([ll, 1, 1, ll, ll], [bb, bb, 1, 1, bb], transform=axsc.transAxes, color="w", edgecolor="k", zorder=1000)
scalebar(axsc, "UR", xoff=7500, yoff=17500, ytoff=3000)

caxsc = fig.add_axes([0.30, 0.94, 0.185, 0.03])
plt.tick_params(
    which="both", bottom=False, top=False, labelbottom=False, labelleft=False
)  # both major and minor ticks are affected  # ticks along the bottom edge are off  # ticks along the top edge are off
plt.colorbar(cm, cax=caxsc, extend="max", label=r"Speed (m yr$^{-1}$)", orientation="horizontal")

for ax, letter, h in zip([axsc, axsc_itslive], "abcde", [0.06, 0.06]):
    ax.fill(
        [0, 0.06, 0.06, 0, 0], [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h], transform=ax.transAxes, color="w", edgecolor="k"
    )
    ax.text(0.005, 0.997, letter, ha="left", va="top", fontsize=14, transform=ax.transAxes)
fig.savefig("../plots/JOG-2024-0020.Figure2.tif", dpi=400)
