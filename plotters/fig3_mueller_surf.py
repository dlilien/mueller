#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import rasterio
from rasterio.plot import plotting_extent
import geopandas as gpd
from scalebar import scalebar

from cartopolar.canadian_maps import CAN, mueller_map
from cartopolar.canadian_maps import AXEL_HEIBERG_TIGHT_CAN_EXTENT as AE
from cartopolar.canadian_maps import MUELLER_CAN_EXTENT as ME

FS = 10

crs = "EPSG:3348"

best_loc_fn = "../RES/UWB/best_loc.gpkg"
best_loc = gpd.read_file(best_loc_fn).to_crs(crs)

heiberg_asp = (AE[3] - AE[2]) / (AE[1] - AE[0])
mueller_asp = (ME[3] - ME[2]) / (ME[1] - ME[0])
zoom_xlim = (6.154e6, 6.220e6)
zoom_ylim = (4.756e6, 4.822e6)
zoom_asp = (zoom_ylim[1] - zoom_ylim[0]) / (zoom_xlim[1] - zoom_xlim[0])
background = rasterio.open("../imagery/sentinel/sentinel_2_cloudless.tif")
# Convert to numpy arrays
red = background.read(3)
green = background.read(2)
blue = background.read(1)
bck_arr = np.dstack((red, green, blue))

surf = rasterio.open("../topography/arcticDEM/reference_dem_64m_can.tif")
trend = rasterio.open("../topography/arcticDEM/aligned_mosaics/muellers_2012-2021_trend_epsg3348_128m_clipped.tif", masked=True)

widthm = 7.05
top = 0.99
heightm = widthm * 0.5 * mueller_asp * 0.95
fig = plt.figure(figsize=(widthm, heightm))
gs = mgridspec.GridSpec(1, 2, left=0.064, right=0.995, bottom=0.05, top=top, wspace=0.137, hspace=0.0)
axsc = fig.add_subplot(gs[0, 0], projection=CAN())
axsc_icesat = fig.add_subplot(gs[0, 1], projection=CAN())

for ax in [axsc, axsc_icesat]:
    mueller_map(ax)
    ax.set_ylim(ME[2], ME[3])
    ax.set_xlim(ME[0], ME[1])
    ax.imshow(blue, cmap="gray", extent=plotting_extent(background))
    gl = ax.gridlines(color="k", linewidth=0.5, draw_labels=True, dms=True)
    gl.right_labels = False
    gl.top_labels = False
    ax.plot(best_loc.geometry.x[0], best_loc.geometry.y[0], marker="*", color="k", markersize=8, zorder=99999)

trend_cm = axsc.imshow(trend.read(1, masked=True), extent=plotting_extent(trend), vmin=-1, vmax=1, cmap="RdBu")

axsc_icesat.show_tif("../topography/icesat2/ATL15_CN_0314_01km_002_01_annual_trend_epsg3348.tif", vmin=-1, vmax=1, cmap="RdBu")

ll, bb = 0.52, 0.83
bb -= 0.15
axsc.fill([ll, 1, 1, ll, ll], [bb, bb, 1, 1, bb], transform=axsc.transAxes, color="w", edgecolor="k", zorder=1000)
scalebar(axsc, "UR", xoff=7500, yoff=17500, ytoff=3000)

caxsc = fig.add_axes([0.30, 0.94, 0.185, 0.03])
plt.tick_params(
    which="both", bottom=False, top=False, labelbottom=False, labelleft=False
)  # both major and minor ticks are affected  # ticks along the bottom edge are off  # ticks along the top edge are off
trend_cbr = plt.colorbar(
    trend_cm, cax=caxsc, label=r"$\partial H/\partial t$ (m yr$^{-1}$)", extend="both", orientation="horizontal"
)
trend_cbr.set_ticks([-1, -0.5, 0, 0.5, 1.0])
trend_cbr.ax.tick_params(labelsize=8)

for ax, letter, h in zip([axsc, axsc_icesat], "abcde", [0.06, 0.06]):
    ax.fill(
        [0, 0.06, 0.06, 0, 0], [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h], transform=ax.transAxes, color="w", edgecolor="k"
    )
    ax.text(0.005, 0.997, letter, ha="left", va="top", fontsize=14, transform=ax.transAxes)

fig.savefig("../plots/JOG-2024-0020.Figure3.tif", dpi=400)
