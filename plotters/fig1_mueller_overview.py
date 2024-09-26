#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy
import cartopy.feature as cfeature
from shapely.geometry import Polygon, box, Point
import matplotlib.gridspec as mgridspec
import rasterio
from rasterio.plot import plotting_extent
from scalebar import scalebar

from cartopolar.canadian_maps import CAN, heiberg_CAN, canadian_inset, mueller_map
from cartopolar.canadian_maps import CAN_EXTENT as CE
from cartopolar.canadian_maps import MUELLER_CAN_EXTENT as ME
from cartopolar.canadian_maps import AXEL_HEIBERG_CAN_EXTENT as AE

ZE = (6198000, 6206000, 4806811.511557467, 4812247.890650768)


def normalize(array):
    array_min, array_max = np.nanmin(array) * 1.0, np.nanmax(array) * 0.7
    return (array - array_min) / (array_max - array_min)


FS = 10
mcrs = "EPSG:3348"
crs = "EPSG:3348"

best_loc_fn = "../RES/UWB/best_loc.gpkg"
best_loc = gpd.read_file(best_loc_fn).to_crs(crs)

camp_loc = gpd.GeoDataFrame(geometry=[Point(-91.687, 79.870)], crs="EPSG:4326").to_crs(crs)
weather_loc = gpd.GeoDataFrame(geometry=[Point(-91.645, 79.861999992)], crs="EPSG:4326").to_crs(crs)
eureka_loc = gpd.GeoDataFrame(geometry=[Point(-85.92919, 79.98984)], crs="EPSG:4326").to_crs(crs)
highest_loc = gpd.GeoDataFrame(geometry=[Point(-91.859, 79.888)], crs="EPSG:4326").to_crs(crs)

asp = (CE[3] - CE[2]) / (CE[1] - CE[0])
heiberg_asp = (AE[3] - AE[2]) / (AE[1] - AE[0])
mueller_asp = (ME[3] - ME[2]) / (ME[1] - ME[0])

mueller_shape = Polygon([(ME[0], ME[2]), (ME[0], ME[3]), (ME[1], ME[3]), (ME[1], ME[2]), (ME[0], ME[2])])
mueller_df = gpd.GeoDataFrame({"mueller": [0]}, geometry=[mueller_shape]).set_crs(mcrs).to_crs(crs)

axel_shape = Polygon([(AE[0], AE[2]), (AE[0], AE[3]), (AE[1], AE[3]), (AE[1], AE[2]), (AE[0], AE[2])])
axel_df = gpd.GeoDataFrame({"axel": [0]}, geometry=[axel_shape]).set_crs(mcrs).to_crs(crs)


ice = gpd.read_file("../../qgis/OSM/muellers_ice.gpkg").to_crs(crs)

rivers = gpd.read_file("../../qgis/OSM/ne_50m_rivers_lake_centerlines_scale_rank.shp").to_crs(crs)
lakes = gpd.read_file("../../qgis/OSM/ne_50m_lakes.shp").to_crs(crs)
states_provinces = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none"
)
glaciers = cfeature.NaturalEarthFeature(category="physical", name="glaciated_areas", scale="50m", facecolor="0.95")
raster = rasterio.open("../../imagery/sentinel/sentinel_2_cloudless.tif")
raster4 = rasterio.open("../../imagery/sentinel/T16XDP_20230802T202849_B04_10m.tif")
raster3 = rasterio.open("../../imagery/sentinel/T16XDP_20230802T202849_B03_10m.tif")
raster2 = rasterio.open("../../imagery/sentinel/T16XDP_20230802T202849_B02_10m.tif")

mueller_ice = gpd.read_file("../../qgis/rgi/rgi60_muellers_simple.gpkg").to_crs(crs)

# Convert to numpy arrays
red = raster.read(3)
green = raster.read(2)
blue = raster.read(1)

# Stack bands
nrg = np.dstack((red, green, blue))

nrg2 = np.dstack((raster4.read(1) / 10000.0, raster3.read(1) / 10000.0, raster2.read(1) / 10000.0)).astype(float)


def threepanel():
    width = 7.8
    height = heiberg_asp * 0.5 * width * 1.2
    fig = plt.figure(figsize=(width, height))
    gs = mgridspec.GridSpec(
        6,
        2,
        left=0.065,
        right=0.995,
        bottom=0.04,
        top=0.99,
        hspace=0.0,
        wspace=0.0,
        height_ratios=[0.53, 0.02, 0.4, 0.15, 0.1, 0.5],
        width_ratios=[0.45, 0.55],
    )
    ax1 = fig.add_subplot(gs[2:, 0], projection=CAN())
    ax_can = fig.add_subplot(gs[0, 0], projection=CAN())
    ax2 = fig.add_subplot(gs[:3, 1], projection=CAN())
    ax3 = fig.add_subplot(gs[5, 1], projection=CAN())
    heiberg_CAN(ax1)
    mueller_map(ax2)
    mueller_map(ax3)
    ax3.set_xlim(*ZE[:2])
    ax3.set_ylim(*ZE[2:])
    canadian_inset(fig, 0.26, 0.70, width=0.1, ax=ax_can)

    extent_box = box(ZE[0], ZE[2], ZE[1], ZE[3])
    ax2.add_geometries([extent_box], ax3.projection, ec='r', fc='none')

    ax1.imshow(nrg, extent=plotting_extent(raster))
    mueller_df.plot(ax=ax1, facecolor="none", edgecolor="orange", alpha=1.0, zorder=999999)

    gl = ax1.gridlines(draw_labels=True, color="k", linewidth=0.5)
    gl.right_labels = False
    gl.top_labels = False
    # ax1.gridlines(color='k', linewidth=0.5)

    places = gpd.read_file("../../logistics/survey_planning/plot_names.gpkg").to_crs(ax1.projection.crs)
    for place, loc in zip(places["name"], places["geometry"]):
        if "ller Ice" in place:
            ax1.annotate(
                place.replace(" Ice", "\nIce"),
                (loc.x, loc.y),
                xytext=(45, 25),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>", color="b"),
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", fc="w", ec="b"),
                zorder=999999,
            )
        if "cie Ice" in place:
            ax1.annotate(
                place.replace(" Ice", "\nIce"),
                (loc.x, loc.y),
                xytext=(-30, -15),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>", color="b"),
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="w", ec="b"),
                zorder=9999999,
            )
        if "White" in place:
            ax1.annotate(
                place.replace(" ", "\n"),
                (loc.x, loc.y),
                xytext=(30, -25),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>", color="b"),
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", fc="w", ec="b"),
                zorder=999999,
            )

    ax1.annotate(
        "Princess\nMargaret\nRange",
        (6198000, 4802000),
        xytext=(-40, -25),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-|>", color="b"),
        ha="right",
        va="center",
        bbox=dict(boxstyle="round", fc="w", ec="b"),
        zorder=9999999,
    )
    ax1.annotate(
        "Iceberg\nGlacier",
        (6190208, 4760821),
        xytext=(-35, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-|>", color="b"),
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="w", ec="b"),
        zorder=9999999,
    )
    ax1.annotate(
        "Eureka",
        (6317443, 4827838),
        xytext=(0, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-|>", color="b"),
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", fc="w", ec="b"),
        zorder=9999999,
    )
    # ax1.annotate("Airdrop\nGlacier", (6159000, 4800000), xytext=(-35, 10), textcoords="offset points", arrowprops=dict(arrowstyle="-|>", color="b"), ha="right", va="bottom", bbox=dict(boxstyle="round", fc="w", ec="b"), zorder=9999999)

    ax1.plot([AE[0] + 30000, AE[0] + 80000], [AE[2] + 18000, AE[2] + 18000], color="k", linewidth=4)
    ax1.text(AE[0] + 55000, AE[2] + 8000, "50 km", ha="center", va="center", color="k")

    # ax_can = canadian_inset(fig, 0.26, 0.70, height=0.29)
    ax_can.set_facecolor("lightskyblue")
    ax_can.add_feature(cartopy.feature.LAND, facecolor="palegoldenrod")
    for scale in range(1, 3):
        rivers.loc[rivers["scalerank"] == scale].plot(ax=ax_can, color="lightskyblue", linewidth=1.0 / scale)
    for scale in range(1):
        lakes.loc[lakes["scalerank"] == scale].plot(ax=ax_can, facecolor="lightskyblue", edgecolor="none")

    ax_can.add_feature(states_provinces, edgecolor="gray", linewidth=0.5)
    ice.plot(ax=ax_can, color="w", edgecolor="w")
    ax_can.add_feature(glaciers, edgecolor="0.95")
    axel_df.plot(ax=ax_can, facecolor="none", edgecolor="r", alpha=1, zorder=999998)

    ax2.imshow(blue, extent=plotting_extent(raster), cmap="gray")
    surf_ds = rasterio.open("../../topography/arcticDEM/reference_dem_64m_can.tif")
    surf = surf_ds.read(1, masked=True)
    cm = ax2.imshow(surf, extent=plotting_extent(surf_ds), vmin=0, vmax=1800, cmap="terrain", alpha=1.00)
    cax = fig.add_axes([0.525, 0.40, 0.45, 0.02])
    cbr = plt.colorbar(cm, cax, label="Elevation (m)", orientation="horizontal", extend="max")
    cbr.ax.set_xticks([0, 600, 1200, 1800])
    ax2.plot(best_loc.geometry.x, best_loc.geometry.y, marker="*", color="k", markersize=8, zorder=99999)

    def path_length(path):
        v = path.vertices
        dv = np.diff(v, axis=0)
        return np.sum(np.sqrt(np.sum(dv**2, axis=-1)))

    CS = ax2.contour(np.flipud(surf), extent=plotting_extent(surf_ds), levels=np.arange(0, 2250, 250), colors='k', lw=0.5, linewidths=0.5)
    deleted_paths = []
    for c in CS.collections:
        paths = c.get_paths()
        if len(paths) > 1:
            paths.sort(key=path_length, reverse=True)
            for p in paths[1:]:
                deleted_paths.append((c, p))
            del paths[1:]
    fmt = {level: "{:d} m".format(int(level)) for level in CS.levels[-4:]}
    ax2.clabel(CS, CS.levels[-4:], inline=True, fmt=fmt, fontsize=8)
    for c, p in deleted_paths:
        c.get_paths().append(p)

    # Zoomed-in map
    ax3.imshow(nrg2, extent=plotting_extent(raster2))
    # ax3.imshow(surf, extent=plotting_extent(surf_ds), vmin=0, vmax=1800, cmap="terrain", alpha=1.00)
    ax3.plot(best_loc.geometry.x, best_loc.geometry.y, marker="*", color="k", markersize=8, zorder=99999)
    ax3.text(best_loc.geometry.x - 100, best_loc.geometry.y, "Best drill site\n& 6.62-m core", ha='right', va='bottom', zorder=99999)

    ax3.plot(highest_loc.geometry.x, highest_loc.geometry.y, marker="o", color="C4", markersize=8, zorder=99999)
    ax3.text(highest_loc.geometry.x - 100, highest_loc.geometry.y + 100, "Ice-cap\nsummit", ha='right', va='bottom', zorder=99999)

    ax3.plot(camp_loc.geometry.x, camp_loc.geometry.y, marker="^", color="C0", markersize=8, zorder=99999)
    ax3.text(camp_loc.geometry.x, camp_loc.geometry.y + 100, "Camp, 67-cm pit,\n& 1.6-m core", ha='center', va='bottom', zorder=99999)
    ax3.plot(weather_loc.geometry.x, weather_loc.geometry.y, marker="2", color="C1", markersize=8, zorder=99999)
    ax3.text(weather_loc.geometry.x - 100, weather_loc.geometry.y, "Weather station", ha='right', va='top', zorder=99999)

    CS2 = ax3.contour(np.flipud(surf), extent=plotting_extent(surf_ds), levels=np.arange(1800, 2250, 50), colors='0.6', lw=0.5, linewidths=0.5)
    fmt = {level: "{:d} m".format(int(level)) for level in CS2.levels}
    ax3.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=8)

    ax3.plot([ZE[1] - 500, ZE[1] - 2500], [ZE[3] - 250, ZE[3] - 250], lw=2, color='k')
    ax3.text(ZE[1] - 1500, ZE[3] - 500, "2 km", ha='center', va='top')

    for feature in mueller_ice.iterrows():
        for i, poly in enumerate(feature[1].geometry.geoms):
            try:
                x, y = poly.boundary.xy
                ax2.plot(x, y, color="k")
            except NotImplementedError:
                for ls in poly.boundary.geoms:
                    x, y = ls.xy
                    ax2.plot(x, y, color="k")

    scalebar(ax2, 'UR', yoff=3000, xoff=4000, ytoff=3000)

    ax_can.add_feature(cartopy.feature.BORDERS, edgecolor="k", zorder=100)
    for ax, letter, asp, w in zip([ax_can, ax1, ax2, ax3], "abcde", [0.5, 1, 1, 0.49], [1, 1, 1, 1]):
        h = 0.060 / asp
        ax.fill(
            [0, 0.065 * w, 0.065 * w, 0, 0],
            [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h],
            transform=ax.transAxes,
            color="w",
            edgecolor="k",
            zorder=99998,
        )
        ax.text(0.005, 0.993, letter, ha="left", va="top", fontsize=14, transform=ax.transAxes, zorder=99999)

    gl = ax2.gridlines(draw_labels=True, dms=True, color="k", linewidth=0.5)
    gl.right_labels = False
    gl.top_labels = False
    fig.savefig("../plots/JOG-2024-0020.Figure1.tif", dpi=400)


if __name__ == "__main__":
    threepanel()
