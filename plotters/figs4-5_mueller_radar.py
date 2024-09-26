#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.gridspec as mgridspec
import rasterio
from rasterio.plot import plotting_extent
import xarray as xr
import rioxarray as rxr

from cartopolar.canadian_maps import CAN, mueller_map
from cartopolar.canadian_maps import MUELLER_CAN_EXTENT as ME
from scalebar import scalebar

from impdar.lib import RadarData, plot

vel_fn = "../velocity/insar/sentinel_1/TC/S1_v_can.tif"
vel_arr = rxr.open_rasterio(vel_fn, masked=True)[0, :, :]

crs = "EPSG:3348"
pick_fns = {
    "MCoRDS": ["../RES/icebridge/mcords/l2/IRMCR2_muellers_20211202.gpkg"],
    "HICARS": ["../RES/BAS_RES/BAS_RES.gpkg"],
    "AWI NB": ["../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230509_03/muller_awi_nb_sar_comb.gpkg"],
    "AWI UWB": [
        "../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230508_09/muller_awi_uwb_sar1_comb.gpkg",
        "../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230508_10/muller_awi_uwb_sar2_comb.gpkg",
    ],
    "Ground UWB": [
        "../RES/UWB/20230522_concatenated_interpolated_dx1_picked_cropped.gpkg",
        "../RES/UWB/20230523_concatenated_interpolated_dx1_picked_cropped.gpkg",
    ],
}
name_dict = {
    "MCoRDS": "THICK",
    "HICARS": "THK",
    "AWI NB": "thickness",
    "AWI UWB": "thickness",
    "Ground UWB": "L1_depth",
}
color_dict = {
    "MCoRDS": "#1b9e77",
    "HICARS": "#d95f02",
    "AWI NB": "#7570b3",
    "AWI UWB": "#e7298a",
    "Ground UWB": "C0",
}
picks_dict = {name: [gpd.read_file(fn).to_crs(crs) for fn in pfns] for name, pfns in pick_fns.items()}
for name in ["AWI NB", "AWI UWB"]:
    for df in picks_dict[name]:
        df[name_dict[name]][df["maxdiff"].values > 30.0] = np.nan
best_loc_fn = "../RES/UWB/best_loc.gpkg"
best_loc = gpd.read_file(best_loc_fn).to_crs(crs)
x, y = np.hstack([p.geometry.x for p in picks_dict["Ground UWB"]]), np.hstack([p.geometry.y for p in picks_dict["Ground UWB"]])
hpad = 250.0
wpad = 1500.0

ZE = (
    np.nanmin(x) - wpad,
    np.nanmax(x) + wpad,
    np.nanmin(y) - hpad,
    np.nanmax(y) + hpad,
)
zoom_asp = (ZE[3] - ZE[2]) / (ZE[1] - ZE[0])
mueller_asp = (ME[3] - ME[2]) / (ME[1] - ME[0])


def res_pub_map_2p():
    background = rasterio.open("../imagery/sentinel/sentinel_2_cloudless.tif")
    # Convert to numpy arrays
    blue = background.read(1)

    width = 7.05
    bottom = 0.08
    top = 0.995
    left = 0.06
    right = 0.995
    height = width * mueller_asp / 2.0 * 0.95

    fig = plt.figure(figsize=(width, height))
    gs = mgridspec.GridSpec(
        1,
        3,
        wspace=0.0,
        hspace=0.05,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width_ratios=[1, 0.14, 1],
    )
    axover = fig.add_subplot(gs[0, 0], projection=CAN())
    axthick = fig.add_subplot(gs[0, 2], projection=CAN())
    for ax in [axover, axthick]:
        mueller_map(ax)
        ax.set_ylim(ME[2], ME[3])
        ax.set_xlim(ME[0], ME[1])
        ax.imshow(blue, cmap="gray", extent=plotting_extent(background), zorder=1)
        gl = ax.gridlines(color="k", linewidth=0.5, draw_labels=True, dms=True)
        gl.right_labels = False
        gl.top_labels = False
        ax.plot(
            best_loc.geometry.x[0],
            best_loc.geometry.y[0],
            marker="*",
            color="k",
            markersize=8,
            zorder=99999,
        )

    axthick.contour_tif(vel_fn, levels=[5], colors="k")

    inc_cutoff = 100.0  # increment to separate lines
    all_x = []
    all_y = []
    all_thicks = []
    for name, pick_list in picks_dict.items():
        x, y = np.hstack([p.geometry.x for p in pick_list]), np.hstack([p.geometry.y for p in pick_list])
        inc_dist = np.hstack(([0], np.sqrt(np.diff(x) ** 2.0 + np.diff(y) ** 2.0)))
        mask = inc_dist < inc_cutoff
        y[~mask] = np.nan

        axover.plot(x, y, label=name, color=color_dict[name])
        for picks in pick_list:
            stride = 1
            p = picks[name_dict[name]][::stride]
            m = ~np.isnan(p)
            if 'maxdiff' in picks:
                md = picks['maxdiff'][::stride]
                m = np.logical_and(m, md < 30.0)
            all_x.append(picks.geometry.x[::stride][m])
            all_y.append(picks.geometry.y[::stride][m])
            all_thicks.append(picks[name_dict[name]][::stride][m])
            thick_cm = axthick.scatter(
                picks.geometry.x,
                picks.geometry.y,
                s=4,
                c=picks[name_dict[name]],
                vmin=0,
                vmax=750,
                zorder=2,
            )
    x = np.hstack(all_x)
    y = np.hstack(all_y)
    thick = np.hstack(all_thicks)
    x = xr.DataArray(x, dims="v")
    y = xr.DataArray(y, dims="v")
    vel = vel_arr.interp(x=x, y=y)
    vel[vel < 5.0] = 0.0
    vel[vel >= 5.0] = 1.0
    comask = np.logical_and(~np.isnan(thick), ~np.isnan(vel))
    print(np.corrcoef(thick[comask], vel[comask]))

    axover.legend(loc="upper right", fontsize=10, facecolor="white", framealpha=1)

    axthick.plot(
        [ZE[0], ZE[1], ZE[1], ZE[0], ZE[0]],
        [ZE[2], ZE[2], ZE[3], ZE[3], ZE[2]],
        color="r",
    )

    cax = fig.add_axes([0.78, 0.955, 0.205, 0.025])
    axthick.fill(
        [0.77, 1.0, 1.0, 0.77, 0.77],
        # [0.84, 0.84, 1.0, 1.0, 0.84],
        [0.71, 0.71, 1.0, 1.0, 0.71],
        transform=fig.transFigure,
        color="w",
        edgecolor="k",
        zorder=998,
    )
    scalebar(axthick, "UR", xoff=7500, yoff=17500, ytoff=3000)

    thick_cbr = plt.colorbar(
        thick_cm,
        extend="max",
        label=r"Ice thickness (m)",
        cax=cax,
        orientation="horizontal",
    )
    thick_cbr.set_ticks([0, 250, 500, 750])
    thick_cbr.ax.tick_params(labelsize=10)

    for ax, letter, asp in zip([axover, axthick], "abcde", [mueller_asp, mueller_asp, zoom_asp]):
        h = 0.075 / asp
        ax.fill(
            [0, 0.05, 0.05, 0, 0],
            [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h],
            transform=ax.transAxes,
            color="w",
            edgecolor="k",
            zorder=99998,
        )
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    fig.tight_layout(pad=0.0)
    fig.savefig("../plots/JOG-2024-0020.Figure4.tif", dpi=400)


def uwb_radargrams():
    fns = [
        "../RES/UWB/20230522_concatenated_interpolated_dx1_picked_cropped.mat",
        "../RES/UWB/20230523_concatenated_interpolated_dx1_picked_cropped.mat",
    ]
    # Do this slowly because of huge matrix
    rd1 = RadarData.RadarData(fns[0])
    # rd1.hcrop(5000, "right")
    right1 = 2980 * 3
    left1 = 1240 * 3
    rd1.hcrop(right1, "right")
    rd1.hcrop(left1, "left")
    rd1.reverse()

    ind1 = np.nanargmin((rd1.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd1.y_coord - best_loc.geometry.y[0]) ** 2.0)

    rd2 = RadarData.RadarData(fns[1])
    # rd2.hcrop(1900, "right")
    right2 = 663 * 3
    left2 = 0
    rd2.hcrop(right2, "right")
    ind2 = np.nanargmin((rd2.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd2.y_coord - best_loc.geometry.y[0]) ** 2.0)

    dists = np.array([rd.dist[-1] for rd in [rd1, rd2]])

    left = 0.082
    right = 0.985
    top = 0.985
    bottom = 0.09
    fig = plt.figure(figsize=(7.05, 5))
    gs = mgridspec.GridSpec(
        2,
        6,
        wspace=0.0,
        hspace=0.22,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width_ratios=[dists[1], 0.3, dists[0] - dists[1] - 1.0, 0.15, 0.1, 0.45],
    )
    axzoom = fig.add_subplot(gs[1, 2], projection=CAN())
    cax = fig.add_subplot(gs[1, -2])

    axes = []
    for i, (rd, ind, vmin, vmax) in enumerate([(rd1, ind1, 35, 70), (rd2, ind2, 35, 70)]):
        if i == 0:
            axes.append(fig.add_subplot(gs[i, :]))
        else:
            axes.append(fig.add_subplot(gs[i, 0]))
        plot.plot_radargram(
            rd,
            xdat="dist",
            ydat="depth",
            cmap="gray_r",
            clims=(vmin, vmax),
            ax=axes[i],
            fig=fig,
        )
        plot.plot_picks(
            rd,
            rd.dist,
            rd.nmo_depth,
            colors=["gold"],
            just_middle=True,
            ax=axes[i],
            fig=fig,
        )
        # axes[i].plot(rd.dist[ind], 606.9, marker="*", color="b")
        axes[i].arrow(rd.dist[ind], 750, 0, -143.1, color="b", zorder=99, head_width=0.05, head_length=30, length_includes_head=True)
    for ax in axes:
        ax.set_ylim(750, 0)
        ax.set_yticks([0, 250, 500, 750])

    background = rasterio.open("../imagery/sentinel/sentinel_2_cloudless.tif")
    # Convert to numpy arrays
    blue = background.read(1)
    mueller_map(axzoom)
    axzoom.imshow(blue, cmap="gray", extent=plotting_extent(background), zorder=1)
    gl = axzoom.gridlines(color="k", linewidth=0.5, draw_labels=True, dms=True, zorder=99999)
    gl.right_labels = False
    gl.top_labels = False
    axzoom.plot(
        best_loc.geometry.x[0],
        best_loc.geometry.y[0],
        marker="*",
        color="k",
        markersize=5,
        zorder=99999,
    )
    axzoom.set_ylim(ZE[2], ZE[3])
    axzoom.set_xlim(ZE[0], ZE[1])

    inc_cutoff = 100.0  # increment to separate lines
    for name, pick_list in picks_dict.items():
        x, y = np.hstack([p.geometry.x for p in pick_list]), np.hstack([p.geometry.y for p in pick_list])
        inc_dist = np.hstack(([0], np.sqrt(np.diff(x) ** 2.0 + np.diff(y) ** 2.0)))
        mask = inc_dist < inc_cutoff
        y[~mask] = np.nan

        for picks in pick_list:
            if name in ["AWI UWB", "AWI NB"]:
                axzoom.plot(
                    picks.geometry.x.values,
                    picks.geometry.y.values,
                    color=color_dict[name],
                    zorder=1.5
                )
                thick_cm = axzoom.scatter(
                    picks.geometry.x[picks['maxdiff'] < 30.0],
                    picks.geometry.y[picks['maxdiff'] < 30.0],
                    s=4,
                    c=picks[name_dict[name]][picks['maxdiff'] < 30.0],
                    vmin=0,
                    vmax=750,
                    zorder=2,
                )
            if name == "Ground UWB":
                thick_cm = axzoom.scatter(
                    picks.geometry.x,
                    picks.geometry.y,
                    s=4,
                    c=picks[name_dict[name]],
                    vmin=0,
                    vmax=750,
                    zorder=2,
                )

    thick_cbr = plt.colorbar(
        thick_cm,
        extend="max",
        label=r"Ice thickness (m)",
        cax=cax,
        orientation="vertical",
    )
    thick_cbr.set_ticks([0, 250, 500, 750])
    thick_cbr.ax.tick_params(labelsize=10)

    axzoom.plot(
        (ZE[0] + 500, ZE[0] + 1500),
        (ZE[2] + 1000, ZE[2] + 1000),
        linewidth=4,
        color="k",
        clip_on=False,
        zorder=1000,
    )
    axzoom.text(
        ZE[0] + 1000,
        ZE[2] + 500,
        "1 km",
        color="k",
        clip_on=False,
        zorder=1000,
        ha="center",
    )
    axzoom.text(
        picks_dict["Ground UWB"][0].geometry.x[right1],
        picks_dict["Ground UWB"][0].geometry.y[right1],
        "a",
        ha="right",
    )
    axzoom.text(
        picks_dict["Ground UWB"][0].geometry.x[left1],
        picks_dict["Ground UWB"][0].geometry.y[left1],
        "a'",
    )
    axzoom.text(
        picks_dict["Ground UWB"][1].geometry.x[left2],
        picks_dict["Ground UWB"][1].geometry.y[left2],
        "b",
        ha="right",
    )
    axzoom.text(
        picks_dict["Ground UWB"][1].geometry.x[right2],
        picks_dict["Ground UWB"][1].geometry.y[right2],
        "b'",
        ha="left",
    )
    axzoom.text(
        picks_dict["AWI UWB"][0].geometry.x[1430],
        picks_dict["AWI UWB"][0].geometry.y[1430],
        "S1",
        ha="center",
    )
    axzoom.text(
        picks_dict["AWI UWB"][0].geometry.x[1325],
        picks_dict["AWI UWB"][0].geometry.y[1325],
        "S1'",
        ha="center",
    )

    for ax, letter in zip(axes, "ab"):
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )
        ax.text(
            0.995,
            0.993,
            letter + "'",
            ha="right",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    for ax, letter, asp in zip([axzoom], "cde", [zoom_asp]):
        h = 0.075 / asp
        ax.fill(
            [0, 0.05, 0.05, 0, 0],
            [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h],
            transform=ax.transAxes,
            color="w",
            edgecolor="k",
            zorder=99998,
        )
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    CS = axzoom.contour_tif("../topography/arcticDEM/reference_dem_64m_can.tif", levels=[1800, 1850, 1900], colors="k")
    fmt = {level: "{:d} m".format(int(level)) for level in [1800, 1850, 1900]}
    axzoom.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8)
    fig.savefig("../plots/JOG-2024-0020.Figure5.tif", dpi=400)


def uwb_radargrams_supp():
    fns = [
        "../RES/UWB/20230522_cat_dx1combined_Int2_fk_picked_cropped.mat",
        "../RES/UWB/20230523_cat_dx1combined_Int2_fk_picked_cropped.mat",
    ]
    # Do this slowly because of huge matrix
    rd1 = RadarData.RadarData(fns[0])
    # rd1.hcrop(5000, "right")
    rd1.hcrop(8944, "right")
    rd1.hcrop(3738, "left")
    rd1.reverse()

    ind1 = np.nanargmin((rd1.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd1.y_coord - best_loc.geometry.y[0]) ** 2.0)

    rd2 = RadarData.RadarData(fns[1])
    # rd2.hcrop(1900, "right")
    rd2.hcrop(1983, "right")
    ind2 = np.nanargmin((rd2.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd2.y_coord - best_loc.geometry.y[0]) ** 2.0)

    dists = np.array([rd.dist[-1] for rd in [rd1, rd2]])

    left = 0.075
    right = 0.985
    top = 0.985
    bottom = 0.09
    fig = plt.figure(figsize=(7.05, 5))
    gs = mgridspec.GridSpec(
        2,
        6,
        wspace=0.0,
        hspace=0.22,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width_ratios=[dists[1], 0.3, dists[0] - dists[1] - 1.0, 0.15, 0.1, 0.45],
    )
    axzoom = fig.add_subplot(gs[1, 2], projection=CAN())
    cax = fig.add_subplot(gs[1, -2])

    axes = []
    for i, (rd, ind, vmin, vmax) in enumerate([(rd1, ind1, 35, 70), (rd2, ind2, 35, 70)]):
        if i == 0:
            axes.append(fig.add_subplot(gs[i, :]))
        else:
            axes.append(fig.add_subplot(gs[i, 0]))
        plot.plot_radargram(
            rd,
            xdat="dist",
            ydat="depth",
            cmap="gray_r",
            clims=(vmin, vmax),
            ax=axes[i],
            fig=fig,
        )
        plot.plot_picks(
            rd,
            rd.dist,
            rd.nmo_depth,
            colors=["gold"],
            just_middle=True,
            ax=axes[i],
            fig=fig,
        )
        axes[i].arrow(rd.dist[ind], 750, 0, -143.1, color="b", zorder=99, head_width=0.05, head_length=30, length_includes_head=True)
    for ax in axes:
        ax.set_ylim(750, 0)
        ax.set_yticks([0, 250, 500, 750])

    background = rasterio.open("../imagery/sentinel/sentinel_2_cloudless.tif")
    # Convert to numpy arrays
    blue = background.read(1)
    mueller_map(axzoom)
    axzoom.imshow(blue, cmap="gray", extent=plotting_extent(background), zorder=1)
    gl = axzoom.gridlines(color="k", linewidth=0.5, draw_labels=True, dms=True, zorder=99999)
    gl.right_labels = False
    gl.top_labels = False
    axzoom.plot(
        best_loc.geometry.x[0],
        best_loc.geometry.y[0],
        marker="*",
        color="k",
        markersize=5,
        zorder=99999,
    )
    axzoom.set_ylim(ZE[2], ZE[3])
    axzoom.set_xlim(ZE[0], ZE[1])

    inc_cutoff = 100.0  # increment to separate lines
    for name, pick_list in picks_dict.items():
        x, y = np.hstack([p.geometry.x for p in pick_list]), np.hstack([p.geometry.y for p in pick_list])
        inc_dist = np.hstack(([0], np.sqrt(np.diff(x) ** 2.0 + np.diff(y) ** 2.0)))
        mask = inc_dist < inc_cutoff
        y[~mask] = np.nan

        for picks in pick_list:
            if name in ["AWI UWB", "AWI NB"]:
                axzoom.plot(
                    picks.geometry.x.values,
                    picks.geometry.y.values,
                    color=color_dict[name],
                    zorder=1.5
                )
                thick_cm = axzoom.scatter(
                    picks.geometry.x[picks['maxdiff'] < 30.0],
                    picks.geometry.y[picks['maxdiff'] < 30.0],
                    s=4,
                    c=picks[name_dict[name]][picks['maxdiff'] < 30.0],
                    vmin=0,
                    vmax=750,
                    zorder=2,
                )
            if name == "Ground UWB":
                thick_cm = axzoom.scatter(
                    picks.geometry.x,
                    picks.geometry.y,
                    s=4,
                    c=picks[name_dict[name]],
                    vmin=0,
                    vmax=750,
                    zorder=2,
                )

    thick_cbr = plt.colorbar(
        thick_cm,
        extend="max",
        label=r"Ice thickness (m)",
        cax=cax,
        orientation="vertical",
    )
    thick_cbr.set_ticks([0, 250, 500, 750])
    thick_cbr.ax.tick_params(labelsize=10)

    axzoom.plot(
        (ZE[0] + 500, ZE[0] + 1500),
        (ZE[2] + 1000, ZE[2] + 1000),
        linewidth=4,
        color="k",
        clip_on=False,
        zorder=1000,
    )
    axzoom.text(
        ZE[0] + 1000,
        ZE[2] + 500,
        "1 km",
        color="k",
        clip_on=False,
        zorder=1000,
        ha="center",
    )
    axzoom.text(
        picks_dict["Ground UWB"][0].geometry.x[8944],
        picks_dict["Ground UWB"][0].geometry.y[8944],
        "a",
        ha="right",
    )
    axzoom.text(
        picks_dict["Ground UWB"][0].geometry.x[3738],
        picks_dict["Ground UWB"][0].geometry.y[3738],
        "a'",
    )
    axzoom.text(
        picks_dict["Ground UWB"][1].geometry.x[0],
        picks_dict["Ground UWB"][1].geometry.y[0],
        "b",
        ha="right",
    )
    axzoom.text(
        picks_dict["Ground UWB"][1].geometry.x[1983],
        picks_dict["Ground UWB"][1].geometry.y[1983],
        "b'",
        ha="left",
    )
    axzoom.text(
        picks_dict["AWI UWB"][0].geometry.x[1430],
        picks_dict["AWI UWB"][0].geometry.y[1430],
        "S1",
        ha="center",
    )
    axzoom.text(
        picks_dict["AWI UWB"][0].geometry.x[1325],
        picks_dict["AWI UWB"][0].geometry.y[1325],
        "S1'",
        ha="center",
    )

    for ax, letter in zip(axes, "ab"):
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )
        ax.text(
            0.995,
            0.993,
            letter + "'",
            ha="right",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    for ax, letter, asp in zip([axzoom], "cde", [zoom_asp]):
        h = 0.075 / asp
        ax.fill(
            [0, 0.05, 0.05, 0, 0],
            [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h],
            transform=ax.transAxes,
            color="w",
            edgecolor="k",
            zorder=99998,
        )
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    plt.savefig("../plots/mueller_uwb_res_supp.png", dpi=300)


def uwb_radargrams_comp():
    fns = [
        "../RES/UWB/20230522_cat_dx1combined_Int2_fk_picked_cropped.mat",
        "../RES/UWB/20230523_cat_dx1combined_Int2_fk_picked_cropped.mat",
        "../RES/UWB/20230522_cat_dx1combined_Int2_raw_cropped.mat",
        "../RES/UWB/20230523_cat_dx1combined_Int2_raw_cropped.mat",
    ]
    # Do this slowly because of huge matrix
    rd1 = RadarData.RadarData(fns[0])
    rd3 = RadarData.RadarData(fns[2])
    rd1.reverse()
    ind1 = np.nanargmin((rd1.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd1.y_coord - best_loc.geometry.y[0]) ** 2.0)
    halfwidth = 250
    rd1.hcrop(ind1 + halfwidth, "right")
    rd1.hcrop(ind1 - halfwidth, "left")
    rd3.reverse()
    rd3.hcrop(ind1 + halfwidth, "right")
    rd3.hcrop(ind1 - halfwidth, "left")

    rd2 = RadarData.RadarData(fns[1])
    rd4 = RadarData.RadarData(fns[3])
    ind2 = np.nanargmin((rd2.x_coord - best_loc.geometry.x[0]) ** 2.0 + (rd2.y_coord - best_loc.geometry.y[0]) ** 2.0)
    rd2.hcrop(ind2 + halfwidth, "right")
    rd2.hcrop(ind2 - halfwidth, "left")
    rd4.hcrop(ind2 + halfwidth, "right")
    rd4.hcrop(ind2 - halfwidth, "left")

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
    axes = [ax1, ax2, ax3, ax4]
    vmin = 35
    vmax = 65
    for rd, ax in zip([rd1, rd3, rd2, rd4], axes):
        rd.dist -= halfwidth / 1000
        rd.dist *= 1000
        plot.plot_radargram(
            rd,
            xdat="dist",
            ydat="depth",
            cmap="gray_r",
            clims=(vmin, vmax),
            ax=ax,
            fig=fig,
        )
        # ax.axis('equal')
        # plt.plot(0, 606.9, marker='*', color='b', markersize=8)
    for ax in axes:
        ax.set_ylim(700, 200)
        # ax.set_yticks([200, 500, 750])
        ax.set_xlim(-halfwidth, halfwidth)
        ax.set_xticks([-halfwidth, 0, halfwidth])
    for rd, ax in zip([rd1, rd1, rd2, rd2], axes):
        ax.plot(rd.dist, rd.nmo_depth[rd.picks.samp2[0, :].astype(int)], color="gold")
        ax.arrow(0, 700, 0, -93.1, color="b", zorder=99, head_width=0.05, head_length=30, length_includes_head=True)

    for ax, letter in zip(axes, "abcd"):
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )

    ax1.set_xlabel("")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax4.set_ylabel("")
    ax2.set_xlabel("Distance from site (m southeast)")
    ax4.set_xlabel("Distance from site (m northeast)")
    fig.tight_layout(pad=0.05)
    fig.savefig("../plots/mueller_uwb_res_comp.png", dpi=300)


def awi_radargrams():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(7, 6))
    rd_fns = [
        "../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230508_09/muller_awi_uwb_sar1_comb.mat",
        "../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230509_03/muller_awi_nb_sar_comb.mat",
    ]
    lims = [(1325, 1430), (2819, 2924)]
    for fn, lim, ax in zip(rd_fns, lims, [ax1, ax2]):
        rd = RadarData.RadarData(fn)
        rd.picks.smooth(24)
        rd.nmo(0.0)
        rd.hcrop(lim[1] * 3, "right", "tnum")
        rd.hcrop(lim[0] * 3, "left", "tnum")
        rd.crop(1000, "bottom", "depth", rezero=False, zero_trig=False)
        rd.crop(100, "top", "depth")
        rd.reverse()
        samps = rd.picks.samp2.copy()
        mask = (
            np.maximum(
                np.maximum(np.abs(samps[1, :] - samps[2, :]), np.abs(samps[1, :] - samps[3, :])),
                np.abs(samps[3, :] - samps[2, :]),
            )
            * rd.dt
            * 1.68e8
            / 2.0
            > 20.0
        )
        rd.picks.samp2[1:, mask] = np.nan
        plot.plot_radargram(
            rd,
            xdat="dist",
            ydat="depth",
            cmap="gray_r",
            fig=fig,
            ax=ax,
        )
        plot.plot_picks(
            rd,
            rd.dist,
            rd.nmo_depth,
            colors=["r", "C0", "C1", "C2"],
            fig=fig,
            ax=ax,
            just_middle=True,
        )
        rd.picks.samp2 = samps.copy()
        rd.picks.samp2[1:, ~mask] = np.nan
        rd.picks.samp2[0, :] = np.nan
        plot.plot_picks(
            rd,
            rd.dist,
            rd.nmo_depth,
            linestyle="dotted",
            colors=["r", "C0", "C1", "C2"],
            fig=fig,
            ax=ax,
            just_middle=True,
        )

    for ax, letter in zip([ax1, ax2], "abcd"):
        h = 0.08
        ax.fill(
            [0, 0.03, 0.03, 0, 0],
            [1.0 - h, 1.0 - h, 1.0, 1.0, 1.0 - h],
            transform=ax.transAxes,
            color="w",
            edgecolor="k",
            zorder=99998,
        )
        ax.text(
            0.005,
            0.993,
            letter,
            ha="left",
            va="top",
            fontsize=14,
            transform=ax.transAxes,
            zorder=99999,
        )
    ax1.set_xlabel("")
    ax1.text(
        0.0,
        1.08,
        "S1",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
        zorder=99999,
    )
    ax1.text(
        1.0,
        1.08,
        "S1'",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
        zorder=99999,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig("../plots/JOG-2024-0020.SuppFigure1.tif", dpi=300)


if __name__ == "__main__":
    res_pub_map_2p()
    uwb_radargrams()
    awi_radargrams()
