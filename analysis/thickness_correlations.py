#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""

import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

crs = "EPSG:3348"
pick_fns = {
    "MCoRDS": ["../RES/icebridge/mcords/l2/IRMCR2_muellers_20211202.gpkg"],
    "HICARS": ["../RES/BAS_RES/BAS_RES.gpkg"],
    "AWI NB": ["../RES/AWI/2023_Canada_Polar5/CSARP_standard/20230509_03/muller_awi_nb_sar_comb.gpkg"],
    "Ground UWB": [
        "../RES/UWB/20230522_cat_dx1combined_Int2_fk_picked_cropped.gpkg",
        "../RES/UWB/20230523_cat_dx1combined_Int2_fk_picked_cropped.gpkg",
    ],
}
name_dict = {
    "MCoRDS": "THICK",
    "HICARS": "THK",
    "AWI NB": "thickness",
    "Ground UWB": "L1_depth",
}

picks = {name: pd.concat([gpd.read_file(fn).to_crs(crs) for fn in file_list]) for name, file_list in pick_fns.items()}
all_picks = gpd.GeoDataFrame(
    pd.concat(
        [pick[[name_dict[name], "geometry"]].rename(columns={name_dict[name]: "thick"}) for name, pick in picks.items()]
    )
)
all_picks = all_picks.set_geometry("geometry", drop=True).set_crs(crs)

zinck_sia_thick_fn = "../topography/zinck_results/muller_thickness_SIAinversion_can.tif"
zinck_pism_thick_fn = "../topography/zinck_results/muller_thickness_PISM_can.tif"
zinck_comb_thick_fn = "../topography/zinck_results/muller_thickness_SIAandPISM_can.tif"
zinck_mcmcns_thick_fn = "../topography/zinck_results/muller_thickness_MCMCnosliding_can.tif"
zinck_mcmcs_thick_fn = "../topography/zinck_results/muller_thickness_MCMCwithsliding_can.tif"
millan_thick_fn = "../topography/millan_2022/THICKNESS_MILLAN_HEIBERG_can.tif"

raster_thicks = {
    "SIA": rxr.open_rasterio(zinck_sia_thick_fn),
    "PISM": rxr.open_rasterio(zinck_pism_thick_fn),
    "MCMCs": rxr.open_rasterio(zinck_mcmcs_thick_fn),
    "MCMCns": rxr.open_rasterio(zinck_mcmcns_thick_fn),
    "Millan": rxr.open_rasterio(millan_thick_fn),
    "Comb": rxr.open_rasterio(zinck_comb_thick_fn),
}
interped_thicks = {}


x = xr.DataArray(all_picks.geometry.x, dims="H")
y = xr.DataArray(all_picks.geometry.y, dims="H")

for name, arr in raster_thicks.items():
    interped_thicks[name] = arr.interp(x=x, y=y).values.flatten()


for name, thick in interped_thicks.items():
    print("For", name)
    comb_mask = np.logical_and(~np.isnan(all_picks["thick"].values), ~np.isnan(thick))
    print("Bias {} m".format(np.nanmean(all_picks["thick"].values[comb_mask] - thick[comb_mask])))
    print("Standard deviation {} m".format(np.nanstd(all_picks["thick"].values[comb_mask] - thick[comb_mask])))
    print("Correlation is {}".format(np.corrcoef(all_picks["thick"].values[comb_mask], thick[comb_mask])[0, 1]))
