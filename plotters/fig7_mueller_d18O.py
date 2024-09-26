#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""

from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
import glob
import tqdm


def get_eureka_data():
    station_ids_daily = [2401200, 2401208, 2401203]
    station_ids_hourly = [2401199]
    data_dir = "../climate/eureka_weather_stations/"
    station_data = {}
    plt.figure()
    for station_id in station_ids_hourly:
        # comp_T[str(station_id)] = comp_T["Queens"].copy()
        file_glob = f"en_climate_hourly_NU_{station_id}_??-????_P1H.csv"
        files = glob.glob(data_dir + file_glob)
        data_list = [pd.read_csv(fn) for fn in files]
        data = pd.concat(data_list).sort_values("Date/Time (LST)")
        data["Date/Time (LST)"] = pd.to_datetime(data["Date/Time (LST)"])
        data = data.set_index("Date/Time (LST)")
        data = data.groupby(pd.Grouper(freq="1d")).sum()
        data["Date"] = data.index
        station_data[station_id] = data
        plt.plot(data["Date"].values, data["Precip. Amount (mm)"].values)

    for station_id in station_ids_daily:
        # comp_T[str(station_id)] = comp_T["Queens"].copy()
        file_glob = f"en_climate_daily_NU_{station_id}_????_P1D.csv"
        files = glob.glob(data_dir + file_glob)
        data_list = [pd.read_csv(fn) for fn in files]
        data = (
            pd.concat(data_list).sort_values("Date/Time").rename(columns={"Total Precip (mm)": "Precip. Amount (mm)"})
        )
        data["Date"] = pd.to_datetime(data["Date/Time"])
        data["Date/Time (LST)"] = pd.to_datetime(data["Date/Time"])
        station_data[station_id] = data.set_index("Date/Time (LST)")
        plt.plot(data["Date"].values, data["Precip. Amount (mm)"].values)

    eureka_mean = pd.concat(
        df.rename(columns={"Precip. Amount (mm)": str(sid) + "mm"}) for sid, df in station_data.items()
    )
    eureka_mean["Acc"] = eureka_mean[[str(si) + "mm" for si in station_ids_hourly + station_ids_daily]].mean(axis=1)

    eureka_mean = eureka_mean[["Acc"]]

    eureka_data = eureka_mean.groupby(pd.Grouper(freq="1d")).mean()
    eureka_annual = eureka_mean[["Acc"]].groupby(pd.Grouper(freq="1y")).sum()

    plt.figure()
    plt.plot(eureka_data.index.values, eureka_data["Acc"].values)
    plt.plot(eureka_annual.index.values, eureka_annual["Acc"].values)

    eureka_temp = pd.read_csv("../climate/eureka_weather_stations/eureka_temps.csv").rename(
        columns={"Temperature_21049794_deg_C": "Temp"}
    )
    eureka_temp["Date"] = pd.to_datetime(eureka_temp["Date/Time (LST)"])
    eureka_temp["Date/Time (LST)"] = pd.to_datetime(eureka_temp["Date/Time (LST)"])
    eureka_temp = eureka_temp.set_index("Date/Time (LST)")

    eureka_all = (
        pd.concat([eureka_data, eureka_temp]).sort_values("Date/Time (LST)")[["Acc", "Temp"]].iloc[-365 * 80 : -146]
    )
    eureka_all = eureka_all.groupby(pd.Grouper(freq="1d")).sum()
    eureka_all["Total Acc"] = np.cumsum(eureka_all["Acc"].values[::-1])[::-1]

    total_acc = np.linspace(0, eureka_all["Total Acc"].values[0], eureka_all.shape[0])
    ta, un_ind = np.unique(eureka_all["Total Acc"].values[::-1], return_index=True)
    T = eureka_all["Temp"].values[::-1][un_ind]
    temp_v_acc = interp1d(ta, T)(total_acc)
    eureka_all["Linearized Temp"] = temp_v_acc[::-1]

    plt.figure()
    plt.plot(ta, T)
    plt.plot(total_acc, temp_v_acc)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(eureka_all.index.values, eureka_all["Acc"].values)
    ax2.plot(eureka_all.index.values, eureka_all["Temp"].values)
    ax2.plot(eureka_all.index.values, eureka_all["Linearized Temp"].values)
    ax3.plot(eureka_all.index.values, eureka_all["Total Acc"].values)

    return eureka_all, total_acc[::-1]


eureka_all, total_acc = get_eureka_data()

camp_core = np.genfromtxt("../climate/field_data/core_d18O/camp_core.csv", names=True, delimiter=",")
site_core = np.genfromtxt("../climate/field_data/core_d18O/target_site_d18O.csv", names=True, delimiter=",")
shallow_density = np.genfromtxt("../climate/field_data/shallow_density.csv", names=True, delimiter=",")
ice_layers = np.genfromtxt("../climate/field_data/ice_layers.txt")

big_peaks = []
med_peaks = []
small_peaks = []
for i in range(4, len(site_core["d_18O"]) - 4):
    if site_core["d_18O"][i] == np.min(site_core["d_18O"][i - 4 : i + 5]):
        big_peaks.append(i)
    elif site_core["d_18O"][i] == np.min(site_core["d_18O"][i - 2 : i + 3]):
        med_peaks.append(i)
    elif site_core["d_18O"][i] == np.min(site_core["d_18O"][i - 1 : i + 2]):
        small_peaks.append(i)

layer_cats = np.array([0.95, 0.66, 0.25])
layers_per_cat = np.array([len(big_peaks), len(med_peaks), len(small_peaks)])
layers = np.hstack([[p for i in range(l)] for p, l in zip(layer_cats, layers_per_cat)])
print("Print mean number of layers", np.sum(layers))
print("Std of layers ", np.sqrt(np.sum((1.0 - layers))))
num_layers = np.arange(0, np.sum(layers_per_cat) + 1, dtype=float)
num_layers[0] = 0.5  # avoids issues with division by zero--assumes 6.62 m acc in 2023
nl_max = 15

layer_probs = np.zeros_like(num_layers, dtype=float)
layer_probs[0] = np.prod((1.0 - layers))


def T(i, layers):
    return np.sum((layers / (1.0 - layers)) ** i)


for k in range(1, len(layer_probs)):
    layer_probs[k] = (
        1.0 / k * np.sum([(-1.0) ** (i - 1.0) * layer_probs[k - i] * T(i, layers) for i in range(1, k + 1)])
    )
layer_probs[-1] = np.prod(layers)

snow_dens, ice_dens = 550.0, 910.0
most_prob_dens = 700.0
deep_dens = np.linspace(snow_dens, ice_dens, 1001)
sigma_dens = 100.0
densmin, densmax = (snow_dens - most_prob_dens) / sigma_dens, (ice_dens - most_prob_dens) / sigma_dens
dens_max = np.where(deep_dens == most_prob_dens)[0]
deep_probs = truncnorm.pdf(deep_dens, densmin, densmax, loc=most_prob_dens, scale=sigma_dens)


def to_kg(deep_den):
    return (
        np.min(shallow_density["Density"]) * shallow_density["start_cm"][0]
        + np.sum((shallow_density["stop_cm"] - shallow_density["start_cm"]) * shallow_density["Density"])
        + (662.0 - shallow_density["stop_cm"][-1]) * deep_den
    ) / 100


total_weight = np.array([to_kg(deep_den) for deep_den in deep_dens])

TW, NL = np.meshgrid(total_weight, num_layers)
DP, NP = np.meshgrid(deep_probs, layer_probs)
ie_yr = TW / NL / 910.0
average = np.average(ie_yr, weights=DP * NP)
print("Mean accumulation is {} m i.e. / yr".format(average))
print("Accumulation std is {} m i.e. / yr".format(np.sqrt(np.average((ie_yr - average) ** 2.0, weights=DP * NP))))

A = np.ones((len(shallow_density["stop_cm"]), 2))
A[:, 0] = (shallow_density["stop_cm"] + shallow_density["start_cm"]) / 2.0
trend = np.linalg.lstsq(A, shallow_density["Density"], rcond=None)
plt.figure()
plt.plot(shallow_density["Density"], A[:, 0], marker="o", linestyle="none")
plt.plot(trend[0][0] * np.arange(200) + trend[0][1], np.arange(200))

plt.figure()
plt.axhline(45, color="0.6")
plt.axhline(65, color="0.6", linestyle="dashed")
plt.plot(camp_core["d_18O"], (camp_core["Start"] + camp_core["End"]) / 2.0)
plt.plot(site_core["d_18O"], (site_core["Start"] + site_core["End"]) / 2.0)
plt.ylim(662, 0)

final_date = "2023-05-26"

normalized_timeseries_len_in_yrs = 25
normalized_timeseries_v_t = eureka_all["Temp"].values[-365 * normalized_timeseries_len_in_yrs :].copy()
normalized_timeseries_v_t -= np.nanmean(normalized_timeseries_v_t)
normalized_timeseries_v_t /= np.nanstd(normalized_timeseries_v_t)

normalized_timeseries_v_acc = eureka_all["Linearized Temp"].values[-365 * normalized_timeseries_len_in_yrs :].copy()
normalized_timeseries_v_acc -= np.nanmean(normalized_timeseries_v_acc)
normalized_timeseries_v_acc /= np.nanstd(normalized_timeseries_v_acc)

normalized_d18O = site_core["d_18O"].copy()
normalized_d18O -= np.nanmean(normalized_d18O)
normalized_d18O /= np.nanstd(normalized_d18O)

site_core_depths = np.hstack(
    [[0.0], (shallow_density["stop_cm"] + shallow_density["start_cm"]) / 2.0, [shallow_density["stop_cm"][-1], 10000]]
)
site_core_densities = np.hstack(
    [[np.min(shallow_density["Density"])], shallow_density["Density"].flatten(), [most_prob_dens, most_prob_dens]]
)
site_core_interval_weights = (site_core["End"] - site_core["Start"]) * interp1d(site_core_depths, site_core_densities)(
    (site_core["Start"] + site_core["End"]) / 2.0
)
site_core_weight_above_depth = np.hstack(([0.0], np.cumsum(site_core_interval_weights)))
site_core_fractional_depths = site_core_weight_above_depth / site_core_weight_above_depth[-1]

d18O_interper = interp1d(site_core_weight_above_depth, np.hstack(([normalized_d18O[0]], normalized_d18O)))

lens = np.arange(10, 25)
dayrange = np.arange(365 * lens[0], 365 * lens[-1])
corrs_perlen = np.zeros_like(dayrange, dtype=float)
best_offsets = np.zeros_like(dayrange, dtype=float)
corrs_perlen_lin = np.zeros_like(dayrange, dtype=float)
best_offsets_lin = np.zeros_like(dayrange, dtype=float)

offsets = np.arange(-61, 62, dtype=int)
# offsets = [0]
corrs_offset = np.zeros_like(offsets, dtype=float)
corrs_offset_lin = np.zeros_like(offsets, dtype=float)


def get_start_date(record_len):
    dates = pd.date_range(end=final_date, periods=record_len, freq="1D")
    return dates[0]


def get_new_record(record_len):
    weights = np.linspace(0.0, site_core_weight_above_depth[-1], record_len)
    new_record = d18O_interper(weights[::-1])
    dates = pd.date_range(end=final_date, periods=record_len, freq="1D")
    return dates, new_record


def get_eureka_temp_by_acc(record_len, offset):
    date = get_start_date(record_len)
    pt_in_tempdata = np.where(eureka_all.index.values == date)[0][0]
    rel_eureka_dat = eureka_all.iloc[pt_in_tempdata + offset + record_len : pt_in_tempdata + offset : -1].copy()
    rel_eureka_dat["Scaled acc"] = (rel_eureka_dat["Total Acc"] - rel_eureka_dat["Total Acc"][0]) / (
        rel_eureka_dat["Total Acc"][-1] - rel_eureka_dat["Total Acc"][0]
    )

    eureka_temp_by_acc = np.zeros((len(site_core_fractional_depths) - 1,))
    for k in range(len(eureka_temp_by_acc)):
        mask = np.logical_and(
            rel_eureka_dat["Scaled acc"].values <= site_core_fractional_depths[k + 1],
            rel_eureka_dat["Scaled acc"].values > site_core_fractional_depths[k],
        )
        # Edge cases because of big days of accumulation or ends of record w/o accumulation
        if np.sum(mask) == 0:
            eureka_temp_by_acc[k] = rel_eureka_dat["Temp"].values[
                np.where(rel_eureka_dat["Scaled acc"].values >= site_core_fractional_depths[k])[0][0]
            ]
        elif np.sum(rel_eureka_dat["Acc"].values[mask]) == 0.0:
            eureka_temp_by_acc[k] = np.average(rel_eureka_dat["Temp"].values[mask])
        else:
            eureka_temp_by_acc[k] = np.average(
                rel_eureka_dat["Temp"].values[mask], weights=rel_eureka_dat["Acc"].values[mask]
            )
    return eureka_temp_by_acc


def get_eureka_temp_by_t(record_len, offset):
    date = get_start_date(record_len)
    pt_in_tempdata = np.where(eureka_all.index.values == date)[0][0]
    rel_eureka_dat = eureka_all.iloc[pt_in_tempdata + offset + record_len : pt_in_tempdata + offset : -1].copy()
    rel_eureka_dat["Scaled t"] = np.arange(record_len) / record_len

    eureka_temp_by_t = np.zeros((len(site_core_fractional_depths) - 1,))
    for k in range(len(eureka_temp_by_t)):
        mask = np.logical_and(
            rel_eureka_dat["Scaled t"].values <= site_core_fractional_depths[k + 1],
            rel_eureka_dat["Scaled t"].values > site_core_fractional_depths[k],
        )
        # Edge cases because of big days of accumulation or ends of record w/o accumulation
        if np.sum(mask) == 0:
            eureka_temp_by_t[k] = rel_eureka_dat["Temp"].values[
                np.where(rel_eureka_dat["Scaled t"].values >= site_core_fractional_depths[k])[0][0]
            ]
        else:
            eureka_temp_by_t[k] = np.average(rel_eureka_dat["Temp"].values[mask])
    return eureka_temp_by_t


rerun = False
if rerun:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        eureka_all.index.values[-365 * normalized_timeseries_len_in_yrs :],
        normalized_timeseries_v_t,
        color="k",
        linewidth=2,
    )
    ax.plot(
        eureka_all.index.values[-365 * normalized_timeseries_len_in_yrs :],
        normalized_timeseries_v_acc,
        color="0.6",
        linewidth=2,
    )

    goodfig, goodax = plt.subplots(figsize=(14, 8))
    goodax.plot(
        eureka_all.index.values[-365 * normalized_timeseries_len_in_yrs :],
        normalized_timeseries_v_t,
        color="k",
        linewidth=2,
        label="T vs t",
    )
    goodax.plot(
        eureka_all.index.values[-365 * normalized_timeseries_len_in_yrs :],
        normalized_timeseries_v_acc,
        color="0.6",
        linewidth=2,
        label=r"T vs $\dot{a}$",
    )

    for i, record_len in enumerate(tqdm.tqdm(dayrange)):
        for j, offset in enumerate(offsets):
            eureka_temp_by_acc = get_eureka_temp_by_acc(record_len, offset)
            eureka_temp_by_t = get_eureka_temp_by_t(record_len, offset)
            corrs_offset_lin[j] = np.corrcoef(eureka_temp_by_acc, site_core["d_18O"])[0, 1]
            corrs_offset[j] = np.corrcoef(eureka_temp_by_t, site_core["d_18O"])[0, 1]

        best_offsets_lin[i] = offsets[np.argmax(corrs_offset_lin)]
        corrs_perlen_lin[i] = np.max(corrs_offset_lin)

        best_offsets[i] = offsets[np.argmax(corrs_offset)]
        corrs_perlen[i] = np.max(corrs_offset)

    ind = np.argmax(corrs_perlen)
    ind_lin = np.argmax(corrs_perlen_lin)

    np.save("cache/corrs_perlen", corrs_perlen)
    np.save("cache/corrs_perlen_lin", corrs_perlen_lin)
    np.save("cache/best_offsets", best_offsets)
    np.save("cache/best_offsets_lin", best_offsets_lin)
else:
    corrs_perlen = np.load("cache/corrs_perlen.npy")
    corrs_perlen_lin = np.load("cache/corrs_perlen_lin.npy")
    best_offsets = np.load("cache/best_offsets.npy")
    best_offsets_lin = np.load("cache/best_offsets_lin.npy")
    ind = np.argmax(corrs_perlen)
    ind_lin = np.argmax(corrs_perlen_lin)

print(
    "Maximized at {} with {} day ({} yr) record and {} day offset for linearized T".format(
        corrs_perlen_lin[ind_lin], dayrange[ind_lin], dayrange[ind_lin] / 365.25, best_offsets_lin[ind_lin]
    )
)
corr_acc_lin = total_weight / (dayrange[ind_lin] / 365.25)
average = np.average(corr_acc_lin, weights=deep_probs)
print(
    "Implies {} cm i.e./yr with {} uncertainty".format(
        average, np.sqrt(np.average((corr_acc_lin - average) ** 2.0, weights=deep_probs))
    )
)
print(
    "Maximized at {} with {} day ({} yr) record and {} day offset for linearized T".format(
        corrs_perlen[ind], dayrange[ind], dayrange[ind] / 365.25, best_offsets[ind]
    )
)
corr_acc = total_weight / (dayrange[ind] / 365.25)
average = np.average(corr_acc, weights=deep_probs)
print(
    "Implies {} cm i.e./yr with {} uncertainty".format(
        average, np.sqrt(np.average((corr_acc - average) ** 2.0, weights=deep_probs))
    )
)

dates, new_record = get_new_record(dayrange[ind_lin])
pt_in_tempdata = np.where(eureka_all.index.values == dates[0])[0][0]
real_span = (
    dates[-1] - eureka_all.index.values[np.max(np.where(eureka_all["Total Acc"].values >= total_acc[pt_in_tempdata]))]
).days
# print('Actually maximized at {} with {} day ({} yr) record and {} day offset for linearized T'.format(corrs_perlen_lin[ind_lin], real_span, real_span / 365.25, best_offsets_lin[ind_lin]))
plt.figure(figsize=(6.5, 4))
plt.plot(dayrange / 365, corrs_perlen, label="T vs t")
plt.plot(dayrange / 365, corrs_perlen_lin, label=r"T vs $\dot{a}$")
plt.xlabel("Years of accumulation in 6.62 m core")
plt.ylabel(r"Cross-correlation between $\delta^{18}$O and T")
plt.xlim(dayrange[0] / 365, dayrange[-1] / 365)
plt.legend(loc="best", frameon=False)
plt.tight_layout(pad=0.1)
plt.savefig("../plots/JOG-2024-0020.SuppFigure2.pdf")

plt.figure()
plt.plot(shallow_density["Density"], A[:, 0], marker="o", linestyle="none")
plt.plot(trend[0][0] * np.arange(200) + trend[0][1], np.arange(200))


fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(3.0, 4.0))
ax1.axhline(1.05, color="salmon", linestyle="dashed", label="May 4\n2021")
ax1.fill_between(
    [-50, 0],
    [
        0.45,
        0.45,
    ],
    [0.65, 0.65],
    color="lightskyblue",
)
ax1.plot(site_core["d_18O"], (site_core["Start"] + site_core["End"]) / 2.0 / 100.0, color="k", label="6.6-m core")
ax1.plot(camp_core["d_18O"], (camp_core["Start"] + camp_core["End"]) / 2.0 / 100.0, color="maroon", label="1.6-m core")
ax1.plot(
    site_core["d_18O"][big_peaks],
    (site_core["Start"] + site_core["End"])[big_peaks] / 2.0 / 100.0,
    color="C2",
    label="High conf",
    linestyle="none",
    marker="o",
)
ax1.plot(
    site_core["d_18O"][med_peaks],
    (site_core["Start"] + site_core["End"])[med_peaks] / 2.0 / 100.0,
    color="C1",
    label="Med conf",
    linestyle="none",
    marker="s",
)
ax1.plot(
    site_core["d_18O"][small_peaks],
    (site_core["Start"] + site_core["End"])[small_peaks] / 2.0 / 100.0,
    color="C0",
    label="Low conf",
    linestyle="none",
    marker="d",
)

ax1.set_ylim(6.62, 0)
ax1.set_xlim(-45, -20)
ax1.set_ylabel("Depth (m)")
ax1.set_xlabel(r"$\delta^{18}$O (‰)")
ax1.legend(loc="lower left", frameon=False, fontsize=9)

fig.tight_layout(pad=0.01)

gs = gridspec.GridSpec(1, 4, left=0.05, right=0.92, bottom=0.21, top=0.9, wspace=0.40, width_ratios=(1, 1, 0.07, 1))
fig = plt.figure(figsize=(7.9, 4.5))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.9, 4.0))
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 3])
scaley = 1.0e5 * 0.91

ax1.plot(
    site_core["d_18O"],
    (site_core["Start"] + site_core["End"]) / 2.0 / 100.0,
    color="k",
    label=r"6.6-m core $\delta^{18}$O",
)
ax1.plot(
    camp_core["d_18O"],
    (camp_core["Start"] + camp_core["End"]) / 2.0 / 100.0,
    color="maroon",
    label=r"1.6-m core $\delta^{18}$O",
)
ax1.set_ylim(6.62, 0)
ax1.set_xlim(-45, -20)
ax1.set_ylabel("Depth (m)")
ax1.set_xlabel(r"$\delta^{18}$O (‰)")
ax1.plot(
    site_core["d_18O"][big_peaks],
    (site_core["Start"] + site_core["End"])[big_peaks] / 2.0 / 100.0,
    color="C2",
    label="High conf.",
    linestyle="none",
    marker="o",
)
for peaks, color in zip([big_peaks, med_peaks, small_peaks], ["C2", "C1", "C0"]):
    for p in peaks:
        for ax in [ax2, ax3]:
            # ax.axhline(site_core_weight_above_depth[p + 1] / scaley, color=color)
            pass
ax1.plot(
    site_core["d_18O"][med_peaks],
    (site_core["Start"] + site_core["End"])[med_peaks] / 2.0 / 100.0,
    color="C1",
    label="Med conf.",
    linestyle="none",
    marker="s",
)
ax1.plot(
    site_core["d_18O"][small_peaks],
    (site_core["Start"] + site_core["End"])[small_peaks] / 2.0 / 100.0,
    color="C0",
    label="Low conf.",
    linestyle="none",
    marker="d",
)

ax1.fill_between(
    [-50, 0],
    [
        0.45,
        0.45,
    ],
    [0.65, 0.65],
    color="lightskyblue",
    zorder=0.1,
    label="Coarse grains",
)
ax1.axhline(1.05, color="salmon", linestyle="dashed", label="May 4, 2021", zorder=0.1)
for i, ice_layer in enumerate(ice_layers):
    if i == 0:
        label = "Ice layers"
    else:
        label = None
    ax1.axhline(ice_layer / 100.0, color="lightblue", linestyle="dashed", label=label, zorder=0.1)


rec = site_core["d_18O"]

ax2ty = ax2.twiny()
ax2tx = ax2ty.twinx()
ax3ty = ax3.twiny()
ax3tx = ax3ty.twinx()
ax2.set_zorder(ax2tx.get_zorder() + 2)
ax2ty.set_zorder(ax2tx.get_zorder() + 1)
ax2.patch.set_visible(False)
ax3.set_zorder(ax3tx.get_zorder() + 2)
ax3ty.set_zorder(ax3tx.get_zorder() + 1)
ax3.patch.set_visible(False)

dates, new_record = get_new_record(dayrange[ind])
pt_in_tempdata = np.where(eureka_all.index.values == dates[0])[0][0]
bo = -int(best_offsets[ind])
ax2ty.plot(
    eureka_all["Temp"].values[pt_in_tempdata - bo : pt_in_tempdata - bo + dayrange[ind]],
    np.linspace(site_core_weight_above_depth[-1] / scaley, 0, dayrange[ind]),
    label="{:4.1f}-yr record".format(dayrange[ind] / 365),
    color="dimgray",
    zorder=1,
)
eureka_temp_by_t = get_eureka_temp_by_t(int(dayrange[int(ind)]), int(best_offsets[int(ind)]))
ax2ty.plot(
    eureka_temp_by_t,
    (site_core_weight_above_depth[1:] + site_core_weight_above_depth[:-1]) / 2.0 / scaley,
    color="darkgrey",
    zorder=1,
)
ax2tx.set_ylim(dates[0], dates[-1])

date = get_start_date(dayrange[ind_lin])
pt_in_tempdata = np.where(eureka_all.index.values == date)[0][0]
bo = -int(best_offsets_lin[ind_lin])

date = get_start_date(int(dayrange[int(ind_lin)]))
pt_in_tempdata = np.where(eureka_all.index.values == date)[0][0]
rel_eureka_dat = eureka_all.iloc[
    pt_in_tempdata
    + int(best_offsets_lin[int(ind_lin)])
    + int(dayrange[int(ind_lin)]) : pt_in_tempdata
    + int(best_offsets_lin[int(ind_lin)]) : -1
].copy()
rel_eureka_dat["Plot acc"] = rel_eureka_dat["Total Acc"] - rel_eureka_dat["Total Acc"][0]


eureka_temp_by_acc = get_eureka_temp_by_acc(int(dayrange[int(ind_lin)]), int(best_offsets_lin[int(ind_lin)]))
ax3ty.plot(
    eureka_temp_by_acc,
    (site_core_weight_above_depth[1:] + site_core_weight_above_depth[:-1]) / 2.0 / scaley,
    color="darkgrey",
    zorder=1,
)
ax3tx.plot(
    rel_eureka_dat["Temp"].values,
    rel_eureka_dat["Plot acc"].values / 1000.0,
    color="dimgrey",
    label="{:4.1f}-yr record".format(real_span / 365.25),
    zorder=1,
)

ax3tx.set_ylim(rel_eureka_dat["Plot acc"][-1] / 1000, 0)

ax2.plot(np.hstack(([rec[0]], rec)), site_core_weight_above_depth / scaley, color="k", zorder=99999)
ax2.set_ylim(site_core_weight_above_depth[-1] / scaley, 0)
ax3.plot(np.hstack(([rec[0]], rec)), site_core_weight_above_depth / scaley, color="k", zorder=99999)
ax3.set_ylim(site_core_weight_above_depth[-1] / scaley, 0)
ax2.set_ylabel("Acc on ice cap (m i.e.)")
ax3.set_ylabel("Acc on ice cap (m i.e.)")
ax2.set_xlabel(r"$\delta^{18}$O (‰)")
ax3.set_xlabel(r"$\delta^{18}$O (‰)")
ax2ty.set_xlim(-50, 25)
ax3ty.set_xlim(-50, 25)
ax2ty.tick_params(axis="x", labelcolor="dimgray")
ax3ty.tick_params(axis="x", labelcolor="dimgrey")
ax2tx.tick_params(axis="y", labelcolor="dimgray")
ax3tx.tick_params(axis="y", labelcolor="dimgrey")
ax2ty.set_xlabel("Eureka Temp. (°C)", color="dimgray")
ax3ty.set_xlabel("Eureka Temp. (°C)", color="dimgrey")
ax2tx.set_ylabel("Date", color="dimgray")
ax3tx.set_ylabel("Precip. at Eureka (m)", color="dimgrey")

ax1.set_ylim(6.62, 0)
ax1.set_ylabel("Depth (m)")
ax1.set_xlabel(r"$\delta^{18}$O (‰)")
ax1.plot([], [], color="dimgray", label=r"Eureka $T$, full res")
ax1.plot([], [], color="darkgray", label=r"Eureka $T$, resampled")
ax1.legend(loc="upper left", fontsize=9, frameon=True, bbox_to_anchor=(-0.11, -0.14), ncol=5)

for ax, letter in zip([ax1, ax2, ax3], "abcde"):
    ax.set_xlim(-45, -20)
    ax.text(0.005, 0.993, letter, ha="left", va="top", fontsize=14, transform=ax.transAxes, zorder=99999)
fig.savefig("../plots/JOG-2024-0020.Figure7.pdf")
