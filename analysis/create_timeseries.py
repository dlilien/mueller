# coding: utf-8

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob


time_fmt = "%m/%d/%y %H:%M:%S"
station_ids = [2401199, 2401200, 2401208, 2401203]

correct_to_this_station = 2401200
correct_using_this_station = 2401199

modern_acc = 0.25

data_dir = "../climate/eureka_weather_stations/"

t0 = datetime.datetime(2021, 5, 4)
now = datetime.datetime(2023, 8, 21)

queens_temp_name = "Temperature_21049794_deg_C"
queens_data = pd.read_csv(data_dir + "../queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
queens_data["Date_Time"] = pd.to_datetime(queens_data["Date_Time"], format=time_fmt)
short_q = queens_data.loc[(queens_data["Date_Time"] > t0) & (queens_data["Date_Time"] < now), :]
short_q = short_q.set_index("Date_Time")

comp_T = short_q[[queens_temp_name]].copy()
comp_T = comp_T.rename(columns={queens_temp_name: "Queens"})

station_data = {}
for station_id in station_ids:
    # comp_T[str(station_id)] = comp_T["Queens"].copy()
    file_glob = f"en_climate_hourly_NU_{station_id}_??-????_*.csv"
    files = glob.glob(data_dir + file_glob)
    data_list = [pd.read_csv(fn) for fn in files]
    data = pd.concat(data_list).sort_values("Date/Time (LST)")
    data["Date/Time (LST)"] = pd.to_datetime(data["Date/Time (LST)"])

    data = data.rename(columns={"Date/Time (LST)": "Date_Time"})
    station_data[station_id] = data.set_index("Date_Time")

eureka_all = station_data[station_ids[0]].copy()
for station_id, df in station_data.items():
    eureka_all[str(station_id)] = df["Temp (°C)"]
for i, station_id1 in enumerate(station_ids):
    for station_id2 in station_ids[i + 1 :]:
        print(
            "Station {} is {:8.6f}°C colder than station {}".format(
                station_id1, np.nanmean(eureka_all[str(station_id1)] - eureka_all[str(station_id2)]), station_id2
            )
        )
corrections = {station_id: 0.0 for station_id in station_data}
intermediate_correction = np.nanmean(
    eureka_all[str(correct_to_this_station)] - eureka_all[str(correct_using_this_station)]
)
for station_id in station_data:
    if station_id not in [correct_to_this_station]:
        corrections[station_id] = intermediate_correction + np.nanmean(
            eureka_all[str(correct_using_this_station)] - eureka_all[str(station_id)]
        )

for station_id, df in station_data.items():
    df["Temp (°C)"] += corrections[station_id]

print("After correction")
eureka_all = station_data[station_ids[0]].copy()
for station_id, df in station_data.items():
    eureka_all[str(station_id)] = df["Temp (°C)"]
for i, station_id1 in enumerate(station_ids):
    for station_id2 in station_ids[i + 1 :]:
        print(
            "Station {} is {:8.6f}°C colder than station {}".format(
                station_id1, np.nanmean(eureka_all[str(station_id1)] - eureka_all[str(station_id2)]), station_id2
            )
        )

for station_id, data in station_data.items():
    short = data.loc[(data.index > t0) & (data.index < now), :]
    print(short)
    comp_T[str(station_id)] = short["Temp (°C)"]

eureka_offset = np.nanmean(
    comp_T[[str(station_id) for station_id in station_ids]].mean(axis=1).values - comp_T["Queens"]
)
print("The ice cap is {}°C colder than station mean".format(eureka_offset))


def get_daily_temp(queens_met_data, station_data, offset):
    # We lump the three Eureka A stations together
    eureka_mean = pd.concat(df.rename(columns={"Temp (°C)": str(sid)}) for sid, df in station_data.items())
    eureka_mean[queens_temp_name] = eureka_mean[[str(si) for si in station_ids]].mean(axis=1)
    # eureka_mean = station_data[station_ids[1]].copy()
    # eureka_mean[queens_temp_name] = eureka_mean["Temp (°C)"]

    eureka_mean = eureka_mean[[queens_temp_name]]
    daily_mean = eureka_mean.groupby(pd.Grouper(freq="1d")).mean()
    daily_mean = daily_mean.loc[daily_mean.index.year > 1958]
    daily_mean[queens_temp_name][0] = daily_mean[queens_temp_name][:365].mean()
    # annual_mean = daily_mean.groupby(pd.Grouper(freq="1y")).mean()
    # return annual_mean.index.year[6:-1] + 0.5, annual_mean[queens_temp_name].values[6:-1] - offset
    return (
        daily_mean.index.year + daily_mean.index.day_of_year / (365.0 + 1.0 * daily_mean.index.is_leap_year),
        daily_mean[queens_temp_name].values - offset,
    )


def get_smooth_temp(queens_met_data, station_data, offset):
    # We lump the three Eureka A stations together
    eureka_mean = pd.concat(df.rename(columns={"Temp (°C)": str(sid)}) for sid, df in station_data.items())
    eureka_mean[queens_temp_name] = eureka_mean[[str(si) for si in station_ids]].mean(axis=1)
    # eureka_mean = station_data[station_ids[1]].copy()
    # eureka_mean[queens_temp_name] = eureka_mean["Temp (°C)"]

    eureka_mean = eureka_mean[[queens_temp_name]]
    daily_mean = eureka_mean.groupby(pd.Grouper(freq="1d")).mean()
    daily_mean = daily_mean.loc[daily_mean.index.year > 1958]
    daily_mean = daily_mean.loc[daily_mean.index.year < 2023]

    # So we dont make the preceding 20 years too cold
    # daily_mean[queens_temp_name][0] = daily_mean[queens_temp_name][:12].mean()

    annual_mean = daily_mean.groupby(pd.Grouper(freq="1y")).mean()
    # return daily_mean.index.year + daily_mean.index.day_of_year / (365.0 + 1.0 * daily_mean.index.is_leap_year), daily_mean[queens_temp_name].values - offset
    return annual_mean.index.year + 0.5, annual_mean[queens_temp_name].values - offset


eureka_temp_age_smooth, eureka_temp_data_smooth = get_smooth_temp(queens_data, station_data, eureka_offset)
eureka_temp_years, eureka_temp_data = get_daily_temp(queens_data, station_data, eureka_offset)

v09_temp_df = pd.read_csv("../climate/other_ice_cores/Vinther_etal_2009_temp.csv", delimiter="\t")
v09_acc_df = pd.read_csv("../climate/other_ice_cores/Vinther_etal_2009_acc.csv", delimiter="\t", header=55)
ngrip_df = pd.read_csv("../climate/other_ice_cores/kindler_2014_ngrip.csv", delimiter="\t", skiprows=[1])

mask = ~np.isnan(v09_acc_df["NGRIP Acc. Rate (m ice/yr)"].values)
v09_age_acc = v09_acc_df["Age b2k (yrs)"].values[mask]
v09_acc_rough = v09_acc_df["NGRIP Acc. Rate (m ice/yr)"].values[mask]
v09_acc_smooth = v09_acc_df["NGRIP Smoothed Acc. Rate (m ice/yr)"].values[mask]

corner_freq = 2.0 / 30.0
b, a = butter(3, corner_freq, "low")
v09_acc_smoothish = filtfilt(b, a, v09_acc_rough, padlen=12)

v09_age_temp = v09_temp_df["Age (yrs b2k)"].values - 10.0
v09_temp_anom = v09_temp_df["20-yr avg temp anomaly (C)"].values
v09_temp_anom_smooth = v09_temp_df["Smoothed temp anomaly (C)"].values

# ngrip_age = ngrip_df['age ss09sea06bm '].values
ngrip_age = ngrip_df["age GICC05modelext"].values
ngrip_acc = ngrip_df["accumulation, tuned"].values
ngrip_temp = ngrip_df["temperature"].values

acc_interper = interp1d(ngrip_age, ngrip_acc)
temp_interper = interp1d(ngrip_age, ngrip_temp)
modern_temp_interper = interp1d(v09_age_temp, v09_temp_anom)

overlap_temp = temp_interper(v09_age_temp[v09_age_temp >= ngrip_age[0]])
overlap_acc = acc_interper(v09_age_acc[v09_age_acc >= ngrip_age[0]])
modern_overlap_temp = modern_temp_interper(2000.0 - eureka_temp_years[2000.0 - eureka_temp_years > 30])

scale_acc = np.nanmean(v09_acc_smoothish[v09_age_acc >= ngrip_age[0]] / overlap_acc)
modern_acc_scale = modern_acc / v09_acc_smooth[0]

offset_temp = np.nanmean(v09_temp_anom[v09_age_temp >= ngrip_age[0]] - overlap_temp)
modern_offset_temp = np.nanmean(eureka_temp_data[2000.0 - eureka_temp_years > 30] - modern_overlap_temp)

mask = ngrip_age > 11700 - 10.0
modernmask = v09_age_temp > (2000.0 - np.min(eureka_temp_years))
all_ages_temp = np.hstack(
    ([-24], 2000.0 - eureka_temp_years[::-1], v09_age_temp[modernmask], ngrip_age[mask], ngrip_age[mask][-1] + 1.0e6)
)
all_temps = np.hstack(
    (
        [eureka_temp_data_smooth[-1]],
        eureka_temp_data[::-1],
        v09_temp_anom[modernmask] + modern_offset_temp,
        ngrip_temp[mask] + offset_temp + modern_offset_temp,
        ngrip_temp[mask][-1] + offset_temp + modern_offset_temp,
    )
)

all_ages_temp_smooth = np.hstack(
    (
        [-24],
        2000.0 - eureka_temp_age_smooth[::-1],
        v09_age_temp[modernmask],
        ngrip_age[mask],
        ngrip_age[mask][-1] + 1.0e6,
    )
)
all_temps_smooth = np.hstack(
    (
        [eureka_temp_data_smooth[-1]],
        eureka_temp_data_smooth[::-1],
        v09_temp_anom[modernmask] + modern_offset_temp,
        ngrip_temp[mask] + offset_temp + modern_offset_temp,
        ngrip_temp[mask][-1] + offset_temp + modern_offset_temp,
    )
)

mask = ngrip_age > 11695 - 10.0
all_ages_acc = np.hstack(
    ([-24], v09_age_acc[v09_age_acc <= 11695 - 10.0], ngrip_age[mask], ngrip_age[mask][-1] + 1.0e6)
)
all_accs = np.hstack(
    (
        modern_acc,
        v09_acc_smoothish[v09_age_acc <= 11695 - 10.0] * modern_acc_scale,
        ngrip_acc[mask] * scale_acc * modern_acc_scale,
        ngrip_acc[mask][-1] * scale_acc * modern_acc_scale,
    )
)

acc_df = pd.DataFrame(np.vstack((-all_ages_acc, all_accs)).T, columns=["Age (yrs a2k)", "Acc (m/yr)"])
temp_df = pd.DataFrame(np.vstack((-all_ages_temp, all_temps)).T, columns=["Age (yrs a2k)", "Temp (C)"])
smooth_temp_df = pd.DataFrame(
    np.vstack((-all_ages_temp_smooth, all_temps_smooth)).T, columns=["Age (yrs a2k)", "Temp (C)"]
)

acc_df.to_csv("../climate/other_ice_cores/combined/combined_acc.csv")
temp_df.to_csv("../climate/other_ice_cores/combined/combined_temp.csv")
smooth_temp_df.to_csv("../climate/other_ice_cores/combined/combined_temp_smooth.csv")

plt.figure(num="Accumulation")
plt.plot(v09_age_acc, v09_acc_rough, color="C0")
plt.plot(all_ages_acc, all_accs, color="k", linestyle="dotted")
plt.plot(v09_age_acc, v09_acc_smoothish, color="r")
plt.plot(v09_age_acc, v09_acc_smooth, color="c")
plt.plot(ngrip_age, ngrip_acc, color="C2")
plt.ylabel("Acc (m/yr)")
plt.xlabel("Years b2k")

plt.figure(num="Temperature")
plt.plot(v09_age_temp, v09_temp_anom, color="C0")
plt.plot(v09_age_temp, v09_temp_anom, color="c")
plt.plot(ngrip_age, ngrip_temp, color="C2")
plt.plot(all_ages_temp, all_temps, color="k")
plt.plot(all_ages_temp_smooth, all_temps_smooth, color="0.6")
plt.ylabel("Temp (C))")
plt.xlabel("Years b2k")
plt.show()
