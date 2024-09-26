#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
import pandas as pd
import datetime as dt
import glob
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from math import floor


station_ids = [2401199, 2401200, 2401208, 2401203]

# Exclude the oddball
eua_station_ids = [2401200, 2401208, 2401203]

correct_to_this_station = 2401200
correct_using_this_station = 2401199


def wind_chill(T, v_mps):
    v_kmph = v_mps * 3.6
    return np.minimum(13.12 + 0.6215 * T - 11.37 * v_kmph**0.16 + 0.3965 * T * v_kmph**0.16, T)


def analyze(queens_met_data, station_data):
    queens_data = queens_met_data.set_index("Date_Time")

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

    comp_T = queens_data.loc[(queens_data.index > t0) & (queens_data.index < now), :][[queens_temp_name]].copy()
    comp_T = comp_T.rename(columns={queens_temp_name: "Queens"})
    for sid, df in station_data.items():
        short = df.loc[(df.index > t0) & (df.index < now), :]
        comp_T[str(sid)] = df["Temp (°C)"]
    for station_id in station_ids:
        diff = np.nanmean(comp_T[str(station_id)] - comp_T["Queens"])
        if np.isfinite(diff):
            print("The ice cap is {}°C colder than station {}".format(diff, station_id))
    eureka_offset = np.nanmean(
        comp_T[[str(station_id) for station_id in station_ids]].mean(axis=1).values - comp_T["Queens"]
    )
    print("The ice cap is {}°C colder than station mean".format(eureka_offset))

    for name, months in [("Summer", [6, 7, 8]), ("Winter", [12, 1, 2]), ("Spring", [3, 4, 5]), ("Fall", [9, 10, 11])]:
        seasonal_data = comp_T.loc[comp_T.index.month.isin(months)]
        for station_id in station_ids:
            print(
                "In {:s}, the ice cap is {}°C colder than station {}".format(
                    name, np.nanmean(seasonal_data[str(station_id)] - seasonal_data["Queens"]), station_id
                )
            )
        eureka_offset = np.nanmean(
            seasonal_data[[str(station_id) for station_id in station_ids]].mean(axis=1).values - seasonal_data["Queens"]
        )
        print("In {:s}, the ice cap is {}°C colder than station mean".format(name, eureka_offset))

    demeaned_data = {
        station_id: (ts[["Temp (°C)"]] - ts[["Temp (°C)"]].mean()).groupby(pd.Grouper(freq="1h")).mean()
        for station_id, ts in station_data.items()
    }
    mmean_data = {
        station_id: ts.groupby([ts.index.month, ts.index.day, ts.index.hour]).mean()
        for station_id, ts in demeaned_data.items()
    }
    eureka_mean = mmean_data[station_ids[0]].copy()
    for station_id, df in mmean_data.items():
        eureka_mean[str(station_id)] = df["Temp (°C)"]
    eureka_mean[queens_temp_name] = eureka_mean[[str(si) for si in station_ids]].mean(axis=1)
    # eureka_mean.rename(columns={0: "Temp"}, inplace=True)

    demeaned_queens = queens_data[[queens_temp_name]].copy() - queens_data[[queens_temp_name]].mean()
    demeaned_queens[queens_temp_name] = (
        demeaned_queens.values
        - eureka_mean.loc[
            [
                (month, day, hour)
                for month, day, hour in zip(
                    demeaned_queens.index.month, demeaned_queens.index.day, demeaned_queens.index.hour
                )
            ]
        ].values
    )

    unseasonal_data = {sid: data.copy() for sid, data in demeaned_data.items()}
    for sid, data in unseasonal_data.items():
        unseasonal_data[sid]["Temp (°C)"] = (
            unseasonal_data[sid].values
            - eureka_mean.loc[
                [
                    (month, day, hour)
                    for month, day, hour in zip(
                        unseasonal_data[sid].index.month,
                        unseasonal_data[sid].index.day,
                        unseasonal_data[sid].index.hour,
                    )
                ]
            ].values
        )

    comp_T = demeaned_queens.loc[(demeaned_queens.index > t0) & (demeaned_queens.index < now), :][
        [queens_temp_name]
    ].copy()
    comp_T = comp_T.rename(columns={queens_temp_name: "Queens"})

    for sid, data in unseasonal_data.items():
        short = data.loc[(data.index > t0) & (data.index < now), :]
        comp_T[str(sid)] = short["Temp (°C)"]
    print("After removing the seasonal (to the hour) cycle, correlation is")
    print(comp_T.corr())

    demeaned_data = {
        station_id: (ts[["Temp (°C)"]] - ts[["Temp (°C)"]].mean()).groupby(pd.Grouper(freq="1d")).mean()
        for station_id, ts in station_data.items()
    }
    mmean_data = {
        station_id: ts.groupby([ts.index.month, ts.index.day]).mean() for station_id, ts in demeaned_data.items()
    }
    eureka_mean = mmean_data[station_ids[0]].copy()
    for station_id, df in mmean_data.items():
        eureka_mean[str(station_id)] = df["Temp (°C)"]
    eureka_mean[queens_temp_name] = eureka_mean[[str(si) for si in station_ids]].mean(axis=1)
    # eureka_mean.rename(columns={0: "Temp"}, inplace=True)

    demeaned_queens = queens_data[[queens_temp_name]].copy() - queens_data[[queens_temp_name]].mean()
    demeaned_queens[queens_temp_name] = (
        demeaned_queens.values
        - eureka_mean.loc[
            [
                (month, day)
                for month, day in zip(
                    demeaned_queens.index.month,
                    demeaned_queens.index.day,
                )
            ]
        ].values
    )

    unseasonal_data = {sid: data.copy() for sid, data in demeaned_data.items()}
    for sid, data in unseasonal_data.items():
        unseasonal_data[sid]["Temp (°C)"] = (
            unseasonal_data[sid].values
            - eureka_mean.loc[
                [(month, day) for month, day in zip(unseasonal_data[sid].index.month, unseasonal_data[sid].index.day)]
            ].values
        )

    comp_T = demeaned_queens.loc[(demeaned_queens.index > t0) & (demeaned_queens.index < now), :][
        [queens_temp_name]
    ].copy()
    comp_T = comp_T.rename(columns={queens_temp_name: "Queens"})

    for sid, data in unseasonal_data.items():
        short = data.loc[(data.index > t0) & (data.index < now), :]
        comp_T[str(sid)] = short["Temp (°C)"]
    print("After removing the seasonal (to the day) cycle, correlation is")
    print(comp_T.corr())

    # We lump the three Eureka A stations together
    eureka_mean = pd.concat(
        df.rename(columns={"Temp (°C)": str(sid), "Precip. Amount (mm)": str(sid) + "mm"})
        for sid, df in station_data.items()
    )
    eureka_mean[queens_temp_name] = eureka_mean[[str(si) for si in station_ids]].mean(axis=1)
    eureka_mean["Acc"] = eureka_mean[[str(si) + "mm" for si in station_ids]].mean(axis=1)
    # eureka_mean = station_data[station_ids[1]].copy()
    # eureka_mean[queens_temp_name] = eureka_mean["Temp (°C)"]

    eureka_mean = eureka_mean[[queens_temp_name, "Acc"]]

    acc_tot = eureka_mean.groupby(pd.Grouper(freq="1d")).sum()
    daily_mean = eureka_mean.groupby(pd.Grouper(freq="1d")).mean()
    annual_mean = daily_mean.groupby(pd.Grouper(freq="1YE")).mean()

    daily_mean.to_csv("../climate/eureka_weather_stations/eureka_temps.csv")
    acc_tot.to_csv("../climate/eureka_weather_stations/daily_acc.csv")

    def plot_trends(ax):
        # ax.plot(eureka_mean.index, eureka_mean[queens_temp_name], label='Eureka weather station', color='0.6')
        ax.plot(
            daily_mean.index.values, daily_mean[queens_temp_name].values, label="Eureka weather station", color="0.6"
        )
        ax.plot(
            queens_data.index.values, queens_data[queens_temp_name].values, label="Ice-cap weather station", color="k"
        )
        # ax.plot(annual_mean.index, annual_mean[queens_temp_name])
        # plt.title("Eureka mean")

        # Deal with the fact that some years are sparsely sampled and thus biased
        good_data = annual_mean.loc[(annual_mean.index.year > 1972) & (annual_mean.index.year < 2016), :]
        A = np.ones((good_data.shape[0], 2))
        A[:, 0] = good_data.index.year.values
        b = good_data[queens_temp_name].values
        trend = np.linalg.lstsq(A, b, rcond=None)[0]
        print("Partial at Eureka is {:f}°C per decade".format(trend[0] * 10.0))
        good_data = annual_mean.loc[(annual_mean.index.year > 1972) & (annual_mean.index.year < 2024), :]
        A = np.ones((good_data.shape[0], 2))
        A[:, 0] = good_data.index.year.values
        b = good_data[queens_temp_name].values
        trend = np.linalg.lstsq(A, b, rcond=None)[0]
        print("Recent at Eureka is {:f}°C per decade".format(trend[0] * 10.0))
        years = np.arange(1973, 2025, 0.5)
        ax.plot(
            [dt.datetime(floor(year), 1 + int((year % 1) * 6), 1, 0, 0, 0) for year in years],
            years * trend[0] + trend[1],
            color="C4",
        )
        ax.text(
            dt.datetime(1990, 1, 1, 0, 0, 0),
            3.0 + 1990 * trend[0] + trend[1],
            "Trend 1973–2024, {:1.2} °C per decade".format(trend[0] * 10.0),
            color="C4",
            rotation=1,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round", fc="w", ec="C4", alpha=0.8),
        )
        years = np.arange(1953, 1973.5, 0.5)
        ax.plot(
            [dt.datetime(floor(year), 1 + int((year % 1) * 6), 1, 0, 0, 0) for year in years],
            years * trend[0] + trend[1],
            linestyle="dashed",
            color="C4",
        )

        good_data = annual_mean.loc[
            (annual_mean.index.year <= 1972) & ~np.isnan(annual_mean[queens_temp_name].values), :
        ]
        A = np.ones((good_data.shape[0], 2))
        A[:, 0] = good_data.index.year.values
        b = good_data[queens_temp_name].values
        trend = np.linalg.lstsq(A, b, rcond=None)[0]
        print("Previous at Eureka is {:f}°C per decade".format(trend[0] * 10.0))
        years = np.arange(1973, 2025, 0.5)
        ax.plot(
            [dt.datetime(floor(year), 1 + int((year % 1) * 6), 1, 0, 0, 0) for year in years],
            years * trend[0] + trend[1],
            color="C2",
            linestyle="dashed",
        )
        years = np.arange(1953, 1973.5, 0.5)
        ax.plot(
            [dt.datetime(floor(year), 1 + int((year % 1) * 6), 1, 0, 0, 0) for year in years],
            years * trend[0] + trend[1],
            color="C2",
        )
        ax.text(
            dt.datetime(1981, 1, 1, 0, 0, 0),
            1981 * trend[0] + trend[1] - 3.0,
            "Trend 1959–1972, {:1.2} °C per decade".format(trend[0] * 10.0),
            color="C2",
            rotation=-1,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", fc="w", ec="C2", alpha=0.8),
        )

        good_data = annual_mean.loc[annual_mean.index.year < 2024, :]
        good_data = good_data.loc[~np.isnan(good_data[queens_temp_name]), :]
        A = np.ones((good_data.shape[0], 2))
        A[:, 0] = good_data.index.year.values
        b = good_data[queens_temp_name].values
        trend = np.linalg.lstsq(A, b, rcond=None)[0]
        years = np.arange(1953, 2024, 0.5)
        # plt.plot([dt.datetime(floor(year), 1 + int((year % 1) * 6), 1, 0, 0, 0) for year in years], years * trend[0] + trend[1])
        print("Overall at Eureka is {:f}°C per decade".format(trend[0] * 10.0))
        print("Implying {:f}°C total change".format(trend[0] * 71))

    return plot_trends


data_dir = "../climate/eureka_weather_stations/"

t0 = datetime.datetime(2021, 5, 4)
now = datetime.datetime(2024, 6, 1)

queens_temp_name = "Temperature_21049794_deg_C"
time_fmt = "%m/%d/%y %H:%M:%S"
queens_data = pd.read_csv(data_dir + "../queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
queens_data["Date_Time"] = pd.to_datetime(queens_data["Date_Time"], format=time_fmt)
short_q = queens_data.loc[(queens_data["Date_Time"] > t0) & (queens_data["Date_Time"] < now), :]
short_q = short_q.set_index("Date_Time")

span = datetime.timedelta(days=1096)
avgs = [
    queens_data.loc[
        (queens_data["Date_Time"] > t0 + datetime.timedelta(days=t_off))
        & (queens_data["Date_Time"] < t0 + span + datetime.timedelta(days=t_off)),
        :,
    ][queens_temp_name].mean()
    for t_off in range(27)
]
print("Average temperature on top of ice cap is: {:f}".format(np.mean(avgs)))
print("Minimum temperature on top of ice cap is: {:f}".format(queens_data[queens_temp_name].min()))
print("Maximum temperature on top of ice cap is: {:f}".format(queens_data[queens_temp_name].max()))

comp_T = short_q[[queens_temp_name]].copy()
comp_T = comp_T.rename(columns={queens_temp_name: "Queens"})

station_data = {}


fig1, ax1 = plt.subplots()
fig, ax = plt.subplots()
for station_id in station_ids:
    # comp_T[str(station_id)] = comp_T["Queens"].copy()
    file_glob = f"en_climate_hourly_NU_{station_id}_??-????_*.csv"
    files = glob.glob(data_dir + file_glob)
    data_list = [pd.read_csv(fn) for fn in files]
    data = pd.concat(data_list).sort_values("Date/Time (LST)")
    data["Date/Time (LST)"] = pd.to_datetime(data["Date/Time (LST)"])

    station_data[station_id] = data.set_index("Date/Time (LST)").drop(
        columns=[
            "Station Name",
            "Longitude (x)",
            "Latitude (y)",
            "Climate ID",
            "Time (LST)",
            "Temp Flag",
            "Dew Point Temp Flag",
            "Rel Hum Flag",
            "Precip. Amount Flag",
            "Wind Dir Flag",
            "Wind Spd Flag",
            "Visibility (km)",
            "Visibility Flag",
            "Stn Press Flag",
            "Hmdx",
            "Hmdx Flag",
            "Wind Chill",
            "Wind Chill Flag",
            "Weather",
        ]
    )

    short = data.loc[(data["Date/Time (LST)"] > t0) & (data["Date/Time (LST)"] < now), :]
    short = short.rename(columns={"Date/Time (LST)": "Date_Time"})
    short = short.set_index("Date_Time")
    comp_T[str(station_id)] = short["Temp (°C)"]

    data.plot("Date/Time (LST)", "Temp (°C)", ax=ax, label=str(station_id))
    data.plot("Date/Time (LST)", "Precip. Amount (mm)", ax=ax1, label=str(station_id))
    data.plot("Date/Time (LST)", "Temp (°C)")
    plt.title(str(station_id))


print("Overall correlation")
print(comp_T.corr())

plot_trends = analyze(queens_data, station_data)

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plot_trends(ax)

    mean_data = {}
    mean_data = {
        station_id: ts.groupby([ts.index.month, ts.index.day, ts.index.hour]).median()
        for station_id, ts in station_data.items()
    }
    # max_data = {station_id: ts.groupby([ts.index.month, ts.index.day]).max() for station_id, ts in station_data.items()}
    # min_data = {station_id: ts.groupby([ts.index.month, ts.index.day]).min() for station_id, ts in station_data.items()}
    max_data = {
        station_id: ts[["Temp (°C)"]].groupby(pd.Grouper(freq="1d")).max() for station_id, ts in station_data.items()
    }
    min_data = {
        station_id: ts[["Temp (°C)"]].groupby(pd.Grouper(freq="1d")).min() for station_id, ts in station_data.items()
    }
    mmax_data = {station_id: ts.groupby([ts.index.month, ts.index.day]).median() for station_id, ts in max_data.items()}
    mmin_data = {station_id: ts.groupby([ts.index.month, ts.index.day]).median() for station_id, ts in min_data.items()}

    for dv in [mmax_data, mmin_data]:
        for station_id, df in dv.items():
            df.index = df.index.rename(["month", "day"])
            df.index = pd.to_datetime(
                "1972-"
                + df.index.get_level_values(0).astype(int).map("{:02d}".format)
                + "-"
                + df.index.get_level_values(1).astype(int).map("{:02d}".format)
                + ",00:00:00",
                format="%Y-%m-%d,%H:%M:%S",
            )

    for dv in [mean_data]:
        for station_id, df in dv.items():
            df.index = df.index.rename(["month", "day", "hour"])
            df.index = pd.to_datetime(
                "1972-"
                + df.index.get_level_values(0).astype(int).map("{:02d}".format)
                + "-"
                + df.index.get_level_values(1).astype(int).map("{:02d}".format)
                + ","
                + df.index.get_level_values(2).astype(int).map("{:02d}".format)
                + ":00:00",
                format="%Y-%m-%d,%H:%M:%S",
            )

    queens_data.plot("Date_Time", "Temperature_21049794_deg_C", ax=ax, label="Met station (ca. 1850 m)")

    fig, ax = plt.subplots()
    for station_id in station_ids:
        mean_data[station_id].plot(y="Temp (°C)", use_index=True, ax=ax)

    wx_dates = pd.to_datetime(queens_data["Date_Time"])

    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.axhline(0.0, color='k', linestyle='dashed')
    # ax.plot(mean_data[station_ids[1]].index, mean_data[station_ids[1]]["Temp (°C)"], color='k', label='Eureka mean')
    ax.fill_between(
        mmin_data[station_ids[1]].index,
        mmin_data[station_ids[1]]["Temp (°C)"],
        mmax_data[station_ids[1]]["Temp (°C)"],
        color="0.6",
    )
    ax.plot(mmin_data[station_ids[1]].index, mmin_data[station_ids[1]]["Temp (°C)"], color="b", label="Eureka mean low")
    ax.plot(
        mmax_data[station_ids[1]].index, mmax_data[station_ids[1]]["Temp (°C)"], color="r", label="Eureka mean high"
    )

    ax.grid()
    for year in range(2021, 2025):
        year_data = queens_data.loc[
            (wx_dates < dt.datetime(year + 1, 1, 1, 0, 0, 0)) & (wx_dates >= dt.datetime(year, 1, 1, 0, 0, 0))
        ]
        # year_data['Day_time'] = year_data['Date_Time'].dt.strftime('%m-%d, %H:%M:%S').copy()
        year_data.loc[:, "Date_Time"] = year_data["Date_Time"].apply(lambda x: x.replace(year=1972))

        l = ax.plot(year_data["Date_Time"], year_data["Temperature_21049794_deg_C"], linewidth=1.5, label=str(year))
        l = ax.plot(
            year_data["Date_Time"],
            wind_chill(year_data["Temperature_21049794_deg_C"], year_data["Wind Speed_20354828_m/s"]),
            linewidth=1.5,
            linestyle="dashed",
            color=l[0].get_color(),
        )

    # ax.set_xlim(dt.datetime(2021, 4, 1, 0, 0, 0), dt.datetime(2024, 4, 1, 0, 0, 0))
    ax.set_xlim(dt.datetime(1972, 1, 1, 0, 0, 0), dt.datetime(1973, 1, 1, 0, 0, 0))
    ax.set_ylim(-50, 20)

    ax.legend(loc="best")

    month_fmt = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(lambda x, y: "")

    plt.xlabel("Date")
    plt.ylabel(r"Temperature ($^\circ$C)")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.grid()
    for year in range(2021, 2025):
        year_data = queens_data.loc[
            (wx_dates < dt.datetime(year + 1, 1, 1, 0, 0, 0)) & (wx_dates >= dt.datetime(year, 1, 1, 0, 0, 0))
        ]
        # year_data['Day_time'] = year_data['Date_Time'].dt.strftime('%m-%d, %H:%M:%S').copy()
        year_data.loc[:, "Date_Time"] = year_data["Date_Time"].apply(lambda x: x.replace(year=1972))

        ax.plot(year_data["Date_Time"], year_data["Wind Speed_20354828_m/s"], linewidth=1.5, label=str(year))

    ax.set_xlim(dt.datetime(1972, 1, 1, 0, 0, 0), dt.datetime(1973, 1, 1, 0, 0, 0))

    ax.legend(loc="best")

    month_fmt = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(lambda x, y: "")

    plt.xlabel("Date")
    plt.ylabel(r"Wind speed (m/s)")
    plt.show()
