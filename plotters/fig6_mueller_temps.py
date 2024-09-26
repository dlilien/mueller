#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import matplotlib.pyplot as plt


import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
from matplotlib import ticker, gridspec

from plot_mueller_weather_stations import plot_trends
import numpy as np

time_fmt = "%m/%d/%y %H:%M:%S"


def three_panel_series():

    wx_df = pd.read_csv("../climate/queens_met_station/QueensUniversity_007.txt", header=2, delimiter="\t")
    wx_df["Date_Time"] = pd.to_datetime(wx_df["Date_Time"], format=time_fmt)

    a_year = 365 * 24
    mean_temps = np.zeros((wx_df.shape[0] - a_year + 1,))
    for startind in range(len(mean_temps)):
        mean_temps[startind] = np.nanmean(wx_df["Temperature_21049794_deg_C"].values[startind : startind + a_year])
    print(mean_temps)
    print("Standardized mean temp at icecap summit is {:4.2f} C".format(np.mean(mean_temps)))

    times = pd.read_csv("../climate/sentinel_backscatter/summit_timeseries.csv", parse_dates=["time"])
    val16 = pd.read_csv("../climate/sentinel_backscatter/averageMelt_1600m.csv", skip_blank_lines=False)
    val16.columns = ["1600–1700 m"]
    val17 = pd.read_csv("../climate/sentinel_backscatter/averageMelt_1700m.csv", skip_blank_lines=False)
    val17.columns = ["1700–1800 m"]
    val18 = pd.read_csv("../climate/sentinel_backscatter/averageMelt_1800m.csv", skip_blank_lines=False)
    val18.columns = [">1800 m"]

    ts = pd.concat((times, val16, val17, val18), axis=1)  # .sort_values('time')

    gs = gridspec.GridSpec(
        4, 1, hspace=0.05, height_ratios=[1, 0.1, 1, 1], left=0.09, right=0.97, bottom=0.08, top=0.995
    )
    fig = plt.figure(figsize=(7.8, 6.5))
    ax_eureka = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[2, 0])
    ax = fig.add_subplot(gs[3, 0])

    for name, color in [("1600–1700 m", "C0"), ("1700–1800 m", "C1"), (">1800 m", "k")]:
        ax.plot(ts["time"].values, ts[name].values, color=color, label=name)

    ax.set_xlim(dt.datetime(2016, 1, 1, 0, 0, 0), dt.datetime(2025, 1, 1, 0, 0, 0))
    ax1.set_xlim(dt.datetime(2016, 1, 1, 0, 0, 0), dt.datetime(2025, 1, 1, 0, 0, 0))
    ax.set_ylim(-25, 5)
    ax1.set_ylim(-50, 25)

    plot_trends(ax_eureka)
    ax_eureka.set_xlim(dt.datetime(1959, 1, 1, 0, 0, 0), dt.datetime(2025, 1, 1, 0, 0, 0))
    ax_eureka.set_ylim(-60, 25)
    ax_eureka.legend(loc="lower left")

    ax.legend(loc="lower left")

    @ticker.FuncFormatter
    def month_fmt(x, pos):
        if (pos % 3) == 1:
            return "Jul"
        else:
            return ticker.NullFormatter()(x, pos)

    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_minor_formatter(month_fmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    dates = []
    for year in range(2017, 2025):
        year_data = ts.loc[
            (ts["time"] < dt.datetime(year + 1, 1, 1, 0, 0, 0)) & (ts["time"] >= dt.datetime(year, 1, 1, 0, 0, 0))
        ]
        ind = year_data[year_data[">1800 m"] < -4.5].first_valid_index()
        ind2 = year_data[year_data[">1800 m"] < -4.5].last_valid_index()
        if ind is not None:
            t = year_data["time"][ind]
            dates.append(t)
            v = year_data[">1800 m"][ind]
            if t.year in [2016, 2017, 2020]:
                ax.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(30, -10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="k"),
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round", fc="w"),
                )
            elif t.year in [2023]:
                ax.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(15, -45),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="k"),
                    ha="left",
                    # ha="right",
                    va="center",
                    bbox=dict(boxstyle="round", fc="w"),
                )
            elif t.year in [2019]:
                ax.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(-15, -28),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="k"),
                    ha="right",
                    va="center",
                    bbox=dict(boxstyle="round", fc="w"),
                )
            else:
                ax.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(-25, -15),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="k"),
                    ha="right",
                    va="center",
                    bbox=dict(boxstyle="round", fc="w"),
                )
            # ax1.axvline(t, color='k', linewidth=0.5)
            print("Sentinel", year_data["time"][ind2].strftime("%b %d, %Y"))

    if False:
        for year in range(2016, 2023):
            year_data = ts.loc[
                (ts["time"] < dt.datetime(year + 1, 1, 1, 0, 0, 0)) & (ts["time"] >= dt.datetime(year, 1, 1, 0, 0, 0))
            ]
            ind = year_data[year_data["1600–1700 m"] < -4.0].first_valid_index()
            if ind is not None:
                t = year_data["time"][ind]
                if t not in dates:
                    v = year_data["1600-1700 m"][ind]
                    if t > dt.datetime(2017, 1, 1, 0, 0, 0):
                        ax.annotate(
                            t.strftime("%b %d, %Y"),
                            (mdates.date2num(t), v),
                            xytext=(-25, -15),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="-|>", color="C0"),
                            ha="right",
                            va="top",
                            color="C0",
                        )
                    else:
                        ax.annotate(
                            t.strftime("%b %d,\n%Y"),
                            (mdates.date2num(t), v),
                            xytext=(-20, -25),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="-|>", color="C0"),
                            ha="center",
                            va="top",
                            color="C0",
                            bbox=dict(boxstyle="round", fc="w", ec="C0"),
                        )

    ax.set_ylabel("Surface reflectivity (dB)")
    ax1.set_ylabel(r"Temperature ($^\circ$C)")
    ax_eureka.set_ylabel(r"Temperature ($^\circ$C)")

    # ax1.axhline(0.0, color="k", linestyle="dashed")

    ax.grid()
    ax1.grid()
    ax_eureka.text(0.01, 0.99, "a", ha="left", va="top", fontsize=14, transform=ax_eureka.transAxes)
    ax.text(0.01, 0.99, "c", ha="left", va="top", fontsize=14, transform=ax.transAxes)
    # ax.text(0.01, 0.92, "a", ha="left", va="bottom", fontsize=14, weight="bold", transform=ax.transAxes)
    # ax.text(0.03, 0.92, "Sentinel 1 backscatter", ha="left", va="bottom", fontsize=10, transform=ax.transAxes)
    ax1.text(0.01, 0.99, "b", ha="left", va="top", fontsize=14, transform=ax1.transAxes)
    # ax1.text(0.01, 0.92, "b", ha="left", va="bottom", fontsize=14, weight="bold", transform=ax1.transAxes)
    # ax1.text(0.03, 0.92, "Weather station", ha="left", va="bottom", fontsize=10, transform=ax1.transAxes)
    ax1.plot(wx_df["Date_Time"].values, wx_df["Temperature_21049794_deg_C"].values, linewidth=0.5, color="0.6")

    daily_df = wx_df.groupby(pd.Grouper(key="Date_Time", freq="1D"))
    ax1.plot(
        daily_df.min().index.values,
        daily_df.min()["Temperature_21049794_deg_C"].values,
        linewidth=1,
        color="b",
        label="Daily low",
    )
    ax1.plot(
        daily_df.max().index.values,
        daily_df.max()["Temperature_21049794_deg_C"].values,
        linewidth=1,
        color="r",
        label="Daily high",
    )
    ax1.legend(loc="lower left")
    wx_dates = pd.to_datetime(wx_df["Date_Time"])

    for year in range(2016, 2025):
        year_data = wx_df.loc[
            (wx_dates < dt.datetime(year + 1, 1, 1, 0, 0, 0)) & (wx_dates >= dt.datetime(year, 1, 1, 0, 0, 0))
        ]
        ind = year_data[year_data["Temperature_21049794_deg_C"] > 0.0].first_valid_index()
        ind2 = year_data[year_data["Temperature_21049794_deg_C"] > 0.0].last_valid_index()
        if ind is not None:
            t = wx_dates[ind]
            v = year_data["Temperature_21049794_deg_C"][ind]
            if t > dt.datetime(2024, 1, 1, 0, 0, 0):
                ax1.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(5, 15),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="r"),
                    ha="center",
                    va="bottom",
                    color="r",
                    bbox=dict(boxstyle="round", fc="w", ec="r"),
                )
            elif t > dt.datetime(2023, 1, 1, 0, 0, 0):
                ax1.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(-4, 35),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="r"),
                    ha="center",
                    va="bottom",
                    color="r",
                    bbox=dict(boxstyle="round", fc="w", ec="r"),
                )
            elif t > dt.datetime(2022, 1, 1, 0, 0, 0):
                ax1.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(-20, 30),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="r"),
                    ha="right",
                    va="top",
                    color="r",
                    bbox=dict(boxstyle="round", fc="w", ec="r"),
                )
            else:
                ax1.annotate(
                    t.strftime("%b %d, %Y"),
                    (mdates.date2num(t), v),
                    xytext=(-30, 15),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-|>", color="r"),
                    ha="right",
                    va="center",
                    color="r",
                    bbox=dict(boxstyle="round", fc="w", ec="r"),
                )

        if ind2 is not None:
            print("Weather station high", wx_dates[ind2].strftime("%b %d, %Y"))
    for year in range(2016, 2024):
        year_data = daily_df.min().loc[
            (daily_df.min().index < dt.datetime(year + 1, 1, 1, 0, 0, 0))
            & (daily_df.min().index >= dt.datetime(year, 1, 1, 0, 0, 0))
        ]
        ind = year_data[year_data["Temperature_21049794_deg_C"] > 0.0].first_valid_index()
        ind2 = year_data[year_data["Temperature_21049794_deg_C"] > 0.0].last_valid_index()
        if ind is not None:
            t = ind
            v = year_data["Temperature_21049794_deg_C"][ind]
            if t > dt.datetime(2023, 1, 1, 0, 0, 0):
                xoff = 5
            else:
                xoff = -10
            ax1.annotate(
                t.strftime("%b %d, %Y"),
                (mdates.date2num(t), v),
                xytext=(xoff, -80),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>", color="b"),
                ha="center",
                va="top",
                color="b",
                bbox=dict(boxstyle="round", fc="w", ec="b"),
            )
        if ind2 is not None:
            print("Weather station", ind2.strftime("%b %d, %Y"))

    ax1.axes.xaxis.set_ticklabels([])
    fig.savefig("../plots/JOG-2024-0020.Figure6.pdf")


if __name__ == "__main__":
    three_panel_series()
