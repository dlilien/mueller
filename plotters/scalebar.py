#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""

def scalebar(ax, loc, length_km=20, xoff=8000, yoff=9000, ytoff=1000):
    LL = [ax.get_xlim()[0], ax.get_ylim()[0]]
    UR = [ax.get_xlim()[1], ax.get_ylim()[1]]
    if loc == "LL":
        x1 = LL[0] + xoff
        x2 = LL[0] + xoff + length_km * 1000
        y = LL[1] + yoff
    if loc == "LR":
        x1 = UR[0] - xoff
        x2 = UR[0] - xoff - length_km * 1000
        y = LL[1] + yoff
    if loc == "UR":
        x1 = UR[0] - xoff
        x2 = UR[0] - xoff - length_km * 1000
        y = UR[1] - yoff
    ax.plot([x1, x2], [y, y], color="k", linewidth=4, zorder=10000)
    ax.text((x1 + x2) / 2.0, y - ytoff, "{:d} km".format(length_km), ha="center", va="center", color="k", zorder=10000)
