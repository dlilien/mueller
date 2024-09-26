#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import datetime

# Physical constants
year = 365.25 * 24 * 60 * 60
capacity = 2090  # 2.108 * year ** 2.0  # kJ / kg / K
conductivity = 2.35  # W / m / K
diffusivity = 1.02e-6 * year
density = 910.0

Hmax = 1210.0
Hfinal = 605.79

acc_modern = 0.25
time_of_temp_measurement = 23.0 + (datetime.datetime(2023, 5, 26) - datetime.datetime(2023, 1, 1)).total_seconds() / (
    365.0 * 24 * 60 * 60
)

basal_heat_flux = 54.0e-3 * year
