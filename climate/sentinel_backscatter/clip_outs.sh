#! /bin/sh
#
# clip_outs.sh
# Copyright (C) 2022 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.
#

for f in *[0-9].tif; do
    rm -f ${f%.*}_clipped.tif
    gdalwarp -t_srs EPSG:3413 -tr 50 50 -of GTiff -cutline /home/dlilien/work/muellers/rgi/rgi60_muellers_corrected.gpkg -cl rgi60_muellers -dstalpha -dstnodata -9999.0 -crop_to_cutline $f ${f%.*}_clipped.tif
done

