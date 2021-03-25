# CityCAT on DAFNI

[![build](https://github.com/OpenCLIM/citycat-dafni/workflows/build/badge.svg)](https://github.com/OpenCLIM/citycat-dafni/actions)

## Features
- [Generate return period events from UKCP18](#return-periods)
- [Fit storm profile to rainfall event](#storm-profile)
- [Extract domain from asc files](#dem)
- [Extract buildings and green areas from gpkg and shp files](#buildings-green-areas)
- Create archive of results
- Create GeoTIFF of max depth
- [Interpolate depths to fill building gaps](#interpolate)
- [Create PNG map of max depth](#png)
- Create metadata JSON file

## Parameters
The following parameters must be provided as environment variables (in uppercase and with spaces replaced by underscores). 
See [model-definition.yml](https://github.com/OpenCLIM/citycat-dafni/blob/master/model-definition.yml) for further details.
- Rainfall mode
- Return period
- Total depth
- Duration
- Post-event duration
- Size
- X
- Y
- Pooling radius
- Open boundaries

## Dataslots
Data is made available to the model at the following paths. The spatial projection of all datasets is assumed to be 
British National Grid. 
- inputs/dem
- inputs/ukcp
- inputs/buildings
- inputs/green_areas

## Usage 
`docker build -t citycat-dafni . && docker run -v "data:/data" --env PYTHONUNBUFFERED=1 --env RAINFALL_MODE=return_period --env TOTAL_DEPTH=40 --env SIZE=0.1 --env DURATION=1 --env POST_EVENT_DURATION=0 --env RETURN_PERIOD=100 --env X=258722 --env Y=665028 --env POOLING_RADIUS=20 --env OPEN_BOUNDARIES=True --name citycat-dafni citycat-dafni `

## <a name="return-periods">Generate return period events from UKCP18</a>
If `RAINFALL_MODE` is set to "return period" then the data in the `UKCP18` dataslot will be used to derive a rainfall total.
The netCDF is first clipped to an area generated using a combination of `X`, `Y`, `SIZE` and `POOLING_RADIUS`.
Then, the annual maxium rainfall totals for the given duration are calculated using pandas.
Finally, the [lmoments3](https://github.com/openhydrology/lmoments3) package is used to fit a generalised extreme value distribution from which the rainfall total corresponding to the `RETURN_PERIOD` is extracted.

## <a name="storm-profile">Fit storm profile to rainfall event</a>
A storm profile is fitted to the rainfall total using the `DURATION` of the event and a static unit profile.
The unit profile is scaled so that its total depth matches the rainfall total for the event.

## <a name="dem">Extract domain from asc files</a>
Rasterio is used to merge and crop all asc files in the `dem` dataslot using a boundary defined by `X`, `Y` and `SIZE`.
A nodata value of -9999 is set as this is required by CityCAT.

## <a name="buildings-green-areas">Extract buildings and green areas from gpkg and shp files</a>
All shp and gpkg files in the `buildings` and `green_areas` dataslots are merged and clipped to the domain boundary 
using geopandas. Cells within building polygons are removed from the domain and infiltration is allowed at cells within 
green areas polygons.

## <a name="interpolate">Interpolate depths to fill building gaps</a>
The gaps in the domain created where buildings exist are filled using inverse distance weighting and a new GeoTIFF is created.
This was carried out to allow point geometries to be used when calculating impacts.
If the original results are used without interpolation, then there is likely to be missing data at asset locations.

## <a name="png">Create PNG map of max depth</a>
Rasterio is used to create a 1280 x 960 pixel map of maximum.
This image is intended to enable quick interpretation of results but it may not always provide a good visualisation as 
the colour scale has a maximum value of 1m.