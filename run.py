import os
import shutil  # must be imported before GDAL
from rasterio.merge import merge
import rasterio as rio
from rasterio.io import MemoryFile
from citycatio import Model, output
import pandas as pd
import subprocess
import xarray as xr
from glob import glob
from shapely.geometry import Point
from lmoments3 import distr
import geopandas as gpd
import rioxarray as rx
from rasterio.crs import CRS
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.fill import fillnodata
from datetime import datetime
import numpy as np
from shapely.geometry import box
import json
from matplotlib.colors import ListedColormap

data_path = os.getenv('DATA_PATH', '/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

# Read environment variables
rainfall_mode = os.getenv('RAINFALL_MODE')
rainfall_total = int(os.getenv('TOTAL_DEPTH'))
size = float(os.getenv('SIZE')) * 1000  # convert from km to m
duration = int(os.getenv('DURATION'))
post_event_duration = int(os.getenv('POST_EVENT_DURATION'))
return_period = int(os.getenv('RETURN_PERIOD'))
x = int(os.getenv('X'))
y = int(os.getenv('Y'))
pooling_radius = int(os.getenv('POOLING_RADIUS')) * 1000  # convert from km to m
open_boundaries = (os.getenv('OPEN_BOUNDARIES').lower() == 'true')
permeable_areas = os.getenv('PERMEABLE_AREAS')
roof_storage = float(os.getenv('ROOF_STORAGE'))

nodata = -9999

if rainfall_mode == 'return_period':
    # Get rainfall values within pooling radius
    ds = xr.open_dataset(glob(os.path.join(inputs_path, 'ukcp/pr*'))[0]).pr.rio.set_crs('EPSG:27700')
    df = ds.rio.clip([Point(x, y).buffer(pooling_radius)]).to_dataframe().pr.dropna().reset_index()

    # Calculate rolling sum for duration
    rolling = df.set_index('time').groupby(
       ['projection_x_coordinate', 'projection_y_coordinate']).pr.rolling(duration).sum().reset_index()

    # Get annual maxima
    amax = rolling.groupby(['projection_x_coordinate', 'projection_y_coordinate',
                            [t.year for t in rolling.time.values]]).pr.max().reset_index(drop=True)

    # Fit GEV and find rainfall total
    params = distr.gev.lmom_fit(amax.values)
    fitted_gev = distr.gev(**params)
    rainfall_total = float(fitted_gev.ppf(return_period / len(amax)))

print(f'Rainfall Total:{rainfall_total}')

unit_profile = np.array([0.017627993, 0.027784045, 0.041248418, 0.064500665, 0.100127555, 0.145482534, 0.20645758,
                         0.145482534, 0.100127555, 0.064500665, 0.041248418, 0.027784045, 0.017627993])

# Fit storm profile
rainfall_times = np.linspace(start=0, stop=duration*3600, num=len(unit_profile))

unit_total = sum((unit_profile + np.append(unit_profile[1:], [0])) / 2 *
                 (np.append(rainfall_times[1:], rainfall_times[[-1]]+1)-rainfall_times))

rainfall = pd.DataFrame(list(unit_profile*rainfall_total/unit_total/1000) + [0, 0],
                        index=list(rainfall_times) + [duration*3600+1, duration*3600+2])

# Create run directory
run_path = os.path.join(outputs_path, 'run')
if not os.path.exists(run_path):
    os.mkdir(run_path)

# Read and clip DEM
dem_path = os.path.join(inputs_path, 'dem')
dem_datasets = [rio.open(os.path.join(dem_path, os.path.abspath(p))) for p in glob(os.path.join(dem_path, '*.asc'))]
bounds = x-size/2, y-size/2, x+size/2, y+size/2
array, transform = merge(dem_datasets, bounds=bounds, precision=50, nodata=nodata)
assert array[array != nodata].size > 0, "No DEM data available for selected location"


def read_geometries(path):
    paths = glob(os.path.join(inputs_path, path, '*.gpkg'))
    paths.extend(glob(os.path.join(inputs_path, path, '*.shp')))
    print(f'Files in {path} directory: {[os.path.basename(p) for p in paths]}')
    geometries = gpd.read_file(paths[0], bbox=bounds) if len(paths) > 0 else None
    if len(paths) > 1:
        for path in paths[1:]:
            geometries = geometries.append(gpd.read_file(path, bbox=bounds))
    return geometries


# Read buildings
buildings = read_geometries('buildings')

# Read green areas
green_areas = read_geometries('green_areas')


with MemoryFile() as dem:
    with dem.open(driver='GTiff', transform=transform, width=array.shape[1], height=array.shape[2], count=1,
                  dtype=rio.float32, nodata=nodata) as dataset:
        bounds = dataset.bounds
        dataset.write(array)

    # Create input files
    Model(
        dem=dem,
        rainfall=rainfall,
        duration=3600*duration+3600*post_event_duration,
        output_interval=600,
        open_external_boundaries=open_boundaries,
        buildings=buildings,
        green_areas=green_areas,
        use_infiltration=True,
        permeable_areas={'polygons': 0, 'impermeable': 1, 'permeable': 2}[permeable_areas],
        roof_storage=roof_storage
    ).write(run_path)

# Copy executable
shutil.copy('citycat.exe', run_path)

start_timestamp = pd.Timestamp.now()

# Run executable
if os.name == 'nt':
   subprocess.call('cd {run_path} & citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)
else:
   subprocess.call('cd {run_path} && wine64 citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)

end_timestamp = pd.Timestamp.now()

# Delete executable
os.remove(os.path.join(run_path, 'citycat.exe'))

# Archive results files
surface_maps = os.path.join(run_path, 'R1C1_SurfaceMaps')
shutil.make_archive(surface_maps, 'zip', surface_maps)

# Create geotiff
geotiff_path = os.path.join(run_path, 'max_depth.tif')
netcdf_path = os.path.join(run_path, 'R1C1_SurfaceMaps.nc')

output.to_geotiff(os.path.join(surface_maps, 'R1_C1_max_depth.csv'), geotiff_path, srid=27700)

output.to_netcdf(surface_maps, out_path=netcdf_path, srid=27700,
                 attributes=dict(
                    rainfall_mode=rainfall_mode,
                    rainfall_total=rainfall_total,
                    size=size,
                    duration=duration,
                    post_event_duration=post_event_duration,
                    return_period=return_period,
                    x=x,
                    y=y,
                    pooling_radius=pooling_radius,
                    open_boundaries=str(open_boundaries),
                    permeable_areas=permeable_areas))

a = xr.open_dataset(netcdf_path)

velocity = xr.ufuncs.sqrt(a.x_vel**2+a.y_vel**2).astype(np.float64)
max_velocity = velocity.max(dim='time').round(3)
max_velocity = max_velocity.where(xr.ufuncs.isfinite(max_velocity), other=output.fill_value)
max_velocity.rio.set_crs('EPSG:27700')
max_velocity.rio.set_nodata(output.fill_value)
max_velocity.rio.to_raster(os.path.join(run_path, 'max_velocity.tif'))

vd_product = velocity * a.depth
max_vd_product = vd_product.max(dim='time').round(3)
max_vd_product = max_vd_product.where(xr.ufuncs.isfinite(max_vd_product), other=output.fill_value)
max_vd_product.rio.set_crs('EPSG:27700')
max_vd_product.rio.set_nodata(output.fill_value)
max_vd_product.rio.to_raster(os.path.join(run_path, 'max_vd_product.tif'))

# Create depth map
with rio.open(geotiff_path) as ds:
    f, ax = plt.subplots()

    cmap = ListedColormap(['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c',
                           '#08306b', 'black'])
    cmap.set_bad(color='lightgrey')
    cmap.colorbar_extend = 'max'

    im = show(ds, ax=ax, cmap=cmap, vmin=0, vmax=1).get_images()[0]

    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_artist(ScaleBar(1, frameon=False))
    f.colorbar(im, label='Water Depth (m)')
    f.savefig(os.path.join(run_path, 'max_depth.png'), dpi=200, bbox_inches='tight')

    # Create interpolated GeoTIFF
    with rio.open(os.path.join(run_path, 'max_depth_interpolated.tif'), 'w', **ds.profile) as dst:
        dst.write(fillnodata(ds.read(1), mask=ds.read_masks(1)), 1)

description = f'A {size/1000}x{size/1000}km domain centred at {x},{y} was simulated for ' \
              f'{duration+post_event_duration}hrs, which took ' \
              f'{round((end_timestamp-start_timestamp).total_seconds()/3600, 1)}hrs to complete. '

if rainfall_mode == 'return_period':
    description += f'The {return_period}yr {duration}hr event was generated using a radius of {pooling_radius/1000}m. '

description += f'Total depth of rainfall was {int(round(rainfall_total, 0))}mm. '
if post_event_duration > 0:
    description += f'Following the {duration}hr event, the simulation continued for {post_event_duration}hrs. '

if buildings is not None and len(buildings) > 0:
    description += f'{len(buildings)} buildings were extracted from the domain. '

if green_areas is not None and len(green_areas) > 0:
    description += f'{len(green_areas)} green areas where infiltration can take place were defined. '

description += f'The boundaries of the domain were set to {"open" if open_boundaries else "closed"}.'

if roof_storage > 0:
    description += ' There was {}m of roof storage.'

geojson = json.dumps({
    'type': 'Feature',
    'properties': {},
    'geometry': gpd.GeoSeries(box(*bounds), crs='EPSG:27700').to_crs(epsg=4326).iloc[0].__geo_interface__})

# Create metadata file
metadata = f"""{{
  "@context": ["metadata-v1"],
  "@type": "dcat:Dataset",
  "dct:language": "en",
  "dct:title": "CityCAT Output ({pd.Timestamp.now().round('s').strftime('%H:%M %d/%m/%y')})",
  "dct:description": "{description}",
  "dcat:keyword": [
    "citycat"
  ],
  "dct:subject": "Environment",
  "dct:license": {{
    "@type": "LicenseDocument",
    "@id": "https://creativecommons.org/licences/by/4.0/",
    "rdfs:label": null
  }},
  "dct:creator": [{{"@type": "foaf:Organization"}}],
  "dcat:contactPoint": {{
    "@type": "vcard:Organization",
    "vcard:fn": "DAFNI",
    "vcard:hasEmail": "support@dafni.ac.uk"
  }},
  "dct:created": "{datetime.now().isoformat()}Z",
  "dct:PeriodOfTime": {{
    "type": "dct:PeriodOfTime",
    "time:hasBeginning": null,
    "time:hasEnd": null
  }},
  "dafni_version_note": "created",
  "dct:spatial": {{
    "@type": "dct:Location",
    "rdfs:label": null
  }},
  "geojson": {geojson}
}}
"""
with open(os.path.join(outputs_path, 'metadata.json'), 'w') as f:
    f.write(metadata)
