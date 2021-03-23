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
    rainfall_total = fitted_gev.ppf(return_period / len(amax))

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

# Read buildings
building_paths = glob(os.path.join(inputs_path, 'buildings/*.gpkg'))
building_paths.extend(glob(os.path.join(inputs_path, 'buildings/*.shp')))
print(f'Files in buildings directory: {[os.path.basename(p) for p in building_paths]}')
buildings = gpd.read_file(building_paths[0], bbox=bounds) if len(building_paths) > 0 else None
if len(building_paths) > 1:
    for building_path in building_paths[1:]:
        buildings = buildings.append(gpd.read_file(building_path, bbox=bounds))


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
        buildings=buildings
    ).write(run_path)

# Copy executable
shutil.copy('citycat.exe', run_path)

# Run executable
if os.name == 'nt':
   subprocess.call('cd {run_path} & citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)
else:
   subprocess.call('cd {run_path} && wine64 citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)

# Delete executable
os.remove(os.path.join(run_path, 'citycat.exe'))

# Archive results files
surface_maps = os.path.join(run_path, 'R1C1_SurfaceMaps')
shutil.make_archive(surface_maps, 'zip', surface_maps)

# Create geotiff
geotiff_path = os.path.join(run_path, 'max_depth.tif')
output.to_geotiff(os.path.join(surface_maps, 'R1_C1_max_depth.csv'), geotiff_path,
                  crs=CRS.from_epsg(27700))

# Create depth map
with rio.open(geotiff_path) as ds:
    f, ax = plt.subplots()
    im = show(ds, ax=ax, cmap='Blues').get_images()[0]

    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_artist(ScaleBar(ds.transform[0]))
    f.colorbar(im, label='Water Depth (m)')
    f.savefig(os.path.join(run_path, 'max_depth.png'), dpi=200)

    # Create interpolated GeoTIFF
    with rio.open(os.path.join(run_path, 'max_depth_interpolated.tif'), 'w', **ds.profile) as dst:
        dst.write(fillnodata(ds.read(1), mask=ds.read_masks(1)), 1)

description = ''
for variable in ['rainfall_mode', 'rainfall_total', 'size', 'duration', 'post_event_duration', 'return_period', 'x',
                 'y', 'pooling_radius', 'open_boundaries']:
    description += f'{variable}={globals()[variable]}, '


# Create metadata file
metadata = f"""{{
  "@context": ["metadata-v1"],
  "@type": "dcat:Dataset",
  "dct:language": "en",
  "dct:title": "CityCAT Output",
  "dct:description": "{description[:-2]}",
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
  "geojson": {{}}
}}
"""
with open(os.path.join(outputs_path, 'metadata.json'), 'w') as f:
    f.write(metadata)
