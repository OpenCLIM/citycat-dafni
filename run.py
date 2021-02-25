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
from shapely.geometry import box
import geopandas as gpd
import rioxarray as rx
from rasterio.crs import CRS
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.fill import fillnodata

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
rainfall = pd.DataFrame(([rainfall_total / (3600*duration) / 1000] * 2) + [0, 0],
                        index=[0, duration*3600, duration*3600+1, duration*3600+2])

# Create run directory
run_path = os.path.join(outputs_path, 'run')
if not os.path.exists(run_path):
    os.mkdir(run_path)

# Read and clip DEM
dem_path = os.path.join(inputs_path, 'dem')
dem_datasets = [rio.open(os.path.join(dem_path, p)) for p in os.listdir(dem_path) if p.endswith('.asc')]
bounds = x-size/2, y-size/2, x+size/2, y+size/2
array, transform = merge(dem_datasets, bounds=bounds)

# Read buildings
building_paths = glob(os.path.join(inputs_path, 'buildings/*.gpkg'))
building_paths.extend(glob(os.path.join(inputs_path, 'buildings/*.shp')))
print(f'Files in buildings directory: {[os.path.basename(p) for p in building_paths]}')
buildings = gpd.read_file(building_paths[0], bbox=bounds) if len(building_paths) > 0 else None
if len(building_paths) > 1:
    for building_path in building_paths[1:]:
        buildings.append(gpd.read_file(building_path, bbox=bounds))


with MemoryFile() as dem:
    with dem.open(driver='GTiff', transform=transform, width=array.shape[1], height=array.shape[2], count=1,
                  dtype=rio.float32) as dataset:
        bounds = dataset.bounds
        dataset.write(array)

    # Create input files
    Model(dem=dem, rainfall=rainfall, duration=3600*duration+3600*post_event_duration,
          output_interval=600, open_boundaries=gpd.GeoDataFrame(geometry=[box(*bounds).buffer(100)]),
          buildings=buildings).write(run_path)

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

    with rio.open(os.path.join(run_path, 'max_depth_interpolated.tif'), 'w', **ds.profile) as dst:
        dst.write(fillnodata(ds.read(1), mask=ds.read_masks(1)), 1)
