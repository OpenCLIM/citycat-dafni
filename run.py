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
import geopandas as gpd
import rioxarray as rx
from rasterio.plot import show
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.fill import fillnodata
from datetime import datetime
import numpy as np
from shapely.geometry import box
import json
from matplotlib.colors import ListedColormap
from zipfile import ZipFile
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap

import random
import string
import logging
from pathlib import Path
from os.path import isfile, join, isdir

# Set up paths
data_path = os.getenv('DATA_PATH', '/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)
    
parameters_path = os.path.join(inputs_path, 'parameters')
print('parameters_path:',parameters_path)
udm_para_in_path = os.path.join(inputs_path, 'udm_parameters')

# Set up log file
logger = logging.getLogger('citycat-dafni')
logger.setLevel(logging.INFO)
log_file_name = 'citycat-dafni-%s.log' %(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))
fh = logging.FileHandler( Path(join(data_path, outputs_path)) / log_file_name)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info('Log file established!')
logger.info('--------')

logger.info('Paths have been setup')    
    
# If the UDM model preceeds the CityCat model in the workflow, a zip file should appear in the inputs folder
# Check if the zip file exists
archive = glob(inputs_path + "/**/*.zip", recursive = True)
logger.info(archive)

# Look to see if a parameter file has been added
parameter_file = glob(parameters_path + "/*.csv", recursive = True)
print('parameter_file:', parameter_file)

# If a parameter.csv is available: read the variables from the document
if len(parameter_file) == 1 :
    file_path = os.path.splitext(parameter_file[0])
    print('Filepath:',file_path)
    filename=file_path[0].split("/")
    print('Filename:',filename[-1])

    parameters = pd.read_csv(os.path.join(parameters_path + '/' + filename[-1] + '.csv'))

    name = filename[-1]
    name = name.replace('-parameters','')
    rainfall_mode = parameters.loc[3][1]
    rainfall_total = int(parameters.loc[4][1])
    duration = int(parameters.loc[5][1])
    open_boundaries = parameters.loc[6][1]
    if open_boundaries.lower()=='true':
        open_boundaries=True
    else :
        open_boundaries=False
    permeable_areas = parameters.loc[7][1]
    roof_storage = float(parameters.loc[8][1])
    post_event_duration = int(parameters.loc[9][1])
    output_interval = int(parameters.loc[10][1])
    size = parameters.loc[11][1]
    if size != None:
        size = float(size)
    x = parameters.loc[12][1]
    y = parameters.loc[13][1]
    if x != None:
         x = int(x)
    if y != None:
        y = int(y)

    # discharge_parameter = float(parameters.loc[15][1])
    #time_horizon = parameters.loc[X][1]
    #return_period = int(parameters.loc[X][1])

# If no parameter file is available, the user needs to define the parameters
if len(parameter_file) == 0 :
    name = os.getenv('NAME')
    rainfall_mode = os.getenv('RAINFALL_MODE')
    rainfall_total = int(os.getenv('TOTAL_DEPTH'))
    duration = int(os.getenv('DURATION'))
    open_boundaries = (os.getenv('OPEN_BOUNDARIES').lower() == 'true')
    permeable_areas = os.getenv('PERMEABLE_AREAS')
    roof_storage = float(os.getenv('ROOF_STORAGE'))
    post_event_duration = int(os.getenv('POST_EVENT_DURATION'))
    output_interval = int(os.getenv('OUTPUT_INTERVAL'))
    size = os.getenv('SIZE') 
    x = os.getenv('X')
    y = os.getenv('Y')
    if size != None:
        size = float(size)*1000
    if x != None:
        x = int(x)
    if y != None:
        y = int(y)
    
    #time_horizon = os.getenv('TIME_HORIZON')
    #return_period = int(os.getenv('RETURN_PERIOD'))
    #discharge_parameter = float(os.getenv('DISCHARGE'))

nodata = -9999


def read_geometries(path, bbox=None):
    logger.info('---- In read geometries function')
    paths = glob(os.path.join(inputs_path, path, '*.gpkg'))
    paths.extend(glob(os.path.join(inputs_path, path, '*.shp')))
    print(f'Files in {path} directory: {[os.path.basename(p) for p in paths]}')
    logger.info(f'---- Files in {path} directory to read in: {[os.path.basename(p) for p in paths]}')
    
    # set a default value
    geometries = None
    
    if len(paths) > 0:
        logger.info('-------- Reading in %s' %path)
        geometries = gpd.read_file(paths[0], bbox=bbox)
        logger.info('-------- Number of features read now: %s' %geometries.shape[0])
              
    if len(paths) > 1:
        for path in paths[1:]:
            logger.info('-------- Reading in %s' %path)
            geometries = geometries.append(gpd.read_file(path, bbox=bounds))
            logger.info('-------- Number of features read now: %s' %geometries.shape[0])
            
    logger.info('---- Completed read geometries funtion')
    return geometries

logger.info('--------')
logger.info('Starting to run code')

logger.info('Setting boundary')
boundary = read_geometries('boundary')

if boundary is None:
    bounds = x-size/2, y-size/2, x+size/2, y+size/2
else:
    bounds = boundary.geometry.total_bounds.tolist()

logger.info('Checking if rainfall period being used')
if rainfall_mode == 'return_period':
    uplifts = pd.read_csv(
        os.path.join(inputs_path,
                     'future-drainage',
                     f'Uplift_{time_horizon if time_horizon != "baseline" else "2050"}_{duration}hr_Pr_{return_period}'
                     f'yrRL_Grid.csv'),
        header=1)

    uplifts = gpd.GeoDataFrame(uplifts,
                               geometry=gpd.points_from_xy(uplifts.easting, uplifts.northing).buffer(2500, cap_style=3))
    if boundary is not None:
        row = uplifts[uplifts.intersects(boundary.geometry.unary_union)].mean()
    else:
        row = uplifts[uplifts.intersects(box(*bounds))].mean()

    rainfall_total = row[f'ReturnLevel.{return_period}']

    if time_horizon != 'baseline':

        rainfall_total *= float(((100 + row['Uplift_50']) / 100))

logging.info(f'Rainfall Total: {rainfall_total}')
print(f'Rainfall Total: {rainfall_total}')

unit_profile = np.array([0.017627993, 0.027784045, 0.041248418, 0.064500665, 0.100127555, 0.145482534, 0.20645758,
                         0.145482534, 0.100127555, 0.064500665, 0.041248418, 0.027784045, 0.017627993])

# Fit storm profile
logger.info('Fitting rainfall to sotrm profile')
rainfall_times = np.linspace(start=0, stop=duration*3600, num=len(unit_profile))

unit_total = sum((unit_profile + np.append(unit_profile[1:], [0])) / 2 *
                 (np.append(rainfall_times[1:], rainfall_times[[-1]]+1)-rainfall_times))

rainfall = pd.DataFrame(list(unit_profile*rainfall_total/unit_total/1000) + [0, 0],
                        index=list(rainfall_times) + [duration*3600+1, duration*3600+2])

# Create run directory
logging.info('Creating run directory')
run_path = os.path.join(outputs_path, 'run')
if not os.path.exists(run_path):
    os.mkdir(run_path)

# Read and clip DEM
logger.info('Reading and clipping DEM')
dem_path = os.path.join(inputs_path, 'dem')
dem_datasets = [rio.open(os.path.join(dem_path, os.path.abspath(p))) for p in glob(os.path.join(dem_path, '*.asc'))]

array, transform = merge(dem_datasets, bounds=bounds, precision=50, nodata=nodata)
assert array[array != nodata].size > 0, "No DEM data available for selected location"

# Read buildings
logger.info('Reading buildings')
buildings = read_geometries('buildings', bbox=bounds)

# Read green areas
logger.info('Reading green areas')
green_areas = read_geometries('green_areas', bbox=bounds)

total_duration = 3600*duration+3600*post_event_duration

# Create discharge timeseries
logger.info('Creating discharge timeseries')
if discharge_parameter > 0:
    discharge = pd.Series([discharge_parameter, discharge_parameter], index=[0, total_duration])

    # Divide by the length of each cell
    discharge = discharge.divide(5)

    flow_polygons = gpd.read_file(glob(os.path.join(inputs_path, 'flow_polygons', '*'))[0]).geometry
else:
    discharge = None
    flow_polygons = None

logger.info('Creating DEM dataset and boundary dataset')
dem = MemoryFile()
with dem.open(driver='GTiff', transform=transform, width=array.shape[2], height=array.shape[1], count=1,
              dtype=rio.float32, nodata=nodata) as dataset:
    bounds = dataset.bounds
    dataset.write(array)

# if boundary is not None:
#     array, transform = mask(dem.open(), boundary.geometry, crop=True)
#     dem = MemoryFile()
#     with dem.open(driver='GTiff', transform=transform, width=array.shape[2], height=array.shape[1], count=1,
#                   dtype=rio.float32, nodata=nodata) as dataset:
#         bounds = dataset.bounds
#         dataset.write(array)

# Create input files
logger.info('Creating input files')
Model(
    dem=dem,
    rainfall=rainfall,
    duration=total_duration,
    output_interval=output_interval,
    open_external_boundaries=open_boundaries,
    buildings=buildings,
    green_areas=green_areas,
    use_infiltration=True,
    permeable_areas={'polygons': 0, 'impermeable': 1, 'permeable': 2}[permeable_areas],
    roof_storage=roof_storage,
    flow=discharge,
    flow_polygons=flow_polygons

).write(run_path)

# Copy executable
logger.info('Preparing CityCat')
shutil.copy('citycat.exe', run_path)

start_timestamp = pd.Timestamp.now()

# Run executable
logger.info('Running CityCat......')
if os.name == 'nt':
    subprocess.call('cd {run_path} & citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)
else:
    subprocess.call('cd {run_path} && wine64 citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)

end_timestamp = pd.Timestamp.now()

logger.info('....CityCat completed!')

# Delete executable
logger.info('Deleting CityCAT model')
os.remove(os.path.join(run_path, 'citycat.exe'))

# Archive results files
logger.info('Archiving results')
surface_maps = os.path.join(run_path, 'R1C1_SurfaceMaps')
shutil.make_archive(surface_maps, 'zip', surface_maps)

# Create geotiff
logger.info('Creating outputs')
geotiff_path = os.path.join(run_path, 'max_depth.tif')
netcdf_path = os.path.join(run_path, 'R1C1_SurfaceMaps.nc')

output.to_geotiff(os.path.join(surface_maps, 'R1_C1_max_depth.csv'), geotiff_path, srid=27700)

output.to_netcdf(surface_maps, out_path=netcdf_path, srid=27700,
                 attributes=dict(
                    rainfall_mode=rainfall_mode,
                    rainfall_total=float(rainfall_total),
                    size=size,
                    duration=duration,
                    post_event_duration=post_event_duration,
                    return_period=return_period,
                    x=x,
                    y=y,
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

# # Create depth map
# with rio.open(geotiff_path) as ds:
#     f, ax = plt.subplots()

#     cmap = ListedColormap(['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c',
#                            '#08306b', 'black'])
#     cmap.set_bad(color='lightgrey')
#     cmap.colorbar_extend = 'max'

#     im = show(ds, ax=ax, cmap=cmap, vmin=0, vmax=1).get_images()[0]

#     ax.set_xticks([])
#     ax.set_yticks([])

#     ax.add_artist(ScaleBar(1, frameon=False))
#     f.colorbar(im, label='Water Depth (m)')
#     f.savefig(os.path.join(run_path, 'max_depth.png'), dpi=200, bbox_inches='tight')

# Create a depth map, with the boundary and max water levels

dpi = 300

#Plotting the Raster and the ShapeFile together
fig, ax = plt.subplots(1, 1, dpi = dpi)
cmap = mpl.cm.Blues

plt.subplots_adjust(left = 0.10 , bottom = 0, right = 0.90 , top =1)

#Bounds for the raster
bounds_depth =  [0.01, 0.05, 0.10, 0.15, 0.30, 0.50, 0.80, 1.00] #you could change here the water depth of your results
norm = mpl.colors.BoundaryNorm(bounds_depth, cmap.N)

axins = inset_axes(ax,
                   width="2%", # width of colorbar in % of plot width
                   height="45%", # height of colorbar in % of plot height
                   loc=2, #topright location
                   bbox_to_anchor=(1.01, 0, 1, 1), #first number: space relative to plot (1.0 = no space between cb and plot)
                   bbox_transform=ax.transAxes,
                   borderpad=0) 

if len(boundary) != 0:
    boundary.boundary.plot(edgecolor = 'black', lw = 0.5, ax = ax) #lw = 0.05 -> entire area #0.2 #0.80 for zoom

citycat_outputs = rio.open(geotiff_path, mode ='r')
#The line below correspond to the raster
show(citycat_outputs, ax = ax, title = 'max_water_depth', cmap = 'Blues', norm = norm)

#Plotting the colorbar for the raster file Water Depth:
plt.colorbar(mpl.cm.ScalarMappable(cmap = cmap, norm = norm),
             ax = ax,
             cax = axins,
             extend = 'both',
             format='%.2f',
             ticks = bounds_depth,
             spacing = 'uniform',
             orientation = 'vertical',
             label = 'Water Depth in m')

plt.savefig(os.path.join(run_path, 'max_depth.png'), dpi=dpi, bbox_inches='tight')

    
# Create interpolated GeoTIFF
with rio.open(geotiff_path) as ds:
    with rio.open(os.path.join(run_path, 'max_depth_interpolated.tif'), 'w', **ds.profile) as dst:
        dst.write(fillnodata(ds.read(1), mask=ds.read_masks(1)), 1)

title = f'{name} {x},{y} {size/1000}km {duration}hr'
description = f'A {size/1000}x{size/1000}km domain centred at {x},{y} was simulated for ' \
              f'{duration+post_event_duration}hrs, which took ' \
              f'{round((end_timestamp-start_timestamp).total_seconds()/3600, 1)}hrs to complete. '

if rainfall_mode == 'return_period':
    description += f'The {return_period}yr {duration}hr event was extracted from the UKCP18 baseline (1980-2000)'
    if time_horizon != 'baseline':
        description += f' and uplifted by {row["Uplift_50"]}%'
    description += '. '

    title += f' {time_horizon} {return_period}yr'

description += f'Total depth of rainfall was {int(round(rainfall_total, 0))}mm. '
title += f' {int(round(rainfall_total, 0))}mm'
if post_event_duration > 0:
    description += f'Following the {duration}hr event, the simulation continued for {post_event_duration}hrs. '

if buildings is not None and len(buildings) > 0:
    description += f'{len(buildings)} buildings were extracted from the domain. '

if green_areas is not None and len(green_areas) > 0:
    description += f'{len(green_areas)} green areas where infiltration can take place were defined. '

description += f'The boundaries of the domain were set to {"open" if open_boundaries else "closed"}.'

if roof_storage > 0:
    description += f' There was {roof_storage}m of roof storage.'
    title += f' storage={roof_storage}m'

if discharge is not None:
    description += f' A flow of {discharge_parameter} cumecs was used as a boundary condition.'
    title += f' {discharge_parameter}m3/s'

udm_para_out_path = os.path.join(outputs_path, 'udm_parameters')
if not os.path.exists(udm_para_out_path):
    os.mkdir(udm_para_out_path)

meta_data_txt = glob(udm_para_in_path + "/**/metadata.txt", recursive = True)
meta_data_csv = glob(udm_para_in_path + "/**/metadata.csv", recursive = True)
attractors = glob(udm_para_in_path + "/**/attractors.csv", recursive = True)
constraints = glob(udm_para_in_path + "/**/constraints.csv", recursive = True)

if len(meta_data_txt)==1:
    src = meta_data_txt[0]
    dst = os.path.join(udm_para_out_path,'metadata.txt')
    shutil.copy(src,dst)

if len(meta_data_csv)==1:
    src = meta_data_csv[0]
    dst = os.path.join(udm_para_out_path,'metadata.csv')
    shutil.copy(src,dst)

if len(attractors)==1:
    src = attractors[0]
    dst = os.path.join(udm_para_out_path,'attractors.csv')
    shutil.copy(src,dst)

if len(constraints)==1:
    src = constraints[0]
    dst = os.path.join(udm_para_out_path,'constraints.csv')
    shutil.copy(src,dst)

geojson = json.dumps({
    'type': 'Feature',
    'properties': {},
    'geometry': gpd.GeoSeries(box(*bounds), crs='EPSG:27700').to_crs(epsg=4326).iloc[0].__geo_interface__})
print(title)

# Create metadata file
logger.info('Building metadata file for DAFNI')
metadata = f"""{{
  "@context": ["metadata-v1"],
  "@type": "dcat:Dataset",
  "dct:language": "en",
  "dct:title": "{title}",
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
with open(os.path.join(run_path, 'metadata.json'), 'w') as f:
    f.write(metadata)
