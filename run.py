import os
import shutil  # must be imported before GDAL
from rasterio.merge import merge
import rasterio as rio
from rasterio.io import MemoryFile
from citycatio import Model
import pandas as pd
import subprocess
import json
import xarray as xr
from glob import glob
from shapely.geometry import Point
import rioxarray as rx

data_path = os.getenv('DATA_PATH', '/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')

# Read environment variables
size = float(os.getenv('SIZE')) * 1000  # convert from km to m
duration = int(os.getenv('DURATION'))
return_period = int(os.getenv('RETURN_PERIOD'))
x = int(os.getenv('X'))
y = int(os.getenv('Y'))
pooling_radius = int(os.getenv('POOLING_RADIUS')) * 1000  # convert from km to m

# Get rainfall total
rainfall = xr.open_dataset(glob(os.path.join(inputs_path, 'ukcp/pr*'))[0]).pr.rio.set_crs('EPSG:27700')

df = rainfall.rio.clip([Point(x, y).buffer(pooling_radius)]).to_dataframe().pr.dropna().reset_index()

rolling = df.set_index('time').groupby(
   ['projection_x_coordinate', 'projection_y_coordinate']).pr.rolling(duration).sum().reset_index()

years = [t.year for t in rolling.time.values]

amax = rolling.groupby(['projection_x_coordinate', 'projection_y_coordinate', years]).pr.max().reset_index(drop=True)

times_exceeded = int(round(len(amax) / return_period, 0))

rainfall_total = amax.sort_values(ascending=False).iloc[times_exceeded]
print(f'Rainfall Total:{rainfall_total}')

# Create run directory
run_path = os.path.join(outputs_path, 'run')
if not os.path.exists(run_path):
    os.mkdir(run_path)

# Read and clip DEM
dem_path = os.path.join(inputs_path, 'dem')
dem_datasets = [rio.open(os.path.join(dem_path, p)) for p in os.listdir(dem_path) if p.endswith('.asc')]
xmin, ymin, xmax, ymax = x-size/2, y-size/2, x+size/2, y+size/2
array, transform = merge(dem_datasets, bounds=(xmin, ymin, xmax, ymax))
with MemoryFile() as dem:
    with dem.open(driver='GTiff', transform=transform, width=array.shape[1], height=array.shape[2], count=1,
                  dtype=rio.float32) as dataset:
        dataset.write(array)

    # Create input files
    Model(dem=dem, rainfall=pd.DataFrame([rainfall_total / (3600*duration) / 1000] * 2), duration=3600*duration,
          output_interval=600).write(run_path)

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

# Create metadata file
metadata = {
   "@context": [
      "metadata-v1"
   ],
   "@type": "dcat:Dataset",
   "dct:title": f"CityCAT Output {return_period}yrs {duration}hr {size}km domain",
   "dct:description": f"CityCAT Output {return_period}yrs {duration}hr {size}km domain",
   "dct:identifier": [
   ],
   "dct:subject": "Environment",
   "dcat:theme": [

   ],
   "dct:language": "en",
   "dcat:keyword": [
      "CityCAT", "Flooding", "Newcastle"
   ],
   "dct:conformsTo": {
      "@id": None,
      "@type": "dct:Standard",
      "label": None
   },
   "dct:spatial": {
      "@id": 2641673,
      "@type": "dct:Location",
      "rdfs:label": "Newcastle upon Tyne, United Kingdom"
   },
   "geojson": {
      "type": "Feature",
      "properties": {

      },
      "geometry": {
         "type": "Polygon",
         "coordinates": [
            [
               [
                  -1.534381542200191,
                  55.03117248153787
               ],
               [
                  -1.7312260677551188,
                  55.03117248153787
               ],
               [
                  -1.7312260677551188,
                  54.960169178330425
               ],
               [
                  -1.534381542200191,
                  54.960169178330425
               ],
               [
                  -1.534381542200191,
                  55.03117248153787
               ]
            ]
         ]
      }
   },
   "dct:PeriodOfTime": {
      "type": "dct:PeriodOfTime",
      "time:hasBeginning": None,
      "time:hasEnd": None
   },
   "dct:accrualPeriodicity": None,
   "dct:creator": [],
   "dct:publisher": {
      "@id": None,
      "@type": "foaf:Organization",
      "foaf:name": None,
      "internalID": None
   },
   "dcat:contactPoint": {
      "@type": "vcard:Organization",
      "vcard:fn": "Fergus McClean",
      "vcard:hasEmail": "fergus.mcclean@newcastle.ac.uk"
   },
   "dct:license": {
      "@type": "LicenseDocument",
      "@id": "https://creativecommons.org/licences/by/4.0/",
      "rdfs:label": None
   },
   "dct:rights": None,
   "dafni_version_note": "created output",
}

with open(os.path.join(outputs_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
