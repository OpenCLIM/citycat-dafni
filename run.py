import os
import shutil  # must be imported before GDAL
from rasterio.merge import merge
import rasterio as rio
from rasterio.io import MemoryFile
from citycatio import Model
import pandas as pd
import subprocess

rainfall_total = float(os.getenv('RAIN', 40))

dem_path = os.getenv('DEM', '/data/inputs/dem')

size = int(os.getenv('SIZE', '20'))

dem_datasets = [rio.open(os.path.join(dem_path, p)) for p in os.listdir(dem_path) if p.endswith('.asc')]

xmin, ymin = 420000, 560000

array, transform = merge(dem_datasets, bounds=(xmin, ymin, xmin+size, ymin+size))

run_path = '/data/outputs/run'

if not os.path.exists(run_path):
    os.mkdir(run_path)

dem = MemoryFile()

with dem.open(driver='GTiff', transform=transform, width=array.shape[1], height=array.shape[2],
              count=1, dtype=rio.float32) as dataset:
    dataset.write(array)

dem.seek(0)

model = Model(dem=dem, rainfall=pd.DataFrame([rainfall_total / 3600 / 1000] * 2), duration=3600, output_interval=600)

model.write(run_path)

shutil.copy('citycat.exe', run_path)

subprocess.call('cd {run_path} && wine64 citycat.exe -r 1 -c 1'.format(run_path=run_path), shell=True)

os.remove(os.path.join(run_path, 'citycat.exe'))

surface_maps = os.path.join(run_path, 'R1C1_SurfaceMaps')

shutil.make_archive(surface_maps, 'zip', surface_maps)
