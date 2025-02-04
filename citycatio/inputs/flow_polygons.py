import geopandas as gpd
from ..utils import geoseries_to_string
import os


class FlowPolygons:
    """Areas corresponding to flow series

    Args:
        data: Table containing flow polygons
    """
    def __init__(self, data: gpd.GeoDataFrame):
        assert type(data) == gpd.GeoDataFrame
        self.data = data

    def write(self, path):
        with open(os.path.join(path, 'BCs_flow.txt'), 'w') as f:
            f.write(geoseries_to_string(self.data.geometry))
