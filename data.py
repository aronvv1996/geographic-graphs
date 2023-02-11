import csv
import geopandas as gpd
import json
import networkx as nx


class DataLoader():

    def __init__(self, data_folder='data'):
        '''
        The 'DataLoader' class contains methods aimed at loading various kinds
        of databases (road network, worldwide cities coordinates, country
        network).
        '''
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        self.data_folder = data_folder

        self.roads_file = 'groads-v1-global-gdb\\gROADS_v1.gdb'
        self.cities_file = 'simplemaps_worldcities_basicv1.75\\worldcities.csv'
        self.countryborders_file = 'country_borders\\country_adj_full.json'

    def load_roads(self, continent=None, country=None):
        '''
        Loads the worldwide road GeoDataBase network. It is possible to load 
        the network of a single country or continent of choice.
        '''
        gdb_roads = f'{self.data_folder}\\{self.roads_file}'

        if (country is not None):
            return gpd.read_file(gdb_roads, mask=self.world[self.world.name == country])

        if (continent is not None):
            return gpd.read_file(gdb_roads, mask=self.world[self.world.continent == continent])

        return gpd.read_file(gdb_roads)
    
    def load_cities(self, continent=None, country=None):
        '''
        Loads the cities CSV database and converts it to a GeoPandas
        GeoDataFrame. It is possible to load the network of a single country or
        continent of choice.
        '''
        csv_cities = open(f'{self.data_folder}\\{self.cities_file}', encoding='utf8')
        DR = list(csv.DictReader(csv_cities))
        lat = [float(x['lat']) for x in DR]
        lng = [float(x['lng']) for x in DR]
        gdf_cities = gpd.GeoDataFrame(DR, geometry=gpd.points_from_xy(lng, lat), crs='EPSG:4326')

        if (country is not None):
            return gpd.tools.sjoin(gdf_cities, self.world[self.world.name == country])

        if (continent is not None):
            return gpd.tools.sjoin(gdf_cities, self.world[self.world.continent == continent])

        return gdf_cities
    
    def load_countryborders(self):
        '''
        Loads the bordering countries JSON database and converts it into a
        Python dictionary.
        '''
        json_file = open(f'{self.data_folder}\\{self.countryborders_file}')
        return json.load(json_file)