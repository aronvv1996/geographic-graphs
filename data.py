import csv
import geopandas as gpd


class DataLoader():

    def __init__(self, data_folder='data'):
        '''
        The 'DataLoader' class contains methods for loading various kinds of
        databases (road network, city network, country network, ...).
        '''
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        self.data_folder = data_folder

        self.roads_file = 'groads-v1-global-gdb\\gROADS_v1.gdb'
        self.cities_file = 'simplemaps_worldcities_basicv1.75\\worldcities.csv'
        self.borders_file = 'country-borders\\GEODATASOURCE-COUNTRY-BORDERS.CSV'
        self.centroids_file = 'world-countries-centroids\\countries.csv'

    def load_roads(self, continent=None, country=None):
        '''
        Loads the worldwide GeoDataBase road network. It is possible to load 
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
        lon = [float(x['lng']) for x in DR]
        gdf_cities = gpd.GeoDataFrame(DR, geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')

        if (country is not None):
            return gpd.tools.sjoin(gdf_cities, self.world[self.world.name == country])

        if (continent is not None):
            return gpd.tools.sjoin(gdf_cities, self.world[self.world.continent == continent])

        return gdf_cities
    
    def load_countryborders(self):
        '''
        Loads the bordering countries CSV database and converts it into a
        Python dictionary.
        '''
        csv_borders = open(f'{self.data_folder}\\{self.borders_file}')
        DR = list(csv.DictReader(csv_borders))
        countries = set([x['country_code'] for x in DR])
        borders = [(x['country_code'],x['country_border_code']) for x in DR if x['country_border_code'] != '']

        dictionary = {}
        for country in countries:
            dictionary[country] = [y for (x,y) in borders if x == country]

        dictionary['JP'] = ['RU']
        dictionary['RU'] += ['JP']

        dictionary['GB'] = ['IE', 'FR']
        dictionary['IE'] = ['GB']
        dictionary['FR'] += ['GB']

        dictionary['GL'] = ['CA']
        dictionary['CA'] += ['GL']

        dictionary['AU'] = ['PG']
        dictionary['PG'] += ['AU']

        dictionary['LK'] = ['IN']
        dictionary['IN'] += ['LK']

        dictionary['TW'] = ['CN']
        dictionary['CN'] += ['TW']

        dictionary['PH'] = ['MY']
        dictionary['MY'] += ['PH']

        dictionary['CY'] = ['TR']
        dictionary['TR'] += ['CY']

        dictionary['XK'] = ['RS', 'MK', 'AL', 'ME']
        dictionary['RS'] += ['XK']
        dictionary['RS'].remove('AL')
        dictionary['MK'] += ['XK']
        dictionary['AL'] += ['XK']
        dictionary['AL'].remove('RS')
        dictionary['ME'] += ['XK']
        
        return dictionary

    def load_countrycentroids(self):
        '''
        Loads the country centroids CSV database and converts it into a Python
        dictionary.
        '''
        csv_centroids = open(f'{self.data_folder}\\{self.centroids_file}')
        DR = list(csv.DictReader(csv_centroids))
        countries = [x['ISO'] for x in DR]
        centroids_lat = [float(x['latitude']) for x in DR]
        centroids_lon = [float(x['longitude']) for x in DR]
        centroids = list(zip(centroids_lat, centroids_lon))
        dictionary = dict(zip(countries, centroids))

        dictionary['MA'] = (31.791702, -7.09262) #morocco
        dictionary['EH'] = (24.215527, -12.885834) #western sahara
        dictionary['HK'] = (22.396428, 114.109497) #hong kong
        dictionary['MO'] = (22.198745, 113.543873) #macau
        dictionary['XK'] = (42.602636, 20.902977) #kosovo
        dictionary['RS'] = (44.016521, 21.005859) #serbia
        dictionary['TW'] = (23.69781, 120.960515) #taiwan
        dictionary['AX'] = (60.1785, 19.9156) #aland islands
        dictionary['ES'] = (40.365008336683836, -3.6516251409956983) #spain (without canaries)

        return dictionary

    def convert_alpha2_to_name(self):
        '''
        Returns a dictionary that links the alpha 2 code of a country ('AD')
        into the name of the country ('Andorra'). Used for compatibility with
        GeoPandas.
        '''
        csv_borders = open(f'{self.data_folder}\\{self.borders_file}')
        DR = list(csv.DictReader(csv_borders))
        alpha2 = [x['country_code'] for x in DR]
        name = [x['country_name'] for x in DR]
        dictionary = dict(zip(alpha2, name))

        dictionary['RU'] = 'Russia'
        dictionary['BN'] = 'Brunei'
        dictionary['CY'] = 'Cyprus'
        dictionary['GQ'] = 'Eq. Guinea'
        dictionary['IR'] = 'Iran'
        dictionary['PS'] = 'Palestine'
        dictionary['EH'] = 'W. Sahara'
        dictionary['SB'] = 'Solomon Is.'
        dictionary['VE'] = 'Venezuela'
        dictionary['GM'] = 'Gambia'
        dictionary['SY'] = 'Syria'
        dictionary['KR'] = 'South Korea'
        dictionary['SS'] = 'S. Sudan'
        dictionary['LA'] = 'Laos'
        dictionary['KP'] = 'North Korea'
        dictionary['CD'] = 'Dem. Rep. Congo'
        dictionary['BA'] = 'Bosnia and Herz.'
        dictionary['SZ'] = 'eSwatini'
        dictionary['MD'] = 'Moldova'
        dictionary['DO'] = 'Dominican Rep.'
        dictionary['CF'] = 'Central African Rep.'
        dictionary['CI'] = 'Côte d\'Ivoire'
        dictionary['TW'] = 'Taiwan'
        dictionary['VN'] = 'Vietnam'
        dictionary['FK'] = 'Falkland Is.'
        dictionary['GB'] = 'United Kingdom'
        dictionary['BO'] = 'Bolivia'
        dictionary['TZ'] = 'Tanzania'
        dictionary['XK'] = 'Kosovo'
        
        return dictionary