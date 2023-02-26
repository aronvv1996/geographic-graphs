import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import scipy as sc

from clustering import ClusteringMethods
from data import DataLoader
from graph import GraphMethods
from plots import PlotMethods
from randomgraph import RandomGraphs

import warnings
warnings.filterwarnings('ignore', message='The GeoDataFrame you are attempting to plot is empty. Nothing has been displayed.')


cm = ClusteringMethods()
dl = DataLoader()
gm = GraphMethods()
pm = PlotMethods()
rg = RandomGraphs()

cb = dl.load_countryborders()
cc = dl.load_countrycentroids()
WG = gm.countryborders_to_weightedgraph(cb, cc)
n = WG.number_of_nodes()
pos = list(nx.get_node_attributes(WG, 'pos').values())

i=70
G = rg.relative_neighborhood(n, pole_angle=i, pos=pos)


pm.plotSP_worldgraph(G, figsize=(20,20), filename=f'RNG_worldgraph_SP')
pm.plot2D_worldgraph(G, figsize=(15,10), filename=f'RNG_worldgraph_2D')
pm.plot3D_worldgraph(G, figsize=(20,20), n_frames=72, filename=f'RNG_worldgraph_3D', keep_frames=True)
print(gm.write_results(G, save_as_file=False))
exit()

country = 'Netherlands'
roads = dl.load_roads(country=country)
G = gm.roadnetwork_to_graph(roads, weighted=False)

cities = dl.load_cities(country=country)
#print(cities)
city_names = list(cities.apply(lambda x: x['city'], axis=1))
city_coords = list(cities.apply(lambda x: [y for y in x['geometry'].coords][0], axis=1))
nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), edge_color='lightgrey', node_size=3)

G_city = nx.Graph()
for c in range(len(cities)):
    G_city.add_node(city_names[c], pos=city_coords[c])
nx.draw(G_city, pos=nx.get_node_attributes(G_city, 'pos'), node_size=10, node_color='red', with_labels=True, font_size=5)
plt.show()