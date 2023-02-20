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
weights = [WG.get_edge_data(e[0], e[1], 'weight')['weight'] for e in WG.edges]

i=70
k=2

G = rg.k_nearest_neighbors(WG.number_of_nodes(), k, pole_angle=i)
degrees = [val for (node, val) in G.degree()]
mean = np.mean(degrees)
triangles = int(sum(nx.triangles(G).values())/3)
conn_comp = nx.number_connected_components(G)
clustering_coeff = nx.average_clustering(G)
clique_number = nx.graph_clique_number(G)
pm.plotSP_worldgraph(G, figsize=(20,20), filename=f'{k}NN_worldgraph_{i}_SP')
pm.plot2D_worldgraph(G, figsize=(15,10), filename=f'{k}NN_worldgraph_{i}')
pm.plot3D_worldgraph(G, figsize=(20,20), n_frames=72, filename=f'{k}NN_worldgraph_{i}',
                        title=f'{k}-nearest-neighbors random graph',
                        figtext=f'number of nodes: {G.number_of_nodes()}\n'+
                                f'number of edges: {G.number_of_edges()}\n'+
                                f'average degree: {mean:.3f}\n'+
                                f'number of triangles: {triangles}\n'+
                                f'connected components: {conn_comp}\n'+
                                f'clustering coefficient: {clustering_coeff:.3f}')


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