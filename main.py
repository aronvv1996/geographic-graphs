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


cm = ClusteringMethods()
dl = DataLoader()
gm = GraphMethods()
pm = PlotMethods()
rg = RandomGraphs()

cb = dl.load_countryborders()
cc = dl.load_countrycentroids()
G = gm.countryborders_to_weightedgraph(cb, cc)
pm.cluster_world(2,figsize=(15,20), type='BC')

exit()

for n in range(nr_runs):
    i = (90,0)
    alpha = epsilon/r
    pos = _uniform_random_points_sphere(nr_points, alpha)
    if (len(pos)%2) == 1:
        pos.pop()
    pos_j = pos[:len(pos)//2]
    pos_k = pos[len(pos)//2:]
    trials = len(pos_j)
    successes = 0
    for (j,k) in zip(pos_j, pos_k):
        if (geopy.distance.great_circle(j,k) < epsilon):
            successes += 1
    results.append(successes/trials)
    print(f'{n} -- {successes}/{trials}')

print(results)
print(np.mean(results))
plt.figure(figsize=(10,10))
plt.hist(results)
plt.show()

exit()

pos = rg._uniform_random_points_sphere(171, pole_angle=90)
for l in np.linspace(0.5,1.5,21):
    print(l)
    G = rg.relative_neighborhood(171, λ=l, pole_angle=90, pos=pos)
    pm.plot2D_worldgraph(G, figsize=(20,15), filename=f'RNG_2Dlambda{int(l*100)}')
    pm.plot3D_worldgraph(G, figsize=(20,20), n_frames=72, filename=f'RNG_3Dlambda{int(l*100)}')
    pm.plotSP_worldgraph(G, figsize=(20,20), filename=f'RNG_SPlambda{int(l*100)}')
    gm.write_results(G, filename=f'RNG_lambda{int(l*100)}_results', full_results=False)

exit()

nr_runs = 2500

res = []
for n in range(nr_runs):
    G = rg.ε_neighborhood(n=171, ε=2000)
    #pm.plot2D_worldgraph(G, figsize=(20,15), filename=f'3NN_maxlength_{e}km', show=False)
    res.append(int(sum(nx.triangles(G).values())/3))
    print(n)

print(np.mean(res))
print(np.var(res))
plt.figure(figsize=(10,10))
plt.hist(res, bins=50)
plt.show()

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