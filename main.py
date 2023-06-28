import geopandas as gpd
import geopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon

from clustering import ClusteringMethods
from data import DataLoader
from graph import GraphMethods
from plots import PlotMethods
from randomgraph import RandomGraphs

dl = DataLoader()
gm = GraphMethods()
pm = PlotMethods()
rg = RandomGraphs()

from pyvis.network import Network

cb = dl.load_countryborders()
WG = gm.countryborders_to_graph(cb)
WG = gm.delete_singletons(WG)

n=171
G = rg.ε_neighborhood(n=n, ε=1836.2)

print(gm.tvd(nx.Graph(), WG))

exit()
degs = [d for n,d in WG.degree() if d != 0]
c = Counter(degs)

def tvd(counter):
    TVD = 0
    mx = max(max(c.keys()),max(counter.keys()))
    WG_c = [0]*(mx+1)
    SG_c = [0]*(mx+1)
    for L in range(mx+1):
        WG_c[L] = c[L]
        SG_c[L] = counter[L]
    for L in range(1,mx+2):
        for subset in combinations(range(mx+1), L):
            WG_tot = 0
            SG_tot = 0
            for element in subset:
                WG_tot += c[element]/171
                SG_tot += counter[element]/171
            temp_TVD = abs(WG_tot - SG_tot)
            if temp_TVD > TVD:
                TVD = temp_TVD
    return TVD

nr_runs = 100
n = 171

best_edges = [0]*14
best_triangles = [0]*14
best_dist = [1]*14
best_graphs = [0]*14*3

for run in range(nr_runs):
    print(f'run no. {run}')
    nodes = rg._uniform_random_points_sphere(n=n)
    distances = rg.compute_spherical_distances(nodes)
    Graphs = []

    Graphs.append(rg.ε_neighborhood(n=n, ε=1800, pos=nodes, distances=distances))
    Graphs.append(rg.ε_neighborhood(n=n, ε=1900, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=2, max_edgelength=1700, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=3, max_edgelength=1700, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=4, max_edgelength=1900, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=4, max_edgelength=2100, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=5, max_edgelength=1800, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=5, max_edgelength=2000, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=6, max_edgelength=1800, pos=nodes, distances=distances))
    Graphs.append(rg.k_nearest_neighbors(n=n, k=6, max_edgelength=1900, pos=nodes, distances=distances))
    Graphs.append(rg.relative_neighborhood(n=n, λ=0.95, pos=nodes, distances=distances))
    Graphs.append(rg.relative_neighborhood(n=n, λ=1.15, pos=nodes, distances=distances))
    Graphs.append(rg.relative_neighborhood(n=n, λ=1.30, pos=nodes, distances=distances))
    Graphs.append(rg.beta_skeleton(n=n, β=1, pos=nodes, distances=distances))

    for Graph in range(len(Graphs)):
        G = Graphs[Graph]
        edges = G.number_of_edges()
        triangles = int(sum(nx.triangles(G).values())/3)
        degs = [d for n,d in G.degree()]
        dist = tvd(Counter(degs))

        if abs(332-edges) < abs(332-best_edges[Graph]) or best_edges[Graph] == 0:
            best_edges[Graph] = edges
            best_graphs[Graph*3] = G

        if abs(173-triangles) < abs(173-best_triangles[Graph]) or best_triangles[Graph] == 0:
            best_triangles[Graph] = triangles
            best_graphs[Graph*3+1] = G
        
        if dist < best_dist[Graph] or best_dist[Graph] == 1:
            best_dist[Graph] = dist
            best_graphs[Graph*3+2] = G

i=0
for G in best_graphs:
    pm.plot2Dw_worldgraph(G, figsize=(8,6), file_name=f'graph_{i}', show_world=False)
    i += 1

print(best_edges)
print(best_triangles)
print(best_dist)

exit()

runs = 10
res = []

for x in range(runs):
    print(x)
    G = rg.beta_skeleton(n=171, β=1)
    res.append(G.number_of_edges())

print(np.mean(res))

exit()
distances = rg.compute_spherical_distances(nodes)

G = rg.gabriel(n=171, pos=nodes, distances=distances)
pm.plot2D_worldgraph(G, file_name='2D_1')

G = rg.beta_skeleton(n=171, β=1, pos=nodes, distances=distances)
pm.plot2D_worldgraph(G, file_name='2D_2')
exit()

for b in np.arange(1,2,0.05):
    print(b)
    G = rg.beta_skeleton(n=171, β=b, pos=nodes, distances=distances)
    pm.plot2D_worldgraph(G, file_name=f'{int(b*100)}skeleton_2D')
    pm.plotSP_worldgraph(G, file_name=f'{int(b*100)}skeleton_SP')

exit()

dist = rg.compute_spherical_distances((i,j))[(i,j)]
nr_runs = 1000
n = 100
counter = 0

for run in range(nr_runs):
    print(f'{run}/{nr_runs}')
    nodes = rg._uniform_random_points_sphere(n=n-2)
    nodes += [i,j]
    G = rg.k_nearest_neighbors(n=n, k=1, pos=nodes)
    if ((i,j) in G.edges()):
        counter += 1

prob = counter/nr_runs
print(prob)