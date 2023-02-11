import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sc

from clustering import ClusteringMethods
from data import DataLoader
from graph import GraphMethods

dl = DataLoader()
gm = GraphMethods()
cm = ClusteringMethods()

def cluster_world(maxK, figsize):
    '''
    Generate and save multiplots of different clustering methods applied to
    the bordering countries graph with various amounts of clusters.
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cb = dl.load_countryborders()
    print(type(cb))
    G = gm.countryborders_to_graph(cb)
    G = gm.biggest_component(G)

    clusters_HCS, k_HCS = cm.highly_connected_subgraphs(G)
    colormap_HCS = cm.generate_coloring(G, clusters_HCS)

    for k in range(2,maxK+1):
        clusters_GM = cm.girvan_newman(G, k)
        colormap_GM = cm.generate_coloring(G, clusters_GM)
        clusters_SP_un = cm.spectral(G, k, version='unnormalized')
        colormap_SP_un = cm.generate_coloring(G, clusters_SP_un)
        clusters_SP_SM = cm.spectral(G, k, version='normalized-SM')
        colormap_SP_SM = cm.generate_coloring(G, clusters_SP_SM)
        clusters_SP_NJW = cm.spectral(G, k, version='normalized-NJW')
        colormap_SP_NJW = cm.generate_coloring(G, clusters_SP_NJW)

        fig, axs = plt.subplots(figsize=figsize, ncols=3, nrows=2)
        fig.tight_layout()
        world.plot(color='lightgrey', ax=axs[0,0])
        world.plot(color='lightgrey', ax=axs[0,1])
        world.plot(color='lightgrey', ax=axs[1,0])
        world.plot(color='lightgrey', ax=axs[1,1])
        world.plot(color='lightgrey', ax=axs[1,2])
        for node in G.nodes:
            world[world.name == node].plot(color=colormap_GM[node], ax=axs[0,0])
            world[world.name == node].plot(color=colormap_HCS[node], ax=axs[0,1])
            world[world.name == node].plot(color=colormap_SP_un[node], ax=axs[1,0])
            world[world.name == node].plot(color=colormap_SP_SM[node], ax=axs[1,1])
            world[world.name == node].plot(color=colormap_SP_NJW[node], ax=axs[1,2])
        axs[0,0].set_title(f'Girvan Newman\n{k} clusters')
        axs[0,1].set_title(f'Highly Connected Subgraphs\n{k_HCS} clusters')
        axs[1,0].set_title(f'Unnormalized spectral clustering\n{k} clusters')
        axs[1,1].set_title(f'Symmetrically normalized spectral clustering\n{k} clusters')
        axs[1,2].set_title(f'Left normalized spectral clustering\n{k} clusters')
        axs[0,2].axis('off')
        
        plt.savefig(f'figures\\cluster_world\\multiclustering_{k}')

def road_network(country, figsize, version='all'):
    '''
    '''
    roads = dl.load_roads(country=country, continent=None)

    if (version == 'all' or version == 'normal'):
        G = gm.roadnetwork_to_graph(roads, weighted=False)
        plt.figure(figsize=figsize)
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), edge_color='lightgrey', node_size=3)
        country = country.replace('.', '')
        plt.savefig(f'figures\\road_network\\{country}')

    if (version == 'all' or version == 'smooth'):
        G = gm.roadnetwork_to_graph(roads, weighted=False)
        G_smooth = gm.smoothen_graph(G)
        plt.figure(figsize=figsize)
        nx.draw(G_smooth, pos=nx.get_node_attributes(G_smooth, 'pos'), edge_color='lightgrey', node_size=3)
        country = country.replace('.', '')
        plt.savefig(f'figures\\road_network\\{country}_smooth')

    if (version == 'all' or version == 'weighted'):
        G_weighted = gm.roadnetwork_to_graph(roads, weighted=True)
        plt.figure(figsize=figsize)
        nx.draw(G_weighted, pos=nx.get_node_attributes(G_weighted, 'pos'), edge_color='lightgrey', node_size=3)
        pos = nx.get_node_attributes(G_weighted, 'pos')
        labels = nx.get_edge_attributes(G_weighted, 'weight')
        labels = {e: "%.2f" % w for e,w in labels.items()}
        nx.draw_networkx_edge_labels(G_weighted, pos=pos, edge_labels=labels)
        country = country.replace('.', '')
        plt.savefig(f'figures\\road_network\\{country}_weighted')

def cluster_world_dist(maxK, figsize):
    '''
    '''
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country_geometries = world.apply(lambda x: x['geometry'], axis=1)
    country_names = world.apply(lambda x: x['name'], axis=1)
    centroids = dict(zip(country_names, country_geometries.centroid.map(lambda p: [y for y in p.coords][0])))

    centroids.update({'Bahrain': (50.637772, 25.930414)})
    centroids.update({'St. Lucia': (-60.978893, 13.909444)})
    centroids.update({'St. Vincent and the Grenadines': (-61.287228, 12.984305)})
    centroids.update({'Andorra': (1.601554, 42.546245)})

    cb = dl.load_countryborders()
    G = gm.countryborders_to_weightedgraph(cb, centroids)
    
    plt.figure(figsize=figsize)
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), edge_color='lightgrey', node_size=3)
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_edge_attributes(G, 'weight')
    labels = {e: "%.2f" % w for e,w in labels.items()}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    plt.show()


#cluster_world(maxK=2, figsize=(30,20))
#road_network('Italy', figsize=(40,40))


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