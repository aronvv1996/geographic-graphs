import geopy.distance
import networkx as nx
import numpy as np


class GraphMethods():

    def __init__(self):
        '''
        The class 'GraphMethods' contains methods converting various kinds of
        databases into networkx graphs, and a variety of graph operations.
        '''

    def countryborders_to_graph(self, cb):
        '''
        Converts the countryborders dictionary into a graph G. G has countries
        as its set of nodes, and each set of bordering countries is connected
        via an edge in G.
        '''
        G = nx.Graph()
        nodes = list(cb.keys())
        edges = [(x,y) for x in nodes for y in cb[x]]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        return G
    
    def countryborders_to_weightedgraph(self, cb, centroids):
        '''
        Converts the countryborders dictionary into a weighted graph G. G has
        countries as its set of nodes, and each set of bordering countries is
        connected via an edge in G. The weight of each edge is computed as the
        distance between the centroids of the incident countries.
        '''
        G = nx.Graph()
        nodes = list(cb.keys())
        edges = [(x,y) for x in nodes for y in cb[x]]
        weights = []
        for edge in edges:
            e0 = edge[0]
            e1 = edge[1]
            c0 = centroids[e0]
            c1 = centroids[e1]
            weights.append(geopy.distance.geodesic(c0, c1).km)
        weighted_edges = [(e0, e1, w) for ((e0, e1), w) in zip(edges, weights)]
        for node in nodes:
            pos_lat = centroids[node][0]
            pos_lng = centroids[node][1]
            G.add_node(node, pos=(pos_lng, pos_lat))
        G.add_weighted_edges_from(weighted_edges)
        
        return G

    def roadnetwork_to_graph(self, gdf, weighted=False):
        '''
        Converts the road-network GeoDataFrame to a networkx Graph. Setting
        'weighted' to True creates a weighted graph where each edge has
        weight corresponding to the length of the road in kilometers.
        '''
        gdf = gdf.explode(index_parts=True)
        coordinates = gdf.apply(lambda x: [y for y in x['geometry'].coords], axis=1)

        if not weighted:
            G = nx.Graph()
            nodes = set([x for y in coordinates for x in y])
            edges = [(r[i],r[i+1]) for r in coordinates for i in range(len(r)-1)]

            for node in nodes:
                G.add_node(node, pos=node)
            G.add_edges_from(edges)

        if weighted:
            G = nx.Graph()
            road_lengths = gdf.apply(lambda x: x['LENGTH_KM'], axis=1)
            edges = [(x[0], x[-1]) for x in coordinates]
            nodes = set([x for y in edges for x in y])
            weighted_edges = [(e0, e1, w) for ((e0, e1), w) in zip(edges, road_lengths)]

            for node in nodes:
                G.add_node(node, pos=node)
            G.add_weighted_edges_from(weighted_edges)

        return G
    
    def add_cities_to_graph(self, G, cities):
        '''
        Computes the closest node to each city's coordinates in graph G.
        Returns the graph where nodes are labeled with their corresponding
        cities.
        '''
        city_names = list(cities.apply(lambda x: x['city'], axis=1))
        city_coords = list(cities.apply(lambda x: [y for y in x['geometry'].coords][0], axis=1))
        
        for c in range(len(city_names)):
            node_name = city_names[c]
            node_coords = city_coords[c]
            distances = [(x-node_coords[0])**2 + (y-node_coords[1])**2 for (x,y) in G.nodes]
            closest_node = G.nodes[np.argmin(distances)]

    def biggest_component(self, G):
        '''
        Returns the biggest connected component of graph G.
        '''
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        return G.subgraph(Gcc[0])
        
    def smoothen_graph(self, G):
        '''
        Returns the smallest homeomorphic graph to G by smoothing out nodes of
        degree 2.
        '''
        G = G.copy()

        for node in list(G.nodes()):
            if (G.degree(node) == 2):
                edges = list(G.edges(node))
                G.add_edge(edges[0][1], edges[1][1])
                G.remove_node(node)
        
        return G