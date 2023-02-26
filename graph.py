import geopy.distance
import networkx as nx
import numpy as np
import scipy as sc

from randomgraph import RandomGraphs


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
            lat = centroids[node][0]
            lon = centroids[node][1]
            G.add_node(node, pos=(lat, lon))
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

    def write_results(self, G, path='figures', filename='results.txt', save_as_file=True):
        '''
        Computes a bunch of statistics about the given Graph G, and saves them
        locally. Optionally prints them in the terminal.
        '''
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        degrees = [val for (node, val) in G.degree()]
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        num_triangles = int(sum(nx.triangles(G).values())/3)
        num_conn_comp = nx.number_connected_components(G)
        clustering_coeff = nx.average_clustering(G)
        clique_number = nx.graph_clique_number(G)
        pos_dict = nx.get_node_attributes(G, 'pos')
        pos = [coords for node, coords in pos_dict.items()]
        lat_nodes = [lat for (lat,lon) in pos]
        lon_nodes = [lon for (lat,lon) in pos]
        sorted_lat_nodes = sorted(pos_dict.keys(), key = lambda x: pos_dict[x][0])
        sorted_lon_nodes = sorted(pos_dict.keys(), key = lambda x: pos_dict[x][1])
        mean_lat = np.mean(lat_nodes)
        std_lat = np.std(lat_nodes)
        max_lat = np.max(lat_nodes)
        max_lat_lbl = sorted_lat_nodes[-1]
        min_lat = np.min(lat_nodes)
        min_lat_lbl = sorted_lat_nodes[0]
        circmean_lon = sc.stats.circmean(lon_nodes)
        circstd_lon = sc.stats.circstd(lon_nodes)
        max_lon = np.max(lon_nodes)
        max_lon_lbl = sorted_lon_nodes[-1]
        min_lon = np.min(lon_nodes)
        min_lon_lbl = sorted_lon_nodes[0]
        dist_dict = RandomGraphs.compute_geodesic_distances(pos)
        distances = [dist_dict[(pos_dict[n0],pos_dict[n1])] for (n0,n1) in G.edges()]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        BC = self.biggest_component(G)
        num_nodes_BC = BC.number_of_nodes()
        num_edges_BC = BC.number_of_edges()
        degrees_BC = [val for (node, val) in BC.degree()]
        mean_degree_BC = np.mean(degrees_BC)
        std_degree_BC = np.std(degrees_BC)
        num_triangles_BC = int(sum(nx.triangles(BC).values())/3)
        num_conn_comp_BC = nx.number_connected_components(BC)
        clustering_coeff_BC = nx.average_clustering(BC)
        clique_number_BC = nx.graph_clique_number(BC)
        pos_dict_BC = nx.get_node_attributes(BC, 'pos')
        pos_BC = [coords for node, coords in pos_dict_BC.items()]
        lat_nodes_BC = [lat for (lat,lon) in pos_BC]
        lon_nodes_BC = [lon for (lat,lon) in pos_BC]
        sorted_lat_nodes_BC = sorted(pos_dict_BC.keys(), key = lambda x: pos_dict_BC[x][0])
        sorted_lon_nodes_BC = sorted(pos_dict_BC.keys(), key = lambda x: pos_dict_BC[x][1])
        mean_lat_BC = np.mean(lat_nodes_BC)
        std_lat_BC = np.std(lat_nodes_BC)
        max_lat_BC = np.max(lat_nodes_BC)
        max_lat_lbl_BC = sorted_lat_nodes_BC[-1]
        min_lat_BC = np.min(lat_nodes_BC)
        min_lat_lbl_BC = sorted_lat_nodes_BC[0]
        circmean_lon_BC = sc.stats.circmean(lon_nodes_BC)
        circstd_lon_BC = sc.stats.circstd(lon_nodes_BC)
        max_lon_BC = np.max(lon_nodes_BC)
        max_lon_lbl_BC = sorted_lon_nodes_BC[-1]
        min_lon_BC = np.min(lon_nodes_BC)
        min_lon_lbl_BC = sorted_lon_nodes_BC[0]
        dist_dict_BC = RandomGraphs.compute_geodesic_distances(pos_BC)
        distances_BC = [dist_dict_BC[(pos_dict_BC[n0],pos_dict_BC[n1])] for (n0,n1) in BC.edges()]
        mean_dist_BC = np.mean(distances_BC)
        std_dist_BC = np.std(distances_BC)

        results = (f'{num_edges} & {mean_degree:.3f} & {std_degree:.3f} & {num_triangles} & {num_conn_comp} & '+
        f'{clustering_coeff:.3f} & {clique_number} & {mean_dist:.3f} & {std_dist:.3f} \n'+
        f'{num_edges_BC} & {mean_degree_BC:.3f} & {std_degree_BC:.3f} & {num_triangles_BC} & {num_conn_comp_BC} & '+
        f'{clustering_coeff_BC:.3f} & {clique_number_BC} & {mean_dist_BC:.3f} & {std_dist_BC:.3f}')

        results = ('GENERAL GRAPH STATISTICS:\n'+
                f'Number of nodes: {num_nodes}\n'+
                f'Number of edges: {num_edges}\n'+
                f'Mean vertex degree: {mean_degree:.3f}\n'+
                f'STD vertex degree: {std_degree:.3f}\n'+
                f'Number of triangles: {num_triangles}\n'+
                f'Number of connected components: {num_conn_comp}\n'+
                f'Clustering coefficient: {clustering_coeff:.3f}\n'+
                f'Clique number: {clique_number}\n\n'+
                f'Mean latitude: {mean_lat:.3f}\n'+
                f'STD latitude: {std_lat:.3f}\n'+
                f'Maximum latitude: {max_lat:.3f} - {max_lat_lbl}\n'+
                f'Minimum latitude: {min_lat:.3f} - {min_lat_lbl}\n'+
                f'Circular mean longitude: {circmean_lon:.3f}\n'+
                f'Circular STD longitude: {circstd_lon:.3f}\n'+
                f'Maximum longitude: {max_lon:.3f} - {max_lon_lbl}\n'+
                f'Minimum longitude: {min_lon:.3f} - {min_lon_lbl}\n\n'+
                f'Mean edge distance: {mean_dist:.3f} km\n'+
                f'STD edge distance: {std_dist:.3f} km\n\n\n'+
                'BIGGEST COMPONENT STATISTICS:\n'+
                f'Number of nodes: {num_nodes_BC}\n'+
                f'Number of edges: {num_edges_BC}\n'+
                f'Mean vertex degree: {mean_degree_BC:.3f}\n'+
                f'STD vertex degree: {std_degree_BC:.3f}\n'+
                f'Number of triangles: {num_triangles_BC}\n'+
                f'Number of connected components: {num_conn_comp_BC}\n'+
                f'Clustering coefficient: {clustering_coeff_BC:.3f}\n'+
                f'Clique number: {clique_number_BC}\n\n'+
                f'Mean latitude: {mean_lat_BC:.3f}\n'+
                f'STD latitude: {std_lat_BC:.3f}\n'+
                f'Maximum latitude: {max_lat_BC:.3f} - {max_lat_lbl_BC}\n'+
                f'Minimum latitude: {min_lat_BC:.3f} - {min_lat_lbl_BC}\n'+
                f'Circular mean longitude: {circmean_lon_BC:.3f}\n'+
                f'Circular STD longitude: {circstd_lon_BC:.3f}\n'+
                f'Maximum longitude: {max_lon_BC:.3f} - {max_lon_lbl_BC}\n'+
                f'Minimum longitude: {min_lon_BC:.3f} - {min_lon_lbl_BC}\n\n'+
                f'Mean edge distance: {mean_dist_BC:.3f} km\n'+
                f'STD edge distance: {std_dist_BC:.3f} km')

        if save_as_file:
            with open(f'{path}\\{filename}', 'w') as f:
                f.write(results)
        return results