import geopy.distance
import networkx as nx
import numpy as np
import os

from randomgraph import RandomGraphs


class GraphMethods():

    def __init__(self, results_folder='results'):
        '''
        The class 'GraphMethods' contains methods converting various kinds of
        databases into networkx graphs, and a variety of graph operations.
        '''
        self.results_folder = results_folder

    @staticmethod
    def biggest_component(G):
        '''
        Returns the biggest connected component of graph G.
        '''
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        return G.subgraph(Gcc[0])
    
    @staticmethod
    def delete_singletons(G):
        '''
        Returns a copy of Graph G where all singletons (vertices with degree 0)
        are removed.
        '''
        G = G.copy()
        for node in list(G.nodes()):
            if (G.degree(node) == 0):
                G.remove_node(node)

        return G
    
    @staticmethod
    def smoothen_graph(G):
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

    @staticmethod
    def countryborders_to_graph(cb):
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
    
    @staticmethod
    def countryborders_to_weightedgraph(cb, centroids):
        '''
        Converts the countryborders dictionary into a weighted graph G. G has
        countries as its set of nodes, and each set of bordering countries is
        connected via an edge in G. The weight of each edge is computed as the
        great-circle distance between the centroids of the incident countries.
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
            weights.append(geopy.distance.great_circle(c0, c1).km)
        weighted_edges = [(e0, e1, w) for ((e0, e1), w) in zip(edges, weights)]

        for node in nodes:
            lat = centroids[node][0]
            lon = centroids[node][1]
            G.add_node(node, pos=(lat, lon))
        G.add_weighted_edges_from(weighted_edges)
        
        return G

    @staticmethod
    def roadnetwork_to_graph(gdf, weighted=False):
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

    def write_results(self, G, form, save_to_file=True, file_name='results'):
        '''
        Computes a bunch of statistics about the given Graph G, and saves them
        locally. The "latex"-form generates results easily inserted in tabular
        form, the "text"-form generates results in more readible text.
        '''
        assert form in ['latex', 'text'], 'Results can be shown in "latex"-form or in "text"-form.'

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        degrees = [val for (node, val) in G.degree()]
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        min_degree = min(degrees)
        max_degree = max(degrees)
        num_singletons = degrees.count(0)
        num_triangles = int(sum(nx.triangles(G).values())/3)
        num_conn_comp = nx.number_connected_components(G)
        clustering_coeff = nx.average_clustering(G)
        clique_number = nx.graph_clique_number(G)
        pos_dict = nx.get_node_attributes(G, 'pos')
        pos = [coords for node, coords in pos_dict.items()]
        dist_dict = RandomGraphs.compute_spherical_distances(pos)
        distances = [dist_dict[(pos_dict[n0],pos_dict[n1])] for (n0,n1) in G.edges()]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        BC = self.biggest_component(G)
        num_nodes_BC = BC.number_of_nodes()
        num_edges_BC = BC.number_of_edges()
        degrees_BC = [val for (node, val) in BC.degree()]
        mean_degree_BC = np.mean(degrees_BC)
        std_degree_BC = np.std(degrees_BC)
        min_degree_BC = min(degrees_BC)
        max_degree_BC = max(degrees_BC)
        num_triangles_BC = int(sum(nx.triangles(BC).values())/3)
        clustering_coeff_BC = nx.average_clustering(BC)
        clique_number_BC = nx.graph_clique_number(BC)
        pos_dict_BC = nx.get_node_attributes(BC, 'pos')
        pos_BC = [coords for node, coords in pos_dict_BC.items()]
        dist_dict_BC = RandomGraphs.compute_spherical_distances(pos_BC)
        distances_BC = [dist_dict_BC[(pos_dict_BC[n0],pos_dict_BC[n1])] for (n0,n1) in BC.edges()]
        mean_dist_BC = np.mean(distances_BC)
        std_dist_BC = np.std(distances_BC)

        if (form == 'latex'):
            results = str(f'{num_edges} & {mean_degree:.3f} & {std_degree:.3f} & '+
                f'{min_degree} & {max_degree} & {num_singletons} & '+          
                f'{num_triangles} & {num_conn_comp} & {clustering_coeff:.3f} & '+
                f'{clique_number} & {mean_dist:.3f} & {std_dist:.3f} \\\\ \n'+
                f'{num_edges_BC} & {mean_degree_BC:.3f} & {std_degree_BC:.3f} & '+
                f'{min_degree_BC} & {max_degree_BC} & '+          
                f'{num_triangles_BC} & {clustering_coeff_BC:.3f} & '+
                f'{clique_number_BC} & {mean_dist_BC:.3f} & {std_dist_BC:.3f} \\\\')

        if (form == 'text'):
            results = ('GENERAL GRAPH STATISTICS:\n'+
                    f'Number of nodes: {num_nodes}\n'+
                    f'Number of edges: {num_edges}\n'+
                    f'Mean vertex degree: {mean_degree:.3f}\n'+
                    f'STD vertex degree: {std_degree:.3f}\n'+
                    f'Minimum vertex degree: {min_degree}\n'+
                    f'Maximum vertex degree: {max_degree}\n'+
                    f'Number of singletons: {num_singletons}\n'+
                    f'Number of triangles: {num_triangles}\n'+
                    f'Number of connected components: {num_conn_comp}\n'+
                    f'Clustering coefficient: {clustering_coeff:.3f}\n'+
                    f'Clique number: {clique_number}\n\n'+
                    f'Mean edge distance: {mean_dist:.3f} km\n'+
                    f'STD edge distance: {std_dist:.3f} km\n\n\n'+
                    'BIGGEST COMPONENT STATISTICS:\n'+
                    f'Number of nodes: {num_nodes_BC}\n'+
                    f'Number of edges: {num_edges_BC}\n'+
                    f'Mean vertex degree: {mean_degree_BC:.3f}\n'+
                    f'STD vertex degree: {std_degree_BC:.3f}\n'+
                    f'Minimum vertex degree: {min_degree_BC}\n'+
                    f'Maximum vertex degree: {max_degree_BC}\n'+
                    f'Number of triangles: {num_triangles_BC}\n'+
                    f'Clustering coefficient: {clustering_coeff_BC:.3f}\n'+
                    f'Clique number: {clique_number_BC}\n\n'+
                    f'Mean edge distance: {mean_dist_BC:.3f} km\n'+
                    f'STD edge distance: {std_dist_BC:.3f} km')

        if save_to_file:
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)
            with open(f'{self.results_folder}\\{file_name}', 'w') as f:
                f.write(results)

        return results