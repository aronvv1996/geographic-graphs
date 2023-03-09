import itertools
import networkx as nx
import numpy as np
import scipy as sc
from sklearn.cluster import KMeans


class ClusteringMethods():

    def __init__(self):
        '''
    	The 'ClusteringMethods' class contains methods which apply a variety of
        clustering methods to networkx Graph objects.
        '''

    @staticmethod
    def generate_coloring(G, clustering, random_colors=False):
        '''
        Generates a coloring that maps nodes to colors depending on what cluster
        they belong to. If the number of clusters k is over 10, random colors
        are selected.
        '''
        k = len(clustering)
        nodes = G.nodes()
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        if random_colors or (k > 10):
            color_list = ["#%06x" % np.random.randint(0, 0xFFFFFF) for _ in range(k)]

        color_map = [color_list[i] for node in nodes for i in range(k) if node in clustering[i]]
        dictionary = dict(zip(G.nodes, color_map))

        return dictionary

    def girvan_newman(self, G, k):
        '''
        Applies the Girvan-Newman clustering algorithm to networkx graph G with
        k clusters, and returns a list of k sets of nodes belonging to
        different clusters.
        '''
        assert 2 <= k <= G.number_of_nodes(), "Number of clusters must be in the range [2, |V|], where |V| is the number of nodes of G."

        comp = nx.algorithms.community.girvan_newman(G)
        num_conn_comp = nx.number_connected_components(G)
        clusters = itertools.islice(comp, k-num_conn_comp-1, k-num_conn_comp)

        return next(clusters)

    def highly_connected_subgraphs(self, G, base_case=True):
        '''
        Applies the recursive HCS-clustering algorithm to networkx graph G. The
        number of clusters cannot be specified beforehand. Returns a list of k 
        sets of nodes belonging to different clusters, and k.
        '''
        assert nx.is_connected(G), "Graph must be connected."

        G = G.copy()
        min_edge_cut = nx.algorithms.connectivity.minimum_edge_cut(G)
        
        if not (len(min_edge_cut) > G.number_of_nodes() / 2):
            G.remove_edges_from(min_edge_cut)
            subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

            if (len(subgraphs) == 2):
                H1 = self.highly_connected_subgraphs(subgraphs[0], base_case=False)
                H2 = self.highly_connected_subgraphs(subgraphs[1], base_case=False)
                G = nx.compose(H1, H2)

        if base_case:
            clusters = list(nx.connected_components(G))
            k = len(clusters)
            return (clusters, k)
        else:
            return G

    def spectral(self, G, k, version):
        '''
        Applies the spectral clustering algorithm to networkx graph G with k
        clusters, and returns a list of k sets of nodes belonging to different
        clusters.

        Three versions are available which all use a different Laplacian matrix
        corresponding to G:
            - the regular Laplacian matrix (unnormalized);
            - the symmetrically normalized Laplacian matrix (normalized-SM);
            - the left normalized Laplacian matrix (normalized-NJW).
        '''
        assert 2 <= k <= G.number_of_nodes() - 2, "Number of clusters must be in the range [2, |V|-2], where |V| is the number of nodes of G."
        assert version in ['unnormalized', 'normalized-SM', 'normalized-NJW'], "Version must be 'unnormalized', 'normalized-SM', or 'normalized-NJW'."

        if (version == 'unnormalized'):
            laplacian = nx.laplacian_matrix(G).asfptype()
            eigenvals, eigenvecs = sc.sparse.linalg.eigs(laplacian, k=k, which='SM')
            eigenvecs = [x.real for x in eigenvecs]

        if (version == 'normalized-SM'):
            laplacian = nx.laplacian_matrix(G).toarray()
            degree_matrix = np.diag([x[1] for x in G.degree])
            normalized_laplacian = np.matmul(np.linalg.pinv(degree_matrix), laplacian)
            eigenvals, eigenvecs = sc.sparse.linalg.eigs(normalized_laplacian, k=k, which='SM')
            eigenvecs = [x.real for x in eigenvecs]

        if (version == 'normalized-NJW'):
            normalized_laplacian = nx.normalized_laplacian_matrix(G).asfptype()
            eigenvals, unnormalized_eigenvecs = sc.sparse.linalg.eigs(normalized_laplacian, k=k, which='SM')
            unnormalized_eigenvecs = [x.real for x in unnormalized_eigenvecs]
            eigenvecs = [[x/np.linalg.norm(row) for x in row] for row in unnormalized_eigenvecs]

        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit(eigenvecs).labels_
        clusters = [set([n for (n,l) in zip(G.nodes,labels) if l == i]) for i in range(k)]

        return clusters