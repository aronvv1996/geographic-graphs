import itertools
import networkx as nx


class ClusteringMethods():

    def __init__(self):
        '''
    	The 'ClusteringMethods' class contains methods which apply a variety of
        clustering methods to networkx Graph objects.
        '''

    @staticmethod
    def create_example_graph():
        '''
        Create a small example graph with 12 nodes.
        '''
        G = nx.Graph()
        nodes = range(1,12)
        edges = [(1,2),(1,11),(1,12),(2,3),(2,12),(3,4),(3,11),(3,12),(4,5),
                 (4,6),(4,10),(5,6),(6,7),(7,8),(7,9),(7,10),(8,9),(8,10),
                 (9,10),(10,11),(11,12)]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        return G

    @staticmethod
    def _assert_validity_graph(G, k=None):
        '''
        '''
        assert nx.is_connected(G), "Graph must be connected."
        if (k is not None):
            assert 2 <= k <= G.number_of_nodes(), "Number of clusters must be in the range [2, number of nodes in G]."

    def girvan_newman(self, G, k):
        '''
        Description. Graph G, k clusters
        '''
        self._assert_validity_graph(G, k)
        comp = nx.algorithms.community.girvan_newman(G)
        clusters = itertools.islice(comp, k-2, k-1)

        return next(clusters)

    def highly_connected_subgraphs(self, G, base_case=True):
        '''
        Description. Graph G
        '''
        self._assert_validity_graph(G)
        min_edge_cut = nx.algorithms.connectivity.minimum_edge_cut(G)
        
        if not (len(min_edge_cut) > G.number_of_nodes() / 2):
            G.remove_edges_from(min_edge_cut)
            subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

            if (len(subgraphs) == 2):
                H1 = self.highly_connected_subgraphs(subgraphs[0], base_case=False)
                H2 = self.highly_connected_subgraphs(subgraphs[1], base_case=False)
                G = nx.compose(H1, H2)

        if base_case:
            return list(nx.connected_components(G))
        else:
            return G


cm = ClusteringMethods()
G = cm.create_example_graph()
print(cm.highly_connected_subgraphs(G,6))