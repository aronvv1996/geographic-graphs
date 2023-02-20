import itertools
import networkx as nx
import numpy as np
import geopy.distance

class RandomGraphs():

    def __init__(self):
        '''
        The class 'RandomGraphs' contains methods that produce random graphs
        according to various models.
        '''

    @staticmethod
    def compute_geodesic_distances(coords):
        '''
        Given a list of (lat, lon) coordinates, computes the distances in
        kilometers between all pairs of nodes and returns a dictionary.
        '''
        dictionary = {}
        for c in coords:
            dictionary[(c, c)] = 0
        for pair in itertools.combinations(coords, 2):
            dist = geopy.distance.geodesic(pair[0], pair[1]).km
            dictionary[(pair[0], pair[1])] = dist
            dictionary[(pair[1], pair[0])] = dist
        return dictionary

    def _uniform_random_points_sphere(self, n, pole_angle=90):
        '''
        Returns the (lat,lon)-coordinates of n random uniformly distributed
        points on a sphere. When 'pole_angle' is strictly less than 90, there
        will be no points p such that |lat(p)| > pole_angle.
        '''
        assert 0 <= pole_angle <= 90, 'Value of \'pole_angle\' must be in the interval [0, 90].'

        a = np.sin(np.deg2rad(pole_angle))/2
        coords = self._uniform_random_points_plane(n, x0=0, x1=1, y0=0.5-a, y1=0.5+a)
        u = [c[0] for c in coords]
        v = [c[1] for c in coords]
        ϕ = [2*np.pi*x-np.pi for x in u]
        θ = [np.arccos(2*y-1)-np.pi/2 for y in v]
        lon = [np.rad2deg(x) for x in ϕ]
        lat = [np.rad2deg(y) for y in θ]

        return list(zip(lat, lon))

    def _uniform_random_points_plane(self, n, x0, x1, y0, y1):
        '''
        Returns the (x,y)-coordinates of n random uniformly distributed points
        on the rectangle [x0,x1]*[y0,y1].
        '''
        x_samples = np.random.uniform(x0, x1, size=n)
        y_samples = np.random.uniform(y0, y1, size=n)
        return list(zip(x_samples, y_samples))

    def erdos_renyi(self, n, p):
        '''
        Returns a random Erdős-Rényi graph with n nodes and probability p.
        '''
        G = nx.erdos_renyi_graph(n, p)
        return G

    def random_geometric(self, n, r):
        '''
        Returns a random geometric graph. Nodes are placed uniformly at random
        in a unit square. Every node is connected to all other nodes within the
        euclidean distance r.
        '''
        G = nx.random_geometric_graph(n, r)
        return G

    def ε_neighborhood(self, n, ε, pole_angle=90):
        '''
        Returns the ε-neighborhood similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        all other nodes within a geodesic distance of ε kilometers.
        '''
        G = nx.Graph()
        nodes = self._uniform_random_points_sphere(n, pole_angle)
        distances = self.compute_geodesic_distances(nodes)
        edges = [(x,y) for x in nodes for y in nodes if 0 < distances[(x,y)] < ε]
        for node in nodes:
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))
        return G

    def k_nearest_neighbors(self, n, k, pole_angle=90):
        '''
        Returns the K-nearest-neighbors similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        its k nearest neighbors.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        nodes = self._uniform_random_points_sphere(n, pole_angle)
        distances = self.compute_geodesic_distances(nodes)
        edges = []
        for node in nodes:
            dist = {k: v for k, v in distances.items() if k[0]==node}
            k_smallest_dist = dict(sorted(dist.items(), key = lambda x: x[1])[1:k+1])
            edges += k_smallest_dist.keys()
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))
        return G

    def mutual_k_nearest_neighbors(self, n, k, pole_angle=90):
        '''
        Returns the mutual-K-nearest-neighbors similarity graph constructed from
        a set of randomly distributed nodes on the globe. A pair of nodes is
        connected if both nodes are in the k nearest neighbors of the other node.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        nodes = self._uniform_random_points_sphere(n, pole_angle)
        distances = self.compute_geodesic_distances(nodes)
        edges = []
        for node in nodes:
            dist = {k: v for k, v in distances.items() if k[0]==node}
            k_smallest_dist = dict(sorted(dist.items(), key = lambda x: x[1])[1:k+1])
            edges += k_smallest_dist.keys()
            G.add_node(node, pos=node)
        edges = [(x,y) for (x,y) in edges if (y,x) in edges]
        G.add_edges_from(set(edges))
        return G

    def relative_neighborhood(self, n, pole_angle=90):
        '''
        Returns the Random Neighborhood Graph constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if they are
        at least as close to each other as they are to any other node.
        '''
        G = nx.Graph()
        nodes = self._uniform_random_points_sphere(n, pole_angle)
        distances = self.compute_geodesic_distances(nodes)
        edges = []
        for i in nodes:
            for j in nodes:
                if (i==j):
                    continue
                d_max = [max(distances[(k,i)], distances[(k,j)]) for k in nodes if k!=i and k!=j]
                if distances[(i,j)] <= min(d_max):
                    edges.append((i,j))

        for node in nodes:
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))
        return G