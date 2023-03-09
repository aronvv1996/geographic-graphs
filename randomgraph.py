import geopandas as gpd
import geopy.distance
import itertools
import networkx as nx
import numpy as np


class RandomGraphs():

    def __init__(self):
        '''
        The class 'RandomGraphs' contains methods that produce random graphs
        according to various models.
        '''
        self.radius = 6_371.009 #km
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    @staticmethod
    def compute_spherical_distances(coords):
        '''
        Given a list of (lat, lon) coordinates, computes the great-circle distances
        in kilometers between all pairs of nodes and returns a dictionary.
        '''
        dictionary = {}
        for c in coords:
            dictionary[(c, c)] = 0
        for pair in itertools.combinations(coords, 2):
            dist = geopy.distance.great_circle(pair[0], pair[1]).km
            dictionary[(pair[0], pair[1])] = dist
            dictionary[(pair[1], pair[0])] = dist
        
        return dictionary

    @staticmethod
    def coords_to_cartesian(coords, radius=None):
        '''
        Converts a list of (lat,lon) coordinates into (x,y,z) coordinates on a
        3D sphere with given radius.
        '''
        if radius is None:
            radius = self.radius
        coords = [(np.deg2rad(lat), np.deg2rad(lon)) for (lat,lon) in coords]
        x = [radius*np.cos(lat)*np.cos(lon) for (lat,lon) in coords]
        y = [radius*np.cos(lat)*np.sin(lon) for (lat,lon) in coords]
        z = [radius*np.sin(lat)             for (lat,lon) in coords]

        return (x, y, z)

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
    
    def _uniform_random_points_world(self, n):
        '''
        Returns the (lat,lon)-coordinates of n randomly uniformly distributed
        points on the worldwide landmass.
        '''
        #TODO: Add this.

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

    def ε_neighborhood(self, n, ε, pole_angle=90, pos=None, distances=None):
        '''
        Returns the ε-neighborhood similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        all other nodes within a geodesic distance of ε kilometers. The position
        of nodes can be predefined through 'pos'.
        '''
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:            
            distances = self.compute_spherical_distances(nodes)

        edges = [(x,y) for x in nodes for y in nodes if 0 < distances[(x,y)] < ε]
        for node in nodes:
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))

        return G

    def k_nearest_neighbors(self, n, k, pole_angle=90, max_edgelength=None, pos=None, distances=None):
        '''
        Returns the K-nearest-neighbors similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        its k nearest neighbors. The position of nodes can be predefined
        through 'pos'.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)

        edges = []
        for node in nodes:
            dist = {k: v for k, v in distances.items() if k[0] == node if v > 0}
            if max_edgelength is not None:
                dist = {k: v for k, v in dist.items() if v <= max_edgelength}
            k_smallest_dist = dict(sorted(dist.items(), key = lambda x: x[1])[0:k])
            edges += k_smallest_dist.keys()
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))

        return G

    def mutual_k_nearest_neighbors(self, n, k, pole_angle=90, pos=None, distances=None):
        '''
        Returns the mutual-K-nearest-neighbors similarity graph constructed from
        a set of randomly distributed nodes on the globe. A pair of nodes is
        connected if both nodes are in the k nearest neighbors of the other node.
        The position of nodes can be predefined through 'pos'.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)
        
        edges = []
        for node in nodes:
            dist = {k: v for k, v in distances.items() if k[0] == node if v > 0}
            k_smallest_dist = dict(sorted(dist.items(), key = lambda x: x[1])[0:k])
            edges += k_smallest_dist.keys()
            G.add_node(node, pos=node)
        edges = [(x,y) for (x,y) in edges if (y,x) in edges]
        G.add_edges_from(set(edges))

        return G

    def relative_neighborhood(self, n, λ=1, pole_angle=90, pos=None, distances=None):
        '''
        Returns the Random Neighborhood Graph constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if they are
        at least as close to each other as they are to any other node.
        The position of nodes can be predefined through 'pos'.
        '''
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)
        
        edges = []
        for i in nodes:
            for j in nodes:
                if (i==j):
                    continue
                d_max = [max(distances[(k,i)], distances[(k,j)]) for k in nodes if k!=i and k!=j]
                if distances[(i,j)] <= λ*min(d_max):
                    edges.append((i,j))
        
        for node in nodes:
            G.add_node(node, pos=node)
        G.add_edges_from(set(edges))

        return G

    def minimum_spanning_tree(self, n, pole_angle=90, pos=None, distances=None):
        '''
        Returns the Minimum Spanning Tree constructed from a set of randomly
        distributed nodes on the globe. Nodes are connected into one connected
        component in such a way that the sum of distances of all edges is
        minimized. The position of nodes can be predefined through 'pos'.
        '''
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)

        edges = [(i, j, distances[(i,j)]) for i in nodes for j in nodes]

        for node in nodes:
            G.add_node(node, pos=node)
        G.add_weighted_edges_from(edges)
        MST = nx.minimum_spanning_tree(G)

        return MST

    #TODO: Fix.
    def delaunay_triangulation(self, n, pole_angle=90, pos=None, distances=None):
        '''
        Returns the Delaunay Triangulation constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if their
        corresponding tiles in the Voronoi diagram share an edge. The position
        of nodes can be predefined through 'pos'.
        '''
        G = nx.Graph()
        if pos is None:
            nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        assert len(nodes) >= 4, 'Number of nodes must be at least 4.'
        if distances is None:
            distances = self.compute_spherical_distances(nodes)    

        #Initial triangle
        for n in nodes[:4]:
            G.add_node(n, pos=n)
        edges = [(x,y) for x in nodes[:4] for y in nodes[:4] if y!=x]
        n0, n1, n2, n3 = nodes[:4]
        triangles = [[n0,n1,n2], [n0,n1,n3], [n0,n2,n3], [n1,n2,n3]]

        #Iterative process
        for n in nodes[4:]:
            G.add_node(n, pos=n)
            cavity = []
            for Δ in triangles:
                n1, n2, n3 = Δ
                cartesian_coords = self.coords_to_cartesian([n1,n2,n3,n], radius=1)
                x1, x2, x3, x = cartesian_coords[0]
                y1, y2, y3, y = cartesian_coords[1]
                z1, z2, z3, z = cartesian_coords[2]

                T = np.array([[1 , 1 , 1 , 1],
                              [x1, x2, x3, x],
                              [y1, y2, y3, y],
                              [z1, z2, z3, z]])

                vol = np.linalg.det(T)
                if (vol < 0):
                    cavity += [(n1,n2), (n2,n3), (n1,n3)]
                    triangles.remove(Δ)

            edges = [e for e in cavity if cavity.count(e)==1]
            triangles += [[n, e0, e1] for (e0,e1) in edges]

        edges = [((n1,n2), (n2,n3), (n1,n3)) for (n1,n2,n3) in triangles]
        edges = [i for j in edges for i in j]
        G.add_edges_from(edges)

        return G