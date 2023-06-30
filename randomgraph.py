import geopandas as gpd
import geopy.distance
import itertools
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point


class RandomGraphs():

    def __init__(self):
        '''
        The class 'RandomGraphs' contains methods that produce random graphs
        according to various models.
        '''
        self.radius = 6_371.009 #km
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def compute_midpoint(self, coords_1, coords_2):
        '''
        Given two sets of (lat, lon) coordinates, computes the midpoint in
        (lat, lon) coordinates.
        '''
        assert coords_1 + coords_2 != [0,0], 'Midpoint of antipodal points is undefined.'
        p = [(x+y)/2 for (x,y) in self.coords_to_cartesian([coords_1, coords_2])]
        midpoint = [x*self.radius/np.linalg.norm(p) for x in p]

        return self.cartesian_to_coords(midpoint)
    
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

    def coords_to_cartesian(self, coords, radius=None):
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
    
    def cartesian_to_coords(self, cartesian, radius=None):
        '''
        Converts a (x,y,z) coordinate on a 3D sphere with given radius into a
        (lat, lon) coordinate.
        '''
        if radius is None:
            radius = self.radius
        x, y, z = cartesian
        lat = np.arcsin(z/radius)
        if (z != radius):
            i = max(min(x/(radius*np.sqrt(1-(z/radius)**2)),1),-1)
            lon = np.arccos(i)*np.sign(y)
        else:
            lon = 0 #north/south pole -> lon undefined

        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
        return (lat, lon)

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
    
    def _uniform_random_points_world(self, n, pole_angle=90):
        '''
        Returns the (lat,lon)-coordinates of n randomly uniformly distributed
        points on the worldwide landmass.
        '''
        world = self.world.explode(index_parts=True)
        polygon_list = list(world.geometry)
        final_node_list = []

        while (len(final_node_list) < n):
            rem_points = n - len(final_node_list)
            node_list = self._uniform_random_points_sphere(n=int(3.5*rem_points), pole_angle=pole_angle)
            for node in node_list:
                point = Point(node[1], node[0])
                for polygon in polygon_list:
                    if (point.within(polygon)):
                        final_node_list.append(node)
                        break
        
        return final_node_list
    
    def check_edge_on_land(self, edge, div):
        '''
        Given an edge of the graph, subdivides the edge into div+1 evenly
        spaces points andchecks whether they all lie on the world landmass.
        If so, returns true.
        '''
        x, y, z = self.coords_to_cartesian(edge)
        Δx = x[1]-x[0]
        Δy = y[1]-y[0]
        Δz = z[1]-z[0]
        p_list = [(x[0] + Δx*i/div, y[0] + Δy*i/div, z[0] + Δz*i/div) for i in range(div+1)]
        norms = [np.linalg.norm(p) for p in p_list]
        p_list_norm = [(x*self.radius/norm, y*self.radius/norm, z*self.radius/norm) for ((x,y,z),norm) in zip(p_list,norms)]
        p_coords = [self.cartesian_to_coords(p) for p in p_list_norm]

        world = self.world.explode(index_parts=True)
        polygon_list = list(world.geometry)
        for coord in p_coords:
            success = False
            point = Point(coord[1], coord[0])
            for polygon in polygon_list:
                if (point.within(polygon)):
                    success = True
                    break
            if not success:
                return False
        return True

    def ε_neighborhood(self, n, ε, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the ε-neighborhood similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        all other nodes within a geodesic distance of ε kilometers.
        '''
        def __repr__(self):
            return 'hehe function created by awesome programmer'
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
                nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:            
            distances = self.compute_spherical_distances(nodes)

        edges = [(x,y) for x in nodes for y in nodes if 0 < distances[(x,y)] < ε]
        for node in nodes:
            G.add_node(node, pos=node)
        edges = set(edges)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)

        return G

    def k_nearest_neighbors(self, n, k, ε, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the K-nearest-neighbors similarity graph constructed from a set
        of randomly distributed nodes on the globe. Every node is connected to
        its k nearest neighbors.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
                nodes = self._uniform_random_points_sphere(n, pole_angle)
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)

        edges = []
        for node in nodes:
            dist = {k: v for k, v in distances.items() if k[0] == node if v > 0}
            if ε is not None:
                dist = {k: v for k, v in dist.items() if v <= ε}
            k_smallest_dist = dict(sorted(dist.items(), key = lambda x: x[1])[0:k])
            edges += k_smallest_dist.keys()
            G.add_node(node, pos=node)
        
        edges = set(edges)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)

        return G

    def mutual_k_nearest_neighbors(self, n, k, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the mutual-K-nearest-neighbors similarity graph constructed from
        a set of randomly distributed nodes on the globe. A pair of nodes is
        connected if both nodes are in the k nearest neighbors of the other node.
        '''
        assert 1 <= k <= n-1, 'k must take values in the interval [1, n-1].'
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
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
        edges = set(edges)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)

        return G

    def relative_neighborhood(self, n, λ=1, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the Random Neighborhood Graph constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if they are
        at least as close to each other as they are to any other node.
        '''
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
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
        edges = set(edges)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)

        return G

    def gabriel(self, n, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the Gabriel Graph constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if there
        are no other nodes within the smallest circle passing through
        both nodes. For the sake of speed, the definition in terms of distances
        is applied, which only holds for the plane but is accurate enough
        on the sphere.
        '''
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
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
                d = [np.sqrt(distances[(k,i)]**2 + distances[(k,j)]**2) for k in nodes if k!=i and k!=j]
                if distances[(i,j)] <= min(d):
                    edges.append((i,j))
        
        for node in nodes:
            G.add_node(node, pos=node)
        edges = set(edges)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)

        return G       

    def minimum_spanning_tree(self, n, pole_angle=90, pos=None, distances=None):
        '''
        Returns the Minimum Spanning Tree constructed from a set of randomly
        distributed nodes on the globe. Nodes are connected into one connected
        component in such a way that the sum of distances of all edges is
        minimized.
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

    def delaunay_triangulation(self, n, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the Delaunay Triangulation constructed from a set of randomly
        distributed nodes on the globe. A pair of nodes is connected if their
        corresponding tiles in the Voronoi diagram share an edge.
        '''
        G = nx.Graph()
        if pos is None:
            if land_filter:
                nodes = self._uniform_random_points_world(n, pole_angle)
            if not land_filter:
                nodes = self._uniform_random_points_sphere(n, pole_angle)
        nodes = [(round(lat,6),round(lon,6)) for (lat,lon) in nodes] # quick fix for floating errors
        if pos is not None:
            nodes = pos
        if distances is None:
            distances = self.compute_spherical_distances(nodes)

        virtual_nodes = [(lat,lon-360) for (lat,lon) in nodes]+nodes+[(lat,lon+360) for (lat,lon) in nodes]
        virtual_triangles = [(virtual_nodes[a], virtual_nodes[b], virtual_nodes[c]) for (a,b,c) in Delaunay(virtual_nodes).simplices]
        virtual_edges = [((a,b), (b,c), (c,a)) for (a,b,c) in virtual_triangles]
        virtual_edges = set([i for j in virtual_edges for i in j])

        edges = []
        for edge in virtual_edges:
            n1 = edge[0]
            n2 = edge[1]
            if (n1[1] > n2[1]):
                n1 = edge[1]
                n2 = edge[0]
            if (n1[1] < -180 and n2[1] >= -180):
                edges.append((n2,(n1[0],round(n1[1]+360,6))))
            if (n2[1] >= 180 and n1[1] < 180):
                edges.append((n1,(n2[0],round(n2[1]-360,6))))
            if (n1[1] >= -180 and n2[1] < 180):
                edges.append((n1,n2))

        for node in nodes:
            G.add_node(node, pos=node)
        if land_filter:
            for edge in edges:
                if (self.check_edge_on_land(edge, div=10)):
                    G.add_edge(edge[0], edge[1])
        if not land_filter:
            G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))

        return G
    
    def beta_skeleton(self, n, β, pole_angle=90, land_filter=False, pos=None, distances=None):
        '''
        Returns the Beta Skeleton Graph constructed from a set of randomly
        distributed nodes on the globe. For the sake of speed, some 
        simplifications are made, such as when computing the coordinates
        of points c1 and c2.
        '''
        assert 1 <= β <= 2, 'Beta takes values in the interval [1,2].'
        G = self.gabriel(n=n, pole_angle=pole_angle, land_filter=land_filter, pos=pos, distances=distances)
        nodes = G.nodes()

        for e in G.edges:
            i, j = e
            r = β/2*(geopy.distance.great_circle(i,j).km)
            c1 = ((1-β/2)*i[0] + β/2*j[0], (1-β/2)*i[1] + β/2*j[1])
            c2 = ((1-β/2)*j[0] + β/2*i[0], (1-β/2)*j[1] + β/2*i[1])
            d_max = [max(geopy.distance.great_circle(k,c1).km, geopy.distance.great_circle(k,c2).km) for k in nodes if k!=i and k!=j]
            if r > min(d_max):
                G.remove_edge(i,j)

        return G