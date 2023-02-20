import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
import os
from PIL import Image

from clustering import ClusteringMethods
from data import DataLoader
from graph import GraphMethods


class PlotMethods():

    def __init__(self):
        '''
        The 'PlotMethods' class contains methods for creating plots, figures,
        and animations and saving them locally.
        '''
        self.cm = ClusteringMethods()
        self.dl = DataLoader()
        self.gm = GraphMethods()

    @staticmethod
    def switch_coords(pos_dictionary):
        '''
        Given a node position dictionary in (lat,lon) coordinates, returns the
        same dictionary in (lon,lat) coordinates, and viceversa.
        '''
        return {node: (l2,l1) for node, (l1,l2) in pos_dictionary.items()}

    @staticmethod
    def coords_to_cartesian(coords, radius=None):
        '''
        Converts a list of (lat,lon) coordinates into (x,y,z) coordinates on a
        3D sphere with given radius.
        '''
        if radius is None:
            radius = 6_371 #km
        coords = [(np.deg2rad(lat), np.deg2rad(lon)) for (lat,lon) in coords]
        x = [radius*np.cos(lat)*np.cos(lon) for (lat,lon) in coords]
        y = [radius*np.cos(lat)*np.sin(lon) for (lat,lon) in coords]
        z = [radius*np.sin(lat)             for (lat,lon) in coords]
        return (x, y, z)

    @staticmethod
    def coords_to_stereographic_projection(coords):
        '''
        Converts (lat,lon) coordinates into (x,y) coordinates on a stereographic
        projection of the globe onto a plane from the northpole. The positive
        x-axis is the prime meridian (longitude 0).
        '''
        lat = coords[0]
        lon = coords[1]
        #polar coordinates
        r = 90-lat
        θ = np.deg2rad(lon)
        #cartesian coordinates on SP
        x = r*np.cos(θ)
        y = r*np.sin(θ)
        return (x,y)

    def save_clustering_fig(self, G, colormap, path, filename, title, figsize):
        '''
        Given a graph G of countries, and a colormap describing a clustering of
        the countries, generates and saves a worldmap showing countries in their
        assigned colors.
        '''
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        country_name = self.dl.convert_alpha2_to_name()
        ax = world.plot(color='lightgrey', figsize=figsize)
        plt.tight_layout()
        for node in G.nodes:
            world[world.name == country_name[node]].plot(color=colormap[node], ax=ax)
        world[world.name == 'Somaliland'].plot(color=colormap['SO'], ax=ax)
        world[world.name == 'N. Cyprus'].plot(color=colormap['CY'], ax=ax)
        ax.set_title(title)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\{filename}')
        plt.close()

    def cluster_world(self, maxK, figsize):
        '''
        Generates and saves plots of different clustering methods applied
        to the bordering countries graph with various amounts of clusters.
        For methods of clustering where the number of clusters k can be specified,
        generates all clusterings for k in the interval [2, maxK].
        '''
        cb = self.dl.load_countryborders()
        G = self.gm.countryborders_to_graph(cb)
        G = self.gm.biggest_component(G)

        clusters_HCS, k_HCS = self.cm.highly_connected_subgraphs(G)
        colormap_HCS = self.cm.generate_coloring(G, clusters_HCS)
        self.save_clustering_fig(G, colormap_HCS,
                                    path='figures\\cluster_world\\HCS',
                                    filename=f'clustering_{k_HCS}',
                                    title=f'Highly Connected Subgraphs\n{k_HCS} clusters',
                                    figsize=figsize)

        for k in range(2,maxK+1):
            clusters_GM = self.cm.girvan_newman(G, k)
            colormap_GM = self.cm.generate_coloring(G, clusters_GM)
            self.save_clustering_fig(G, colormap_GM,
                                        path='figures\\cluster_world\\GM',
                                        filename=f'clustering_{k}',
                                        title=f'Girvan Newman\n{k} clusters',
                                        figsize=figsize)

            clusters_SP_un = self.cm.spectral(G, k, version='unnormalized')
            colormap_SP_un = self.cm.generate_coloring(G, clusters_SP_un)
            self.save_clustering_fig(G, colormap_SP_un,
                                        path='figures\\cluster_world\\SP_un',
                                        filename=f'clustering_{k}',
                                        title=f'Unnormalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

            clusters_SP_SM = self.cm.spectral(G, k, version='normalized-SM')
            colormap_SP_SM = self.cm.generate_coloring(G, clusters_SP_SM)
            self.save_clustering_fig(G, colormap_SP_SM,
                                        path='figures\\cluster_world\\SP_SM',
                                        filename=f'clustering_{k}',
                                        title=f'Symmetrically normalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

            clusters_SP_NJW = self.cm.spectral(G, k, version='normalized-NJW')
            colormap_SP_NJW = self.cm.generate_coloring(G, clusters_SP_NJW)
            self.save_clustering_fig(G, colormap_SP_NJW,
                                        path='figures\\cluster_world\\SP_NJW',
                                        filename=f'clustering_{k}',
                                        title=f'Left normalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

    #TODO: Add weighted versions of HCS and Girvan-Newman clustering
    def cluster_world_dist(self, maxK, figsize):
        '''
        Generates and saves plots of different clustering methods applied
        to the bordering countries graph (weighted with distances between country
        centroids) with various amounts of clusters. For methods of clustering
        where the number of clusters k can be specified, generates all clusterings
        for k in the interval [2, maxK].
        '''
        cb = self.dl.load_countryborders()
        centroids = self.dl.load_countrycentroids()
        G = self.gm.countryborders_to_weightedgraph(cb, centroids)
        G = self.gm.biggest_component(G)

        for k in range(2,maxK+1):
            clusters_SP_un = self.cm.spectral(G, k, version='unnormalized')
            colormap_SP_un = self.cm.generate_coloring(G, clusters_SP_un)
            self.save_clustering_fig(G, colormap_SP_un,
                                        path='figures\\cluster_world_dist\\SP_un',
                                        filename=f'clustering_{k}',
                                        title=f'Unnormalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

            clusters_SP_SM = self.cm.spectral(G, k, version='normalized-SM')
            colormap_SP_SM = self.cm.generate_coloring(G, clusters_SP_SM)
            self.save_clustering_fig(G, colormap_SP_SM,
                                        path='figures\\cluster_world_dist\\SP_SM',
                                        filename=f'clustering_{k}',
                                        title=f'Symmetrically normalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

            clusters_SP_NJW = self.cm.spectral(G, k, version='normalized-NJW')
            colormap_SP_NJW = self.cm.generate_coloring(G, clusters_SP_NJW)
            self.save_clustering_fig(G, colormap_SP_NJW,
                                        path='figures\\cluster_world_dist\\SP_NJW',
                                        filename=f'clustering_{k}',
                                        title=f'Left normalized spectral clustering\n{k} clusters',
                                        figsize=figsize)

    def road_network(self, country, figsize, version='all'):
        '''
        Generates and saves a planar embedding of the road network worldwide or
        of a country of choice. Choices of 'versions' are:
            - normal: full road network;
            - smooth: stretches of road are simplified to their endpoints;
            - weighted: the lengths of all roads are displayed;
            - all: saves all three of the above options.
        '''
        roads = self.dl.load_roads(country=country, continent=None)
        path = 'figures\\road_network'
        if not os.path.exists(path):
            os.makedirs(path)

        if (version == 'all' or version == 'normal'):
            G = self.gm.roadnetwork_to_graph(roads, weighted=False)
            plt.figure(figsize=figsize)
            pos = self.switch_coords(nx.get_node_attributes(G, 'pos'))
            nx.draw(G, pos=pos, edge_color='lightgrey', node_size=3)
            country = country.replace('.', '')
            plt.savefig(f'{path}\\{country}')
            plt.close()

        if (version == 'all' or version == 'smooth'):
            G = self.gm.roadnetwork_to_graph(roads, weighted=False)
            G_smooth = self.gm.smoothen_graph(G)
            plt.figure(figsize=figsize)
            pos = self.switch_coords(nx.get_node_attributes(G, 'pos'))
            nx.draw(G_smooth, pos=pos, edge_color='lightgrey', node_size=3)
            country = country.replace('.', '')
            plt.savefig(f'{path}\\{country}_smooth')
            plt.close()

        if (version == 'all' or version == 'weighted'):
            G_weighted = self.gm.roadnetwork_to_graph(roads, weighted=True)
            plt.figure(figsize=figsize)
            nx.draw(G_weighted, pos=nx.get_node_attributes(G_weighted, 'pos'), edge_color='lightgrey', node_size=3)
            pos = self.switch_coords(nx.get_node_attributes(G, 'pos'))
            labels = nx.get_edge_attributes(G_weighted, 'weight')
            labels = {e: "%.2f" % w for e,w in labels.items()}
            nx.draw_networkx_edge_labels(G_weighted, pos=pos, edge_labels=labels)
            country = country.replace('.', '')
            plt.savefig(f'{path}\\{country}_weighted')
            plt.close()

    def plot2D_worldgraph(self, G, figsize, filename='2Dgraph', show=False):
        '''
        Generates and saves a 2D embedding of Graph G onto a (lat, lon)-plot.
        '''
        path = 'figures\\plot2D_graph'
        plt.figure(figsize=figsize)
        ax = plt.axes()
        pos = self.switch_coords(nx.get_node_attributes(G, 'pos'))
        nx.draw(G, pos=pos, node_color='black', edge_color='lightgrey', node_size=3, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.axis('on')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\{filename}')
        if show:
            plt.show()
        plt.close()

    def plotSP_worldgraph(self, G, figsize, filename='SPgraph', show=False):
        '''
        Generates and saves a 2D embedding of Graph G as a stereographic
        projection of the globe onto a plane from the north pole.
        '''
        path = 'figures\\plot2D_graph'
        plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.axhline(y=0, color='lightgrey', lw=1)
        ax.axvline(x=0, color='lightgrey', lw=1)
        southpole = Circle((0,0), 180, color='grey', lw=0.5, fill=False)
        equator = Circle((0,0), 90, color='grey', lw=0.5, fill=False)
        parallels = [Circle((0,0), r, color='lightgrey', lw=0.5, fill=False) for r in range(0,180,10)]
        for circle in parallels:
            ax.add_patch(circle)
        ax.add_patch(southpole)
        ax.add_patch(equator)
        plt.xlim(-190, 190)
        plt.ylim(-190, 190)

        pos = nx.get_node_attributes(G, 'pos')
        SP_pos = {node: self.coords_to_stereographic_projection(coords) for node, coords in pos.items()}
        nx.draw(G, pos=SP_pos, node_color='black', edge_color='grey', node_size=3, ax=ax)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\{filename}')
        if show:
            plt.show()
        plt.close()

    def plot3D_worldgraph(self, G, figsize, filename='3Dgraph', n_frames=72,
                                animate=True, radius=None, show=False, keep_frames=False,
                                title=None, figtext=None):
        '''
        Generates and saves a 3D embedding of Graph G with given (lat, lon)
        coordinates for nodes on a 3D sphere of given radius.
        '''
        assert 1 <= n_frames <= 1000, 'The number of frames must be in the interval [1, 1000].'
        if radius is None:
            radius = 6_371 #km
        path = 'figures\\plot3D_graph'
        plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        x, y, z = self.coords_to_cartesian(G.nodes(), radius)
        ax.scatter3D(x, y, z, s=3, color='black')
        for edge in G.edges:
            n0 = edge[0]
            n1 = edge[1]
            x0, y0, z0 = self.coords_to_cartesian([n0], radius)
            x1, y1, z1 = self.coords_to_cartesian([n1], radius)
            ax.plot([x0, x1], [y0, y1], [z0,z1], color='lightgrey')
        ax.set_xlim3d(-radius, radius)
        ax.set_ylim3d(-radius, radius)
        ax.set_zlim3d(-radius, radius)
        ax.dist = 7
        ax.axis('equal')
        ax.set_axis_off()
        plt.title(title, fontdict={'fontsize': 45})
        plt.figtext(0.5, 0.1, figtext, fontdict={'fontsize': 25}, ha='center')
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(n_frames):
            ax.view_init(elev=10, azim=i*360/n_frames)
            plt.savefig(f'{path}\\{filename}_{i:03}.png')
        if show:
            plt.show()
        plt.close()
        if animate:
            self._animate_frames(path=path, filename=filename, keep_frames=keep_frames)

    def _animate_frames(self, path, filename, keep_frames=False):
        '''
        Loads a set of frames, turns them into an animated gif file, and saves
        the gif.
        '''
        frames = []
        for file in os.listdir(path):
            if (f'{filename}_' in str(file)):
                frames.append(Image.open(os.path.join(path,file)))
        imageio.mimsave(f'{path}\\{filename}.gif', frames)
        if keep_frames:
            frames_dir = f'{path}\\{filename}_frames'
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            for file in os.listdir(path):
                if file.endswith(".png"):
                    os.rename(os.path.join(path,file), os.path.join(frames_dir,file))
        if not keep_frames:
            for file in os.listdir(path):
                if file.endswith(".png"):
                    os.remove(os.path.join(path,file))
        