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

    def __init__(self, figures_folder='figures'):
        '''
        The 'PlotMethods' class contains methods for creating plots, figures,
        and animations and saving them locally.
        '''
        self.cm = ClusteringMethods()
        self.dl = DataLoader()
        self.gm = GraphMethods()

        self.figures_folder = figures_folder
        self.radius = 6_371.009 #km
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    @staticmethod
    def switch_coords(pos_dictionary):
        '''
        Given a node position dictionary in (lat,lon) coordinates, returns the
        same dictionary in (lon,lat) coordinates, and viceversa.
        '''
        return {node: (l2,l1) for node, (l1,l2) in pos_dictionary.items()}

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
    
    @staticmethod
    def _animate_frames(path, filename, keep_frames=False):
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

    def save_clustering_fig(self, G, colormap, figsize, path, file_name='clustering_map', title=None):
        '''
        Given a graph G of countries, and a colormap describing a clustering of
        the countries, generates and saves a worldmap showing countries in their
        assigned colors.
        '''
        country_name = self.dl.convert_alpha2_to_name()
        ax = self.world.plot(color='lightgrey', figsize=figsize)
        plt.tight_layout()
        for node in G.nodes:
            if (country_name[node] in list(self.world.name)):
                self.world[self.world.name == country_name[node]].plot(color=colormap[node], ax=ax)
        self.world[self.world.name == 'Somaliland'].plot(color=colormap['SO'], ax=ax)
        self.world[self.world.name == 'N. Cyprus'].plot(color=colormap['CY'], ax=ax)
        ax.set_title(title)

        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\{file_name}')
        plt.close()

    def cluster_world(self, maxK, figsize, type='BC'):
        '''
        Generates and saves plots of different clustering methods applied
        to the bordering countries graph with various amounts of clusters.
        For methods of clustering where the number of clusters k can be specified,
        generates all clusterings for k in the interval [2, maxK].
        The type of clustering map can be:
            - "BC": includes only the biggest component of the Graph;
            - "NS": includes all nodes except singletons (islands) in the Graph.
        '''
        assert type in ['BC', 'NS'], 'Type of clustering map must be "BC" or "NS".'
        cb = self.dl.load_countryborders()
        G = self.gm.countryborders_to_graph(cb)
        if (type == 'BC'):
            G = self.gm.biggest_component(G)
        if (type == 'NS'):
            G = self.gm.delete_singletons(G)
        minK = nx.number_connected_components(G)+1
        assert maxK >= minK, f'maxK must be at least {minK} (connected components of G).'

        for k in range(minK,maxK+1):
            clusters_GM = self.cm.girvan_newman(G, k)
            colormap_GM = self.cm.generate_coloring(G, clusters_GM)
            self.save_clustering_fig(G, colormap_GM,
                                        figsize=figsize,
                                        path=f'{self.figures_folder}\\cluster_world\\GM',
                                        file_name=f'clustering_{k}',
                                        title=f'Girvan Newman\n{k} clusters')

            clusters_SP_un = self.cm.spectral(G, k, version='unnormalized')
            colormap_SP_un = self.cm.generate_coloring(G, clusters_SP_un)
            self.save_clustering_fig(G, colormap_SP_un,
                                        figsize=figsize,
                                        path=f'{self.figures_folder}\\cluster_world\\SP_un',
                                        file_name=f'clustering_{k}',
                                        title=f'Unnormalized spectral clustering\n{k} clusters')

            clusters_SP_SM = self.cm.spectral(G, k, version='normalized-SM')
            colormap_SP_SM = self.cm.generate_coloring(G, clusters_SP_SM)
            self.save_clustering_fig(G, colormap_SP_SM,
                                        figsize=figsize,
                                        path=f'{self.figures_folder}\\cluster_world\\SP_SM',
                                        file_name=f'clustering_{k}',
                                        title=f'Symmetrically normalized spectral clustering\n{k} clusters')

            clusters_SP_NJW = self.cm.spectral(G, k, version='normalized-NJW')
            colormap_SP_NJW = self.cm.generate_coloring(G, clusters_SP_NJW)
            self.save_clustering_fig(G, colormap_SP_NJW,
                                        figsize=figsize,
                                        path=f'{self.figures_folder}\\cluster_world\\SP_NJW',
                                        file_name=f'clustering_{k}',
                                        title=f'Left normalized spectral clustering\n{k} clusters')

    def cluster_world_dist(self, maxK, figsize, type='BC'):
        '''
        Generates and saves plots of different clustering methods applied
        to the bordering countries graph (weighted with distances between country
        centroids) with various amounts of clusters. For methods of clustering
        where the number of clusters k can be specified, generates all clusterings
        for k in the interval [2, maxK].
        The type of clustering map can be:
            - "BC": includes only the biggest component of the Graph;
            - "NS": includes all nodes except singletons (islands) in the Graph.
        '''
        assert type in ['BC', 'NS'], 'Type of clustering map must be "BC" or "NS".'
        cb = self.dl.load_countryborders()
        G = self.gm.countryborders_to_graph(cb)
        if (type == 'BC'):
            G = self.gm.biggest_component(G)
        if (type == 'NS'):
            G = self.gm.delete_singletons(G)
        minK = nx.number_connected_components(G)+1
        assert maxK >= minK, f'maxK must be at least {minK} (connected components of G).'

        for k in range(2,maxK+1):
            #TODO: Add weighted version of GN-clustering.

            clusters_SP_un = self.cm.spectral(G, k, version='unnormalized')
            colormap_SP_un = self.cm.generate_coloring(G, clusters_SP_un)
            self.save_clustering_fig(G, colormap_SP_un,
                                        figsize=figsize,
                                        path=f'{self.figures_folder}\\cluster_world_dist\\SP_un',
                                        file_name=f'clustering_{k}',
                                        title=f'Unnormalized spectral clustering\n{k} clusters')

            clusters_SP_SM = self.cm.spectral(G, k, version='normalized-SM')
            colormap_SP_SM = self.cm.generate_coloring(G, clusters_SP_SM)
            self.save_clustering_fig(G, colormap_SP_SM,
                                        figsize=figsize,                                     
                                        path=f'{self.figures_folder}\\cluster_world_dist\\SP_SM',
                                        file_name=f'clustering_{k}',
                                        title=f'Symmetrically normalized spectral clustering\n{k} clusters')

            clusters_SP_NJW = self.cm.spectral(G, k, version='normalized-NJW')
            colormap_SP_NJW = self.cm.generate_coloring(G, clusters_SP_NJW)
            self.save_clustering_fig(G, colormap_SP_NJW,
                                        figsize=figsize,                                     
                                        path=f'{self.figures_folder}\\cluster_world_dist\\SP_NJW',
                                        file_name=f'clustering_{k}',
                                        title=f'Left normalized spectral clustering\n{k} clusters')

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
        path = f'{self.figures_folder}\\road_network'
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

    def plot2D_worldgraph(self, G, figsize, file_name='2Dgraph', show=False, show_world=True):
        '''
        Generates and saves a 2D embedding of Graph G onto a (lat, lon)-plot.
        '''
        path = f'{self.figures_folder}\\plots'
        plt.figure(figsize=figsize)
        if show_world:
            ax = self.world.plot(color='lightgrey', figsize=figsize)
        if not show_world:
            ax = plt.axes()
        pos = self.switch_coords(nx.get_node_attributes(G, 'pos'))
        nx.draw(G, pos=pos, node_color='black', edge_color='darkgrey', node_size=3, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.axis('on')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig(f'{path}\\{file_name}')
        if show:
            plt.show()
        plt.close()

    def plotSP_worldgraph(self, G, figsize, file_name='SPgraph', show=False):
        '''
        Generates and saves a 2D embedding of Graph G as a stereographic
        projection of the globe onto a plane from the north pole.
        '''
        path = f'{self.figures_folder}\\plots'
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
        plt.tight_layout()
        plt.savefig(f'{path}\\{file_name}')
        if show:
            plt.show()
        plt.close()

    def plot3D_worldgraph(self, G, figsize, file_name='3Dgraph', n_frames=72,
                                radius=None, keep_frames=False):
        '''
        Generates and saves a 3D embedding of Graph G with given (lat, lon)
        coordinates for nodes on a 3D sphere of given radius.
        '''
        assert 1 <= n_frames <= 1000, 'The number of frames must be in the interval [1, 1000].'
        if radius is None:
            radius = self.radius
        path = f'{self.figures_folder}\\plots'
        plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')

        pos = nx.get_node_attributes(G, 'pos')
        x, y, z = self.coords_to_cartesian(list(pos.values()), radius)
        ax.scatter3D(x, y, z, s=3, color='black')
        ax.set_xlim3d(-radius, radius)
        ax.set_ylim3d(-radius, radius)
        ax.set_zlim3d(-radius, radius)
        ax.axis('equal')
        ax.set_axis_off()

        for edge in G.edges:
            n0 = edge[0]
            n1 = edge[1]
            x0, y0, z0 = self.coords_to_cartesian([pos[n0]], radius)
            x1, y1, z1 = self.coords_to_cartesian([pos[n1]], radius)
            ax.plot([x0, x1], [y0, y1], [z0,z1], color='lightgrey')
        
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(n_frames):
            ax.view_init(elev=10, azim=i*360/n_frames)
            plt.savefig(f'{path}\\{file_name}_{i:03}.png', bbox_inches='tight', pad_inches=-2.5)
        plt.close()

        self._animate_frames(path=path, filename=file_name, keep_frames=keep_frames)