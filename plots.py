import geopandas as gpd
import imageio
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from PIL import Image

from data import DataLoader
from graph import GraphMethods


class PlotMethods():

    def __init__(self, figures_folder='figures'):
        '''
        The 'PlotMethods' class contains methods for creating plots, figures,
        and animations and saving them locally.
        '''
        self.dl = DataLoader()
        self.gm = GraphMethods()

        self.figures_folder = figures_folder
        self.radius = 6_371.009 #km
        self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    @staticmethod
    def switch_coords(pos):
        '''
        Given a node position dictionary in (lat,lon) coordinates, returns the
        same dictionary in (lon,lat) coordinates, and viceversa.
        '''
        if (type(pos) is dict):
            return {node: (l2,l1) for node, (l1,l2) in pos.items()}
        if (type(pos) is list):
            return [(l2,l1) for (l1,l2) in pos]

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
                if (f'{filename}_' in str(file)):
                    os.rename(os.path.join(path,file), os.path.join(frames_dir,file))

        if not keep_frames:
            for file in os.listdir(path):
                if (f'{filename}_' in str(file)):
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

    def plot2D_worldgraph(self, G, figsize=(20,15), file_name='2Dgraph', show=False, show_world=False, fix_wrapping=True):
        '''
        Generates and saves a 2D embedding of Graph G onto a (lat, lon)-plot.
        '''
        path = f'{self.figures_folder}\\plots'
        plt.figure(figsize=figsize)
        ax = plt.axes()
        if show_world:
            self.world.plot(color=[0.9, 0.9, 0.9], figsize=figsize, ax=ax)
        Gw = nx.Graph()
        if fix_wrapping:
            for node in G.nodes():
                Gw.add_node(node, pos=node)
                Gw.add_node((node[0],node[1]+360), pos=(node[0],node[1]+360))
                Gw.add_node((node[0],node[1]-360), pos=(node[0],node[1]-360))
            for edge in G.edges():
                n1 = edge[0]
                n2 = edge[1]
                if (n1[1] > n2[1]):
                    n1 = edge[1]
                    n2 = edge[0]
                if (n2[1]-n1[1] > 180):
                    Gw.add_edge(n2,(n1[0],n1[1]+360))
                    Gw.add_edge(n1,(n2[0],n2[1]-360))
                if (n2[1]-n1[1] <= 180):
                    Gw.add_edge(n1,n2)
        if not fix_wrapping:
            Gw = G.copy()
        pos = self.switch_coords(nx.get_node_attributes(Gw, 'pos'))
        nx.draw(Gw, pos=pos, node_color='black', edge_color='darkgrey', node_size=3, ax=ax)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlim(-180,180)
        plt.ylim(-90,90)
        plt.axis('on')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig(f'{path}\\{file_name}', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plotSP_worldgraph(self, G, figsize=(20,20), file_name='SPgraph', show=False, show_world=False):
        '''
        Generates and saves a 2D embedding of Graph G as a stereographic
        projection of the globe onto a plane from the north pole.
        '''
        path = f'{self.figures_folder}\\plots'
        plt.figure(figsize=figsize)
        ax = plt.axes()
        if show_world:
            world = self.world.explode(index_parts=True)
            polygon_list = list(world[world.name != 'Antarctica'].geometry)
            polygon_list += list(world[world.name == 'Antarctica'].geometry)[:7]
            for polygon in polygon_list:
                coords_2D = self.switch_coords(list(polygon.exterior.coords))
                coords_SP = [self.coords_to_stereographic_projection(c) for c in coords_2D]
                polygon_SP = Polygon(coords_SP, color=[0.9, 0.9, 0.9])
                ax.add_patch(polygon_SP)
            antarctica = list(world[world.name == 'Antarctica'].geometry)[-1]
            coords_2D = self.switch_coords(list(antarctica.exterior.coords))
            coords_2D += [(-90,lon) for lon in range(180,-181,-1)]
            coords_SP = [self.coords_to_stereographic_projection(c) for c in coords_2D]
            polygon_SP = Polygon(coords_SP, color=[0.9, 0.9, 0.9])
            ax.add_patch(polygon_SP)

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

    def plot3D_worldgraph(self, G, figsize=(20,20), file_name='3Dgraph', n_frames=72,
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

    def plot_degreedist(self, degreedist, WG=None, figsize=(10,6), file_name='DegreeDist', show=False):
        '''
        Plots the degree distribution and saves it to the
        'figures' folder. If 'WG' is specified as the worldgraph, will
        overlay the degree distribution of G over that one of the WG.
        '''
        if WG is not None:
            WG_degreedist = [d for n,d in WG.degree() if d!=0]

        path = f'{self.figures_folder}\\degreedist'
        plt.figure(figsize=figsize)
        ax = plt.axes()

        if WG is not None:
            plt.hist(WG_degreedist, bins=np.arange(1,19)-0.5, color='k', alpha=0.1)
            plt.hist(WG_degreedist, bins=np.arange(1,19)-0.5, color='k', histtype='step')
        plt.bar(list(degreedist.keys()), list(degreedist.values()), color='tab:cyan', alpha=0.7, width=0.8, zorder=3)
        ax.set_xticks(range(19))
        plt.grid(axis='y')
        plt.xlabel('Degree')
        plt.tight_layout()
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\{file_name}.png')
        plt.close()