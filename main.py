from collections import Counter
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import time

from data import DataLoader
from graph import GraphMethods
from plots import PlotMethods
from randomgraph import RandomGraphs

dl = DataLoader()
gm = GraphMethods()
pm = PlotMethods()
rg = RandomGraphs()

def create_worldgraph():
    '''
    Creates a Networkx Graph object of the woldgraph.
    '''
    print('Creating the worldgraph.')
    cb = dl.load_countryborders()
    cc = dl.load_countrycentroids()
    WG = gm.countryborders_to_graph(cb, cc)

    return WG

def plot_worldgraph(WG):
    '''
    Plot 2D mercator projection of the worldgraph, a stereographic
    projection of the worldgraph centered around the northpole, 
    and a 3D animated gif of the spinning worldgraph. Saves the plots
    and animation in the 'figures' folder.
    '''
    print('Plotting the worldgraph, saving to \'figures\' folder.')
    pm.plot2D_worldgraph(WG, file_name='WG_2D', show_world=True, fix_wrapping=False)
    pm.plotSP_worldgraph(WG, file_name='WG_SP', show_world=True)
    pm.plot3D_worldgraph(WG, file_name='WG_3D')
    pm.plot_degreedist(None, WG, file_name='WG_degreedist')

def analyze_worldgraph(WG):
    '''
    Returns a list of results of properties of the worldgraph,
    and saves them in the 'results' folder.
    '''
    print('Analyzing the worldgraph, saving to \'results\' folder.')
    gm.write_results(WG, form='text', file_name='WG_results')
    degrees = Counter([d for n,d in WG.degree() if d!=0])

def evaluate_randomgraphs(WG, models, N, num_cores=None):
    '''
    Generates N samples of a selection of random graph models.
    All random graphs have n=171 nodes.
    Returns results of properties to the 'results' folder and
    creates a plot of the average degree distributions in the
    'figures' folder. If 'num_cores' is left unspecified, uses
    all but one core.
    '''
    print('Sampling, evaluating, and plotting random graph models.')
    mean_edges, var_edges, mean_triangles, var_triangles, degreedist = [], [], [], [], []
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()-1
    
    def run_simulation(run, model, **kwargs):
        G = model(**kwargs)
        edges = G.number_of_edges()
        triangles = int(sum(nx.triangles(G).values())/3)
        degree_sequence = sorted((d for n, d in G.degree()), reverse=False)
        degreeCount = Counter(degree_sequence)
        
        t2 = time.time()
        print(f'Run {run+1}/{N} -- Time passed: {int(np.floor(int(t2-t1)/60)):02d}:{int(t2-t1)%60:02d}')
        return (edges, triangles, degreeCount)

    t1 = time.time()
    m = 0
    for model in models:
        m += 1
        print(f'Sampling from graph model {m}/{len(models)}.')
        res = Parallel(n_jobs = num_cores) (delayed(run_simulation)(run, model[0], **model[1]) for run in range(N))
        edges = [r[0] for r in res]
        triangles = [r[1] for r in res]
        degreeCounts = [r[2] for r in res]
                        
        mean_edges.append(np.mean(edges))
        var_edges.append(np.var(edges))
        mean_triangles.append(np.mean(triangles))
        var_triangles.append(np.var(triangles))
        degreeSum = sum(degreeCounts, Counter())
        for item, count in degreeSum.items():
            degreeSum[item] /= N
        pm.plot_degreedist(degreeSum, WG, file_name=f'DegreeDist_{m}')
        degreedist.append(degreeSum)

    for m in range(len(models)):
        print('')
        print(f'Graph model {m+1}/{len(models)} results:')
        print(f'Edges: Mean = {mean_edges[m]}, Var = {var_edges[m]:.4f}.')
        print(f'Triangles: Mean = {mean_triangles[m]}, Var = {var_triangles[m]:.4f}.')
        print(f'Degree Distribution: {degreedist[m]}.')
    return (mean_edges, var_edges, mean_triangles, var_triangles, degreedist)

WG = create_worldgraph()
plot_worldgraph(WG)
#analyze_worldgraph(WG)

models = ((rg.ε_neighborhood, {"n":171, "ε":1900}),
          (rg.k_nearest_neighbors, {"n":171, "k":4, "ε":1900}),
          (rg.relative_neighborhood, {"n":171, "λ":1.15}))

#evaluate_randomgraphs(WG, models, N=100)