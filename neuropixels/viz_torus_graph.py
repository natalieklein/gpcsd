
"""
Visualize torus graphs bootstrap computed by fit_torus_graph.m on Neuropixels data
"""
# %% imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import networkx as nx
import scipy.special

plt.rcParams.update({'font.size': 18})

# %% Load data
np_tg = scipy.io.loadmat('results/neuropixel_tg.mat')
np_tg_boot = scipy.io.loadmat('results/bootstrap_neuropixels.mat')

beta_pvals = np.zeros((8, 8, 2))
counter = 0
for i in range(8):
    for j in range(i+1, 8):
        beta_pvals[i, j, :] = np_tg['beta_pvals'][counter, :]
        beta_pvals[j, i, :] = np_tg['beta_pvals'][counter, :]
        counter += 1
# reorder edges since higher x is at the top of the probe
beta_pvals = beta_pvals[::-1, ::-1, :]

theta_pvals = np.zeros((8, 8, 2))
counter = 0
for i in range(8):
    for j in range(i+1, 8):
        theta_pvals[i, j, :] = np_tg['theta_pvals'][counter, :]
        theta_pvals[j, i, :] = np_tg['theta_pvals'][counter, :]
        counter += 1
theta_pvals = theta_pvals[::-1, ::-1, :]

theta_ci_lb = np.zeros((8, 8, 2))
counter = 0
for i in range(8):
    for j in range(i+1, 8):
        theta_ci_lb[i, j, :] = np.quantile(np_tg_boot['pplv_theta'][counter, :, :], 0.025, axis=0)
        theta_ci_lb[j, i, :] = np.quantile(np_tg_boot['pplv_theta'][counter, :, :], 0.025, axis=0)
        counter += 1
theta_ci_lb = theta_ci_lb[::-1, ::-1, :]

beta_ci_lb = np.zeros((8, 8, 2))
counter = 0
for i in range(8):
    for j in range(i+1, 8):
        beta_ci_lb[i, j, :] = np.quantile(np_tg_boot['pplv_beta'][counter, :, :], 0.025, axis=0)
        beta_ci_lb[j, i, :] = np.quantile(np_tg_boot['pplv_beta'][counter, :, :], 0.025, axis=0)
        counter += 1
beta_ci_lb = beta_ci_lb[::-1, ::-1, :]

pvals_dict = {'beta': beta_pvals, 'theta': theta_pvals}
ci_lb_dict = {'beta':beta_ci_lb, 'theta':theta_ci_lb}

# %%
pval_thresh = 0.001

# Flipped to plot in correct order
csd_loc = {'V1': np.flip(np.array([2260., 2450., 2650., 2785.])), 
           'LM': np.flip(np.array([2215., 2410., 2590., 2720.]))}
csd_loc_graph = {}
for k in csd_loc.keys():
    minx = np.min(csd_loc[k])
    maxx = np.max(csd_loc[k])
    csd_loc_graph[k] = 2 * ((csd_loc[k]-minx)/(maxx-minx) - 0.5)

times = ['0 ms', '70 ms']
n_nodes = 4

f = plt.figure(figsize=[16, 8])
image_counter = 0
for band in ['theta', 'beta']:

    for ti, t in enumerate(times):

        image_counter += 1

        # Visualize between-probe CSD graph
        G = nx.Graph()
        G.add_nodes_from(range(1, n_nodes+1), bipartite=0)
        G.add_nodes_from(range(n_nodes+1, 2*n_nodes+1), bipartite=1)

        edgelist = []
        tstat_list = []
        width_list = []
        ci_lb_list = []
        linestyle_list = []
        for i in range(1, n_nodes+1):
            for j in range(n_nodes+1, 2*n_nodes+1):
                if pvals_dict[band][i-1, j-n_nodes-1, ti] <= pval_thresh:
                    G.add_edge(i, j)
                    ci_lb = ci_lb_dict[band][i-1, j-n_nodes-1, ti]
                    edgelist.append((i, j))
                    ci_lb_list.append(ci_lb)
                    if pvals_dict[band][i-1, j-n_nodes-1, ti] <= 0.0001:
                        linestyle_list.append('solid')
                    else:
                        linestyle_list.append('dashed')
        ci_lb_ord = np.argsort(ci_lb_list)
        edgelist_ordered = [edgelist[i] for i in ci_lb_ord]
        ci_lb_ordered = [ci_lb_list[i] for i in ci_lb_ord]
        linestyle_ordered = [linestyle_list[i] for i in ci_lb_ord]

        print('num edges %d'%G.number_of_edges())
        left_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)
        pos = {}
        pt = np.flip(np.linspace(-1, 1, 4))
        for i in range(1, 2*n_nodes+1):
            if G.nodes[i]['bipartite'] == 0:
                pos[i] = np.array([-1. / 3., pt[i-1]])
            else:
                pos[i] = np.array([1. / 3., pt[i - n_nodes - 1]])

        lab_dict = {1:'2/3', 2:'4', 3:'5', 4:'6', 5:'2/3', 6:'4', 7:'5', 8:'6'}
        
        vmin = 0.2
        vmax = 0.6
        print('CI lb min %0.2f, max %0.2f' % (np.min(ci_lb_ordered), np.max(ci_lb_ordered)))
        from matplotlib.colors import LinearSegmentedColormap
        orig_cmap = plt.cm.Blues
        orig_colors = orig_cmap(np.linspace(vmin, vmax, 100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", orig_colors)
        cmap = matplotlib.cm.get_cmap(cmap, 4)

        plt.subplot(1, 4, image_counter)
        plt.axis('off')
        nx.draw_networkx_nodes(G, pos, node_color='black', node_size=900)
        nx.draw_networkx_labels(G, pos, labels=lab_dict, font_size=18, font_color='white', font_weight='bold')
        for ei in range(len(edgelist_ordered)):
            nx.draw_networkx_edges(G, pos, edgelist=[edgelist_ordered[ei]], width=10, 
                                edge_color=[ci_lb_ordered[ei]], edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
                                style=linestyle_ordered[ei])

        bounds = [0.0, 0.1, 0.3, 0.5, 0.7]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, orientation='horizontal', pad=0.05)
        cb.ax.set_title('95% Bootstrap CI L.B.')
        plt.title('Time %s' % t, fontsize=20)
        if image_counter == 1:
            plt.text(-0.5, 1.15, 'B', fontsize=26)
        if image_counter == 3:
            plt.text(-0.5, 1.15, 'C', fontsize=26)
        plt.text(-0.39, 1.1, 'V1', fontsize=22)
        plt.text(0.3, 1.1, 'LM', fontsize=22)
plt.tight_layout()
plt.show()
# %%
