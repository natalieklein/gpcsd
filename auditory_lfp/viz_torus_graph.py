""" 
Visualize torus graph results (computed in Matlab by torus_graph_fit.m).
Creates Figure 3 in the paper.

"""
import scipy.special
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# %% Load results from Matlab
# model fit on original data
mat = scipy.io.loadmat('results/twoprobe_tg.mat')
ntests = 48*(48-1)/2
phi_lfp = mat['phi_hat_lfp']
phi_csd = mat['phi_hat_csd']
# bootstrapped partial PLV
boot_mat = scipy.io.loadmat('results/bootstrap_pplv_tg.mat')

# layer boundaries for superficial/medium/deep
layerbounds = 100*(np.array([11, 16])-1)
layerlabs = ['Superficial', 'Medium', 'Deep']
probe1boundaries = layerbounds/100.
probe2boundaries = layerbounds/100.

# %% Aggregate model fits
# no marginals
phi_lfp = phi_lfp[2*48:]
phi_csd = phi_csd[2*48:]
# no phase sums
phi_lfp = phi_lfp[:(2*int(ntests))]
phi_csd = phi_csd[:(2*int(ntests))]

ntests = 48*(48-1)/2
alpha = 0.01/ntests

counter = 0
csd_pvals = np.nan * np.ones((48, 48))
lfp_pvals = np.nan * np.ones((48, 48))
for i in range(48):
    for j in range(i+1, 48):
        csd_pvals[i, j] = mat['twoprobe_csd_pvals'][counter]
        csd_pvals[j, i] = mat['twoprobe_csd_pvals'][counter]
        lfp_pvals[i, j] = mat['twoprobe_lfp_pvals'][counter]
        lfp_pvals[j, i] = mat['twoprobe_lfp_pvals'][counter]
        counter += 1

csd_pplv = boot_mat['partial_plv_csd'] # (1128, 100)
csd_pplv_mat = np.zeros((48, 48, 100))
counter = 0
for i in range(48):
    for j in range(i+1, 48):
        csd_pplv_mat[i, j, :] = csd_pplv[counter, :]
        counter = counter + 1

# %% Visualize model fit as matrix/graph
plt.rcParams.update({'font.size': 14})
# Visualize between- and within- with colored p values
minlogp = np.min([np.min(np.log10(csd_pvals[~np.isnan(csd_pvals)])), np.min(np.log10(lfp_pvals[~np.isnan(lfp_pvals)]))])
csd_pvals[csd_pvals > 0.1] = np.nan
lfp_pvals[lfp_pvals > 0.1] = np.nan

cmap = matplotlib.cm.get_cmap('YlOrRd_r', 4)

f = plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.imshow(np.log10(csd_pvals), vmin=-5, vmax=-1, cmap=cmap)
plt.axvline(23.5, color='black')
plt.axhline(23.5, color='black')
plt.hlines(probe1boundaries, xmin=0, xmax=23.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.vlines(probe1boundaries, ymin=0, ymax=23.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.hlines(24+probe2boundaries, xmin=24, xmax=47.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.vlines(24+probe2boundaries, ymin=24, ymax=47.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.xlabel('Lateral            Medial', fontsize=16)
plt.ylabel('Medial           Lateral', fontsize=16)
plt.title('CSD')
plt.text(-4.5, -4, 'A', fontsize=20)
plt.subplot(122)
im = plt.imshow(np.log10(lfp_pvals), vmin=-5, vmax=-1, cmap=cmap)
plt.axvline(23.5, color='black')
plt.axhline(23.5, color='black')
plt.hlines(probe1boundaries, xmin=0, xmax=23.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.vlines(probe1boundaries, ymin=0, ymax=23.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.hlines(23.5+probe2boundaries, xmin=24, xmax=47.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.vlines(23.5+probe2boundaries, ymin=24, ymax=47.5, linestyles='dashed', linewidth=2, color='black', alpha=0.8)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.title('LFP')
plt.xlabel('Lateral            Medial', fontsize=16)
plt.ylabel('Medial           Lateral', fontsize=16)
cbar = f.colorbar(im, ax=f.axes, extend='min', ticks=[-5, -4, -3, -2, -1])
cbar.ax.set_title('log10(p)', {'fontsize': 14}, pad=10)
plt.show()

#%% Visualize between-probe CSD graph
csd_pvals[csd_pvals > alpha] = np.nan
lfp_pvals[lfp_pvals > alpha] = np.nan
G = nx.Graph()
G.add_nodes_from(range(1, 25), bipartite=0)
G.add_nodes_from(range(25, 49), bipartite=1)

for i in range(1, 25):
    for j in range(25, 49):
        if ~np.isnan(csd_pvals[i-1, j-1]):
            G.add_edge(i, j)

print('num edges %d'%G.number_of_edges())
left_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)
pos = nx.bipartite_layout(G, left_nodes, aspect_ratio=0.4)
for i in range(1, 49):
    pos[i][1] = -pos[i][1]

options = {'node_color': 'grey', 'node_size': 50}
f = plt.figure(figsize=(2.5, 4))
nx.draw(G, pos, **options)
plt.text(-0.7, 1.05, 'Lateral', fontsize=16)
plt.text(0.2, 1.05, 'Medial', fontsize=16)
plt.text(-0.55, 0.30, 'Superficial', rotation=90, fontsize=16)
plt.text(-0.55, -0.15, 'Med.', rotation=90, fontsize=16)
plt.text(-0.55, -0.75, 'Deep', rotation=90, fontsize=16)
plt.hlines(-2*(probe1boundaries/24 - 0.5), xmin=-0.6, xmax=0.0, linestyles='dashed', color='grey')
plt.hlines(-2*(probe2boundaries/24 - 0.5), xmin=0.0, xmax=0.6, linestyles='dashed', color='grey')
plt.text(-0.7, 1.25, 'B', fontsize=20)
plt.show()

# %% Bootstrap PPLV graph
locs = [5, 13, 20]
pplv = csd_pplv_mat[:24, 24:, :] # (24, 24, 150)
pplv = pplv[locs, :, :]
pplv = pplv[:, locs, :]
pplv_mean = np.mean(pplv, 2)
pplv_se = np.std(pplv, 2)
pplv_lb = np.quantile(pplv, 0.025, 2)

n_nodes = pplv.shape[0]
n_edges = (n_nodes - 1) * n_nodes /2
pval_thresh = 0.005

# Get layer locs in coordinates suitable for graph plot
csd_loc = -2 * (np.array(locs)/24 - 0.5)

# Visualize between-probe CSD graph
G = nx.Graph()
G.add_nodes_from(range(1, n_nodes+1), bipartite=0)
G.add_nodes_from(range(n_nodes+1, 2*n_nodes+1), bipartite=1)

edgelist = []
width_list = []
ci_lb_list = []
linestyle_list = []
for i in range(1, n_nodes+1):
    for j in range(n_nodes+1, 2*n_nodes+1):
        if csd_pvals[i-1, j-n_nodes-1] <= pval_thresh:
            G.add_edge(i, j)
            ci_lb = pplv_lb[i-1, j-n_nodes-1]
            edgelist.append((i, j))
            ci_lb_list.append(ci_lb)
            width_list.append(10 * pplv_mean[i-1, j-n_nodes-1] ** 4)
            if csd_pvals[i-1, j-n_nodes-1] <= 0.001:
                linestyle_list.append('solid')
            else:
                linestyle_list.append('dashed')
ci_lb_ord = np.argsort(ci_lb_list)
edgelist_ordered = [edgelist[i] for i in ci_lb_ord]
width_ordered = [width_list[i] for i in ci_lb_ord]
ci_lb_ordered = [ci_lb_list[i] for i in ci_lb_ord]
linestyle_ordered = [linestyle_list[i] for i in ci_lb_ord]

print('num edges %d'%G.number_of_edges())
left_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)

pos = {}
pt = np.flip(np.linspace(-1, 1, 4))
for i in range(1, 2*n_nodes+1):
    if G.nodes[i]['bipartite'] == 0:
        pos[i] = np.array([-1. / 2., pt[i-1]])
    else:
        pos[i] = np.array([1. / 2., pt[i - n_nodes - 1]])

lab_dict = {1:'S', 2:'M', 3:'D', 4:'S', 5:'M', 6:'D'}

vrange = [0.0, 0.4]
vmin = vrange[0]
vmax = vrange[1]
print('CI lb min %0.2f, max %0.2f' % (np.min(ci_lb_ordered), np.max(ci_lb_ordered)))

#cmap = matplotlib.cm.get_cmap('Blues', 4)
from matplotlib.colors import LinearSegmentedColormap
colors = plt.cm.Blues(np.linspace(0.2, 1.0, 4))
cmap = LinearSegmentedColormap.from_list('name', colors)
cmap = matplotlib.cm.get_cmap(cmap, 3)

f = plt.figure(figsize=(2.5, 5.0))
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=300)
nx.draw_networkx_labels(G, pos, labels=lab_dict, font_size=14, font_color='white', font_weight='bold')
for ei in range(len(edgelist_ordered)):
    nx.draw_networkx_edges(G, pos, edgelist=[edgelist_ordered[ei]], width=7, 
                               edge_color=[ci_lb_ordered[ei]], edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
                               style=linestyle_ordered[ei])
bounds = [0.0, 0.1, 0.2, 0.3, 0.4]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
cb = plt.colorbar(sm, orientation='horizontal', pad=0.07)
cb.ax.set_title('95% Bootstrap CI L.B.', {'fontsize': 14})
plt.text(-0.65, 1.1, 'Lateral', fontsize=16)
plt.text(0.18, 1.1, 'Medial', fontsize=16)
plt.text(-0.8, 1.2, 'C', fontsize=20)
plt.show()
# %%
