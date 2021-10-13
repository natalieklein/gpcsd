% Torus graph model fitting and bootstrapping using Octave or Matlab
% Loads phase files computed by fit_gpcsd_baseline.py
% DEPRECATED; use torus_graph_fit.py which uses a python version of Torus Graphs.

% Clone torus graphs from https://github.com/natalieklein/torus-graphs
% For Octave (warning: slower than Matlab):
%   Switch to octave branch:
%       git checkout octave
%   See octave_setup.m in torus-graphs directory for dependencies

pkg load statistics
pkg load mapping

addpath(genpath('torus-graphs/')) % may need to change path to point to local files

% Load files generated by fit_gpcsd_baseline.py
load('results/csd_lfp_filt_phases_lateral.mat'); % may need to change path to point to local files
lateral_phases_lfp = lfp;
lateral_phases_csd = csd;

load('results/csd_lfp_filt_phases_medial.mat'); % may need to change path to point to local files
medial_phases_lfp = lfp;
medial_phases_csd = csd;

% fit phase differences submodel
submodels = false(1,3);
submodels(1,2) = true;

% CSD between probe
X = [lateral_phases_csd; medial_phases_csd];
[TG, edges, phi_hat, inference] = torus_graphs(X, [], [], submodels);
twoprobe_csd_pvals = edges.p_vals;
fprintf('Twoprobe CSD TG has %d edges bonf 0.001\n', sum(twoprobe_csd_pvals < 0.001/(24*24)))
phi_hat_csd = phi_hat;

% LFP between probe 
X = [lateral_phases_lfp; medial_phases_lfp];
[TG, edges, phi_hat, inference] = torus_graphs(X, [], [], submodels);
twoprobe_lfp_pvals = edges.p_vals;
fprintf('Twoprobe LFP TG has %d edges bonf 0.001\n', sum(twoprobe_lfp_pvals < 0.001/(24*24)))
phi_hat_lfp = phi_hat;

% Save results -- may need to change path to point to local files
save('results/twoprobe_tg.mat','twoprobe_csd_pvals','twoprobe_lfp_pvals','phi_hat_lfp','phi_hat_csd', '-v7')

% bootstrapping CSD
ntrials = size(medial_phases_lfp, 2);
nboot = 100
partial_plv_csd = zeros(1128, nboot);
for bi = 1:nboot
     fprintf('bootstrap sample %d of %d\n', bi, nboot)
     binds = randsample(ntrials, ntrials, true);
     X = [lateral_phases_csd(:, binds); medial_phases_csd(:, binds)];
     [TG, edges, phi_hat, inference] = torus_graphs(X, [], [], submodels);
    partial_plv_csd(:, bi) = edges.cond_coupling_coeff;
end
% Save results -- may need to change path to point to local files
save('results/bootstrap_pplv_tg.mat',  'partial_plv_csd')