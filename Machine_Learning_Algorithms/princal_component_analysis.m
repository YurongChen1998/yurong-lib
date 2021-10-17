clear all; clc; close all;

% 4 features, 5 samples
M = [1, 0, 0, 0, 2; 
    0, 0, 3, 0, 0;
    0, 0, 0, 0, 0;
    0, 2, 0, 0, 0;];

M = M';
M_mean = mean(M);
M_mean = repmat(M_mean, 5, 1);
M = M - M_mean;
M = M';

MMT = M * M';
[V, ev] = eig(MMT);
[ev, ord] = sort(diag(ev), 'descend');
ev = diag(ev);
V = V(:, ord);
M_trans = M'*V;

reduced_V = V(:, [1:2]);
reduced_M = M'*reduced_V;

% varify using pca() function
[V_pca, ev_pca] = pca(M');
M_pca_trans = M'*V_pca;

