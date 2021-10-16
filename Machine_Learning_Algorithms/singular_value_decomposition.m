clear all; close all; clc

M = [1, 0, 0, 0, 2; 
    0, 0, 3, 0, 0;
    0, 0, 0, 0, 0;
    0, 2, 0, 0, 0;];

%% Implementation SVD with eig
MMT = M * M';
[U, SU] = eig(MMT);
[SU,ord] = sort(diag(SU), 'descend');
SU = diag(sqrt(SU));
U = U(:,ord);

MTM = M' * M;
[V, SV] = eig(MTM);
[SV,ord] = sort(diag(SV), 'descend');
SV = diag(sqrt(SV));
V = V(:,ord);

%% Varifiy the implementation
[U1, S, V1] = svd(M);
M_1 = U1*S*V1';
M_2 = U*SV(1:4, 1:5)*V';


