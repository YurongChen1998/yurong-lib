% Writen by Yurong Chen 2022-01-26
% https://yurongchen1998.github.io/
clear all; clc; close all;

%% Load MNIST Data
data = load('mnist_test.csv');

labels = data(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785);
images = images/255;

images = images';

%% MDS
[u, s, v] = svd(images);
v = v(:, 1:2);
truncted_sigma = sqrt(diag(s)); truncted_sigma = truncted_sigma(1:2, 1); 
truncted_sigma = diag(truncted_sigma);
proj_data = truncted_sigma * v';
proj_data = proj_data';
scatter(proj_data(:, 1), proj_data(:, 2), 30, labels, 'filled');

%% Isomap
close all;
% step 1: construct the k-nearest graph
K_euclidean_distance_matrix = images' * images;
K_euclidean_distance_matrix = (K_euclidean_distance_matrix - min(K_euclidean_distance_matrix))./max(K_euclidean_distance_matrix);
K_euclidean_distance_matrix = 1./K_euclidean_distance_matrix;
K_euclidean_distance_matrix = K_euclidean_distance_matrix - eye(10000) .* K_euclidean_distance_matrix;
[sort_index_matrx, sort_index] = sort(K_euclidean_distance_matrix, 2);
selected_k_nearest_distance = sort_index_matrx(:, 2:11);
selected_k_nearest_index = sort_index(:, 2:11);
K_euclidean_distance_matrix_new = zeros(10000, 10000);
for i = 1:10000
    for j = 1:10
        K_euclidean_distance_matrix_new(i, selected_k_nearest_index(i, j)) = selected_k_nearest_distance(i, j);
    end
end
K_euclidean_distance_matrix_new = K_euclidean_distance_matrix_new + K_euclidean_distance_matrix_new';

% step2: compute the shortest path length graph
G_distance = distances(graph(K_euclidean_distance_matrix_new));

% step3: eigen-decomposition
H = eye(10000) - ones(10000)./10000;
Kernel_matrix = -0.5 * H * G_distance * H;
[eigvector, eigvalue] = eig(Kernel_matrix);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvector = eigvector(:, index);
truncted_eigvector = eigvector(:, 1:2);
truncted_sigma = eigvalue(1:2, 1); 
truncted_sigma = diag(truncted_sigma);
proj_data = truncted_sigma * truncted_eigvector';
proj_data = proj_data';
scatter(proj_data(:, 1), proj_data(:, 2), 30, labels, 'filled');
