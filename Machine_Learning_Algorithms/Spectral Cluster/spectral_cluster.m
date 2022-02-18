% Writen by Yurong Chen 2022-01-30
% https://yurongchen1998.github.io/
clear all; clc; close all;

%% Load MNIST Data
data = load('mnist_test.csv');

labels = data(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785); clear data;
images = images/255;
images = images';

zero_data = images(:, find(labels==0)); zero_label = labels(find(labels==0));
one_data = images(:, find(labels==1)); one_label = labels(find(labels==1));
clear images; clear labels;

data = [zero_data, one_data]; label = [zero_label; one_label];

% step 1: construct the k-nearest graph
K_euclidean_distance_matrix = data' * data;
K_euclidean_distance_matrix = (K_euclidean_distance_matrix - min(K_euclidean_distance_matrix))./max(K_euclidean_distance_matrix);
K_euclidean_distance_matrix = 1./K_euclidean_distance_matrix;
K_euclidean_distance_matrix = K_euclidean_distance_matrix - eye(2115) .* K_euclidean_distance_matrix;
[sort_index_matrx, sort_index] = sort(K_euclidean_distance_matrix, 2);
selected_k_nearest_distance = sort_index_matrx(:, 2:11);
selected_k_nearest_index = sort_index(:, 2:11);

% step2: compute the weight matrix
Weight_matrix = zeros(2115, 10);
for i = 1:2115
    X_N = repmat(data(:, i), 1, 10);
    X_V = zeros(784, 10);
    for j = 1:10
        X_V(:, j) = data(:, selected_k_nearest_index(i, j));
    end
    Gram_matrix = (X_N - X_V)' * (X_N - X_V);
    weight = inv(Gram_matrix) * ones(10, 1);
    Weight_matrix(i, :) = weight ./ sum(weight);
    
end

% step3: compute the projection
Weight_matrix_complete = zeros(2115, 2115);
for i = 1:2115
    for j = 1:10
        Weight_matrix_complete(i, selected_k_nearest_index(i, j)) = Weight_matrix(i, j);
    end
end
I_matirx = eye(2115, 2115);
M_matrix = I_matirx - Weight_matrix_complete;
[eigvector, eigvalue] = eig(M_matrix);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue);
eigvector = eigvector(:, index);
Y = eigvector(:, 2);

predict = zeros(2115, 1);
for i = 1:2115
    if Y(i, 1) > 0
        predict(i, 1) = 0;
    else
        predict(i, 1) = 1;
    end
end
