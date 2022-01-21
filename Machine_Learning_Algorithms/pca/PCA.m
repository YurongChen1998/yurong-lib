% Writen by Yurong Chen 2022-01-16
% https://yurongchen1998.github.io/
clear all; clc; close all;

%% Load Hyperspectral Data
load Indian_pines_corrected.mat;
data = indian_pines_corrected; data = reshape(data, [145*145, 200]);
clear indian_pines_corrected;

load Indian_pines_gt.mat;
labbel = indian_pines_gt; labbel = reshape(labbel, [145*145, 1]);
clear indian_pines_gt;

noise_data = data + 500*rand(145*145, 200);
[number_pixel, band] = size(data);
P = 0.70 ;
idx = randperm(number_pixel)  ;
data_train = noise_data(idx(1:round(P*number_pixel)),:) ; 
label_train = labbel(idx(1:round(P*number_pixel)),:) ; 
data_test = noise_data(idx(round(P*number_pixel)+1:end),:) ;
label_test = labbel(idx(round(P*number_pixel)+1:end),:) ;

%% PCA (Eigen-decomposition and SVD)
% Centering 
mean_data_ = mean(data_train);
mean_data = repmat(mean_data_, 14717, 1);
centered_data = data_train - mean_data; 

% Eigen-decompostion
covariance_matrix = centered_data.' * centered_data;
[eigvector, eigvalue] = eig(covariance_matrix);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvector = eigvector(:, index);
truncted_eigvector = eigvector(:, 1:10); 
new_data = noise_data - repmat(mean_data_, 145*145, 1);
new_data = new_data * truncted_eigvector;
new_data = new_data * truncted_eigvector';

% SVD
[u, s, v] = svd(centered_data');
u = u(:, 1:10);
new_data = noise_data - repmat(mean_data_, 145*145, 1);
new_data = new_data * u;
new_data = new_data * u';

%visualize the pca reconstruction
figure()
imagesc(reshape(noise_data(:, 100), [145, 145]))
colormap gray
figure()
imagesc(reshape(new_data(:, 100), [145, 145]))
colormap gray
close all

%% Dual PCA
v = v(:, 1:10);
inverse_sigma = (1./diag(s)).^2; inverse_sigma = inverse_sigma(1:10, 1);
new_data = centered_data' * v * diag(inverse_sigma) * v' * centered_data * noise_data';
new_data = new_data';
figure()
imagesc(reshape(noise_data(:, 100), [145, 145]))
colormap gray
figure()
imagesc(reshape(new_data(:, 100), [145, 145]))
colormap gray
close all
