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
proj_data = new_data * truncted_eigvector;
figure()
imagesc(reshape(proj_data(:, 1), [145, 145]))
colormap gray
new_data = proj_data * truncted_eigvector';

% SVD
[u, s, v] = svd(centered_data');
u = u(:, 1:10);
new_data = noise_data - repmat(mean_data_, 145*145, 1);
proj_data = new_data * u;
figure()
imagesc(reshape(proj_data(:, 1), [145, 145]))
colormap gray
new_data = proj_data * u';

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
truncted_sigma = 1./diag(s); truncted_sigma = truncted_sigma(1:10, 1);
proj_data = diag(truncted_sigma) * v' * centered_data * noise_data';
proj_data = proj_data';
figure()
imagesc(reshape(proj_data(:, 1), [145, 145]))
colormap gray

new_data = centered_data' * v * diag(inverse_sigma) * v' * centered_data * noise_data';
new_data = new_data';
figure()
imagesc(reshape(noise_data(:, 100), [145, 145]))
colormap gray
figure()
imagesc(reshape(new_data(:, 100), [145, 145]))
colormap gray
close all

%% Kernel PCA (resize the data size)
clear all;
load Indian_pines_corrected.mat;
data = indian_pines_corrected; 
resized_data =  zeros(15, 15, 200);
for i = 1:200
    resized_data(:, :, i) = imresize(data(:, :, i), 0.1); 
end
data = reshape(resized_data, [15*15, 200]);
clear indian_pines_corrected; clear resized_data;

load Indian_pines_gt.mat;
label = indian_pines_gt; 
resized_labbel = imresize(label, 0.1, 'nearest'); 
label = reshape(resized_labbel, [15*15, 1]);
clear indian_pines_gt;  clear resized_labbel;

kernel_matrix = (data * data' + 1).^2;
[eigvector, eigvalue] = eig(kernel_matrix);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvalue = diag(eigvalue);
eigvector = eigvector(:, index);

eigvector = eigvector(:, 1:2);
truncted_sigma = 1./diag(eigvalue); truncted_sigma = truncted_sigma(1:2, 1);
truncted_sigma = diag(truncted_sigma);
proj_data = truncted_sigma * eigvector';
proj_data = proj_data';

