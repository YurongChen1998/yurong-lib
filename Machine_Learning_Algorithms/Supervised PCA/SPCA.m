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

number_point = length(data);

%% construct the Generalized matrix G
kernel_y = label * label';
H = eye(2115) - ones(2115)./2115;
G = data * H * kernel_y * H * data';
[eigvector, eigvalue] = eig(G);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvector = eigvector(:, index);
truncted_eigvector = eigvector(:, 1:2); 
proj_data = (truncted_eigvector' * data)'; 
scatter(proj_data(1:980,1), proj_data(1:980, 2))
hold on
scatter(proj_data(981:2115,1), proj_data(981:2115, 2))
hold off
close all;

%% Using the kernel trick
kernel_x = (data' * data + 1).^2;
kernel_y = label * label';
H = eye(2115) - ones(2115)./2115;
G = H * kernel_y * H * kernel_x;
[eigvector, eigvalue] = eig(G);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvector = eigvector(:, index);
eigvector = eigvector(:, 1:2);
proj_data = eigvector' * kernel_x;
proj_data = proj_data';
scatter(proj_data(1:980,1), proj_data(1:980, 2))
hold on
scatter(proj_data(981:2115,1), proj_data(981:2115, 2))
hold off