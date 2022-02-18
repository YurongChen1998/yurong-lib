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


%% visulize the reconstruction 
kernel_data = data' * data;
figure()
imagesc(kernel_data)
A = kernel_data(1:500, 1:500);
B = kernel_data(1:500, 501:2115);
C = B' * pinv(A) * B;
h_kernel_data = [A, B; B', C'];
figure()
imagesc(h_kernel_data)
norm(h_kernel_data - kernel_data)

close all
%% project data
[eigvector, eigvalue] = eig(kernel_data);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvalue = eigvalue(1:2); eigvalue  = sqrt(eigvalue); eigvalue = diag(eigvalue); 
eigvector = eigvector(:, index);
truncted_eigvector = eigvector(:, 1:2); 
proj_data = (eigvalue * truncted_eigvector')'; 
scatter(proj_data(1:980,1), proj_data(1:980, 2))
hold on
scatter(proj_data(981:2115,1), proj_data(981:2115, 2))
hold off

close all
%% project data using Nystrom Approximation
[eigvector, eigvalue] = eig(A);
eigvalue = diag(eigvalue);
[eigvalue, index] = sort(eigvalue, 'descend');
eigvalue = eigvalue(1:2); eigvalue  = sqrt(eigvalue); eigvalue = diag(eigvalue); 
eigvector = eigvector(:, index);
truncted_eigvector = eigvector(:, 1:2); 
proj_data_R = eigvalue * truncted_eigvector'; 
eigvalue = diag(eigvalue); eigvalue = 1./eigvalue; eigvalue = diag(eigvalue); 
proj_data_S = eigvalue * truncted_eigvector' * B; 
proj_data = [proj_data_R, proj_data_S]';
scatter(proj_data(1:980,1), proj_data(1:980, 2))
hold on
scatter(proj_data(981:2115,1), proj_data(981:2115, 2))
hold off
