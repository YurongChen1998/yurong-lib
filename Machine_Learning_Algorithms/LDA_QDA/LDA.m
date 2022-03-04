% Writen by Yurong Chen 2022-01-16
% https://yurongchen1998.github.io/
clear all; clc; close all;

%% Load Hyperspectral Data
data = load('mnist_test.csv');

labels = data(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785);
images = images/255;

[number, band] = size(images);
P = 0.70 ;
idx = randperm(number)  ;
data_train = images(idx(1:round(P*number)),:) ; 
label_train = labels(idx(1:round(P*number)),:) ; 
data_test = images(idx(round(P*number)+1:end),:) ;
label_test = labels(idx(round(P*number)+1:end),:) ;


%% Linear Discriminative Analysis
% number_class = max(label_train);
% mean_data = zeros(number_class, 784);
% covariance_data_sum = zeros(784, 784);
% for k = 1:number_class
%     label_k = find(label_train==k);
%     data_k = data_train(label_k,:);
%     temp_mean_k = mean(data_k);
%     mean_data(k, :) = temp_mean_k;
%     temp_mean_k = repmat(temp_mean_k, length(label_k), 1);
%     centered_data = data_k - temp_mean_k; 
%     covariance_data = length(label_k) * (centered_data' * centered_data);
%     covariance_data_sum = covariance_data_sum + covariance_data;
% end
% covariance_data = covariance_data ./ length(data_train);
% %covariance_data = inv(covariance_data);
%     
% % prediction
% prediction = zeros(length(data_test), number_class);
% for i = 1:length(data_test)
%     data_point = data_test(i, :);
%     for k = 1:number_class
%         probability = data_point * covariance_data * mean_data(k, :)' - 0.5 .* mean_data(k, :) * covariance_data * mean_data(k, :)';
%         prediction(i, k) = probability;
%     end
% end
% 
% [~,prediction] = max(prediction'); prediction = prediction';

%% QDA
number_class = max(label_train);
mean_data = zeros(number_class, 784);
covariance_data_sum = zeros(number_class, 784, 784);
for k = 1:number_class
    label_k = find(label_train==k);
    data_k = data_train(label_k,:);
    temp_mean_k = mean(data_k);
    mean_data(k, :) = temp_mean_k;
    temp_mean_k = repmat(temp_mean_k, length(label_k), 1);
    centered_data = data_k - temp_mean_k; 
    covariance_data_sum(k, :, :) = centered_data' * centered_data;
end
    
% prediction
prediction = zeros(length(data_test), number_class);
for i = 1:length(data_test)
    data_point = data_test(i, :);
    for k = 1:number_class
        probability = -0.5*log(det(reshape(covariance_data_sum(k, :, :), [784, 784]))) - 0.5*((data_point - mean_data(k, :)) * reshape(covariance_data_sum(k, :, :), [784, 784]) * (data_point - mean_data(k, :))');
        prediction(i, k) = probability;
    end
end

[~,prediction] = max(prediction'); prediction = prediction';