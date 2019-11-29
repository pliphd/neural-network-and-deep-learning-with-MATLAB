% test gnnet using MNIST dataset
% 
%   $Author: Peng Li
%   $Date:   Jan 11, 2019
%

clc; clear; close all; diary off;

DiaryPt = ['./_GNNET_MNIST_README_' datestr(today, 'yyyymmdd') '.txt'];
if exist(DiaryPt, 'file') == 2
    delete(DiaryPt);
end
diary(DiaryPt)

% load MNIST data
[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;

% further split training data set into training (50K) plus validation (10K)
ind_valid = randperm(size(imgDataTrain, 4), 10000);
ind_train = setxor(1:size(imgDataTrain, 4), ind_valid);

imgDataTrain_ = imgDataTrain(:, :, :, ind_train);
labelsTrain_  = labelsTrain(ind_train);

imgDataValid  = imgDataTrain(:, :, :, ind_valid);
labelsValid   = labelsTrain(ind_valid);

% reshape
training_data  = double(reshape(imgDataTrain_, 784, length(labelsTrain_)));
training_label = zeros(10, length(labelsTrain_));
for iS = 1:length(labelsTrain_)
    training_label(double(labelsTrain_(iS)), iS) = 1;
end

valid_data     = double(reshape(imgDataValid, 784, length(labelsValid)));
valid_label    = double(labelsValid);

test_data      = double(reshape(imgDataTest, 784, length(labelsTest)));
test_label     = double(labelsTest);

% init neural network
net = gnnet([784, 30, 10], CrossEntropyCost);
% net.large_weight_initializer; % uncomment to use large weight initializer

% train net using training data
epoch = 30; mini_size = 10; rate = 0.5; lambda = 5;
training_data  = training_data(:, :);   % change to (:, 1:1000) to use the first 1000 training samples
training_label = training_label(:, :);  % change to (:. 1:1000) if using the first 1000 training samples
tic;
out = ...
    net.SGD(training_data./255, training_label, epoch, mini_size, rate, lambda, test_data./255, test_label, 1, 1, 1, 1);
t = toc;

% show
disp('Task: Recogniting handwrite digits using MNIST database');
disp('neural network model:');
disp(net);
fprintf('\ttraining samples: %d\r', size(training_data, 2));
fprintf('\tepoch: %d\r', epoch);
fprintf('\tmini batch size: %d\r', mini_size);
fprintf('\tlearning rate: %.1f\r', rate);
fprintf('\ttesting samples: %d\r', size(test_data, 2));
fprintf('overall time: %.6f\r', t);
diary off;

%% plot
close all;
Pos = CenterFig(18, 8, 'centimeters');
figure('Color', 'w', 'Units', 'centimeters', 'Position', Pos);
if length(out) > 3
    train_accu = out{4}./size(training_data, 2);
    
    axis_accuracy = subplot(121);
    a_train = plot(1:epoch, train_accu, 'k', 'DisplayName', 'Training Accuracy', 'LineWidth', 2); hold on; grid on; grid minor
end

if length(out) > 2
    train_cost = out{3};
    
    axis_cost = subplot(122);
    c_train = plot(1:epoch, train_cost, 'k', 'DisplayName', 'Training Cost', 'LineWidth', 2); hold on;
end

if length(out) > 1
    test_accu  = out{2}./size(test_data, 2);
    
    if ~ (exist('axis_accuracy', 'var') == 1)
        axis_accuracy = subplot(121);
    end
    a_test = plot(axis_accuracy, 1:epoch, test_accu, 'r', 'DisplayName', 'Test Accuracy', 'LineWidth', 2);
    legend(axis_accuracy, 'show', 'Location', 'best');
    
    axis_accuracy.Units = 'norm';
    axis_accuracy.Position = [.06 .15 .41 .8];
    axis_accuracy.YTick = 0:.1:1;
    axis_accuracy.YLim  = [0.8 1];
    xlabel(axis_accuracy, 'Epoch');
end

if ~isempty(out)
    test_cost  = out{1};
    
    if ~ (exist('axis_cost', 'var') == 1)
        axis_cost = subplot(122);
    end
    c_test = plot(axis_cost, 1:epoch, test_cost, 'r', 'DisplayName', 'Test Cost', 'LineWidth', 2);
    legend(axis_cost, 'show', 'Location', 'best');
    grid on;
    
    axis_cost.Units = 'norm';
    axis_cost.Position = [.56 .15 .41 .8];
    axis_cost.YScale = 'log';
    xlabel(axis_cost, 'Epoch');
end