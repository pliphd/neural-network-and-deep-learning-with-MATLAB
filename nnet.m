% NNET A simple neural network
% 
% NNET creates a simple neural network object with properties numLayers,
% sizes, biases, and weights. This class def follows the same procedure as
% what Michael Nielsen has done in his e-book "Neural Networks and Deep
% Learning" in Python. This is corresponding to his file network.py
% 
% NET = NNET(SIZES) creates a neural network NET with size SIZES.
% 
% See also NNETO, GNNET
% This program has been stopped updating. Further implementations were 
% based on NNETO.
% 
%   $Author: Peng Li
%   $Date:   Jan 11, 2019
%

classdef nnet < handle
    properties
        numLayers
        sizes
        biases
        weights
    end
    
    methods
        %% __init__
        function this = nnet(netSizes)
            this.sizes     = netSizes;
            this.numLayers = length(netSizes);
            
            % init biases
            this.biases = cell(this.numLayers - 1, 1);
            for iB = 1:length(this.biases)
                this.biases{iB} = randn(this.sizes(iB+1), 1);
            end
            
            % init weight
            this.weights = cell(this.numLayers - 1, 1);
            for iW = 1:length(this.weights)
                this.weights{iW} = randn(this.sizes(iW+1), this.sizes(iW));
            end
        end
        
        %% FEEDFORWARD
        function out = feedforward(this, in)
            out = in;
            for iL = 1:length(this.biases)
                out = sigmoid(this.weights{iL} * out + this.biases{iL});
            end
        end
        
        %% STOCHASTIC GRADIENT DESCENT to train the nnet
        function SGD(this, training_data, training_label, epochs, mini_batch_size, eta, varargin)
            % varargin for optional test data
            if nargin == 8
                test_data  = varargin{1};
                test_label = varargin{2};
                n_test     = numel(test_label);
            end
            
            n = size(training_label, 2);
            
            % train for EPOCHS times
            for iE = 1:epochs
                % shuffle training data
                ind = randperm(n);
                training_data_  = training_data(:, ind);
                training_label_ = training_label(:, ind);
                
                % minibatch
                n_batch = ceil(n/mini_batch_size);
                for iB  = 1:n_batch
                    mini_batch_data  = training_data_(:, (iB-1)*mini_batch_size+1:min(iB*mini_batch_size, n));
                    mini_batch_label = training_label_(:, (iB-1)*mini_batch_size+1:min(iB*mini_batch_size, n));
                    this.update_mini_batch(mini_batch_data, mini_batch_label, eta);
                end
                
                % evaluate if test_data existing
                if nargin == 8
                    fprintf('Epoch %d: %d / %d\r', iE, this.evaluate(test_data, test_label), n_test);
                else
                    fprintf('Epoch %d complete\r', iE);
                end
            end
        end
        
        %% UPDATE MINI BATCH
        function update_mini_batch(this, mini_batch_data, mini_batch_label, eta)
            nabla_b = cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0);
            nabla_w = cellfun(@(x) zeros(size(x)), this.weights, 'UniformOutput', 0);
            
            % back propoagation to calculate nabla biases and nabla weights
            for iB = 1:size(mini_batch_label, 2)
                [delta_n_b, delta_n_w] = this.backprop(mini_batch_data(:, iB), mini_batch_label(:, iB));
                
                nabla_b = cellfun(@plus, nabla_b, delta_n_b, 'UniformOutput', 0);
                nabla_w = cellfun(@plus, nabla_w, delta_n_w, 'UniformOutput', 0);
            end
            
            % update weights and biases
            this.weights = cellfun(@(x, y) x-(eta ./ size(mini_batch_label, 2))*y, this.weights, nabla_w, 'UniformOutput', 0);
            this.biases  = cellfun(@(x, y) x-(eta ./ size(mini_batch_label, 2))*y, this.biases,  nabla_b, 'UniformOutput', 0);
        end
        
        %% BACK PROPAGATION
        function [nabla_b, nabla_w] = backprop(this, x, y)
            nabla_b = cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0);
            nabla_w = cellfun(@(x) zeros(size(x)), this.weights, 'UniformOutput', 0);
            
            % feed forward
            activation = [{x}; cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0)];  % activation includes inputs
            z          = cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0);         % weighted input
            for iL     = 1:length(this.biases)
                z{iL}  = this.weights{iL} * activation{iL} + this.biases{iL};
                activation{iL+1} = sigmoid(z{iL});
            end
            
            % backward pass
            delta        = (activation{end} - y) .* sigmoid_prime(z{end});
            nabla_b{end} = delta;
            nabla_w{end} = delta * activation{end-1}';
            
            for iL       = 1:(this.numLayers-2)
                delta    = (this.weights{end-iL+1}' * delta) .* sigmoid_prime(z{end-iL});
                nabla_b{end-iL} = delta;
                nabla_w{end-iL} = delta * activation{end-iL-1}';
            end
        end
        
        %% EVALUATION
        function correct = evaluate(this, test_data, test_label)
            test_res = nan(size(test_label));
            for iT = 1:numel(test_label)
                [~, test_res(iT)] = max(this.feedforward(test_data(:, iT)));
            end
            correct  = sum(test_res == test_label);
        end
    end
end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function s_prime = sigmoid_prime(z)
s_prime = sigmoid(z) .* (1 - sigmoid(z));
end