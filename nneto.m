% NNETO A simple neural network
% 
% NNETO creates a simple neural network object with properties numLayers,
% sizes, biases, and weights. This program has been optimized taking
% advantages of MATLAB's matrix computing. The concept follows the same
% procedure as has been used in NNET.
% 
% NET = NNETO(SIZES) creates a neural network NET with size SIZES.
% 
% See also NNET, GNNET
% This program has been stopped updating. Further implementations were 
% based on GNNET.
% 
%   $Author: Peng Li
%   $Date:   Jan 16, 2019
%

classdef nneto < handle
    properties
        numLayers
        sizes
        biases
        weights
    end
    
    methods
        %% __init__
        function this = nneto(netSizes)
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
                % else
                %     fprintf('Epoch %d complete\r', iE);
                end
            end
        end
        
        %% UPDATE MINI BATCH
        function update_mini_batch(this, mini_batch_data, mini_batch_label, eta)
            % back propoagation to calculate nabla biases and nabla weights
            [nabla_b, nabla_w] = this.backprop(mini_batch_data, mini_batch_label);
            
            % update weights and biases
            this.weights = cellfun(@(x, y) x-(eta ./ size(mini_batch_label, 2))*y, this.weights, nabla_w, 'UniformOutput', 0);
            this.biases  = cellfun(@(x, y) x-(eta ./ size(mini_batch_label, 2))*y, this.biases,  nabla_b, 'UniformOutput', 0);
        end
        
        %% BACK PROPAGATION
        function [nabla_b, nabla_w] = backprop(this, X, Y)
            nabla_b = cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0);
            nabla_w = cellfun(@(x) zeros(size(x)), this.weights, 'UniformOutput', 0);
            
            % feed forward
            activation = [{X}; cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0)];  % activation includes inputs
            z          = cellfun(@(x) zeros(size(x)), this.biases, 'UniformOutput', 0);         % weighted input
            for iL     = 1:length(this.biases)
                z{iL}  = this.weights{iL} * activation{iL} + this.biases{iL};
                activation{iL+1} = sigmoid(z{iL});
            end
            
            % backward pass
            delta        = (activation{end} - Y) .* sigmoid_prime(z{end});
            nabla_b{end} = sum(delta, 2);
            nabla_w{end} = delta * activation{end-1}';
            
            for iL       = 1:(this.numLayers-2)
                delta    = (this.weights{end-iL+1}' * delta) .* sigmoid_prime(z{end-iL});
                nabla_b{end-iL} = sum(delta, 2);
                nabla_w{end-iL} = delta * activation{end-iL-1}';
            end
        end
        
        %% EVALUATION
        function correct = evaluate(this, test_data, test_label)
            [~, test_res] = max(this.feedforward(test_data), [], 1);
            correct  = sum(test_res(:) == test_label);
        end
    end
end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function s_prime = sigmoid_prime(z)
s_prime = sigmoid(z) .* (1 - sigmoid(z));
end