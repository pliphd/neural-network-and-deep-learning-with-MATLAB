% GNNET A general simple neural network
% 
% NET = GNNET(SIZES, COST) creates a neural network NET with size SIZES and
% using cost function COST
% 
% See also NNET, NNETO
% 
%   $Author: Peng Li
%   $Date:   Jan 18, 2019
%

classdef gnnet < handle
    properties
        numLayers
        sizes
        biases
        weights
        
        cost
    end
    
    methods
        %% __init__
        function this = gnnet(netSizes, Cost)
            this.sizes     = netSizes;
            this.numLayers = length(netSizes);
            
            % init weights and biases
            this.default_weight_initializer;
            
            % init cost
            this.cost       = Cost;
        end
        
        %% INIT WEIGHT AND BIASES
        function large_weight_initializer(this)
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
        
        function default_weight_initializer(this)
            % init biases
            this.biases = cell(this.numLayers - 1, 1);
            for iB = 1:length(this.biases)
                this.biases{iB} = randn(this.sizes(iB+1), 1);
            end
            
            % init weight
            this.weights = cell(this.numLayers - 1, 1);
            for iW = 1:length(this.weights)
                this.weights{iW} = randn(this.sizes(iW+1), this.sizes(iW)) ./ sqrt(this.sizes(iW));
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
        function out = SGD(this, training_data, training_label, epochs, mini_batch_size, eta, lambda, varargin)
            % varargin for optional evaluation data and demo figure options
            if nargin > 7
                if length(varargin) > 5
                    monitor_training_accurary   = varargin{6};
                    training_accuracy           = [];
                end
                if length(varargin) > 4
                    monitor_training_cost       = varargin{5};
                    training_cost               = [];
                end
                if length(varargin) > 3
                    monitor_evaluation_accurary = varargin{4};
                    evaluation_accuracy         = [];
                end
                if length(varargin) > 2
                    monitor_evaluation_cost     = varargin{3};
                    evaluation_cost             = [];
                end
                if length(varargin) > 1
                    evaluation_data  = varargin{1};
                    evaluation_label = varargin{2};
                    n_evaluation     = numel(evaluation_label);
                end
                if length(varargin) == 1
                    error('wrong parameters.');
                end
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
                    this.update_mini_batch(mini_batch_data, mini_batch_label, eta, lambda, n);
                end
                
                % print
                fprintf('Epoch %d training complete\r', iE);
                
                if exist('monitor_training_cost', 'var') == 1
                    training_cost = [training_cost; this.total_cost(training_data, training_label, lambda, 0)];
                    fprintf('Cost on training data: %f\r', training_cost(end));
                end
                if exist('monitor_training_accurary', 'var') == 1
                    training_accuracy = [training_accuracy; this.accuracy(training_data, training_label, 1)];
                    fprintf('Accuracy on training data: %d / %d\r', training_accuracy(end), n);
                end
                
                if exist('monitor_evaluation_cost', 'var') == 1
                    evaluation_cost = [evaluation_cost; this.total_cost(evaluation_data, evaluation_label, lambda, 1)];
                    fprintf('Cost on evaluation data: %f\r', evaluation_cost(end));
                end
                if exist('monitor_evaluation_accurary', 'var') == 1
                    evaluation_accuracy = [evaluation_accuracy; this.accuracy(evaluation_data, evaluation_label, 0)];
                    fprintf('Accuracy on evaluation data: %d / %d\r', evaluation_accuracy(end), n_evaluation);
                end
                fprintf('\r');
            end
            
            if nargin > 7
                out = {};
                if length(varargin) > 5
                    out{4} = training_accuracy;
                end
                if length(varargin) > 4
                    out{3} = training_cost;
                end
                if length(varargin) > 3
                    out{2} = evaluation_accuracy;
                end
                if length(varargin) > 2
                    out{1} = evaluation_cost;
                end
            else
                out = {};
            end
        end
        
        %% UPDATE MINI BATCH
        function update_mini_batch(this, mini_batch_data, mini_batch_label, eta, lambda, train_size)
            % back propoagation to calculate nabla biases and nabla weights
            [nabla_b, nabla_w] = this.backprop(mini_batch_data, mini_batch_label);
            
            % update weights with regularization parameer lambda
            this.weights = cellfun(@(x, y) (1-eta*(lambda/train_size))*x-(eta ./ size(mini_batch_label, 2))*y, ...
                this.weights, nabla_w, 'UniformOutput', 0);
            % update biases
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
            delta        = this.cost.delta(z{end}, activation{end}, Y);
            nabla_b{end} = sum(delta, 2);
            nabla_w{end} = delta * activation{end-1}';
            
            for iL       = 1:(this.numLayers-2)
                delta    = (this.weights{end-iL+1}' * delta) .* sigmoid_prime(z{end-iL});
                nabla_b{end-iL} = sum(delta, 2);
                nabla_w{end-iL} = delta * activation{end-iL-1}';
            end
        end
        
        %% ACCURACY
        function correct = accuracy(this, data, label, vectorize)
            if vectorize % training data
                [~, res] = max(this.feedforward(data), [], 1);
                [~, RES] = max(label);
                correct  = sum(res(:) == RES(:));
            else % test or validation
                [~, res] = max(this.feedforward(data), [], 1);
                correct  = sum(res(:) == label);
            end
        end
        
        %% TOTAL COST
        function cost  = total_cost(this, data, label, lambda, vectorize)
            activation = this.feedforward(data);
            
            if vectorize
                vec_label = zeros(this.sizes(end), length(label));
                for iS = 1:length(label)
                    vec_label(double(label(iS)), iS) = 1;
                end
            else
                vec_label = double(label);
            end
            
            cost = sum(this.cost.fn(activation, vec_label) ./ size(label, 2)) + ...
                0.5*lambda/size(label, 2)*sum(cellfun(@(x) sum(x(:)), cellfun(@(x) x.^2, this.weights, 'uni', 0)));
        end
    end
end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function s_prime = sigmoid_prime(z)
s_prime = sigmoid(z) .* (1 - sigmoid(z));
end