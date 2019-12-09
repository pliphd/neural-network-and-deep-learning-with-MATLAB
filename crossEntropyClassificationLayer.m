% The default classificationLayer must be preceded by a softmaxLayer. In
% order to use the customized sigmoidLayer, I'm now customizing a
% classificationLayer that can be used in this structure.
% 
% In your MATLAB documentation, search custom deep learning output layer
% for more information
% 
% 
%   $Author: Peng Li
%   $Date:   Dec 02, 2019
%
classdef crossEntropyClassificationLayer < nnet.layer.RegressionLayer
    % Example custom classification layer with cross entropy loss function.
    
    methods
        function layer = crossEntropyClassificationLayer(varargin)
            % Set layer name.
            if nargin == 1
                layer.Name = varargin{1};
            end

            % Set layer description.
            layer.Description = 'Cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the cross entropy loss between
            % the predictions Y and the training targets T.
            
            loss = -sum(sum(sum(T.*log(Y+eps)) ./size(Y, 4)));
            
            % Note this forwardLoss function may appear different from
            % Michael's original expression. But they are essentially the
            % same as the very inner sum covers all classes to be classified
            % specifically,
            % \sigma\limit_1^K(t_{i,j}\times log(y_{i,j})
            % t_{i,j} means the actually class for the ith sample is the
            % jth, y_{i,j} means the output value for the ith sample from
            % the previous layer (sigmoid)
            %
            % I added +eps in the log() function to prevent invalid outcome
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the cross entropy loss with respect to the predictions Y.

            dLdY = (-T./(Y+eps))./size(Y, 4);
            
            % This seems different from the deriatives in Michael's book,
            % too. But if you work through the math using the new cross
            % entropy loss shown in the forwardLoss function, you will get
            % into this.
        end
    end
end