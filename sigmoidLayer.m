% By some historical reason, MATLAB never has ever provided a sigmoid layer
% that directly supports DL, following the instruction in 
% https://www.mathworks.com/help/deeplearning/ug/define-custom-deep-learning-layer.html
% 
% Or in your MATLAB documentation, search custom deep learning layer
% 
% I'm customizing a sigmoid layer that can be used in the DL structure
% 
%   $Author: Peng Li
%   $Date:   Dec 01, 2019
%

classdef sigmoidLayer < nnet.layer.Layer
    methods
        function layer = sigmoidLayer(varargin)
            if nargin == 1
                % Set layer name.
                layer.Name = varargin{1};
            end
            
            % Set layer description.
            layer.Description = "sigmoidLayer";
        end
        
        % uncomment the following predict function if you are using MATLAB
        % 2019b and newer versions
        % function Z = predict(layer, X)
        %     % Z = predict(layer, X) forwards the input data X through the
        %     % layer and outputs the result Z.
        %     Z = sigmoid(X);
        % end
        
        % comment out the following predict and backward function if you
        % are using MATLAB R2019b and newer versions
        % The reason is that starting from R2019b, MATLAB can automatically
        % implement some functions' gradient, including sigmoid function
        % Note that I've applied notations that are different from what
        % Michael used: X-->weighted inputs (z), Z-->activation (a)
        function Z = predict(~, X)
            Z = 1 ./ (1 + exp(-X));
        end
        
        function dLdX = backward(~, ~ ,Z, dLdZ, ~)
            % Backward propagate the derivative of the loss function through
            % the layer
            % See BP1 in Chp2 of Michael's book if you get lost
            %     dC/dz = dC/da * sigma_prime(z)
            dLdX = Z.*(1 - Z) .* dLdZ;
        end
    end
end