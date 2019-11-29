% QUADRATICCOST Quadratic cost function
% 
% See also CROSSENTROPYCOST
% 
%   $Author: Peng Li
%   $Date:   Jan 18, 2019
%

classdef QuadraticCost < handle
    methods (Static)
        function fy = fn(a, y)
            fy = sum(0.5 .* abs(a - y).^2, 1);
        end
        
        function d  = delta(z, a, y)
            d  = (a - y) .* sigmoid_prime(z);
        end
    end
end

function s = sigmoid(z)
s = 1 ./ (1 + exp(-z));
end

function s_prime = sigmoid_prime(z)
s_prime = sigmoid(z) .* (1 - sigmoid(z));
end