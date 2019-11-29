% CROSSENTROPYCOST Cross entropy cost function
% 
% See also QUADRATICCOST
% 
%   $Author: Peng Li
%   $Date:   Jan 18, 2019
%

classdef CrossEntropyCost < handle
    methods (Static)
        function fy = fn(a, y)
            A = a;
            
            a(a == 0) = a(a == 0) + eps; % in case of log(0) -- log(a)
            A(A == 1) = A(A == 1) - eps; % in case of log(0) -- log(1-a)
            fy = -1 .* sum(y .* log(a) + (1-y) .* log(1-A), 1);
        end
        
        function d  = delta(z, a, y)
            d  = (a - y);
        end
    end
end