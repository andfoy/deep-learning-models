function [value,grad] = simpleQuadraticFunction(x)
% this function accepts a 2D vector as input. 
% Its outputs are:
%   value: h(x1, x2) = x1^2 + 3*x1*x2
%   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2 
% Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
% that computeNumericalGradients will use only the first returned value of this function.

value = x(1)^2 + 3*x(1)*x(2);

grad = zeros(2, 1);
grad(1)  = 2*x(1) + 3*x(2);
grad(2)  = 3*x(1);

end
