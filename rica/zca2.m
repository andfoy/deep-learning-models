function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

sigma = x * x' / size(x, 2);
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
Z = ZCAWhite*x;


%%% YOUR CODE HERE %%%
