function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%W1 = reshape(theta(1:hiddenSize * (visibleSize + 1)), ...
                 %hiddenSize, (visibleSize + 1));


%W2 = reshape(theta((1 + (hiddenSize * (visibleSize + 1))):end), ...
                 %visibleSize, (hiddenSize + 1));

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

m = size(data, 2);       
rho = sparsityParam;
Z2 = (W1*data) + repmat(b1, 1, m);
A2 = sigmoid(Z2);
Z3 = (W2*A2) + repmat(b2, 1, m);
A3 = Z3;

J = (1/m) * (0.5 * sum(sum((A3 - data).^2)));
J = J + (lambda/2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
Rho = sum(A2, 2)/m;
KL = (rho * log(rho./Rho)) + ((1 - rho) * log((1 - rho) ./ (1 - Rho)));
KL = sum(KL);
cost = J + beta*KL;

D3 = -(data-A3);
D2 = (W2'*D3);
P = beta*((-rho./Rho)+((1-rho)./(1-Rho)));
D2 = (D2 + repmat(P, 1, m)).*sigmoidGradient(A2);

DELTA_1 = D2 * data';
DELTA_2 = D3 * A2'; 

W1grad = (DELTA_1 / m) + lambda * W1;
W2grad = (DELTA_2 / m) + lambda * W2;
b1grad = (sum(D2, 2))/m;
b2grad = (sum(D3, 2))/m;






 %-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:); b1grad(:); b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(z)
    g = z.*(1-z);
end


