function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.



full_prediction = theta*data;
max_prediction = max(full_prediction);
shrunk_prediction = full_prediction - repmat(max_prediction, numClasses, 1);
exp_shrunk_prediction = exp(shrunk_prediction);
term = sum(exp_shrunk_prediction);
prediction = (exp_shrunk_prediction./repmat(term, numClasses, 1));

log_term = log(prediction);
cost = -1 * sum(sum((groundTruth.*log_term))) / numCases;
cost = cost + ((0.5 * lambda) * sum(sum(theta.^2)));

gp = groundTruth - prediction;
thetagrad = (data*gp')' * (-1.0 / numCases);
thetagrad = thetagrad + (lambda * theta);








% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

