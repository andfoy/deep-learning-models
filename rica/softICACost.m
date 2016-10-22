%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%

m = size(x, 2);

aux1 = sqrt(((W*x).^2) + params.epsilon);
aux2 = (W'*W*x - x).^2;
cost = params.lambda*sum(aux1(:)) + sum(aux2(:))/2;

% % GRADIENT:
aux3 = W'*W*x - x;
part1 = ((W*aux3*(x'))+ ((W*x)*(aux3')));
part2 = ((W*x)./aux1)*x';
Wgrad = part1 + params.lambda*part2;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
