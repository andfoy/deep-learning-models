function [cost,grad,features] = sparseLinearNNCost(theta, visibleSize, hiddenSize, ...
                                                   outputSize, lambda, sparsityParam, beta, data, labels)


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:(hiddenSize*visibleSize) + (hiddenSize*outputSize)), outputSize, hiddenSize);
b1 = theta((hiddenSize*visibleSize) + (hiddenSize*outputSize) + 1:(hiddenSize*visibleSize) + (hiddenSize*outputSize)  + hiddenSize);
b2 = theta((hiddenSize*visibleSize) + (hiddenSize*outputSize) + 1 + hiddenSize:end);


m = size(data, 2);       
rho = sparsityParam;
Z2 = (W1*data) + repmat(b1, 1, m);
A2 = sigmoid(Z2);
Z3 = (W2*A2) + repmat(b2, 1, m);
A3 = Z3;

J = (1/m) * (0.5 * sum(sum((A3 - labels').^2)));
J = J + (lambda/2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
Rho = sum(A2, 2)/m;
KL = (rho * log(rho./Rho)) + ((1 - rho) * log((1 - rho) ./ (1 - Rho)));
KL = sum(KL);
cost = J + beta*KL;

D3 = -(labels'-A3);
D2 = (W2'*D3);
P = beta*((-rho./Rho)+((1-rho)./(1-Rho)));
D2 = (D2 + repmat(P, 1, m)).*sigmoidGradient(A2);

DELTA_1 = D2 * data';
DELTA_2 = D3 * A2'; 

W1grad = (DELTA_1 / m) + lambda * W1;
W2grad = (DELTA_2 / m) + lambda * W2;
b1grad = (sum(D2, 2))/m;
b2grad = (sum(D3, 2))/m;

grad = [W1grad(:) ; W2grad(:); b1grad(:); b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(z)
    g = z.*(1-z);
end
