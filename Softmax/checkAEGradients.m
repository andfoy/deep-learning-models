function checkAEGradients(lambda, beta, X)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

if ~exist('beta', 'var') || isempty(beta)
    beta = 0;
end

visibleSize = 64;
hiddenSize = 5;
sparsityParam = 0.01;
m = 5;

% We generate some 'random' test data
nn_params =initializeParameters(hiddenSize, visibleSize);
% Reusing debugInitializeWeights to generate X
X = X(:, 1:m);

% Unroll parameters


% Short hand for cost function
costFunc = @(x) sparseAutoencoderCost(x, visibleSize, ...
                                      hiddenSize, lambda, ...
                                      sparsityParam, beta, ...
                                      X);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end
