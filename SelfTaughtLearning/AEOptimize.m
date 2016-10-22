function opttheta = AEOptimize(opttheta, numbatch, numiter, inputSize, hiddenSize, data, plt)
clc
if ~exist('data', 'var')
   load data.mat
else
   unlabeledData = data;
end


%inputSize  = 28 * 28;
%hiddenSize = 196;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term  

addpath '~/Deep Learning/SelfTaughtLearning/minFunc/'
addpath '~/plot2svg_20120520'
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = numiter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

for i = 1:numbatch
fprintf('Batch # %d\n', i);
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData), ...
                              opttheta, options);
save theta.mat opttheta;
if exist('plt', 'var')
   W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
   display_network(W1');
   plot2svg(strcat('~/Weights/', int2str(i+10), '.svg'));
   clear W1;
   close all;
end
end
