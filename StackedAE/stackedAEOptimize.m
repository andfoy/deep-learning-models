function stackedAEOptTheta = stackedAEOptimize(stackedAETheta, numbatch, numiter, inputSize, hiddenSizeL2, data, labels, netconfig, numClasses, lambda)

options.Method = 'lbfgs';
options.maxIter = numiter;
options.display = 'on';
stackedAEOptTheta = stackedAETheta;
trainData = data;
trainLabels = labels;

addpath '~/Deep Learning/SelfTaughtLearning/minFunc/';

for i = 1:numbatch
fprintf('Batch # %d\n', i);
[stackedAEOptTheta, cost] = minFunc( @(x) stackedAECost(x, ...
      inputSize, hiddenSizeL2, numClasses, netconfig, ...
      lambda, trainData, trainLabels), ...
      stackedAEOptTheta, options); 
save Stheta.mat stackedAEOptTheta;
end
