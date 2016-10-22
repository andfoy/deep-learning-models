imageChannels = 3;     % number of channels (rgb, so 3)

patchDim   = 8;          % patch dimension
numPatches = 100000;   % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term       

epsilon = 0.1;	       % epsilon for ZCA whitening

load STL10Features.mat

patches = ZCAWhite * patches;


%% STEP 2c: Learn features
%  You will now use your sparse autoencoder (with linear decoder) to learn
%  features on the preprocessed patches. This should take around 45 minutes.

optTheta = initializeParameters(hiddenSize, visibleSize);

W1_init = reshape(optTheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2_init = reshape(optTheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1_init = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2_init = optTheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

alpha = 3e-3;
n_iterations = 4000;

minfunc = @(W1, W2, b1, b2, data) sparseAutoencoderLinearCostGD(W1, W2, b1, ... 
                                  b2, visibleSize, hiddenSize, ...
                                  lambda, sparsityParam, beta, data.inputs);
[W1, W2, b1, b2] = optimizeGD(W1_init, W2_init, b1_init, b2_init, minfunc, data.training, alpha, n_iterations)



% Save the learned features and the preprocessing matrices for use in 
% the later exercise on convolution and pooling
fprintf('Saving learned features and preprocessing matrices...\n');                          
save('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
fprintf('Saved\n');
