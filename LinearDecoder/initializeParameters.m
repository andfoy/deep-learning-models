function theta = initializeParameters(hiddenSize, visibleSize)%fan_out, fan_in)

%% Initialize parameters randomly based on layer sizes.
%r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
%W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
%W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

%b1 = zeros(hiddenSize, 1);
%b2 = zeros(visibleSize, 1);
%W = zeros(fan_out, 1 + fan_in);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
%theta = reshape(sin(1:numel(W)), size(W)) / 10;
% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
%theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];


%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

