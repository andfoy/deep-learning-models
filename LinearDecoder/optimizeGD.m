function [W1, W2, b1, b2] = optimizeGD(W1, W2, b1, b2, gradient_function, training_data, learning_rate, n_iterations)
% This trains a model that's defined by a single matrix of weights.
% <model_shape> is the shape of the array of weights.
% <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
% This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
% This returns the matrix of weights of the trained model.
    momentum_speed1 = zeros(size(W1));
    momentum_speed2 = zeros(size(W2));
    momentum_speed_b1 = zeros(size(b1));
    momentum_speed_b2 = zeros(size(b2));
    mini_batch_size = 401; 
    start_of_next_mini_batch = 1;
    for iteration_number = 1:n_iterations
        mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
        start_of_next_mini_batch = mod(start_of_next_mini_batch + mini_batch_size, size(training_data.inputs, 2));
        [W1_g, W2_g, b1_g, b2_g, cost] = gradient_function(W1, W2, b1, b2, mini_batch);
        fprintf('Iteration %d | Cost: %4.6e | Batch # %d\n', iteration_number, cost, start_of_next_mini_batch);
        momentum_speed1 = 0.9 * momentum_speed1 + W1_g;
        momentum_speed2 = 0.9 * momentum_speed2 + W2_g;
        momentum_speed_b1 = 0.9 * momentum_speed_b1 + b1_g;
        momentum_speed_b2 = 0.9 * momentum_speed_b2 + b2_g;
        W1 = W1 - momentum_speed1 * learning_rate;
        W2 = W2 - momentum_speed2 * learning_rate;
        b1 = b1 - momentum_speed_b1 * learning_rate;
        b2 = b2 - momentum_speed_b2 * learning_rate;
        %displayColorNetwork((W1*ZCAWhite)');
    end
end
