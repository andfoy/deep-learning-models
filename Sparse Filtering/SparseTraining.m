function SparseTraining(data, N)

addpath '~/Deep Learning/Softmax';
addpath '~/nnclass/a4';
         
train_inputs = data.training.inputs;
[optW] = SparseFiltering(N, train_inputs);

save optW optW

for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
    data_name = data_details{1};
    data = data_details{2};
    hid_input = optW * data.inputs; % size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input); % size: <number of hidden units> by <number of data cases>
    %hidden_representation = logistic(input_to_hid * data.inputs);
    lambda = 1e-4;
    options.maxIter = 100;
    softmaxModel = softmaxTrain(n_hid, size(data.targets, 1), lambda, ...
                                hid_output, argmax_over_rows(data.targets), options);
    save strcat('Softmax', char(data_name), '.mat') softmaxModel
    [pred] = softmaxPredict(softmaxModel, hid_output);
    labels = argmax_over_rows(data.targets);
    acc = mean(labels(:) == pred(:));
    fprintf('For the %s data, the accuracy is %0.3f%%\n', data_name ,acc * 100);
end
