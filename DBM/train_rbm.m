function train_rbm(n_hid, lr_rbm, lr_classification, n_iterations, rbm_w, n_iter)

clc; close all;

addpath '~/Deep Learning/Softmax';
addpath '~/Deep Learning/Softmax/minFunc';

load mnistrbm
data_sets = data;
clear data

data.training.inputs = data_sets.training.inputs; %data_sets.validation.inputs data_sets.test.inputs];
data.training.targets = data_sets.training.targets; %data_sets.validation.targets data_sets.test.targets];

if (~exist('rbm_w', 'var')) 
rbm_w = optimize([n_hid, 784], ...
                 @(rbm_w, data) cd1(rbm_w, data.inputs), ...  % discard labels
                 data.training, ...
                 lr_rbm, ...
                 n_iterations, 0);

save rbm_w.mat rbm_w;
elseif (exist('rbm_w', 'var') && exist('n_iter', 'var'))
load rbm_w
rbm_w = optimize([n_hid, 784], ...
                 @(rbm_w, data) cd1(rbm_w, data.inputs), ...  % discard labels
                 data.training, ...
                 lr_rbm, ...
                 n_iter, 1, rbm_w);

save rbm_w.mat rbm_w;
else
load rbm_w
end


input_to_hid = rbm_w;
% calculate the hidden layer representation of the labeled data
hidden_representation = logistic(input_to_hid * data.training.inputs);
% train hid_to_class
data_2.inputs = hidden_representation;
data_2.targets = data.training.targets;
clear data

hid_to_class = optimize([10, n_hid], @(model, data) classification_phi_gradient(model, data), data_2, lr_classification, n_iterations);
clear data_2
for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
    data_name = data_details{1};
    data = data_details{2};
    hid_input = input_to_hid * data.inputs; % size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input); % size: <number of hidden units> by <number of data cases>
    class_input = hid_to_class * hid_output; % size: <number of classes> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    error_rate = mean(double(argmax_over_rows(class_input) ~= argmax_over_rows(data.targets))); % scalar
    acc_rate = mean(argmax_over_rows(class_input) == argmax_over_rows(data.targets))*100;
    loss = -mean(sum(log_class_prob .* data.targets, 1)); % scalar. select the right log class probability using that sum; then take the mean over all data cases.
    fprintf('For the %s data, the classification cross-entropy loss is %f, and the classification error rate (i.e. the misclassification rate) is %f → Accuracy rate: %f\n', data_name, loss, error_rate, acc_rate);
end

for data_details = reshape({'training', data_sets.training, 'validation', data_sets.validation, 'test', data_sets.test}, [2, 3]),
    data_name = data_details{1};
    data = data_details{2};
    hid_input = input_to_hid * data.inputs; % size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input); % size: <number of hidden units> by <number of data cases>
    %hidden_representation = logistic(input_to_hid * data.inputs);
    lambda = 1e-4;
    options.maxIter = 100;
    softmaxModel = softmaxTrain(n_hid, size(data.targets, 1), lambda, ...
                                hid_output, argmax_over_rows(data.targets), options);
    save strcat('Softmax', data_name, '.mat') softmaxModel
    [pred] = softmaxPredict(softmaxModel, hid_output);
    labels = argmax_over_rows(data.targets);
    acc = mean(labels(:) == pred(:));
    fprintf('For the %s data, the accuracy is %0.3f%%\n', data_name ,acc * 100);
end


