function mini_batch = extract_mini_batch(data_set, start_i, n_cases, targets)
    mini_batch.inputs = data_set.inputs(:, start_i : start_i + n_cases - 1);
    if exist('targets', 'var')
       mini_batch.targets = data_set.targets(:, start_i : start_i + n_cases - 1);
    end
end

