function ret = cd11(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    visible_data = sample_bernoulli(visible_data);
    hidden_probability = vis2hid1(rbm_w, visible_data);
    binaryhid = sample_bernoulli(hidden_probability);
    g1 = configuration_goodness_gradient(visible_data, binaryhid);
    visible_probability = hid12vis(rbm_w, binaryhid);
    binaryvis = sample_bernoulli(visible_probability);
    hiddenprob = vis2hid1(rbm_w, binaryvis);
    %binaryhid2 = sample_bernoulli(hiddenprob);
    g2 = configuration_goodness_gradient(binaryvis, hiddenprob);
    ret = g1 - g2;
    ret = ret';
end
