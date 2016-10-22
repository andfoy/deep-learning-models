function ret = cd12(model, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    rbm_w = reshape(model(1:500*784), 500, 784);
    rbm_w2 = reshape(model((500*784)+1:end), 1000, 500);
    visible_data = sample_bernoulli(visible_data);
    hid1_probability = vis_hid1(rbm_w, visible_data);
    binaryhid = sample_bernoulli(hid1_probability);
    g1 = configuration_goodness_gradient(visible_data, binaryhid);
    hid2_probability = hid1_hid2(rbm_w2, binaryhid);
    binaryhid2 = sample_bernoulli(hid2_probability);
    g2 = configuration_goodness_gradient(binaryhid, binaryhid2);
    h2_h1 = hid2_hid1(rbm_w2, binaryhid2);
    binaryh1 = sample_bernoulli(h2_h1);
    visprob = hid1_vis(rbm_w, binaryh1);
    binaryvis = sample_bernoulli(visprob);
    h1prob = vis_hid1(rbm_w, binaryvis);
    %binaryhid2 = sample_bernoulli(hiddenprob);
    g3 = configuration_goodness_gradient(binaryvis, h1prob);
    ret1 = (g1 - g3)';
    h1bin = sample_bernoulli(h1prob);
    h2prob = hid1_hid2(rbm_w2, h1bin);
    g4 = configuration_goodness_gradient(h1bin, h2prob);
    ret2 = (g2 - g4)';
    ret = [ret1(:); ret2(:)];
end
