import numpy as np

def computeNumericalGradient(J, theta):
  numgrad = np.zeros((theta.shape[0], 1));
  perturb = np.zeros((theta.shape[0], 1));
  e = 1e-4
  for p in 0:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;

