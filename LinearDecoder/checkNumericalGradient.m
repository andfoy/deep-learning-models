function [] = checkNumericalGradient()
% This code can be used to check your numerical gradient implementation 
% in computeNumericalGradient.m
% It analytically evaluates the gradient of a very simple function called
% simpleQuadraticFunction (see below) and compares the result with your numerical
% solution. Your numerical gradient implementation is incorrect if
% your numerical solution deviates too much from the analytical solution.
  
% Evaluate the function and gradient at x = [4; 10]; (Here, x is a 2d vector.)
x = [4; 10];
[value, grad] = simpleQuadraticFunction(x);

% Use your code to numerically compute the gradient of simpleQuadraticFunction at x.
% (The notation "@simpleQuadraticFunction" denotes a pointer to a function.)
numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be 2.1452e-12 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
end


  

