function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sum_err = zeros([size(theta)(1) - 1; 1]);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

sum_sq_err = sum((h - y) .^ 2);
theta_sq = theta(2:end)' * theta(2:end);

J = (1/(2*m)) * sum_sq_err + (lambda/(2 * m)) * theta_sq;


grad(1) = 0;

for i=1:m,
	grad(1) += (h(i) - y(i)) * X(i, 1);
end

grad(1) = grad(1)/m;

for i=1:m,
	xi = X(i, 2:end)';
	sum_err += (h(i) - y(i)) * xi;
end

grad(2:end) = ((1/m) * sum_err) + ((lambda/m) * theta(2:end));









% =========================================================================

grad = grad(:);

end
