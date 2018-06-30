function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sum = zeros([size(theta)(1) - 1; 1]);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X* theta);

J = 1.0/m * ( - y' * log(h) - (1 - y)' * log(1 - h));

reg_term =  lambda/(2.0 * m) * (theta(2:end)' * theta(2:end));

J+= reg_term;

grad(1) = 0;

for i=1:m,
	grad(1) += (h(i) - y(i)) * X(i, 1);
end

grad(1) = grad(1)/m;

for i=1:m
	xi = X(i, 2:end)';
   sum += (h(i) - y(i)) * xi;
end

grad(2:end) = ((1/m) * sum) + ((lambda/m) * theta(2:end));



% =============================================================

end