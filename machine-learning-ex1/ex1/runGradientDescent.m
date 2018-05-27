function [theta, J_history] = runGradientDescent(X, y, theta, alpha, num_iters, color)

[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:numel(J_history), J_history, color);
xlabel('Number of iterations');
ylabel('Cost J');
