% 111061702 ex2_3
rng(0, 'v4');  %random seed
%% Generate Data
n = 500;
x = sort(rand(n, 1) * 5);
y = 2*sin(1.5*x);
epsilon = randn(n, 1);
y_epsilon = y + epsilon;

% Separate data into 8:2 randomly
train_val_ratio = 0.8;
n_train = n * train_val_ratio;
index_train = randperm(n, n_train);
index_val = setdiff(1:n, index_train);

x_train = x(index_train);
y_train = y(index_train);
x_val = x(index_val);
y_val = y(index_val);

%% Polynomial Regression
ks = [1, 3, 5];
train_error = zeros(size(ks));
val_error = zeros(size(ks));

for i = 1:length(ks)
    k = ks(i);
    x_matrix = zeros(n_train, k+1);
    x_val_matrix = zeros(n - n_train, k+1);
    
    % Polynomial
    x_matrix(:, 1:k+1) = x_train.^(0:k);
    x_val_matrix(:, 1:k+1) = x_val.^(0:k);
    
    % Normal equation 
    % %vector of coefficient w = (X_TX)-1 X_Ty
    w = (x_matrix' * x_matrix) \ (x_matrix' * y_train);
    y_fit = x_matrix * w;
    y_fit_val = x_val_matrix * w;
    
    % Normalized Sum of Squared Error
    train_error(i) = norm(y_train - y_fit)^2 / n_train;
    val_error(i) = norm(y_val - y_fit_val)^2 / (n - n_train);

    figure;
    hold on;
    plot(x_train, y_train, 'r+');
    plot(x_train, y_fit, 'go');
    xlabel('x');
    ylabel('y');
    legend('Training Samples', 'Fitted Curve');
    title(sprintf('Order %d', k));
    hold off;
end

% Plot Error vs Order
figure;
plot(ks, train_error, 'rx-');
hold on;
plot(ks, val_error, 'bo-');
legend('Training error', 'Validation error');
xlabel('Polynomial order');
ylabel('Regression error');
title('Regression error versus polynomial order');
