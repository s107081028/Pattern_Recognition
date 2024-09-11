% 111061702 ex4_3
rng(0, 'v4');  %random seed
%% Generate dataset
mu1 = [1 1];
mu2 = [0 0];
sigma = [0.2 0; 0 0.2];
n = 100;

% Generate data for class 1
x1 = mvnrnd(mu1, sigma, n);
x1 = x1(sum(x1,2)>=1,:);

% Generate data for class 2
x2 = mvnrnd(mu2, sigma, n);
x2 = x2(sum(x2,2)<=1,:);

% Concatenate data for both classes
x = [x1; x2];
y = [-ones(size(x1,1),1); ones(size(x2,1),1)];

% Plot data sets
figure;
scatter(x1(:,1), x1(:,2), 'r');
hold on;
scatter(x2(:,1), x2(:,2), 'b');
xlabel('X_1');
ylabel('X_2');
legend('Class 1', 'Class 2');

%%
% Perceptron algorithm
perceptron_w = zeros(size(x,2),1);
b = 0;
eta = 0.01;
decay = 0.00001;
for i = 1:1000
    y_pred = sign(x*perceptron_w+b);
    misclassified = find(y.*y_pred<=0);
    if isempty(misclassified)
        break
    end
    j = misclassified(randi(length(misclassified)));
    perceptron_w = perceptron_w + eta*y(j)*x(j,:)';
    b = b + eta*y(j);
    eta = eta - decay;
end
fprintf("Perceptron Algorithm: w = [%f %f], b = %f\n", perceptron_w(1), perceptron_w(2), b);

% Sum-of-squared-error classifier
X = [ones(size(x,1),1) x];
eta = 0.005;
SSE_b = 0;
SSE_w = zeros(2,1);
SSE_w_temp = zeros(2,1);
for iter = 1:1000
    pred = x*SSE_w + SSE_b;
    err = y - pred;
    % Update weights
    SSE_w = SSE_w + eta*x'*err;
    SSE_b = SSE_b + eta*sum(err);
    % Compute loss (sum of squared errors)
    % loss = sum(err.^2) / n;
    lambda = 0.5;
    loss = ((1-lambda)*sum(err.^2) + lambda*sum(SSE_w.^2)) / n;
    
    % Check convergence
    if loss < 1e-6
        break;
    end
    if (SSE_w == SSE_w_temp)
        break;
    end
    SSE_w_temp = SSE_w;
    eta = eta - decay;
end
%SSE_w = (X'*X)\X'*y;
fprintf("Sum-of-squared-error Classifier: w = [%f %f %f]\n", SSE_w(1), SSE_w(2), SSE_b);

%%
% Plot dataset and decision lines
figure;
hold on;

% Plot data points
scatter(x1(:,1), x1(:,2), 'ro');
scatter(x2(:,1), x2(:,2), 'bx');

% Plot decision line for perceptron
fplot(@(x) (-perceptron_w(1)*x-b)/perceptron_w(2), [-0.5 1.5], 'k--');

% Plot decision line for SSE
fplot(@(x) (-SSE_w(1)*x-SSE_b)/SSE_w(2), [-0.5 1.5], 'g-');

legend('Class 1', 'Class 2', 'Perceptron', 'SSE');
xlabel('X_1');
ylabel('X_2');