% 111061702 ex4_1
rng(0, 'v4');  %random seed
%% Generate dataset
mu = [1 1; 4 4; 10 2];
sigma = cat(3, [1 0.4; 0.4 1], [1 -0.6; -0.6 1], [1 0; 0 1]);
n = 500;
x = zeros(n, 2);

for i = 1 : n/4
    x(i*4-3:i*4-2,:) = mvnrnd(mu(2,:), sigma(:,:,2), 2); % generate the first 2 samples from the 2nd Gaussian
    x(i*4-1,:) = mvnrnd(mu(1,:), sigma(:,:,1)); % generate the 3rd sample from the 1st Gaussian
    x(i*4,:) = mvnrnd(mu(3,:), sigma(:,:,3)); % generate the 4th sample from the 3rd Gaussian
end

% Plot data sets
figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');

%% (a) EM algorithm
k = 3;
iter = 0;
max_iter = 100;
tol = 1e-6; % convergence tolerance
p = ones(k,1)/k; % mixing coefficients
mu_em = [2 2; 6 6; 12 2]; % initial means
sigma_em = cat(3, [2 0; 0 2], [2 0; 0 2], [2 0; 0 2]); % initial covariances for components

log_likelihood = 0;
log_likelihood_threshold = -Inf;

while iter < max_iter && abs(log_likelihood - log_likelihood_threshold) > tol
    iter = iter + 1;
    log_likelihood_threshold = log_likelihood;
    
    % Expectation step
    gamma = zeros(n,k);
    for i = 1:k
        gamma(:,i) = p(i) * mvnpdf(x, mu_em(i,:), sigma_em(:,:,i));
    end
    gamma = gamma ./ sum(gamma,2);
    
    % Maximization step
    Nk = sum(gamma,1); % sum of posterior
    for i = 1:k
        mu_em(i,:) = sum(gamma(:,i).*x,1) ./ Nk(i);
        sigma_em(:,:,i) = (x - mu_em(i,:))' * diag(gamma(:,i)) * (x - mu_em(i,:)) ./ Nk(i);
        p(i) = Nk(i) / n;
    end
    
    % Compute log-likelihood
    log_likelihood = sum(log(sum(bsxfun(@times, gamma, p'), 2)));
end

% Results
fprintf('Number of iterations: %d\n', iter);
fprintf('Log-likelihood: %.6f\n', log_likelihood);
disp('Mixture coefficients:'); disp(p);
disp('Means:'); disp(mu_em);
disp('Covariances:'); disp(sigma_em);

%% (d) Plot data sets
figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');
hold on;

% Plot estimated mixture components
colors = ['b', 'g', 'm'];
for i = 1:k
    plot_gaussian_2d(mu_em(i,:), sigma_em(:,:,i), colors(i));
end
legend('Data', 'Component 1', 'Component 2', 'Component 3');

%% (b) EM algorithm
mu = [1 1; 3 3; 6 2];
sigma = cat(3, [1 0.4; 0.4 1], [1 -0.6; -0.6 1], [1 0; 0 1]);
n = 500;
x = zeros(n, 2);

for i = 1 : n/4
    x(i*4-3:i*4-2,:) = mvnrnd(mu(2,:), sigma(:,:,2), 2); % generate the first 2 samples from the 2nd Gaussian
    x(i*4-1,:) = mvnrnd(mu(1,:), sigma(:,:,1)); % generate the 3rd sample from the 1st Gaussian
    x(i*4,:) = mvnrnd(mu(3,:), sigma(:,:,3)); % generate the 4th sample from the 3rd Gaussian
end

% Plot data sets
figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');

k = 3;
iter = 0;
max_iter = 100;
tol = 1e-6;
p = ones(k,1)/k;
mu_em = [2 2; 6 6; 12 2];
sigma_em = cat(3, [2 0; 0 2], [2 0; 0 2], [2 0; 0 2]);

log_likelihood = 0;
log_likelihood_threshold = -Inf;

while iter < max_iter && abs(log_likelihood - log_likelihood_threshold) > tol
    iter = iter + 1;
    log_likelihood_threshold = log_likelihood;
    
    % Expectation step
    gamma = zeros(n,k);
    for i = 1:k
        gamma(:,i) = p(i) * mvnpdf(x, mu_em(i,:), sigma_em(:,:,i));
    end
    gamma = gamma ./ sum(gamma,2);
    
    % Maximization step
    Nk = sum(gamma,1);
    for i = 1:k
        mu_em(i,:) = sum(gamma(:,i).*x,1) ./ Nk(i);
        sigma_em(:,:,i) = (x - mu_em(i,:))' * diag(gamma(:,i)) * (x - mu_em(i,:)) ./ Nk(i);
        p(i) = Nk(i) / n;
    end
    
    % Compute log-likelihood
    log_likelihood = sum(log(sum(bsxfun(@times, gamma, p'), 2)));
end

fprintf('Number of iterations: %d\n', iter);
fprintf('Log-likelihood: %.6f\n', log_likelihood);
disp('Mixture coefficients:'); disp(p);
disp('Means:'); disp(mu_em);
disp('Covariances:'); disp(sigma_em);

figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');
hold on;

colors = ['b', 'g', 'm'];
for i = 1:k
    plot_gaussian_2d(mu_em(i,:), sigma_em(:,:,i), colors(i));
end
legend('Data', 'Component 1', 'Component 2', 'Component 3');

%% (c) EM algorithm
mu = [1 1; 2 2; 3 1];
sigma = cat(3, [1 0.4; 0.4 1], [1 -0.6; -0.6 1], [1 0; 0 1]);
n = 500;
x = zeros(n, 2);

for i = 1 : n/4
    x(i*4-3:i*4-2,:) = mvnrnd(mu(2,:), sigma(:,:,2), 2); % generate the first 2 samples from the 2nd Gaussian
    x(i*4-1,:) = mvnrnd(mu(1,:), sigma(:,:,1)); % generate the 3rd sample from the 1st Gaussian
    x(i*4,:) = mvnrnd(mu(3,:), sigma(:,:,3)); % generate the 4th sample from the 3rd Gaussian
end

% Plot data sets
figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');

k = 3;
iter = 0;
max_iter = 100;
tol = 1e-6;
p = ones(k,1)/k;
mu_em = [1 1; 2 2; 3 1];
sigma_em = cat(3, [2 0; 0 2], [2 0; 0 2], [2 0; 0 2]);

log_likelihood = 0;
log_likelihood_threshold = -Inf;

while iter < max_iter && abs(log_likelihood - log_likelihood_threshold) > tol
    iter = iter + 1;
    log_likelihood_threshold = log_likelihood;
    
    % Expectation step
    gamma = zeros(n,k);
    for i = 1:k
        sigma_em(:,:,i) = (sigma_em(:,:,i)+ sigma_em(:,:,i).') / 2;
        gamma(:,i) = p(i) * mvnpdf(x, mu_em(i,:), sigma_em(:,:,i));
    end
    gamma = gamma ./ sum(gamma,2);
    
    % Maximization step
    Nk = sum(gamma,1);
    for i = 1:k
        mu_em(i,:) = sum(gamma(:,i).*x,1) ./ Nk(i);
        sigma_em(:,:,i) = (x - mu_em(i,:))' * diag(gamma(:,i)) * (x - mu_em(i,:)) ./ Nk(i);
        p(i) = Nk(i) / n;
    end
    
    % Compute log-likelihood
    log_likelihood = sum(log(sum(bsxfun(@times, gamma, p'), 2)));
end

fprintf('Number of iterations: %d\n', iter);
fprintf('Log-likelihood: %.6f\n', log_likelihood);
disp('Mixture coefficients:'); disp(p);
disp('Means:'); disp(mu_em);
disp('Covariances:'); disp(sigma_em);

figure;
scatter(x(:,1), x(:,2), 'r');
xlabel('X_1');
ylabel('X_2');
hold on;

colors = ['b', 'g', 'm'];
for i = 1:k
    plot_gaussian_2d(mu_em(i,:), sigma_em(:,:,i), colors(i));
end
legend('Data', 'Component 1', 'Component 2', 'Component 3');

%% (d) Visualize the Result
function plot_gaussian_2d(mu, sigma, color)
    n = 100;
    x = linspace(mu(1)-3*sqrt(sigma(1,1)), mu(1)+3*sqrt(sigma(1,1)), n);
    y = linspace(mu(2)-3*sqrt(sigma(2,2)), mu(2)+3*sqrt(sigma(2,2)), n);
    [X, Y] = meshgrid(x, y);
    Z = mvnpdf([X(:) Y(:)], mu, sigma);
    Z = reshape(Z, n, n);
    contour(X, Y, Z, color);
end