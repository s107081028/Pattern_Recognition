% 111061702 ex2_4
rng(0, 'v4');  %random seed
%% Generate Data (a)
mu = [1 1];
sigma = [5 3; 3 4];
N = 1000;
X = mvnrnd(mu, sigma, N);

% Plot Data
figure;
hold on;
scatter(X(:, 1), X(:, 2), 'r+');
hold off;
%% Estimation (a)
fprintf('(a)\n');

mu_hat_ML = mean(X);
% mu_hat = sum(X - mu_hat_ML) / N;
fprintf('True Mu:[%g %g]\n', mu(1), mu(2));
fprintf('Unbiased Estimate of Mean mu_hat_ML:[%g %g]\n', mu_hat_ML(1), mu_hat_ML(2));

sigma_ML = cov(X, 1);
sigma_hat_ML = ((N - 1) / N) * sigma_ML;
fprintf('True sigma: [%g %g; %g %g]\n', sigma(1), sigma(2), sigma(3), sigma(4));
fprintf('biased Estimate of Variance sigma_hat_ML: [%g %g; %g %g]\n', sigma_hat_ML(1), sigma_hat_ML(2), sigma_hat_ML(3), sigma_hat_ML(4));

%% Generate Data (b)
mu = [10 5];
sigma = [7 4; 4 5];
N = 1000;
X = mvnrnd(mu, sigma, N);

% Plot Data
figure;
hold on;
scatter(X(:, 1), X(:, 2), 'r+');
hold off;
%% Estimation (b)
fprintf('(b)\n');

mu_hat_ML = mean(X);
fprintf('True Mu:[%g %g]\n', mu(1), mu(2));
fprintf('Unbiased Estimate of Mean mu_hat_ML:[%g %g]\n', mu_hat_ML(1), mu_hat_ML(2));

sigma_ML = cov(X, 1);
sigma_hat_ML = ((N - 1) / N) * sigma_ML;
fprintf('True sigma: [%g %g; %g %g]\n', sigma(1), sigma(2), sigma(3), sigma(4));
fprintf('biased Estimate of Variance sigma_hat_ML: [%g %g; %g %g]\n', sigma_hat_ML(1), sigma_hat_ML(2), sigma_hat_ML(3), sigma_hat_ML(4));
