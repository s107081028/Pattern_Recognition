% 111061702 ex3_2
rng(0, 'v4');  %random seed
%% a.(i) Generate data sets
% Set parameters
n = 100;
mu = [2 4; 2.5 10];
sigma = eye(2);

X1 = mvnrnd(mu(1,:), sigma, n);
X2 = mvnrnd(mu(2,:), sigma, n);

% Plot data sets
figure;
scatter(X1(:,1), X1(:,2), 'r'); 
hold on;
scatter(X2(:,1), X2(:,2), 'b');
xlabel('X_1'); ylabel('X_2'); title('(a) 2-Class');
xlim([-1 6]);
ylim([0 14]);

%% a.(ii) FDR index
% Compute the mean vectors and pooled covariance matrix
mu = mean([X1; X2]);
Sw = cov([X1; X2]);
mu1 = mean(X1);
mu2 = mean(X2);

% Compute the between-class scatter matrix
Sb = (mu1 - mu).'*(mu1 - mu) + (mu2 - mu).'*(mu2 - mu);

% Compute the Fisher's discriminant ratio index for both features
FDR1 = Sb(1,1) / Sw(1,1);
FDR2 = Sb(2,2) / Sw(2,2);
fprintf('(a):\n');
fprintf('FDR1: %g\n', FDR1);
fprintf('FDR2: %g\n', FDR2);

%% b.
n = 100;
mu = [2 4; 2.5 10];
sigma = 0.25*eye(2);

X1 = mvnrnd(mu(1,:), sigma, n);
X2 = mvnrnd(mu(2,:), sigma, n);

figure;
scatter(X1(:,1), X1(:,2), 'r'); 
hold on;
scatter(X2(:,1), X2(:,2), 'b');
xlabel('X_1'); ylabel('X_2'); title('(b) 2-Class');
xlim([-1 6]);
ylim([0 14]);

mu = mean([X1; X2]);
Sw = cov([X1; X2]);
mu1 = mean(X1);
mu2 = mean(X2);

Sb = (mu1 - mu).'*(mu1 - mu) + (mu2 - mu).'*(mu2 - mu);

FDR1 = Sb(1,1) / Sw(1,1);
FDR2 = Sb(2,2) / Sw(2,2);
fprintf('(b):\n');
fprintf('FDR1: %g\n', FDR1);
fprintf('FDR2: %g\n', FDR2);