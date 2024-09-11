% 111061702 ex3_1
rng(0, 'v4');  %random seed
%% a.(i) Generate data sets
% Set parameters
n = 100;
mu = [-10 -10; -10 10; 10 -10; 10 10];
sigma = 0.2*eye(2);

X1 = mvnrnd(mu(1,:), sigma, n);
X2 = mvnrnd(mu(2,:), sigma, n);
X3 = mvnrnd(mu(3,:), sigma, n);
X4 = mvnrnd(mu(4,:), sigma, n);

% Plot data sets
figure;
scatter(X1(:,1), X1(:,2), 'r'); hold on;
scatter(X2(:,1), X2(:,2), 'b');
scatter(X3(:,1), X3(:,2), 'g');
scatter(X4(:,1), X4(:,2), 'm');
xlabel('X_1'); ylabel('X_2'); title('4-Class 2-Dimension');
%legend('Set 1', 'Set 2', 'Set 3', 'Set 4');

%% a.(ii) Compute mean vectors and covariance matrices for each set
mu1 = mean(X1); cov1 = cov(X1);
mu2 = mean(X2); cov2 = cov(X2);
mu3 = mean(X3); cov3 = cov(X3);
mu4 = mean(X4); cov4 = cov(X4);
mu_all = mean(vertcat(X1, X2, X3, X4));

% Compute Sw, within-class scatter matrix
Sw = cov1 + cov2 + cov3 + cov4;
Sw = Sw * 0.25;

% Compute Sb, between-class scatter matrix
Sb = (mu1 - mu_all).'*(mu1 - mu_all) + ...
     (mu2 - mu_all).'*(mu2 - mu_all) + ...
     (mu3 - mu_all).'*(mu3 - mu_all) + ...
     (mu4 - mu_all).'*(mu4 - mu_all);
Sb = Sb * 0.25;

% Compute Sm, mixture scatter matrix
Sm = Sw + Sb;

fprintf('(a)\n');
fprintf('Sw: [%g %g; %g %g]\n', Sw(1), Sw(2), Sw(3), Sw(4));
fprintf('Sb: [%g %g; %g %g]\n', Sb(1), Sb(2), Sb(3), Sb(4));
fprintf('Sm: [%g %g; %g %g]\n', Sm(1), Sm(2), Sm(3), Sm(4));
%% a.(iii) Compute J3
J3 = trace(inv(Sw)*Sm);
fprintf('J3: %g\n', J3);

%% b.
n = 100;
mu = [-1 -1; -1 1; 1 -1; 1 1];
sigma = 0.2*eye(2);

X1 = mvnrnd(mu(1,:), sigma, n);
X2 = mvnrnd(mu(2,:), sigma, n);
X3 = mvnrnd(mu(3,:), sigma, n);
X4 = mvnrnd(mu(4,:), sigma, n);

figure;
scatter(X1(:,1), X1(:,2), 'r'); hold on;
scatter(X2(:,1), X2(:,2), 'b');
scatter(X3(:,1), X3(:,2), 'g');
scatter(X4(:,1), X4(:,2), 'm');
xlabel('X_1'); ylabel('X_2'); title('4-Class 2-Dimension');

mu1 = mean(X1); cov1 = cov(X1);
mu2 = mean(X2); cov2 = cov(X2);
mu3 = mean(X3); cov3 = cov(X3);
mu4 = mean(X4); cov4 = cov(X4);
mu_all = mean(vertcat(X1, X2, X3, X4));

Sw = cov1 + cov2 + cov3 + cov4;
Sw = Sw * 0.25;

Sb = (mu1 - mu_all).'*(mu1 - mu_all) + ...
     (mu2 - mu_all).'*(mu2 - mu_all) + ...
     (mu3 - mu_all).'*(mu3 - mu_all) + ...
     (mu4 - mu_all).'*(mu4 - mu_all);
Sb = Sb * 0.25;

Sm = Sw + Sb;

fprintf('(b)\n');
fprintf('Sw: [%g %g; %g %g]\n', Sw(1), Sw(2), Sw(3), Sw(4));
fprintf('Sb: [%g %g; %g %g]\n', Sb(1), Sb(2), Sb(3), Sb(4));
fprintf('Sm: [%g %g; %g %g]\n', Sm(1), Sm(2), Sm(3), Sm(4));

J3 = trace(inv(Sw)*Sm);
fprintf('J3: %g\n', J3);

%% c.
n = 100;
mu = [-10 -10; -10 10; 10 -10; 10 10];
sigma = 3*eye(2);

X1 = mvnrnd(mu(1,:), sigma, n);
X2 = mvnrnd(mu(2,:), sigma, n);
X3 = mvnrnd(mu(3,:), sigma, n);
X4 = mvnrnd(mu(4,:), sigma, n);

figure;
scatter(X1(:,1), X1(:,2), 'r'); hold on;
scatter(X2(:,1), X2(:,2), 'b');
scatter(X3(:,1), X3(:,2), 'g');
scatter(X4(:,1), X4(:,2), 'm');
xlabel('X_1'); ylabel('X_2'); title('4-Class 2-Dimension');

mu1 = mean(X1); cov1 = cov(X1);
mu2 = mean(X2); cov2 = cov(X2);
mu3 = mean(X3); cov3 = cov(X3);
mu4 = mean(X4); cov4 = cov(X4);
mu_all = mean(vertcat(X1, X2, X3, X4));

Sw = cov1 + cov2 + cov3 + cov4;
Sw = Sw * 0.25;

Sb = (mu1 - mu_all).'*(mu1 - mu_all) + ...
     (mu2 - mu_all).'*(mu2 - mu_all) + ...
     (mu3 - mu_all).'*(mu3 - mu_all) + ...
     (mu4 - mu_all).'*(mu4 - mu_all);
Sb = Sb * 0.25;

Sm = Sw + Sb;

fprintf('(c)\n');
fprintf('Sw: [%g %g; %g %g]\n', Sw(1), Sw(2), Sw(3), Sw(4));
fprintf('Sb: [%g %g; %g %g]\n', Sb(1), Sb(2), Sb(3), Sb(4));
fprintf('Sm: [%g %g; %g %g]\n', Sm(1), Sm(2), Sm(3), Sm(4));

J3 = trace(inv(Sw)*Sm);
fprintf('J3: %g\n', J3);
