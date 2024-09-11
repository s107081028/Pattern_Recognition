% 111061702 ex4_2
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

%% k =2
% Set number of clusters
K = 2;

% Initialize centroids randomly
centroids = x(randperm(size(x, 1), K), :);

% Initialize cluster assignments
cluster_assignments = zeros(size(x, 1), 1);

% Repeat until convergence
while true
    
    % Compute distances to centroids for each data point
    distances = pdist2(x, centroids);
    
    % Assign each data point to the closest centroid
    [~, cluster_assignments_new] = min(distances, [], 2);
    
    % Check for convergence
    if all(cluster_assignments == cluster_assignments_new)
        break;
    end
    
    % Update cluster assignments
    cluster_assignments = cluster_assignments_new;
    
    % Update centroids
    for k = 1:K
        centroids(k,:) = mean(x(cluster_assignments == k, :), 1);
    end
end

% Plot results
figure;
scatter(x(cluster_assignments==1, 1), x(cluster_assignments==1, 2), 'r');
hold on;
scatter(x(cluster_assignments==2, 1), x(cluster_assignments==2, 2), 'b');
hold off;
xlabel('X_1');
ylabel('X_2');
legend('Cluster 1', 'Cluster 2');

%% k = 3
% Set number of clusters
K = 3;

% Initialize centroids randomly
centroids = x(randperm(size(x, 1), K), :);

% Initialize cluster assignments
cluster_assignments = zeros(size(x, 1), 1);

% Repeat until convergence
while true
    
    % Compute distances to centroids for each data point
    distances = pdist2(x, centroids);
    
    % Assign each data point to the closest centroid
    [~, cluster_assignments_new] = min(distances, [], 2);
    
    % Check for convergence
    if all(cluster_assignments == cluster_assignments_new)
        break;
    end
    
    % Update cluster assignments
    cluster_assignments = cluster_assignments_new;
    
    % Update centroids
    for k = 1:K
        centroids(k,:) = mean(x(cluster_assignments == k, :), 1);
    end
end

% Plot results
figure;
scatter(x(cluster_assignments==1, 1), x(cluster_assignments==1, 2), 'r');
hold on;
scatter(x(cluster_assignments==2, 1), x(cluster_assignments==2, 2), 'g');
scatter(x(cluster_assignments==3, 1), x(cluster_assignments==3, 2), 'b');
hold off;
xlabel('X_1');
ylabel('X_2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');

%% k = 4
% Set number of clusters
K = 4;

% Initialize centroids randomly
centroids = x(randperm(size(x, 1), K), :);

% Initialize cluster assignments
cluster_assignments = zeros(size(x, 1), 1);

% Repeat until convergence
while true
    
    % Compute distances to centroids for each data point
    distances = pdist2(x, centroids);
    
    % Assign each data point to the closest centroid
    [~, cluster_assignments_new] = min(distances, [], 2);
    
    % Check for convergence
    if all(cluster_assignments == cluster_assignments_new)
        break;
    end
    
    % Update cluster assignments
    cluster_assignments = cluster_assignments_new;
    
    % Update centroids
    for k = 1:K
        centroids(k,:) = mean(x(cluster_assignments == k, :), 1);
    end
end

% Plot results
figure;
scatter(x(cluster_assignments==1, 1), x(cluster_assignments==1, 2), 'r');
hold on;
scatter(x(cluster_assignments==2, 1), x(cluster_assignments==2, 2), 'g');
scatter(x(cluster_assignments==3, 1), x(cluster_assignments==3, 2), 'b');
scatter(x(cluster_assignments==4, 1), x(cluster_assignments==4, 2), 'm');
hold off;
xlabel('X_1');
ylabel('X_2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4');
