% 111061702 ex3_3
rng(0, 'v4');  %random seed
%% a.
train_data = readtable('mnist_train.csv');

X_train = table2array(train_data);
digit = 3;
indices = find(X_train(:,1) == digit);
N = 2000;
sampled_indices = randsample(indices, N);

% Create data matrix X
X = double(X_train(sampled_indices, 2:end));
Y = X_train(sampled_indices, 1);
mux = mean(X);
X = X - repmat(mux, N, 1);

%% b. 
% Compute covariance matrix of X
C = cov(X);

% Compute eigenvectors and eigenvalues of covariance matrix
[V, D] = eig(C);

% Sort eigenvectors in descending order of eigenvalues
[eigenvalues, indices] = sort(diag(D), 'descend');
coeff = V(:, indices);

%% c.
% Set range of reduced dimensions and sample sizes to evaluate
L = [1, 10, 50, 250, 784];
N = [500, 1000, 1500, 2000];

% Evaluate reconstruction error for different settings of l and N
for i = 1:length(N)
    indices = find(X_train(:,1) == digit);
    Ni = N(i);
    sampled_indices = randsample(indices, Ni);
    X = double(X_train(sampled_indices, 2:end));
    Y = X_train(sampled_indices, 1);
    mux = mean(X);
    X = X - repmat(mux, Ni, 1);
    sampled_indices = randsample(indices, 200);
    X_sample = double(X_train(sampled_indices, 2:end)); % Randomly sample 200 instances from X

    C = cov(X);
    [V, D] = eig(C);
    [eigenvalues, indices] = sort(diag(D), 'descend');
    coeff = V(:, indices);

    for j = 1:length(L)
        Y = X_sample * coeff(:, 1:L(j)); % Project X_sample onto subspace spanned by first L(j) eigenvectors
        X_reconstructed = Y * coeff(:, 1:L(j))'; % Reconstruct X from Y
        if i == 2
            test_image = zeros(28, 28);
            for i1 = 1:28
                for j1 = 1:28
                    test_image(i1, j1) = X_reconstructed(1,i1*28-28+j1);
                end
            end
            figure;
            image(test_image);
            title(sprintf('l = %d', L(j)))
        end
        mse = mean(mean((X_sample - X_reconstructed).^2)); % Compute mean squared error
        fprintf('N = %d, l = %d, MSE = %f\n', N(i), L(j), mse);
    end
end