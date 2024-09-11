% 111061702 ex1_2
%% Generate Data
randn('seed', 0);
m1 = [1 1]';
m2 = [14 7]';
m3 = [16 1]';
Sigma = [5 3; 3 4];
R1 = mvnrnd(m1, Sigma, 334);
R2 = mvnrnd(m2, Sigma, 333);
R3 = mvnrnd(m3, Sigma, 333);
X2 = [R1; R2; R3];

% Ground Truth
Gt = zeros(1000, 1);
Gt(1:334) = 1;
Gt(335:667) = 2;
Gt(668:1000) = 3;

% Plot Data
figure;
hold on;
scatter(R1(:, 1), R1(:, 2), 'r+');
scatter(R2(:, 1), R2(:, 2), 'g+');
scatter(R3(:, 1), R3(:, 2), 'b+');
hold off;

%% Bayesian Case4
Pred_B = zeros(1000, 1);
for i = 1:1000
    x = X2(i, :)';
    Dis_1 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m1)' *  x + (-1/2) * m1' * inv(Sigma) * m1 - 1/2 * log(det(Sigma)) + log(333/1000);
    Dis_2 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m2)' *  x + (-1/2) * m2' * inv(Sigma) * m2 - 1/2 * log(det(Sigma)) + log(333/1000);
    Dis_3 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m3)' *  x + (-1/2) * m3' * inv(Sigma) * m3 - 1/2 * log(det(Sigma)) + log(334/1000);
    [val, idx] = max([Dis_1, Dis_2, Dis_3]);
    Pred_B(i) = idx;
end

% Error Rate
Err_B = error(Pred_B, Gt, 1000);
fprintf('X2: Error Rate of Bayesian = %f\n', Err_B);

%% Euclidean Case1
Pred_E = zeros(1000, 1);
for i = 1:1000
    x = X2(i, :)';
    Dis_1 = sum((x - m1).^2);
    Dis_2 = sum((x - m2).^2);
    Dis_3 = sum((x - m3).^2);
    [val, idx] = min([Dis_1, Dis_2, Dis_3]);
    Pred_E(i) = idx;
end

% Error Rate
Err_E = error(Pred_E, Gt, 1000);
fprintf('X2: Error Rate of Euclidean = %f\n', Err_E);

%% Mahalanobis Case3
Pred_M = zeros(1000, 1);
for i = 1:1000
    x = X2(i, :)';
    Dis_1 = (-1/2) * ((x - m1)' * inv(Sigma) * (x -m1));
    Dis_2 = (-1/2) * ((x - m2)' * inv(Sigma) * (x -m2));
    Dis_3 = (-1/2) * ((x - m3)' * inv(Sigma) * (x -m3));
    [val, idx] = max([Dis_1, Dis_2, Dis_3]);
    Pred_M(i) = idx;
end

% Error Rate
Err_M = error(Pred_M, Gt, 1000);
fprintf('X2: Error Rate of Mahalanobis = %f\n', Err_M);

%% Function
function Err = error(pred, gt, N)
    count = 0;
    for i = 1:1000
        if gt(i) ~= pred(i)
            count = count + 1;
        end
    end
    Err = count / N;
end

