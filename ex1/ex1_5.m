% 111061702 ex1_5
%% Generate Data
randn('seed', 0);
m1 = [1 1]';
m2 = [4 4]';
m3 = [8 1]';
Sigma = 2*eye(2);
R1 = mvnrnd(m1, Sigma, 334);
R2 = mvnrnd(m2, Sigma, 333);
R3 = mvnrnd(m3, Sigma, 333);
X5 = [R1; R2; R3];
R1_prime = mvnrnd(m1, Sigma, 800);
R2_prime = mvnrnd(m2, Sigma, 100);
R3_prime = mvnrnd(m3, Sigma, 100);
X5_prime = [R1_prime; R2_prime; R3_prime];

% Ground Truth
Gt = zeros(1000, 1);
Gt(1:334) = 1;
Gt(335:667) = 2;
Gt(668:1000) = 3;

Gt_prime = zeros(1000, 1);
Gt_prime(1:800) = 1;
Gt_prime(801:900) = 2;
Gt_prime(901:1000) = 3;

% Plot Data
figure;
hold on;
scatter(R1(:, 1), R1(:, 2), 'r+');
scatter(R2(:, 1), R2(:, 2), 'g+');
scatter(R3(:, 1), R3(:, 2), 'b+');
hold off;

figure;
hold on;
scatter(R1_prime(:, 1), R1_prime(:, 2), 'r+');
scatter(R2_prime(:, 1), R2_prime(:, 2), 'g+');
scatter(R3_prime(:, 1), R3_prime(:, 2), 'b+');
hold off;

%% Bayesian Case4
% X5
Pred_B = zeros(1000, 1);
for i = 1:1000
    x = X5(i, :)';
    Dis_1 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m1)' *  x + (-1/2) * m1' * inv(Sigma) * m1 - 1/2 * log(det(Sigma)) + log(334/1000);
    Dis_2 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m2)' *  x + (-1/2) * m2' * inv(Sigma) * m2 - 1/2 * log(det(Sigma)) + log(333/1000);
    Dis_3 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m3)' *  x + (-1/2) * m3' * inv(Sigma) * m3 - 1/2 * log(det(Sigma)) + log(333/1000);
    [val, idx] = max([Dis_1, Dis_2, Dis_3]);
    Pred_B(i) = idx;
end

Err_B = error(Pred_B, Gt, 1000);
fprintf('X5: Error Rate of Bayesian = %f\n', Err_B);

% X5_prime
Pred_B_prime = zeros(1000, 1);
for i = 1:1000
    x = X5_prime(i, :)';
    Dis_1 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m1)' *  x + (-1/2) * m1' * inv(Sigma) * m1 - 1/2 * log(det(Sigma)) + log(800/1000);
    Dis_2 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m2)' *  x + (-1/2) * m2' * inv(Sigma) * m2 - 1/2 * log(det(Sigma)) + log(100/1000);
    Dis_3 = x' * ((-1/2) * inv(Sigma)) * x + (inv(Sigma) * m3)' *  x + (-1/2) * m3' * inv(Sigma) * m3 - 1/2 * log(det(Sigma)) + log(100/1000);
    [val, idx] = max([Dis_1, Dis_2, Dis_3]);
    Pred_B_prime(i) = idx;
end

Err_B_prime = error(Pred_B_prime, Gt_prime, 1000);
fprintf('X5_prime: Error Rate of Bayesian = %f\n', Err_B_prime);

%% Euclidean Case1
% X5
Pred_E = zeros(1000, 1);
for i = 1:1000
    x = X5(i, :)';
    Dis_1 = sum((x - m1).^2);
    Dis_2 = sum((x - m2).^2);
    Dis_3 = sum((x - m3).^2);
    [val, idx] = min([Dis_1, Dis_2, Dis_3]);
    Pred_E(i) = idx;
end

Err_E = error(Pred_E, Gt, 1000);
fprintf('X5: Error Rate of Euclidean = %f\n', Err_E);

% X5_prime
Pred_E_prime = zeros(1000, 1);
for i = 1:1000
    x = X5_prime(i, :)';
    Dis_1 = sum((x - m1).^2);
    Dis_2 = sum((x - m2).^2);
    Dis_3 = sum((x - m3).^2);
    [val, idx] = min([Dis_1, Dis_2, Dis_3]);
    Pred_E_prime(i) = idx;
end

Err_E_prime = error(Pred_E_prime, Gt_prime, 1000);
fprintf('X5_prime: Error Rate of Euclidean = %f\n', Err_E_prime);

%% Draw Conclusion
Plot_Conclusion(Pred_B, X5, "Bayesian X5");
Plot_Conclusion(Pred_E, X5, "Euclidean X5");
Plot_Conclusion(Pred_B_prime, X5_prime, "Bayesian X5'");
Plot_Conclusion(Pred_E_prime, X5_prime, "Euclidean X5'");

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

function Plot_Conclusion(pred, X, T)
    figure;
    hold on;
    for i = 1:1000
        switch pred(i)
            case 1
                scatter(X(i, 1), X(i, 2), 'r+');
            case 2
                scatter(X(i, 1), X(i, 2), 'g+');
            case 3
                scatter(X(i, 1), X(i, 2), 'b+');
        end
    end
    title(T);
    hold off;
end

