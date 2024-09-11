% 111061702 ex1_6
%% Generate Data
randn('seed', 0);
m1 = [1 1]';
m2 = [8 6]';
m3 = [13 1]';
Sigma = 6*eye(2);
R1 = mvnrnd(m1, Sigma, 334);
R2 = mvnrnd(m2, Sigma, 333);
R3 = mvnrnd(m3, Sigma, 333);
X3 = [R1; R2; R3];
R1_z = mvnrnd(m1, Sigma, 334);
R2_z = mvnrnd(m2, Sigma, 333);
R3_z = mvnrnd(m3, Sigma, 333);
Z = [R1_z; R2_z; R3_z];

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
title('X3');
hold off;

figure;
hold on;
scatter(R1_z(:, 1), R1_z(:, 2), 'r+');
scatter(R2_z(:, 1), R2_z(:, 2), 'g+');
scatter(R3_z(:, 1), R3_z(:, 2), 'b+');
title('Z');
hold off;


%% KNN
% Calculate Distance
distance = zeros(1000, 1000);
for i = 1:1000
    for j = 1:1000
        distance(i, j) = (abs(X3(i, 1) - Z(j, 1))^2 + abs(X3(i, 2) - Z(j, 2))^2) ^ 1/2;
    end
end

% k = 1
Pred_KNN = zeros(1000, 1);
[val, idx] = mink(distance, 1, 2);
for i = 1:1000
    Pred_z = zeros(1, 1);
    for j = 1:1
        Pred_z(j) = Gt(idx(i, j));
    end
    Pred_KNN(i) = mode(Pred_z, 2);
end

Err_KNN = error(Pred_KNN, Gt, 1000);
fprintf('X3: Error Rate of KNN(k=1) = %f\n', Err_KNN);
Plot_Conclusion(Pred_KNN, X3, 'KNN(k=1)');

% k = 11
Pred_KNN = zeros(1000, 1);
[val, idx] = mink(distance, 11, 2);
for i = 1:1000
    Pred_z = zeros(1, 11);
    for j = 1:11
        Pred_z(j) = Gt(idx(i, j));
    end
    Pred_KNN(i) = mode(Pred_z, 2);
end

Err_KNN = error(Pred_KNN, Gt, 1000);
fprintf('X3: Error Rate of KNN(k=11) = %f\n', Err_KNN);
Plot_Conclusion(Pred_KNN, X3, 'KNN(k=11)');


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

