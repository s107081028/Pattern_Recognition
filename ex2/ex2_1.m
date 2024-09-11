% 111061702 ex2_1
rng(0, 'v4');  %random seed
%% (a) Generate Data
p = 0.3;
N = 1000;
X = rand(1, N) < p;

%% Estimate
p_ML = sum(X) / N;
fprintf('(a) True p: %g\n', p);
fprintf('Estimate: %g\n', p_ML);

%% (b) Generate Data
p = 0.5;
N = 1000;
X = rand(1, N) < p;

%% Estimate
p_ML = sum(X) / N;
fprintf('(b) True p: %g\n', p);
fprintf('Estimate: %g\n', p_ML);
