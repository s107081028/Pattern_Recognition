% 111061702 ex2_2
rng(0, 'v4');  %random seed
%% Define P function
p = @(x, theta) theta * exp(-theta * x).*(x >= 0);

%% Different Thetas and Ns
thetas = [1/3, 1/2, 1];
for j = 1:length(thetas)
    theta = thetas(j);
    fprintf('When theta = %g\n',theta);

    Ns = [10, 100, 1000];
    x_max = 10 / theta;  % for limit of plot
    
    for i = 1:length(Ns)
        figure;
        hold on;
        N = Ns(i);
     
        x_plt = linspace(0, x_max, 100);
        plot(x_plt, p(x_plt, theta), 'LineWidth', 1, DisplayName='True');

        % theta_ML
        X = exprnd(1/theta, [1, N]);  % exponential random numbers for estimating
        theta_ML = N / sum(X);
        fprintf('  Estimate theta of N = %g: %g\n', N, theta_ML);
        plot(x_plt, p(x_plt, theta_ML), 'LineWidth', 1, 'DisplayName', sprintf('Estimate of N=%g', N));
    
        title(sprintf('theta = %g', theta));
        xlabel('x');
        ylabel('p(x|theta)');
        ylim([0, 1.2*theta]);
        legend('Location', 'best');
        hold off;
    end    
end