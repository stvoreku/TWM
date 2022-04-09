
% przed wykonaniem tego skryptu nale¿y najpierw
% uruchomiæ L3_bake_histograms.m
% zabieg ten oszczêdza powtarzania tych samych d³ugich obliczen
% podczas prototypowania
newX = test_hist;


% Definicja parametrów pod SVM
Gamma = 1;
C = 1;

% Utworzenie modeli SVM
SVMModel_deli_vs_bathroom = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'bathroom','deli'},... 
    'KernelScale', Gamma, 'BoxConstraint', C);

SVMModel_deli_vs_greenhouse = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'greenhouse','deli'},... 
    'KernelScale', Gamma, 'BoxConstraint', C);

[label1,score1] = predict(SVMModel_deli_vs_bathroom, newX)
[label2,score2] = predict(SVMModel_deli_vs_greenhouse, newX);

% jak deli to pierwszy wynik negatywny drugi pozytywny
% jak bathroom/greenhouse to vice versa
% mo¿na wiêc optymalizowaæ po wartoœci bezwzglêdnej

% SVM tuning
% KernelScale = Gamma = Sigma 
% BoxConstraintValue = C = Penalty parameter

% 0.0001 < Gamma < 10
% 0.1 < C < 100

avg_certainty1 = mean(abs(score1(:, 1)))

% --- Grid search --- %
gamma_min = 1;
gamma_max = 5;
gamma_count = 6; % ile liczb w gridzie
constr_min = 1;
constr_max = 5;
constr_count = 6; % ile liczb w gridzie

grid_gamma = linspace(gamma_min, gamma_max, gamma_count)
grid_contraint = linspace(constr_min, constr_max, constr_count)





