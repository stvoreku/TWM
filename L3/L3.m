
% przed wykonaniem tego skryptu nalezy najpierw
% uruchomic L3_bake_histograms.m
% zabieg ten oszczedza powtarzania tych samych dlugich obliczen
% podczas prototypowania
dane_testowe = test_hist;
dane_treningowe = X;

% SVM tuning
% KernelScale = Gamma = Sigma 
% BoxConstraintValue = C = Penalty parameter

% 0.0001 < Gamma < 10
% 0.1 < C < 100

% Przygotowanie "maski" pod punkty
mask = ones(3*cnt_train, 1);
mask(cnt_train:2*cnt_train) = mask(cnt_train:2*cnt_train) * -1;
% wyniki sa w skali (-1, 1), gdzie -1 oznacza np. "jest deli"
% a 1 "nie jest deli". Maska zmienia znak wynikow tak,
% aby wynik poprawny byl ujemny,
% a false positive dodatni;
% wtedy srednia wynikow jest dobrym wyznacznikiem "jakosci" SVM
% ktory mozna minimalizowac w grid search

% --- Grid search --- %
gamma_min = 1;
gamma_max = 5;
gamma_count = 6; % ile liczb w gridzie
constr_min = 1;
constr_max = 5;
constr_count = 6; % ile liczb w gridzie

grid_gamma = linspace(gamma_min, gamma_max, gamma_count);
grid_contraint = linspace(constr_min, constr_max, constr_count);

% init
Gamma = 1;
C = 1;
avg_certainty1_best = 200.0; % arbitrary big numbers
avg_certainty2_best = 200.0; % arbitrary big numbers
best_gamma1 = Gamma;
best_gamma2 = Gamma;
best_constr1 = C;
best_constr2 = C;

for i = 1:gamma_count
    for j = 1:constr_count
        % new SVMs (kernel scale can't be changed)
        Gamma = grid_gamma(i);
        C = grid_contraint(j);
        SVMModel_deli_vs_bathroom = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'bathroom','deli'},...
    'KernelScale', Gamma, 'BoxConstraint', C);

        SVMModel_deli_vs_greenhouse = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'greenhouse','deli'},...
    'KernelScale', Gamma, 'BoxConstraint', C);

        % predict and count the scores
        [label1,score1] = predict(SVMModel_deli_vs_bathroom, dane_treningowe);
        [label2,score2] = predict(SVMModel_deli_vs_greenhouse, dane_treningowe);

        avg_certainty1 = mean(score1(:, 1).*mask);
        avg_certainty2 = mean(score2(:, 1).*mask);

        % save results if they improved
        if avg_certainty1 < avg_certainty1_best
            avg_certainty1_best = avg_certainty1;
            best_gamma1 = Gamma;
            best_constr1 = C;
        end
        if avg_certainty2 < avg_certainty2_best
            avg_certainty2_best = avg_certainty2;
            best_gamma2 = Gamma;
            best_constr2 = C;
        end
    end % constr loop
end % gamma loop

avg_certainty1_best
avg_certainty2_best

% --- Test with new data --- %

SVMModel_deli_vs_bathroom = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'bathroom','deli'},...
    'KernelScale', best_gamma1, 'BoxConstraint', best_constr1);

SVMModel_deli_vs_greenhouse = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'greenhouse','deli'},...
    'KernelScale', best_gamma2, 'BoxConstraint', best_constr2);

[label1,score1] = predict(SVMModel_deli_vs_bathroom, dane_testowe);
[label2,score2] = predict(SVMModel_deli_vs_greenhouse, dane_testowe);

% porownanie wynikow obu SVM
pred1 = string(label1);
pred2 = string(label2);
predicted = pred1;
for i=1:length(predicted)
    tmp = [pred1(i), pred2(i)];
    if ismember("deli", tmp)
        predicted(i) = "deli";
    else
        predicted(i) = pred2(i);
    end
end

original = string(imtest.Labels);
confusionchart(original, predicted);
