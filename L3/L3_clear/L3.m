close all
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
gamma_min = 0.01;
gamma_max = 10;
gamma_count = 10; % ile liczb w gridzie
constr_min = 0.1;
constr_max = 1000;
constr_count = 10; % ile liczb w gridzie

grid_gamma = linspace(gamma_min, gamma_max, gamma_count);
grid_contraint = linspace(constr_min, constr_max, constr_count);
% Opcjonalnie: nieliniowy grid
% grid_gamma = log(grid_gamma + 1);
% grid_contraint = log(grid_contraint + 1);

% init
Gamma = 1;
C = 1;
avg_certainty1_best = 200.0; % arbitrary big numbers
avg_certainty2_best = 200.0; % arbitrary big numbers
best_gamma1 = Gamma;
best_gamma2 = Gamma;
best_constr1 = C;
best_constr2 = C;

for3dplot1 = zeros(3,constr_count*gamma_count);
for3dplot2 = zeros(3,constr_count*gamma_count);
counter = 1;
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
        %zapisanie na potrzeby wykresu 3D
        for3dplot1(:,counter) = [Gamma, C, avg_certainty1];
        for3dplot2(:,counter) = [Gamma, C, avg_certainty2];
        counter = counter +1;
    end % constr loop
end % gamma loop

% Drukowanie wynikow
params1 = [avg_certainty1_best, best_gamma1, best_constr1];
txt1 = sprintf("avg certainty 1: %f \nGamma 1: %f \nConstraint 1: %f", params1);
disp(txt1);

params2 = [avg_certainty2_best, best_gamma2, best_constr2];
txt2 = sprintf("avg certainty 2: %f \nGamma 2: %f \nConstraint 2: %f", params2);
disp(txt2);

% --- Test with new data --- %

SVMModel_deli_vs_bathroom = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'bathroom','deli'},...
    'KernelScale', best_gamma1, 'BoxConstraint', best_constr1);

SVMModel_deli_vs_greenhouse = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'greenhouse','deli'},...
    'KernelScale', best_gamma2, 'BoxConstraint', best_constr2);

[label1,score1] = predict(SVMModel_deli_vs_bathroom, dane_testowe);
[label2,score2] = predict(SVMModel_deli_vs_greenhouse, dane_testowe);

% ocena wynikow
predicted = merge_results(label1, label2);
original = string(imtest.Labels);
cc = confusionchart(original, predicted);
cc.Title = "SVM z doborem parametrow poprzez grid search";

% --- Porownanie z automatem --- %

if true
SVMModel_deli_vs_bathroom = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'bathroom','deli'}, ...
    'OptimizeHyperparameters','auto');

SVMModel_deli_vs_greenhouse = fitcsvm(X,Y,'KernelFunction','gaussian',...
    'Standardize',false,'ClassNames',{'greenhouse','deli'}, ...
    'OptimizeHyperparameters','auto');

[label1,score1] = predict(SVMModel_deli_vs_bathroom, dane_testowe);
[label2,score2] = predict(SVMModel_deli_vs_greenhouse, dane_testowe);
predicted = merge_results(label1, label2);
original = string(imtest.Labels);
figure;
cc = confusionchart(original, predicted);
cc.Title = "SVM z doborem automatycznym";

figure;
plot3(for3dplot1(1,:),for3dplot1(2,:),for3dplot1(3,:),'o','Color','r','MarkerSize',10)
xlabel('Gamma')
ylabel('C')
zlabel('¦rednia pewno¶æ')
figure;
plot3(for3dplot2(1,:),for3dplot2(2,:),for3dplot2(3,:),'o','Color','r','MarkerSize',10)
xlabel('Gamma')
ylabel('C')
zlabel('¦rednia pewno¶æ')
end

% -- funkcja laczaca wyniki z dwoch svm-ow --- %

function predicted = merge_results(label1, label2)
pred1 = string(label1);
pred2 = string(label2);
predicted = pred1;
    for i=1:length(predicted)
        tmp = [pred1(i), pred2(i)];
        if (pred1(i) == "deli") && (pred2(i) == "deli")
            predicted(i) = "deli";
        elseif ismember("greenhouse", tmp) && ismember("bathroom", tmp)
            predicted(i) = pred2(i);  % umyslny blad, nie umiemy ich rozroznic
        elseif ismember("greenhouse", tmp)
            predicted(i) = "greenhouse";
        elseif ismember("bathroom", tmp)
            predicted(i) = "bathroom";
        end
    end
end



