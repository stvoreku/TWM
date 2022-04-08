%% Dobór parametrów klasyfikatorów
%% Parametry działania
% Powtarzalne wyniki
close all ;
clear all;
rng('default') ;

% Liczba obrazów treningowych na klasę
cnt_train = 70 ;

% Liczba obrazów testowych na klasę
cnt_test = 30;

% Wybrane klasy obiektów
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Liczba cech wybierana na każdym obrazie
feats_det = 100;

% Metoda wyboru cech (true - jednorodnie w całym obrazie, false - najsilniejsze)
feats_uniform = true;

% Wielkość słownika
words_cnt = 30 ;

% Detekcja cech
% Ładowanie pełnego zbioru danych z automatycznym podziałem na klasy
% Zbiór danych pochodzi z publikacji: A. Quattoni, and A.Torralba. <http://people.csail.mit.edu/torralba/publications/indoor.pdf 
% _Recognizing Indoor Scenes_>. IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), 2009.
% 
% Pełny zbiór dostępny jest na stronie autorów: <http://web.mit.edu/torralba/www/indoor.html 
% http://web.mit.edu/torralba/www/indoor.html>

imds_full = imageDatastore("indoor_images/", "IncludeSubfolders", true, "LabelSource", "foldernames");
%countEachLabel(imds_full)

% Wybór przykładowych klas i podział na zbiór treningowy i testowy
[imds, imtest] = splitEachLabel(imds_full, cnt_train, cnt_test, 'Include', img_classes);
%countEachLabel(imds)

% Wyznaczenie punktów charakterystycznych we wszystkich obrazach zbioru treningowego
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

% Przygotowanie listy przechowującej indeksy plików i punktów charakterystycznych
file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

% Obliczenie deskryptorów punktów charakterystycznych
all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

% Tworzenie słownika

% Klasteryzacja punktów 
[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 10000);
% Wizualizacja wyliczonych słów

% Wyznaczenie histogramów słów dla każdego obrazu treningowego
file_hist = zeros(files_cnt, words_cnt);
for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

% Wyznaczenie histogramów słów dla każdego obrazu testowego
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

%% Klasyfikator Regresji Logistycznej dla n-klas - pierwsza przymiarka
close all ;
rng('default') ;

featurestrain = file_hist ;
featurestrain = [featurestrain, ones(size(featurestrain,1),1)] ;
[W, cost] = trainsimple(featurestrain, imds.Labels, 0.0000) ;
labelstrain = imds.Labels == unique(imds.Labels)' ;

% Wyniki dla zbioru uczącego
pred = hfun1(W,featurestrain) ;
acc = getAccuracy(pred,labelstrain) * 100 ;
    
% Wyniki dla zbioru testowego
featurestest = test_hist ;
featurestest = [featurestest, ones(size(featurestest,1),1)] ;
labelstest = imtest.Labels == unique(imtest.Labels)' ;
predtest = hfun1(W,featurestest) ;
acctest = getAccuracy(predtest,labelstest) * 100 ;
 
fprintf(1,'Accuracy train: %f\n', acc) ;
fprintf(1,'Accuracy test: %f\n', acctest) ;
 
%% Klasyfikator Regresji Logistycznej dla n-klas - modyfikacja liczby cech (bez regularyzacji)
close all ;
rng('default') ;

feat_counts = 2:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;
for cnt_usedfeatures = feat_counts    
    features = file_hist(:,1:cnt_usedfeatures) ;
    features = [features, ones(size(features,1),1)] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, imds.Labels, 0) ;
    
    costsall = [costsall, median(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, median(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, mean(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% Rysowanie kosztów
figure
plot(feat_counts, costsall, feat_counts, costsvalall, 'LineWidth', 2) ;
title('Funkcja kosztu vs. liczba cech') ;
xlabel('Liczba cech') ; 
ylabel('Funkcja kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji kosztów
figure
plot(feat_counts, costsstd, feat_counts, costsvalstd, 'LineWidth', 2) ;
title('Odch. std. funkcji kosztu vs. liczba cech') ;
xlabel('Liczba cech') ;
ylabel('Odch. std. funkcji kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie skuteczności
figure
plot(feat_counts, accsall, feat_counts, accsvalall, 'LineWidth', 2) ;
title('Skuteczność vs. liczba cech') ;
xlabel('Liczba cech') ;
ylabel('Skuteczność (%)') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji skuteczności
figure
plot(feat_counts, accsstd, feat_counts, accsvalstd, 'LineWidth', 2) ;
title('Odch. std. skuteczności vs. liczba cech') ;
xlabel('Liczba cech') ;
ylabel('Odch. std. skuteczności') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - modyfikacja wielkości zbioru danych (bez regularyzacji)
close all ;
rng('default') ;

%feat_counts = 1:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;

subset_sizes = [] ;
num_features = 5 ;
for subset_prop = 0.2:0.1:0.9 
    part = cvpartition(imds.Labels, 'HoldOut', 1-subset_prop) ;
    
    features = file_hist(part.training(1),1:num_features) ;    
    labels = imds.Labels(part.training(1)) ;
    features = [features, ones(size(features,1),1)] ;
    subset_sizes = [subset_sizes, part.TrainSize] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, labels, 0, 10) ;
    
    costsall = [costsall, median(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, median(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, mean(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% Rysowanie kosztów
figure
plot(subset_sizes, costsall, subset_sizes, costsvalall, 'LineWidth', 2) ;
title(strcat('Funkcja kosztu vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
xlabel('Wielkość zbioru') ; 
ylabel('Funkcja kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji kosztów
figure
plot(subset_sizes, costsstd, subset_sizes, costsvalstd, 'LineWidth', 2) ;
title(strcat('Odch. std. funkcji kosztu vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
xlabel('Wielkość zbioru') ; 
ylabel('Odch. std. funkcji kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie skuteczności
figure
plot(subset_sizes, accsall, subset_sizes, accsvalall, 'LineWidth', 2) ;
title(strcat('Skuteczność vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
xlabel('Wielkość zbioru') ;
ylabel('Skuteczność (%)') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji skuteczności
figure
plot(subset_sizes, accsstd, subset_sizes, accsvalstd, 'LineWidth', 2) ;
title(strcat('Odch. std. skuteczności vs. wielkość zbioru danych numfeatures = ',num2str(num_features))) ;
xlabel('Wielkość zbioru') ;
ylabel('Odch. std. skuteczności') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - wczesne zatrzymanie (bez regularyzacji)
close all ;
rng('default') ;

cnt_usedfeatures = size(file_hist,2) ;

features = file_hist(:,1:cnt_usedfeatures) ;
features = [features, ones(size(features,1),1)] ;
[Ws, costs, costsval, accs, accsval] = crossvalearly(features, imds.Labels, 0) ;
    
% Rysowanie kosztów
figure
%plot(1:length(costs), costs, 1:length(costs), costsval, 'LineWidth', 2) ;
plot(1:length(costs), costs, 1:length(costs), costsval, 'LineWidth', 2) ;
title('Funkcja kosztu vs. liczba iteracji optymalizacji funkcji kosztu') ;
xlabel('Liczba iteracji') ; 
ylabel('Funkcja kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;
ylim([0,4]) ; %Lepsza widoczność

% Rysowanie skuteczności
figure
plot(1:length(accs), accs, 1:length(accsval), accsval, 'LineWidth', 2) ;
title('Skuteczność vs. liczba iteracji optymalizacji funkcji kosztu') ;
xlabel('Liczba iteracji') ; 
ylabel('Skuteczność') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - regularyzacja z normą L2
close all ;
rng('default') ;

%feat_counts = 1:2:size(file_hist,2) ;

costsall = [] ;
costsstd = [] ;
costsvalall = [] ;
costsvalstd = [] ;
accsall = [] ;
accsstd = [] ;
accsvalall = [] ;
accsvalstd = [] ;

%lambdas = logspace(-9,2,15) ;
lambdas = logspace(-4,0,15) ; % Zagęszczenie próby...
features = file_hist ;
features = [features, ones(size(features,1),1)] ;
for lambda = lambdas   
    
    subset_sizes = [subset_sizes, part.TrainSize] ;
    
    [Ws, costs, costsval, accs, accsval] = crossval(features, imds.Labels, lambda, 10) ;
    
    costsall = [costsall, median(costs)] ;
    costsstd = [costsstd, std(costs)] ;
    costsvalall = [costsvalall, median(costsval(~isnan(costsval)))] ;
    costsvalstd = [costsvalstd, std(costsval(~isnan(costsval)))] ;
    accsall = [accsall, mean(accs)] ;
    accsstd = [accsstd, std(accs)] ;
    accsvalall = [accsvalall, median(accsval)] ;
    accsvalstd = [accsvalstd, std(accsval)] ;
end

% Rysowanie kosztów
figure
semilogx(lambdas, costsall, lambdas, costsvalall, 'LineWidth', 2) ;
title('Funkcja kosztu vs. lambda') ;
xlabel('Lambda') ; 
ylabel('Funkcja kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji kosztów
figure
semilogx(lambdas, costsstd, lambdas, costsvalstd, 'LineWidth', 2) ;
title('Odch. std. funkcji kosztu vs. lambda') ;
xlabel('Lambda') ; 
ylabel('Odch. std. funkcji kosztu') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie skuteczności
figure
semilogx(lambdas, accsall, lambdas, accsvalall, 'LineWidth', 2) ;
title('Skuteczność vs. lambda') ;
xlabel('Lambda') ;
ylabel('Skuteczność (%)') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

% Rysowanie niepewności estymacji skuteczności
figure
semilogx(lambdas, accsstd, lambdas, accsvalstd, 'LineWidth', 2) ;
title(strcat('Odch. std. skuteczności vs. lambda ',num2str(num_features))) ;
xlabel('Lambda') ;
ylabel('Odch. std. skuteczności') ;
legend('Zb. treningowy','Zb. walidacyjny') ;

%% Klasyfikator Regresji Logistycznej dla n-klas - z regularyzacją
close all ;
rng('default') ;

featurestrain = file_hist ;
featurestrain = [featurestrain, ones(size(featurestrain,1),1)] ;
[W, cost] = trainsimple(featurestrain, imds.Labels, 0.0007) ;
labelstrain = imds.Labels == unique(imds.Labels)' ;

% Wyniki dla zbioru uczącego
pred = hfun1(W,featurestrain) ;
acc = getAccuracy(pred,labelstrain) * 100 ;
    
% Wyniki dla zbioru testowego
featurestest = test_hist ;
featurestest = [featurestest, ones(size(featurestest,1),1)] ;
labelstest = imtest.Labels == unique(imtest.Labels)' ;
predtest = hfun1(W,featurestest) ;
acctest = getAccuracy(predtest,labelstest) * 100 ;
 
fprintf(1,'Accuracy train: %f\n', acc) ;
fprintf(1,'Accuracy test: %f\n', acctest) ;


%% Funkcje pomocnicze

% Optymalizacja bez ograniczeń z restartami
function [Wout, costout] = optimize1(features,labels)
    Wout = [] ;
    costout = inf ;
    for i=1:10
        W = rand(size(features,2), size(labels,2)) ;
        %cost = costfun1(W(:),features,labels) ;
        options = optimoptions('fminunc','Display','iter', 'MaxFunctionEvaluations', 10e9) ;
        [w,cost,exitflag] = fminunc(@(w) costfun1(w,features,labels),W(:),options) ;        
        if exitflag > 0
             Wout = reshape(w,size(features,2), size(labels,2)) ;
             costout = cost ;
        end
    end
end

function P = getPatch(I, pt, scale, scale_factor)
    x1 = round(pt(1) - 0.5*scale*scale_factor);
    x2 = round(pt(1) + 0.5*scale*scale_factor);
    y1 = round(pt(2) - 0.5*scale*scale_factor);
    y2 = round(pt(2) + 0.5*scale*scale_factor);
    
    [x1, x2, y1, y2] = clipInside(x1, x2, y1, y2, size(I, 1), size(I, 2));
    
    P = imresize(I(y1:y2, x1:x2, :), [64 64]);
end

function [xr1, xr2, yr1, yr2] = clipInside(x1, x2, y1, y2, rows, cols)
    xr1 = min(max(x1, 1), cols);
    xr2 = min(max(x2, 1), cols);
    yr1 = min(max(y1, 1), rows);
    yr2 = min(max(y2, 1), rows);
end

function pts = getFeaturePoints(I, pts_det, pts_uniform)
    if size(I, 3) > 1
        I2 = rgb2gray(I);
    else
        I2 = I;
    end
    
    pts = detectSURFFeatures(I2, 'MetricThreshold', 100);
    if pts_uniform
        pts = selectUniform(pts, pts_det, size(I));
    else
        pts = pts.selectStrongest(pts_det);
    end
end

function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

function [h, P] = visSingleImage(I, pts, feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [dis, lbl] = min(dis, [], 2);
    [~, ids] = sort(dis);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
    P = zeros(words_cnt*64, 30*64, 3, 'uint8');
    pos = zeros(words_cnt, 1);
    for i=1:size(feats, 1)
        id = ids(i);
        x = pos(lbl(id)) * 64;
        pos(lbl(id)) = min(pos(lbl(id)) + 1, 29);
        y = (lbl(id)-1) * 64;
        pat = getPatch(I, pts.Location(id, :), pts.Scale(id), 12);
        pat = insertText(pat, [2, 2], dis(id), 'FontSize', 10, 'BoxOpacity', 0);
        pat = insertText(pat, [1, 1], dis(id), 'FontSize', 10, 'BoxOpacity', 0, 'TextColor', 'white');
        P(y+1:y+64, x+1:x+64, :) = pat;
    end
end

% Wczytanie obrazu i przeskalowanie jeśli jest zbyt duży
function I = readImage(path)
    I = imread(path);
    if size(I,2) > 640
        I = imresize(I, [NaN 640]);
    end
end