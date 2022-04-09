% X,Y?
close all;

% Liczba obrazów treningowych na klasê
cnt_train = 60;

% Liczba obrazów testowych na klasê
cnt_test = 15;

% Wybrane klasy obiektów
img_classes = {'deli', 'greenhouse', 'bathroom'};

% Liczba cech wybierana na ka¿dym obrazie
feats_det = 100;

% Metoda wyboru cech (true - jednorodnie w ca³ym obrazie, false - najsilniejsze)
feats_uniform = true;

% Wielko¶æ s³ownika
words_cnt = 30;


% imagedatastorage to format daj±cy zdjêcia + labele
imds_full = imageDatastore("indoor_images/", "IncludeSubfolders", true, "LabelSource", "foldernames");
countEachLabel(imds_full)

% wykrajamy ma³e imds, jako zestaw do trenowania dla nas
[imds, imtest] = splitEachLabel(imds_full, cnt_train, cnt_test, 'Include', img_classes);
countEachLabel(imds)

%przyk³adowe wy¶wietlenie jednego obrazu z zaznaczonymi wykrytymi obiektami
%po krawêdziach
% I = readImage(imds.Files{43});
% imshow(I); 
% hold on;
% pts = getFeaturePoints(I, feats_det, false);
% plot(pts);

% Wyznaczanie punktów dla wszystkich obrazów
files_cnt = length(imds.Files);
all_points = cell(files_cnt, 1);
total_features = 0;

for i=1:files_cnt
    I = readImage(imds.Files{i});
    all_points{i} = getFeaturePoints(I, feats_det, feats_uniform);
    total_features = total_features + length(all_points{i});
end

file_ids = zeros(total_features, 2);
curr_idx = 1;
for i=1:files_cnt
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 1) = i;
    file_ids(curr_idx:curr_idx+length(all_points{i})-1, 2) = 1:length(all_points{i});
    curr_idx = curr_idx + length(all_points{i});
end

all_features = zeros(total_features, 64, 'single');
curr_idx = 1;
for i=1:files_cnt
    I = readImage(imds.Files{i});
    curr_features = extractFeatures(rgb2gray(I), all_points{i});
    all_features(curr_idx:curr_idx+length(all_points{i})-1, :) = curr_features;
    curr_idx = curr_idx + length(all_points{i});
end

[idx, words, sumd, D] = kmeans(all_features, words_cnt, "MaxIter", 500);

file_hist = zeros(files_cnt, words_cnt);

for i=1:files_cnt
    file_hist(i,:) = histcounts(idx(file_ids(:,1) == i), (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end

X = file_hist;
Y = imds.Labels;

%Przygotowanie histogramów danych testwych
test_hist = zeros(length(imtest.Files), words_cnt);
for i=1:length(imtest.Files)
    I = readImage(imtest.Files{i});
    pts = getFeaturePoints(I, feats_det, feats_uniform);
    feats = extractFeatures(rgb2gray(I), pts);
    test_hist(i,:) = wordHist(feats, words);
end

disp("histograms ready");




