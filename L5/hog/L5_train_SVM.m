% HOGi "uczące"
training_count = 18;  % liczba treningowych zdjec osob

imds_full = imageDatastore("imgs/", "IncludeSubfolders", true, "LabelSource", "foldernames");
countEachLabel(imds_full)
% Przycinamy zbiór żeby było po training_count obrazów per klasa:
[trainingSet, testSet] = splitEachLabel(imds_full, training_count);
countEachLabel(trainingSet)

% Jednak bierzemy cały set - im więcej neg tym lepiej
trainingSet = imds_full;


% Test rozmiaru komórki - potrzebny do obliczeń!
% Rozmiar komórki wybieramy dynamicznie, żeby potem
% rozmiar HOGów był stały
num_cells_x = 8;
num_cells_y = 8;
img = imread("imgs/pos/person_2.png");
[y,x] = getCellSize(img, num_cells_y, num_cells_x);
cellSize = [y x];
[hog_test, vis] = extractHOGFeatures(img,'CellSize',cellSize);
hogFeatureSize = length(hog_test);
% Podgląd powyższego testu
figure;
imshow(img);
hold on;
plot(vis);

% Wyciągamy HOGi ze wszystkich obrazków
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages,hogFeatureSize,'single');

for i=1:numImages
    img = readimage(trainingSet,i);

    [y,x] = getCellSize(img, num_cells_y, num_cells_x);
    cellSize = [y x];

%     imshow(img);
%     hold on;
%     plot(vis);
    [tmp_hog, vis] = extractHOGFeatures(img,'CellSize',cellSize);
    close all;
    imshow(img);
    hold on;
    plot(vis);
    trainingFeatures(i, :) = tmp_hog;
end

trainingLabels = trainingSet.Labels;
% multiclass:
% svm_classifier = fitcecoc(trainingFeatures, trainingLabels);
% binary class:
svm_classifier = fitcsvm(trainingFeatures, trainingLabels);
svm_classifier = fitSVMPosterior(svm_classifier);  % dzięki temu wynikiem są pstwa 0..1

% Test
numImages = numel(testSet.Files);
testFeatures = zeros(numImages,hogFeatureSize,'single');

for i=1:numImages
    img = readimage(testSet,i);

    [y,x] = getCellSize(img, num_cells_y, num_cells_x);
    cellSize = [y x];

%     imshow(img);
%     hold on;
%     plot(vis);
    
    [tmp_hog, vis] = extractHOGFeatures(img,'CellSize',cellSize);
    testFeatures(i, :) = tmp_hog;
end

[predictedLabels, scores] = predict(svm_classifier, testFeatures);
figure;
confusionchart(testSet.Labels, predictedLabels);