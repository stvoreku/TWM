% HOGi "uczące"
N = 18;  % liczba treningowych zdjec osob
M = 18;  % liczba negatywów

% Do starego klasyfikatora:
% img = imread("person_1.jpg");
% [hog1, vis] = extractHOGFeatures(img);
% img = imread("person_2.jpg");
% [hog2, vis] = extractHOGFeatures(img);

% for n=1:N
%     im = imread("imgs/pos/person_"+ n +".png");
%     [hog, vis] = extractHOGFeatures(im);
%     hogs{n} = hog;
% end
% 
% for m=1:M
%     im = imread("imgs/neg/neg ("+ m +").jpg");
%     [hog, vis] = extractHOGFeatures(im);
%     hogs_neg{n} = hog;
% end

imds_full = imageDatastore("imgs/", "IncludeSubfolders", true, "LabelSource", "foldernames");
countEachLabel(imds_full)
% Przycinamy zbiór żeby było po N obrazów per klasa:
[trainingSet, testSet] = splitEachLabel(imds_full, N);
countEachLabel(trainingSet)


% Test rozmiaru komórki - potrzebny do obliczeń!
% Rozmiar komórki wybieramy dynamicznie, żeby potem
% rozmiar HOGów był stały
num_cells_x = 6;
num_cells_y = 8;
img = imread("imgs/pos/person_2.png");
[height, width, colour_planes] = size(img);
y = floor(height/num_cells_y);
x = floor(width/num_cells_x);
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
trainingFeatures = zeros(N,hogFeatureSize,'single');

for i=1:numImages
    img = readimage(trainingSet,i);
    i
    % Dobieramy rozmiar komórki jak wcześniej
    [height, width, colour_planes] = size(img);
    y = floor(height/num_cells_y);
    x = floor(width/num_cells_x);
    curr_cellSize = cellSize;
    cellSize = [y x];

%     imshow(img);
%     hold on;
%     plot(vis);
    
    [tmp_hog, vis] = extractHOGFeatures(img,'CellSize',cellSize);
    trainingFeatures(i, :) = tmp_hog;
end
trainingLabels = trainingSet.Labels;
svm_classifier = fitcecoc(trainingFeatures, trainingLabels);
