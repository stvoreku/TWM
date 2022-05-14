% TODO przyciąć obrazki w neg

% HOGi "uczące"
N = 18;  % liczba treningowych zdjec osob
M = 18;  % liczba negatywów

img = imread("person_1.jpg");
[hog1, vis] = extractHOGFeatures(img);
img = imread("person_2.jpg");
[hog2, vis] = extractHOGFeatures(img);

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

% Obrazek testowy i jego HOG
It = imread("people_1.jpg");
[hogt,vist] = extractHOGFeatures(img);
% figure;
% imshow(It);
% hold on;
% plot(vist);

% Okno ruchome - TODO to pewnie trzeba dopasować do num_cells_x/y
sx = 48;
sy = 128;
step = 8;

scale_step = 0.8;
levels = 5;

scale = 1.0;
thr = 60;

dets = [];
scores = [];
labels = {};

for k=1:levels
    scale
    cur_img = imresize(It, scale);
    
    w = size(cur_img, 2);
    h = size(cur_img, 1);
    
    count_x = floor((w - sx) / step)
    count_y = floor((h - sy) / step)
    
    out = zeros(count_y, count_x);
    
    for j=0:count_y-1
        for i=0:count_x-1
            x = 1+(i*step);
            y = 1+(j*step);
            sub_img = cur_img(y:y+sy, x:x+sx);
            sub_hog = extractHOGFeatures(img,'CellSize',cellSize);

            % Stary klasyfikator
            %dist = min(sum((sub_hog - hog1) .^ 2 ), sum((sub_hog - hog2) .^ 2 ));
            %out(j+1, i+1) = dist;

            % Nowy klasyfikator
            [label, score] = predict(svm_classifier, sub_hog);
            out(j+1, i+1) = score(1);

        end
    end
    
    best_score = min(min(out))

    % heatmapy czegośtam:

    figure;
    imagesc(out)
    figure;
    mins = imregionalmin(out);
    mins(out > thr) = 0;
    imagesc(mins);
    [rs, cs] = find(mins);
    
    for i=1:size(rs)
        x = ((cs(i)-1) * step) / scale + 1;
        y = ((rs(i)-1) * step) / scale + 1;
        w = sx / scale;
        h = sy / scale;
        
        dets = [dets; [x,y,w,h] ];
        scores = [scores, out(rs(i), cs(i))];
        labels{size(dets, 1)} = num2str(out(rs(i), cs(i)), '%.1f');
    end
    
    scale = scale * scale_step;

end

% Surowe wyniki detekcji
% figure;
% imshow(insertObjectAnnotation(It, 'rectangle', dets, labels));

% Filtracja detekcji i NMS
% to chodzi o to że okna zachodzą na siebie podczas przesuwania,
% więc usuwamy zachodzące na siebie prostokąty
% (chyba) preferując ten bliższy do zbioru treningowego
filtered_dets = [];
filtered_scores = [];
tmp_dets = dets;
tmp_scores = scores;
while 1
    [m, i] = min(tmp_scores);
    filtered_dets = [filtered_dets; tmp_dets(i,:)];
    filtered_scores = [filtered_scores, m];
    ratio = bboxOverlapRatio(tmp_dets(i, :), tmp_dets);
    tmp_dets = tmp_dets(ratio < 0.2, :);
    tmp_scores = tmp_scores(ratio < 0.2);
    if size(tmp_dets, 1) < 1
        break
    end
end
cnt = size(filtered_scores, 2);
filtered_lbls = cell(cnt, 1);
for i=1:cnt
    filtered_lbls{i} = num2str(filtered_scores(i), '%.1f');
end

% Wstępna detekcja po filtracji - bez progowania:
figure;
imshow(insertObjectAnnotation(It, "rectangle", filtered_dets, filtered_lbls));

% Tu się dzieją jakieś czary bo na bazie tych dziwnych danych gTruth.mat
% wybierane są prostokąty z ludźmi
gt_rect=gTruth.LabelData.person{1,1};
ann1 = insertObjectAnnotation(It, "rectangle", gt_rect, "person");
figure;
imshow(ann1);

% Finalna detekcja w oparciu o próg
score_thr = 40;
selected_dets = filtered_dets(filtered_scores < score_thr, :);
ratio = bboxOverlapRatio(gt_rect, selected_dets)

[best_overlaps, good_ids] = max(ratio, [], 2)

iou = 0.4;
best_ids = good_ids(best_overlaps > iou)

% Wyświetlanie false pos i "false negative"
pos_neg = zeros(size(selected_dets, 1), 1);
pos_neg(best_ids) = 1;
tp_boxes = selected_dets(pos_neg==1, :);
fp_boxes = selected_dets(pos_neg==0, :);
ann2 = insertObjectAnnotation(ann1, "rectangle", fp_boxes, "", "Color", 'red');
ann2 = insertObjectAnnotation(ann2, "rectangle", tp_boxes, "", "Color", 'green');
imshow(ann2)
