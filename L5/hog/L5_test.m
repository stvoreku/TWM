
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
            out(j+1, i+1) = score(1)*(-1);  % wartości muszą być dodatnie bo potem jest minimalizacja

        end
    end
    
    best_score = min(min(out))

    % heatmapy czegośtam:

%     figure;
%     imagesc(out)
%     figure;
    mins = imregionalmin(out);
    mins(out > thr) = 0;
%     imagesc(mins);
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

