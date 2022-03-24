orgImg = imread("IMG_5772.jpg");


% Blok tworzący i poprawiający maske
[BW1, imge1] = createMaskColorful(orgImg);
[BW2, imge2] = createMaskBright2(orgImg);
BW3 = logical((1 - BW1) + (1 - BW2));

[BW4, imge3] = segmentImageFinal2(orgImg, BW3);

[BW5, properties] = filterRegions(BW4);

% izolacja poszczegolnych skladowych przestrzeni HSV
HSV = rgb2hsv(orgImg);
H = HSV(:,:,1);
S = HSV(:,:,2);
V = HSV(:,:,3);

% detekcja rejonow
props = regionprops(BW5, {"MajorAxisLength", "MinorAxisLength", 'Area', 'Eccentricity', 'EquivDiameter', 'BoundingBox', 'PixelIdxList', 'Centroid'});

%detekcja koloru i ksztaltu
out = orgImg;
for i=1:size(properties, 1)
    if props(i).Area > 500
        avg_h = mean(H(props(i).PixelIdxList));
        avg_s = mean(S(props(i).PixelIdxList));
        avg_v = mean(V(props(i).PixelIdxList));
        
        labelH = sprintf("%.3f", avg_h);
        labelS = sprintf("%.3f", avg_s);
        labelV = sprintf("%.3f", avg_v);
        
        color = hsv2rgb([avg_h, 1, 1]) * 255;
        
        if avg_s < 0.4
            if avg_v > 0.75
                label = " White";
            elseif avg_v > 0.4
                label = " Gray";
            else
                label = " Black";
            end
        else
            if avg_h < 0.1
                label = " Red";
            elseif (avg_h > 0.1) & (avg_h < 0.20)
                label = " Yellow";
            elseif (avg_h > 0.2) & (avg_h < 0.45)
                label = " Green";
            elseif (avg_h > 0.45) & (avg_h < 0.75)
                label = " Blue";
            elseif avg_h > 0.90
                label = " Pink";
            end
        end
        if props(i).Eccentricity < 0.3
            shape_l = "circle";
            shape = "circle";
            pos = [props(i).Centroid, props(i).EquivDiameter/2];
        elseif props(i).MajorAxisLength / props(i).MinorAxisLength <= 1.5
            shape_l = "square";
            shape = "rectangle";
            pos = props(i).BoundingBox;
        else
            shape_l = "rectangle";
            shape = "rectangle";
            pos = props(i).BoundingBox;
        end
        out = insertObjectAnnotation(out, shape, pos, shape_l+ ' ' +label+ 'H:'+labelH+'S:'+labelS+'V:'+labelV) ;
    end
end
imshow(out);