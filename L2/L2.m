orgImg = imread("obraz_testowy.jpg");

% -- Tworzenie i filtracja maski -- %
[BW1, imge1] = createMaskColorful(orgImg);
[BW2, imge2] = createMaskBright2(orgImg);
BW3 = logical((1 - BW1) + (1 - BW2));

[BW4, imge3] = segmentImageFinal2(orgImg, BW3);

[BW5, properties] = filterRegions(BW4);

% -- Izolacja skladowych obrazu w przestrzeni HSV --%
HSV = rgb2hsv(orgImg);
H = HSV(:,:,1);
S = HSV(:,:,2);
V = HSV(:,:,3);

% -- Detekcja rejonow -- %
props = regionprops(BW5, {"Circularity","MajorAxisLength", "MinorAxisLength", 'Area', 'Eccentricity', 'EquivDiameter', 'BoundingBox', 'PixelIdxList', 'Centroid'});

% -- Oznaczanie obiektów -- %
out = orgImg;
for i=1:size(properties, 1)
    if props(i).Area > 500

        avg_h = mean(H(props(i).PixelIdxList));
        avg_s = mean(S(props(i).PixelIdxList));
        avg_v = mean(V(props(i).PixelIdxList));
        
        labelH = sprintf("%.3f", avg_h);
        labelS = sprintf("%.3f", avg_s);
        labelV = sprintf("%.3f", avg_v);
        
        % -- Oznaczenie koloru -- %
        
        % Najpierw sprawdzamy barwe
        if avg_h < 0.1
            color_l = "Red";
        elseif (avg_h > 0.1) & (avg_h < 0.20)
            color_l = "Yellow";
        elseif (avg_h > 0.2) & (avg_h < 0.45)
            color_l = "Green";
        elseif (avg_h > 0.45) & (avg_h < 0.75)
            color_l = "Blue";
            if avg_v < 0.5
                color_l = "Dark blue";
            end
        elseif avg_h > 0.8
            color_l = "Pink";
        end
        % Jezeli kolor jest zbyt jasny (niska saturacja, wysoka wartoœæ)
        % lub zbyt ciemny (niska wartoœæ i saturacja), barwa jest
        % nieistotna; mozna nadpisac kolor
        if (avg_v < 0.32) & (avg_s < 0.4)
            color_l = "Black";
        elseif (avg_s < 0.27)
            if (avg_v > 0.9)
                color_l = "White";
            elseif (avg_v > 0.7)
                color_l = "Grey";
            end
        end
        % UWAGA: ze wzgledu na warunki swietlne nie da sie dobrze wykryc
        % jasnoniebieskiego; jest tak poniewaz na zdjeciu kolor ten
        % jest blizej bieli niz karta biala

        % -- Oznaczenie kszta³tu -- %
        %if props(i).Circularity > 0.8
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

        % Output
        label = color_l+' '+shape_l+' '+ 'H:'+labelH+'S:'+labelS+'V:'+labelV;
        out = insertObjectAnnotation(out, shape, pos, label);
    end
end
imshow(out);