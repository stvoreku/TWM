%imshow(mask)
out = medfilt2(mask, [80 80]);
% imshow(out)
% 
se = strel('rect',[1200,800]);
out2 = imclose(out, se);
%imshowpair(mask,out2,'montage')
montage([mask,out,out2])

%imshow(disparityMap, disparityRange)
