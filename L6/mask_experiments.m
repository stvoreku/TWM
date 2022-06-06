%imshow(mask)
out = medfilt2(mask, [80 120]);
% imshow(out)
% 
imshowpair(mask,out,'montage')

%imshow(disparityMap, disparityRange)
