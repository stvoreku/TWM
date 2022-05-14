function [y, x] = getCellSize(img, num_cells_y, num_cells_x)
[height, width, ~] = size(img);
y = floor(height/num_cells_y);  % może lepiej zaokr. do całkowitych
x = floor(width/num_cells_x);
end