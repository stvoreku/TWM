% Read in a video file.
vidReader = VideoReader('visiontraffic.avi');

videoPlayer = vision.VideoPlayer();

% Create optical flow object.
opticFlow = opticalFlowFarneback;

% skip first still frames
for i=1:90
    frame = readFrame(vidReader);
end

% initialize optical flow
frameGray = rgb2gray(frame);

% estimate optical flow for the first frame (eliminates the noise for the
% actual first detection)
flow = estimateFlow(opticFlow,frameGray); 

ba = vision.BlobAnalysis;

figure(1);
tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact'); 

%Matlab przeciętnie obsługuje dynamiczne rozciąganie macierzy, więc
%na potrzeby przykładu zakładamy że nie pojawi się więcej niż 16 obiektów do
%śledzenia

tracesArrayX = { []; []; []; []; []; []; []; []; []; []; []; []; []; []; []; [] };
tracesArrayY = { []; []; []; []; []; []; []; []; []; []; []; []; []; []; []; [] };
dirArray = { []; []; []; []; []; []; []; []; []; []; []; []; []; []; []; [] };
colors = colormap(turbo(16));

% Estimate the optical flow of objects in the video.
while hasFrame(vidReader)

    frameRGB = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);

    % estimate optical flow
    flow = estimateFlow(opticFlow,frameGray); 
    
    % plot orginal frame
    nexttile(1)
    imshow(frameRGB)  
    
    % calculate speed and direction from Vx and Vy
    dir = flow.Orientation;
    spd = flow.Magnitude;
    
    
    % threshold optical flow for speeds over 2
    thr = spd > 2;
    
    % remove all measurements for speeds lower than 2
    filtdir = zeros(size(dir));
    filtdir(thr) = dir(thr);
    
    % calculate region statistics for thresholded image
    [AREA,CENTROID,BBOX] = step(ba, thr);
    
    % analyze found regions
    bb = [];
    lbl = [];
    cc = [];
    avgdir = [];
    avgspd = [];
    for i=1:size(AREA, 1)
        % leave only regions bigger than 2000 px
        if AREA(i) > 2000
            % extract the bounding box 
            x = BBOX(i,1);
            y = BBOX(i,2);
            w = BBOX(i,3);
            h = BBOX(i,4);
            
            % add the area to the list of labels
            lbl = [lbl AREA(i)];
            
            % store the bounding box
            bb = [bb; BBOX(i, :)];
            
            % compute the average direction inside the bounding box
            dirbb = filtdir(y:y+h-1,x:x+w-1);
            avgdir = [avgdir mean(dirbb(dirbb ~= 0))];
            
            spdbb = spd(y:y+h-1,x:x+w-1);
            avgspd = [avgspd mean(spdbb(spdbb > 2))];
            
            % store the centroid
            cc = [cc; CENTROID(i,:)];
        end
    end
    
    
    % show the annotated image
    nexttile(2)
    imshow(frameRGB)
    % draw the average direction vector
    for i=1:size(bb,1)
        x1 = cc(i, 1);
        y1 = cc(i, 2);
        x2 = x1 + 5*avgspd(i)*cos(avgdir(i));
        y2 = y1 + 5*avgspd(i)*sin(avgdir(i));
        dir1 = avgdir(i);
        assigned = false;
        for j=1:size(tracesArrayX,1)
            %iterujemy po znanych obiektach i patrzymy czy da sie
            %przypasowac. Jesli obiekt jest pusty to nie sprawdzamy.
            if ~isempty(tracesArrayX{j})
%                 pause(0.5)
%                 disp("CHECK FOR")
%                 disp(j)
%                 disp(tracesArrayX{j})
                if abs(tracesArrayX{j}(end) - x1) < 40 && abs(tracesArrayY{j}(end) - y1) < 40 && ~assigned && abs(dirArray{j}(end) - dir1) < 0.3
                    %Kopia z dołu
%                     disp("ASSIGNED")
                    %disp(abs(tracesArrayX{j}(end)- x1))
                    assigned = true;
                    tracesArrayX{j} = [tracesArrayX{j} x1];
                    tracesArrayY{j} = [tracesArrayY{j} y1];
                    dirArray{j} = [dirArray{j} dir1];
                else
%                     disp("Cannot ASSIGN")
%                     disp(tracesArrayX{j}(end))
%                     disp(x1)
%                     disp(abs(tracesArrayX{j}(end)- x1))
                end
            end
        end
            %jeśli nie przypisano do żadnego do wrzucamy na pierwszy wolny
        if ~assigned
            for j=1:size(tracesArrayX,1)
                if isempty(tracesArrayX{j})
                    tracesArrayX{j} = x1;
                    tracesArrayY{j} = y1;
                    dirArray{j} = dir1;
%                     disp("NO ASSIGNED, ADDED")
%                     disp(x1)
%                     disp(y1)
%                     pause(0.5)
                    assigned=true;
                    break 
                end
            end
        end
            %iterujemy po znanych obiektach i patrzymy czy da sie
            %przypasowac

    end
    hold on;
    for k=1:size(tracesArrayX,1)
        line(tracesArrayX{k}, tracesArrayY{k}, 'LineWidth', 3, 'Color', colors(k,:));
    end
    hold off;
    pause(0.001)
end