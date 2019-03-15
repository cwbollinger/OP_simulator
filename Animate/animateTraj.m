clear; close all;
load('C_goodmap_50x50_03_tw_Q.mat');
addpath('C:\Users\DRL-Valkyrie\Google Drive\ROB 534_Sequential Decision Making\Project\Maps')
addpath('C:\Users\DRL-Valkyrie\Google Drive\ROB 534_Sequential Decision Making\Project\Maps\pq')

makeVid = 0;
if makeVid == 1
    v=VideoWriter('newVid','MPEG-4');
    v.FrameRate=10;
    open(v);
end

gifIT = 1;


%Load data
goals = Q.goals;
wp = Q.wp;
mapName = Q.mapName;
time = Q.t;

%% Get a map
showMap = 0;
map = read_map(Q.mapName,showMap);
fig = figure;

%Initialize map
scheduleImage = strcat(pwd,'\schedule.png');
plot_path(map,[],'Path',fig, scheduleImage);

axis equal

%Plot goals
for i = 1:length(goals)
    [goalX, goalY] = state_from_index(map, goals(i));
    [labelI, ~] = get_neighbors(map,goals(i)); %Look where the label should go
    [labelX, labelY] = state_from_index(map,labelI(1)); %Grab the first place
    drawGoal(goalX,goalY,1,fig,i,labelX,labelY);
end

%Make robot position
px = 1/2 * cos(2*pi*linspace(0,1)) + 0; 
py = 1/2 * sin(2*pi*linspace(0,1)) + 0;
rp = patch(px,py,'blue');

if makeVid == 1
    disp('pause to fiddle with frame')
    pause
end

%% Animate the trajectory
%Make a frame for each position of the robot, moving 1 tile at a time
for i = 1:length(time)-1
    %Get path between wp(i) and wp(i+1)
    if wp(i) ~= wp(i+1)
        
        [path, pathLen] = runAstar2D(map, wp(i), wp(i+1));
%         pathLen
%         [wp(i) wp(i+1)]
%         2+2;
    else
        path = wp(i) * ones(1,time(i+1)-time(i));
        pathLen = time(i+1) - time(i);
    end
    
    for j = 1:length(path)
        [x, y] = state_from_index(map,path(j));
        rp = updateRobot(x,y,rp);
        drawnow
        
        if makeVid ==1
            F=getframe(fig);
            writeVideo(v,F);
        else
            pause(.1)
        end
    
        if gifIT == 1
            if i == 1 && j == 1
                filename = 'newGif.gif';
                gif(filename,'frame',gcf);
            else
                gif
            end                
        end
        
    end 
end
if makeVid == 1
    close(v)
end        

