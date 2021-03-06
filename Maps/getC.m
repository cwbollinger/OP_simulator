function [C, wp_loc] = getC(mapName)
%% Get a map
showMap = 0;
map = read_map(mapName,showMap);
fig = figure;

%% Add waypoints
num_wp = 50;
%Find list of valid waypoints
possible_wp = [];
for stateID = 1:map.R*map.C
    [x,y] = state_from_index(map,stateID); 
    if check_hit(map, x, y, 0, 0) == 0
        possible_wp(end+1) = stateID;
    end
end

%hack to make sure wp_loc contains only unique elements (i.e. the same wp
%isn't used twice)
wp_loc = [];
while length(wp_loc) ~= num_wp
    wp_loc = unique(datasample(possible_wp,num_wp));
%     length(wp_loc)
end

%% Initialize C matrix
%This is the path cost matrix
C = zeros(num_wp);

%% Find cost from each waypoint to each waypoint
%For each waypoint
for outer = 1:length(wp_loc)
    %Find the path length from each inner waypoint to each outer waypoint
    for inner = 1:length(wp_loc)
        if wp_loc(outer) ~= wp_loc(inner)           
            [path, pathLen] = runAstar2D(map, wp_loc(outer), wp_loc(inner));
        else
            pathLen = 10e6; %The path cost between a wp and itself is large to discourage taking that action
            path = wp_loc(outer);
        end
        %Update C matrix
        C(outer,inner) = pathLen;
        
%         %Plot path
%         [X, Y] = state_from_index(map,path);
%         XY = [X',Y'];
%         plot_path(map,XY,'Path',fig)
%         hold on;
    end
end
