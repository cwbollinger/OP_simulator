%MHector
%HW 1
function [optPath, pathLen] = runAstar2D(map, startInd, goalInd)
    %% Start queues
    o = pq_init(10000);
    c = [];
    s.b = zeros(1,map.R*map.C); %Parent 
    s.g = zeros(1,map.R*map.C); %Cost to get there
    goalReached = 0;
    %% Set up Astar algorithm
    %Define heuristic
    [goalX, goalY] = state_from_index(map,goalInd);
    h = @(nodeInd) euclid_dist(map,nodeInd,goalX,goalY); 

    %Add initial node to open list
    o = pq_insert(o,startInd,0);

    %% Astar algorithm
    init = 1;
    while o.size ~= 0 %Check to see if we've explored the whole space
        %% Find the best node to expand to  
        %Pop best node from o and put it in c
        [o, state] = pq_pop(o);
        c(end+1) = state;
        if init == 1
            s.b(1) = 0;
            s.g(1) = 0;
            init = 0;
        end

        if state == goalInd
%             disp('goal state reached')
            goalReached = 1;
            break
        else
            %Get the neighboring points
            [nbrs, num_nbrs] = get_neighbors(map,state);

            %Find nodes in o and not c
            nodesToExpand = setdiff(nbrs, c);

            %For each node in o and not c
            for i=1:length(nodesToExpand)
                %Calc priority: f = heuristic + additionalOperatingCost (1) + costToCurrentNode
                priority_temp = 1 * h(nodesToExpand(i)) + 1 + s.g(state);

                if ~any(o.ids == nodesToExpand(i)) %If its not in O
                    %Add x to O
                    o = pq_set(o,nodesToExpand(i),priority_temp);
                    s.b(nodesToExpand(i)) = state;
    %                 state
                    %Update backpointers and cost to go
    %                 s.b(nodesToExpand(i)) = state;
                    s.g(nodesToExpand(i)) = s.g(state)+1;
    %             elseif priority_temp < (s.g(nodesToExpand(i))+h(nodesToExpand(i))) %If the path is better 
                elseif s.g(state) + 1 < s.g(nodesToExpand(i))
    %                 %Update the back pointer and cost
                    s.b(nodesToExpand(i)) = state;
                    s.g(nodesToExpand(i)) = s.g(state) + 1;
                end
            end
        end
    end
    if goalReached ~= 1
        disp('goal not reached')
    end
    %Plot the path
%     close
    optPath = goalInd;
    while optPath(end) ~= startInd
        optPath(end+1) = s.b(optPath(end));
    end
    pathLen = length(optPath)-1;
    optPath = fliplr(optPath);
%     nodesExpanded = length(c);
end