function dist = euclid_dist(map, nodeInd, goalX, goalY)
%euclid_dist Get the heuristic for an input node index
%This hueristic will be euclidean distance from goal    
    %Get node position
    [stateX,stateY] = state_from_index(map,nodeInd);
    
    %Calculate heuristic - Euclidean distance to goal
    dist = sqrt((goalX - stateX)^2 + (goalY - stateY)^2);
    
end

