% [X,Y,DX,DY] = dynamic_state_from_index(map,index)
% By Jeremy Kubica, 2003
% Modified by Geoff Hollinger, 2007
%
% Gets the state vector for an index in a 4D dynamics problem.
% Does NO bounds checking.
function [X,Y] = state_from_index(map,index)

% find the current state from unique index
index = index - 1;
X = floor(index / map.R);
Y = mod(index, map.R);
X = X + 1;
Y = Y + 1;
