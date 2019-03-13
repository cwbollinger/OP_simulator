% index = dynamic_index_from_state(map,x,y,dx,dy)
% By Jeremy Kubica, 2003
% Modified by Geoff Hollinger, 2007
%
% Gets a unique index from the state for a maze with 2D kinematics.
% Does NO bounds checking.
function index = index_from_state(map,x,y)

index = (x-1) * map.R + (y-1);
index = index + 1;