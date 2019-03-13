function roboPatch = drawRobot(x,y,p,fig) %(x,y,patch)
%DRAWGOAL Summary of this function goes here
%   Detailed explanation goes here
    px = p/2 * cos(2*pi*linspace(0,1)) + x; 
    py = p/2 * sin(2*pi*linspace(0,1)) + y;
    figure(fig);
    roboPatch = patch(px,py,'blue');
end

