function rp = updateRobot(xLoc,yLoc,rp)
%UPDATEROBOT Summary of this function goes here
%   Detailed explanation goes here
    px = 1/2 * cos(2*pi*linspace(0,1)) + xLoc; 
    py = 1/2 * sin(2*pi*linspace(0,1)) + yLoc;
    
    rp.Vertices = [px', py'];
end

