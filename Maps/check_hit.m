% Check to see whether there are any occupied cells of MAP between STARTX,
% STARTY and STARTX+DX, STARTY+DY. Also checks if we've left the map.

function hit = check_hit(map, startx, starty, dx, dy)

if(startx < 1 || starty < 1 || startx > map.C || starty > map.R)
    hit = 1;
    return
end
if(map.cells(round(startx),round(starty)) ~= 0)
    hit = 1;
    return
end

nrm = max(abs(dx), abs(dy));
dx = dx / nrm;
dy = dy / nrm;

x = startx;
y = starty;
hit = 0;
for i = 1:nrm
    x = x + dx;
    y = y + dy;
    %off da map
    if(x < 1 || y < 1 || x > map.C || y > map.R)
        hit = 1;
        return
    end
    if (map.cells(round(x), round(y)) ~= 0)
        hit = 1;
        return
    end
end

return


end

