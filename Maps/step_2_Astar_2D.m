%MHector
    rmpath('2D');
    rmpath('4D');
    rmpath('pq');

    addpath('pq\');
    % 2D or 4D?
    addpath('2D');
%     addpath('4D');
%Astar with euclidean distance as heuristic
mapLoc = [pwd,'\','maze2.pgm'];
map = read_map(mapLoc);
e = 10;
tLimit = .05;
tic
out = [];
while e >= 1
    if toc > tLimit
        disp('waaa1')
        break
    end
    
    [a, b] = runAstar(map,e);
    out(end+1,2) = a;
    out(end,3) = b;
    out(end,1) = e;
    
    if toc > tLimit
        disp('waaa2')
        break
    end
    
    e = e - .5 *(e-1);
    if e < 1.001
        e = 1;
   
    end
    if toc > tLimit
        disp('waaa3')
        break
    end
end


% map2 = [pwd,'\','maze2.pgm'];
% runAstar(map1,1)
% runAstar(map2,1)