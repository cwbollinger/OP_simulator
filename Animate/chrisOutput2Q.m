%turn chris' output into Q structure
clear; close all;
mapName = 'C_goodmap_50x50_03';
load(strcat(mapName,'_tw_solution.mat'));
load(strcat(mapName,'.mat'));

Q.mapName = A.mapName;
i = 1:2:21;
wp = problem_solution(i,1)+1;
Q.wp = A.waypointLoc(wp);

t = problem_solution(i,2);
t(1) = 0;
Q.t = round(t/10)';

Q.goals = A.waypointLoc;

save(strcat(mapName,'_tw_Q'),'Q');