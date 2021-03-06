%Make the data structure to keep things tidy.
clear;
strucc = dir(strcat(pwd,'\goodmap_50x50_*'));
for i = 1:length(strucc)
    A.mapName = strucc(i).name;
    [C, wp] = getC(strucc(i).name);
    A.waypointLoc = wp;
    A.C = C;
    save(strcat('C_',A.mapName(1:end-4),'_100_POI'),'A');
    i/length(strucc)*100
end
