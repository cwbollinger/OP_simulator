function map = makeMap(mapName)
    %Make a map (pgm)
    columns = 50;
    rows = 50;

    %% Initialize map
    cells = uint8(zeros(rows,columns));

    %% Modify Map
    obs = makeObstacles(cells);

    for j = 1:numel(obs) %For each shape
        placed = 0; 
    %     while ~placed %While it hasn't been placed
    %     posX = 0; posY = 6; %Choose a location to place it
            rowStart = randi([0, rows - max(obs{j}(:,1))],1);
            colStart = randi([0, columns - max(obs{j}(:,2))],1);
        cellsTemp = cells;
            for i = 1:length(obs{j}) %For each block in the shape
                cellsTemp(rowStart+obs{j}(i,1),colStart+obs{j}(i,2)) = 1; %Put a block in that place
            end
            cells = sign(cells+cellsTemp);
    end 

    %% Save Map
    %View Map
    imagesc(cells)
    axis equal
    %Save map
    imwrite(cells,mapName)
    % Verify
    % loadedMap = read_map(strcat(pwd,'\test01.pgm'));
    % imagesc(loadedMap.cells)
    % axis equal

    map.R = rows;
    map.C = columns;
    map.cells = double(cells);
end
