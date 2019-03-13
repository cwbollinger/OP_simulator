function shape = makeObstacles(map)
   columns = size(map,1);
   rows = size(map,2);
%    budget = randomness * columns*rows; %number of blocks that are obstacles is related to number of blocks in map
    shape = [];
   %Tetris - vertical L blocks
    r = max(2,round(columns/5));
    shape{numel(shape)+1} = [[1:(2*r)]'*0+1, [1:(2*r)]'*1];
    
    %Tetris - vertical L blocks
    r = max(2,round(columns/5));
    shape{numel(shape)+1} = [[1:(2*r)]'*0+1, [1:(2*r)]'*1];
   
    
    %Tetris - horizontal L blocks
    r = max(2,round(columns/5));
    shape{numel(shape)+1} = [[1:(2*r)]'*1, [1:(2*r)]'*0+1];
    
    %Tetris - horizontal L blocks
    r = max(2,round(columns/5));
    shape{numel(shape)+1} = [[1:(2*r)]'*1, [1:(2*r)]'*0+1];

    
    %Diagonal block
    r = max(2,round(columns/5));
    shape{numel(shape)+1} = [[1:(2*r)]'*1, [1:(2*r)]'*1];
%     
%     %Diagonal block
%     r = max(2,round(columns/5));
%     shape{numel(shape)+1} = [[1:(2*r)]'*1, [1:(2*r)]'*1];

    
    

end