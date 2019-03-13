% function pq = pq_swap(pq, i, j)
% swap two nodes in the heap
% By Yang Gu, 2006, ported from Geoff's pathplan code
%  - pq: priority quene
%  - i,j : the index of the node to swap
function pq = pq_swap(pq, i, j)

swapi = pq.ids(j);
pq.ids(j) = pq.ids(i);
pq.ids(i) = swapi;
swapd = pq.priorities(j);
pq.priorities(j) = pq.priorities(i);
pq.priorities(i) = swapd;
% 
% if pq.directory(pq.ids(j)) == pq.emptyID
%     pq.directory(pq.ids(j)) = pq.emptyID;
%     pq.directory(pq.ids(i)) = i;
% elseif pq.directory(pq.ids(i)) == pq.emptyID
%     pq.directory(pq.ids(i)) = pq.emptyID;
%     pq.directory(pq.ids(j)) = j;
% else
    pq.directory(pq.ids(i)) = i;
    pq.directory(pq.ids(j)) = j;
% end