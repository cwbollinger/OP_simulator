% function pq = pq_percolateUp(pq, i)
% We have just made the priority at the given heap index
% numerically smaller.  Swap nodes to restore the heap property.
% By Yang Gu, 2006, ported from Geoff's pathplan code
function pq = pq_percolateUp(pq, i)
par = floor(i/2);
while( (i ~= 1) && (pq.priorities(par) > pq.priorities(i)))
    pq = pq_swap(pq, i, par);
    i = par;
    par = floor(i/2);
end