% function pq = pq_percolateDown(pq, i)
% We have just made the priority at the given heap index
% numerically larger.  Swap nodes to restore the heap property.
% By Yang Gu, 2006, ported from Geoff's pathplan code
function pq = pq_percolateDown(pq, i)

child = i*2;
while (child < pq.size)
    if (child < pq.size - 1 & pq.priorities(child) > pq.priorities(child + 1))
        child = child + 1;
    end
    if ( pq.priorities(i) <= pq.priorities(child))
        break;
    end
    pq = pq_swap(pq, i, child);
    i = child;
    child = i*2;
end
