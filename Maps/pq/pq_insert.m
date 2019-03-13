% function pq = pq_insert(pq, id, priority)
% /**
%      * Insert an element at the given priority.  If the queue is full,
%      * throw an exception.
%      * @param id A number between 0 and getSize()-1.  We can use the
%      * ID later to reset the element's priority.  It is an error to
%      * attempt to put two distinct elements with the same ID in the
%      * queue simultaneously.
%      * @param priority The numerical priority.  (Lower numbers mean the
%      * element will be accessed sooner, which some might say is
%      * reversed from the usual definition of priority.)
%      */
% By Yang Gu, 2006, ported from Geoff's pathplan code
function pq = pq_insert(pq, id, priority)

pq.size = pq.size + 1;

if pq.size > pq.maxsize
    error('priority queue full');
end
pq.priorities(pq.size) = priority;
pq.ids(pq.size) = id;
pq.directory(id) = pq.size;

pq = pq_percolateUp(pq, pq.size);
