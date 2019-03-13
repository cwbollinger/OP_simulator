% function pq = pq_set(pq, id, priority)
% /**
%      * Reset the priority of an element, or insert it if it's not
%      * already there.
%      * @param id The element ID.
%      * @param priority The element priority.
%      */
% By Yang Gu, 2006, ported from Geoff's pathplan code
function pq = pq_set(pq, id, priority)

elemID = pq.directory(id);
if elemID == pq.emptyID
    pq = pq_insert(pq, id, priority);
elseif pq.priorities(elemID) < priority
    pq.priorities(elemID) = priority;
    pq= pq_percolateDown(pq, elemID);
else
    pq.priorities(elemID) = priority;
    pq = pq_percolateUp(pq, elemID);
end





     
     
