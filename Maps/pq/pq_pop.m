% function [pq, res] = pq_pop(pq)
% /**
%      * Remove and return the first element.  (The first element is the
%      * one with smallest numerical priority value, which some might
%      * say is reversed from the usual definition of priority.)  If the
%      * queue was empty, throw an exception.
%      */
% By Yang Gu, 2006, ported from Geoff's pathplan code
function [pq, res] = pq_pop(pq)
if pq.size == 0
    error('popped empty queue');
end
res = pq.ids(1);
pq.lastpri = pq.priorities(1);

pq.directory(pq.ids(1)) = pq.emptyID;

if pq.size > 0
    pq = pq_swap(pq, 1, pq.size);
    pq = pq_percolateDown(pq, 1);
end

pq.size = pq.size - 1;