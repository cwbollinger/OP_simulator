% function pq = pq_init(limit)
% Priority Queue
% /**
%  * Make a new queue.
%  * @param limit The queue can contain at most limit nodes.
%  */
% By Yang Gu, 2006, ported from Geoff's pathplan code
function pq = pq_init(limit)

pq.emptyID = -1;
pq.maxsize = limit;
pq.ids = zeros(1, limit);
pq.directory = pq.emptyID*ones(1, pq.maxsize);
pq.priorities = zeros(1, limit);

pq.lastpri = 0;
pq.size = 0;

% %pq = pq_empty(pq);
% 
% %     /**
% %      * Empty out the queue.
% %      */
% function pq = pq_empty(pq)
% 
% pq.size = 0;
% pq.directory = pq.emptyID*ones(1, pq.maxsize);


