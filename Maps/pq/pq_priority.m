
function [priority] = pq_priority(pq, id)

ind = find(pq.ids == id);

priority=pq.priorities(ind);

end