% function [test] = pq_test(pq, id)
% /**
%      * Test if an id is in pq. Returns 1 if it is.
%      * @param id The element ID.
%      */
% By Geoff Hollinger, 2006, ported from 
% Geoff Gordon's pathplan code
function [test] = pq_test(pq, id)

ind = find(pq.ids == id);
if(ind <= pq.size)
    test = 1;
else
    test = 0;
end

%elemID = pq.directory(id);
%if elemID == pq.emptyID
%    test = 0;
%else
%    test = 1;
%end
