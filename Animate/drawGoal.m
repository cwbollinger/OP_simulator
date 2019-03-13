function [px,py] = drawGoal(x,y,scale,fig,label,labelX,labelY)
%DRAWGOAL Summary of this function goes here
%   Detailed explanation goes here

px = x + scale * [0      .5      .3090   .8090   .1910   0       -.1910  -.8090  -.3090  -.5000]/1.618;
py = y + scale * [-.3249 -.6882  -.1004  .2629   .2629   .8507   .2629   .2629   -.1004  -.6882]/1.618;

figure(fig);
patch(px,py,'yellow')



n = text(labelX,labelY,num2str(label));
n.HorizontalAlignment = 'center';
n.FontSize = 14;
n.FontWeight = 'bold';
2+2;
end

