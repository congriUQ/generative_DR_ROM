%script that dynamically plots the p_f loss

fig = figure('units','normalized','outerposition',[0 0 1 1]);
ax = axes('Parent', fig);
axis(ax, 'fill');


while true
    l = dlmread('./log_pf_loss.txt');
    p = plot(l, 'linewidth', 2.5, 'Parent', ax);
    axis(ax, 'tight');
    axis(ax, 'fill');
    drawnow;
    pause(2)
end