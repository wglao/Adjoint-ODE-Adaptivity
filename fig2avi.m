nf = length(findobj('type','figure'));

if nf>0
    v = VideoWriter('rec_linear.avi');
    v.FrameRate = 5;
    open(v);
    for i=1:nf
        fig = figure(i);
        fig.Position = [200 200 1200 900];
        frame = getframe(gcf);
        writeVideo(v,frame)
        close(fig)
    end
    close(v)
end