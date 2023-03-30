function [t,y] = fwd_euler_march(y0,times,ode_fn)
    t = times;
    dt = times(2:end) - times(1:end-1);
    y = zeros(size(times)); y(1) = y0;
    for t=times
        dy = ode_fn(t,y);
        
        delta_y = dy*dt;
    delta_y = cumsum(delta_y);
    y(2:end) = delta_y + y0;
    return
end