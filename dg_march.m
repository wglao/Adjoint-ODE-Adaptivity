function [t,y] = dg_march(n,k,tspan,y0)
    % solve ode with DG marching in time (element by element)
    Globals1D;
    fem_setup(n,k,tspan)
    y = zeros(size(x));
    yR_prev = y0;
    % advection speed
    a = 1;
    
    % outer time step loop
    for s = 1:K
        hk = x(end,s) - x(1,s);
        M = hk/2 .* inv(V*V');
        S = inv(V*V')*Dr;
        m = zeros(Np,Np); m(1) = -1;
        A = -S'+m-M;
        F = zeros(Np,1); F(1) = yR_prev;
        
        u_s = A\F;
        y(:,s) = u_s;
        yR_prev = u_s(end);
    end
    t=x;
    return
end