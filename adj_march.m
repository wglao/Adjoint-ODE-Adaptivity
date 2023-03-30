function [t,psi] = adj_march(n,k,tspan)
    % solve adjoing ode with CG marching in time (element by element)
    Globals1D;
    fem_setup(n,k,tspan)
    psi = zeros(size(x));
    yR_prev = 0;
    % advection speed
    a = 1;
    
    % outer time step loop
    for s = K:-1:1
        hk = x(end,s) - x(1,s);
        M = hk/2 .* inv(V*V');
        S = inv(V*V')*Dr;
        m = zeros(Np,Np); m(1) = 1;
        A = -S'+m-M;
        A(end,:) = 0;
        A(end) = 1;
        F = zeros(Np,1); F(end) = yR_prev;

        
        
        psi_s = A\F;
        psi(:,s) = psi_s;
        yR_prev = psi_s(end);
    end
    t=x;
    return
end