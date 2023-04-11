function [t,y] = dg_march(Ns,Ks,times,y0)
    % solve ode with DG marching in time (element by element)
    Globals1D;
    t = cell(Ks,1);
    y = cell(Ks,1);
    res = cell(Ks,1);
    yR_prev = y0;
    % advection speed
    a = 1;
    
    % outer time step loop
    for s = 1:Ks
        fem_setup(Ns(s),1,times(s:s+1))
        hk = x(end) - x(1);
        M = hk/2 .* inv(V*V');
        S = inv(V*V')*Dr;
        m = zeros(Np,Np); m(end) = 1;
        A = -S'+m-M;
        F = zeros(Np,1); F(1) = yR_prev;
        
        u_s = A\F;
        y{s} = u_s;
        yR_prev = u_s(end);
        t{s} = x;

        if s==Ks
            p = polyfit(x,u_s,Ns(s));
            u_interp = @(t) polyval(p,t);
            disp('JuH')
            fprintf('%.10e\n',integral(u_interp,x(1),x(end)))
        end
    end
    return
end