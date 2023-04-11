function [t,v,err] = adj_march(Ns,Ks,times)
    % solve adjoint ode with DG marching in time (element by element)
    Globals1D;
    global y1 t1;
    t = cell(Ks,1);
    v = cell(Ks,1);
    err = zeros(Ks,1);
    vL_prev = 0;
    y0 = 1;
    % outer time step loop
    for s = Ks:-1:1
        tk = times(s:s+1);
        fem_setup(Ns(s),1,tk);
        hk = x(1) - x(end);
        M = hk/2 .* inv(V*V'); 
        S = inv(V*V')*Dr;
        m = zeros(Np,Np); m(1) = -1;
        A = -S'+m-M;
        if s == Ks
            F = M*ones(Np,1); F(end) = F(end) - vL_prev;
        else
            F = zeros(Np,1); F(end) = - vL_prev;
        end
%         F = M*ones(Np,1); F(end) = F(end) - vL_prev;
%         F = M*(x.^2); F(end) = F(end) - vL_prev;

        v_s = A\F;
        v{s} = v_s;
        vL_prev = v_s(1);
        t{s} = x;
        
        % polyfit for interpolation
        pu = polyfit(t1{s},y1{s},length(y1{s})-1);
        uh_interp = @(t) polyval(pu,t);
        uh_s = uh_interp(x);
        
        m([1,end]) = [0,1];
        A = -S'+m-M;
        if s == 1
            F = zeros(Np,1); F(1) = y0;
        else
            F = zeros(Np,1); F(1) = y1{s-1}(end);
        end
        err(s) = v_s'*(-A*uh_s + F);
        
    end
    return
end