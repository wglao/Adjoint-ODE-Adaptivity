function [t,v,err] = adj_rec(Ns,Ks,times)
    % solve adjoint ode and reconstruct to higher polynomial degree
    Globals1D;
    global y1 t1;
    t = cell(Ks,1);
    v = cell(Ks,1);
    err = zeros(Ks,1);
    vL_prev = 0;
    y0 = 1;

    linear = false;
%     linear = true;
    H2h = "interpolate";
%     H2h = "reconstruct";


    if linear
        for s = Ks:-1:1
            tk = times(s:s+1);
            fem_setup(Ns(s),1,tk,1);
            hk = x(1) - x(end);
            M = hk/2 .* inv(V*V'); 
            S = inv(V*V')*Dr;
            m = zeros(Np,Np); m(1) = -1;
            A = -S'+m-M;
    %         if s == Ks
    %             F = M*ones(Np,1); F(end) = F(end) - vL_prev;
    %         else
    %             F = zeros(Np,1); F(end) = - vL_prev;
    %         end
            F = M*ones(Np,1); F(end) = F(end) - vL_prev;
    %         F = M*(x.^2); F(end) = F(end) - vL_prev;
    
    %         Reconstruct with Radau Colocation Points
            % Radau order m
            rad_m = Ns(s) + 1;
            rad = radau{rad_m};
            rad_x = tk(1) + (1+rad).*abs(hk)./2;
            
            v_s = A\F;
            % polyfit for interpolation
            pa = polyfit(x,v_s,length(v_s)-1);
            v_interp = @(t) polyval(pa,t);
            v_rec = v_interp(rad_x); v_rec(end+1) = vL_prev;
            
            x_rec = [rad_x; tk(2)];
            pa_rec = polyfit(x_rec,v_rec,rad_m);
            vh_interp = @(t) polyval(pa_rec,t);
                    
            fem_setup(rad_m,1,tk,1);
            M = hk/2 .* inv(V*V'); 
            S = inv(V*V')*Dr;
            m = zeros(Np,Np); m(end) = 1;
            A = -S'+m+M;
            if s == 1
                F = zeros(Np,1); F(1) = y0;
            else
                F = zeros(Np,1); F(1) = y1{s-1}(end);
            end
            
            % polyfit for interpolation
            pu = polyfit(t1{s},y1{s},length(y1{s})-1);
            uh_interp = @(t) polyval(pu,t);
            uh_s = uh_interp(x);
            v_h = vh_interp(x);
            err(s) = v_h'*(-A*uh_s + F);
                    
            v{s} = v_rec;
            vL_prev = v_rec(1);
            t{s} = x_rec;
        end
    else
        for s = Ks:-1:1
            fem_setup(Ns(s),1,times(s:s+1),2*Ns(s))
            hk = x(end) - x(1);
            if H2h == "interpolate"
                % polyfit for interpolation
                pu = polyfit(t1{s},y1{s},length(y1{s})-1);
                uh_interp = @(t) polyval(pu,t);
                x_gq = x(1) + (1+r)./2.*hk;
                uh_s = uh_interp(x_gq);
            elseif H2h == "reconstruct"
            end
            w_tilde = hk./2 .* diag(w.*uh_s);
            M = Phi'*w_tilde*Phi;
        end
    return
end
