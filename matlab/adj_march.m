function [t,v,err] = adj_march(Ns,Ks,times)
    % solve adjoint ode with DG marching in time (element by element)
    Globals1D;
    global y1 t1 y1f t1f;
    t = cell(Ks,1);
    v = cell(Ks,1);
    err = zeros(Ks,1);
    vL_prev = 0;
    y0 = 1;
    
    linear = false;
%     linear = true;
    
    if linear
        for k = Ks:-1:1
            tk = times(k:k+1);
            fem_setup(Ns(k),1,tk,1);
            hk = x(1) - x(end);
            M = hk/2 .* inv(V*V'); 
            S = inv(V*V')*Dr;
            m = zeros(Np,Np); m(1) = -1;
            A = -S'+m-M;
    %         if k == Ks
    %             F = M*ones(Np,1); F(end) = F(end) - vL_prev;
    %         else
    %             F = zeros(Np,1); F(end) = - vL_prev;
    %         end
            F = M*ones(Np,1); F(end) = F(end) - vL_prev;
    %         F = M*(x.^2); F(end) = F(end) - vL_prev;
    
            v_k = A\F;
            v{k} = v_k;
            vL_prev = v_k(1);
            t{k} = x;
            
            % polyfit for interpolation
            pu = polyfit(t1{k},y1{k},length(y1{k})-1);
            uh_interp = @(t) polyval(pu,t);
            uh_k = uh_interp(x);
            
            m([1,end]) = [0,1];
            A = -S'+m+M;
            if k == 1
                F = zeros(Np,1); F(1) = y0;
                else
                F = zeros(Np,1); F(1) = y1{k-1}(end);
            end
            err(k) = v_k'*(-A*uh_k + F);
            if k==1
                p = polyfit(t1{Ks},y1{Ks},Ns(Ks)-1);
                u_interp = @(t) polyval(p,t);
                ph = polyfit(t1f{Ks},y1f{Ks},Ns(Ks));
                uh_interp = @(t) polyval(ph,t);
                uh_k = uh_interp(x);
                bc = 0;
                for kk = 1:Ks
                    if kk == 1
                        bc = y1{1}(1) - 1;
                    else
                        bc = bc + y1{kk}(1) - y1{kk-1}(end);
                    end
                end
            end
        end
    else
        % if nonlinear, construct M with interpolated Uh_k
        for k = Ks:-1:1
            U_k = y1{k};
            tk = t1{k};
            tspan = tk([1,end]);
            fem_setup(Ns(k),1,tspan,2*Ns(k));
            hk = x(1) - x(end);

            % polyfit for interpolation
            pu = polyfit(tk,U_k,Ns(k)-1);
            uh_interp = @(t) polyval(pu,t);
            uh_k = uh_interp(x);
            r_interp = tk(1) + (1+r).*hk./2;
            ur_k = uh_interp(r_interp);
                        
            w_tilde = diag(w.*cos(ur_k));
            M_v = hk/2 .* Phi'*w_tilde*Phi;
            M_k = hk/2 .* inv(V*V');
            S = inv(V*V')*Dr;
            B = zeros(Np,Np); B(1) = -1;
            A = -S'+B-M_v;

            % %     $ J = \int_{k=Nk} (u) dt $      % %
%             if k == Ks
%                 F = M_k*eq(1:Np,Np)'; F(end) = F(end) - vL_prev;
%             else
%                 F = zeros(Np,1); F(end) = - vL_prev;
%             end

            % %     $ J = \int_{\Omega_h} (u) dt $      % %
            F = M_k*ones(Np,1); F(end) = F(end) - vL_prev;
    
            v_k = A\F;
            v{k} = v_k;
            vL_prev = v_k(1);
            t{k} = x;
            
            B([1,end]) = [0,1];
            wfu = w.*(sin(ur_k));
            M_tilde = hk/2 .* Phi'*wfu;
            S = (V*V')\Dr;
            B = zeros(Np,Np); B(end) = -1;
            F = zeros(Np,1);
            if k == 1
                F(1) = y0;
            else
                F(1) = y1{k-1}(end);
            end

            A = -S'-B;

            err(k) = v_k'*(-A*uh_k - M_tilde + F);

        end
    end
    return
end
