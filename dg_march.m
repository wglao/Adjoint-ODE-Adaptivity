function [t,y] = dg_march(Ns,Ks,times,y0,x_true,u_true)
    % solve ode with DG marching in time (element by element)
    Globals1D;
    t = cell(Ks,1);
    y = cell(Ks,1);
    uR_prev = y0;
    
    linear = false;
%     linear = true;
    
    if linear
        for k = 1:Ks
            fem_setup(Ns(k),1,times(k:k+1),1)
            hk = x(end) - x(1);
            M_k = hk/2 .* inv(V*V');
            S = inv(V*V')*Dr;
            B = zeros(Np,Np); B(end) = 1;
            A = -S'+B-M_k;
            F = zeros(Np,1); F(1) = uR_prev;
            
            u_k = A\F;
            y{k} = u_k;
            uR_prev = u_k(end);
            t{k} = x;
        end
    else
        Nps = Ns+1;
        for k = 1:Ks
            fem_setup(Ns(k),1,times(k:k+1),4*Ns(k))
            hk = x(end) - x(1);
            
            % Newton Iteration
            it = 0;
            maxit = 500;
            err = 1;
            tol = 1e-7;
            U_old = uR_prev*ones(Nps(k),1);
%             interpolate true solution
            x_interp = x(1) + (1+r).*hk./2;
            U_e = interp1(x_true,u_true,x_interp);
            wue = w.*U_e;
            Me = hk/2 .* Phi'*wue;
            U_old = (hk/2 .* inv(V*V'))\Me;
            while it<=maxit && err > tol
                % Construct Forward A Matrix and Jacobian                
                % polyfit for interpolation for A(f(u))
                pu = polyfit(x,U_old,Ns(k));
                x_interp = x(1) + (1+r).*hk./2;
                ur_k = polyval(pu, x_interp);
                            
                wfu = w.*(ur_k.^2);
                wdf = diag(w.*2.*ur_k);
                M_tilde = hk/2 .* Phi'*wfu;
                dMtdU = hk/2 .* Phi'*wdf*Phi;
                S = (V*V')\Dr;
                B = zeros(Np,Np); B(end) = -1;
                F = zeros(Np,1); F(1) = uR_prev;

                A = S'+B;
                dRdU = A+dMtdU;

                R = A*U_old + M_tilde + F;
                delta_u = dRdU\R;
                U_next = U_old - delta_u;
                err = norm(U_old-U_next);
                U_old = U_next;
                it = it + 1;
            end
            if it > maxit
                fprintf('element %d did not converge\n',k)
            end
            uR_prev = U_next(end);
            y{k} = U_next;
            t{k} = x;
        end
    end
    return
end
