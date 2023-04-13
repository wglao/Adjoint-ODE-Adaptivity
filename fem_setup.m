function fem_setup(n,k,tspan,n_gq)
    % setup problem
    Globals1D;
    N = n;
    K = k;
    clear("n","k")

    % Generate simple equidistant grid with K elements
    xmax = tspan(2);
    xmin = tspan(1);
    
    Nv = K+1;
    % Generate node coordinates
    VX = (1:Nv);
    for i = 1:Nv
        VX(i) = (xmax-xmin)*(i-1)/(Nv-1) + xmin;
    end
    
    % read element to node connectivity
    EToV = zeros(K, 2);
    for k = 1:K
        EToV(k,1) = k; EToV(k,2) = k+1;
    end
    
    StartUp1D;
    
    [r,w] = JacobiGQ(0,0,n_gq);
    n_r = size(r,1);
    Phi = zeros(n_r,Np);
    p = zeros(Np,1);
    invVT = inv(V');
    for k = 1:n_r
        for i = 1:Np
            for nn = 1:Np
                p(nn) = invVT(i,nn)*JacobiP(r(k),0,0,nn-1);
            end
            Phi(k,i) = sum(p);
        end
    end

        

end