function fem_setup(n,k,tspan)
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

end