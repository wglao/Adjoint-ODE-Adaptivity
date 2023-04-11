% Purpose: declare global variables

global N Nfp Np K
global r x  VX
global Dr LIFT
global nx Fx Fscale
global vmapM vmapP vmapB mapB Fmask
global vmapI vmapO mapI mapO
global rx J
global rk4a rk4b rk4c
global Nfaces EToE EToF
global V invV
global NODETOL

global ode_fn adj_fn
global y0

% Low storage Runge-Kutta coefficients
rk4a = [            0.0 ...
        -567301805773.0/1357537059087.0 ...
        -2404267990393.0/2016746695238.0 ...
        -3550918686646.0/2091501179385.0  ...
        -1275806237668.0/842570457699.0];
rk4b = [ 1432997174477.0/9575080441755.0 ...
         5161836677717.0/13612068292357.0 ...
         1720146321549.0/2090206949498.0  ...
         3134564353537.0/4481467310338.0  ...
         2277821191437.0/14882151754819.0];
rk4c = [             0.0  ...
         1432997174477.0/9575080441755.0 ...
         2526269341429.0/6820363962896.0 ...
         2006345519317.0/3224310063776.0 ...
         2802321613138.0/2924317926251.0];

% Radau abscissas
radau = {};
radau{1} = -1;
radau{2} = [-1; 1/3];
radau{3} = [-1; (1 - sqrt(6))/5; (1 + sqrt(6))/5];
radau{4} = [-1; -0.575319; 0.181066; 0.822824];
radau{5} = [-1; -0.72048; -0.167181; 0.446314; 0.885792];