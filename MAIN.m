% setup problem
Globals1D;

tspan=[0 1];
y0 = 1;

ode_fn = @(t,y) y;
adj_fn = @(t,y) y - kroneckerDelta(t,tspan(end));
obj_fn = @(y) y(end);

v = VideoWriter('dg.avi');
open(v);
% figure;
% ax = gca();
xplot = linspace(0,1);

times = [tspan(1) mean(tspan) tspan(2)];

% [t,y] = dg_march(N,K,tspan,y0);
% [t2,sens] = adj_march(N+1,K,tspan);

[t,y] = fwd_eul_march(y0,times,ode_fn);
opts = odeset('RelTol',1e-6,'AbsTol',1e-7);
[t2,err] = ode45(adj_fn,[tspan(2) tspan(1)],0,opts);


figure;
hold on
plot(xplot,y0*exp(xplot))
plot(t,y)
plot(t2,sens)

close(v)