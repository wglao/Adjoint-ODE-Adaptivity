syms u(t)
ode_fn = diff(u,t) == u.^2;
u_sol(t) = dsolve(ode_fn,u(0)==1);
u_t = matlabFunction(u_sol);