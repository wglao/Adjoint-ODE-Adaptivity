function [err,res] = err_contribution(Ks,Ns,uh,t1)
    Globals1D;
    err = zeros(Ks,1);
    res = cell(Ks,1);
    for i=1:Ks
        u = uh{i};
        tu = t1{i};

%         polyfit for interpolation
        pu = polyfit(tu,u,Ns(i));
        uh_interp = @(t) polyval(pu,t);
        powers = (Ns(i)):-1:1;
        pudiff = pu(1:end-1).*powers;
        dudt = @(t) polyval(pudiff,t);

%         pa = polyfit(ta,a,length(a)-1);
%         adj_interp = @(t) polyval(pa,t);
%         resu = @(t) uh_interp(t) - dudt(t);
%         adj_w = @(t) adj_interp(t) .* resu(t);
%         err(i) = integral(adj_w,tu(1),tu(end));
        syms a(t)
%         a_ode = diff(a,t) == -a;
        a_ode = diff(a,t) == -a -1;
%         a_ode2 = diff(a,t) == -a;
        adj_eq(t) = dsolve(a_ode,a(1)==0);
        adj_eq_f = matlabFunction(adj_eq);
%         tn = ta(1);
%         adj_eq2(t) = dsolve(a_ode2,a(tn)==adj_eq_f(tn));
%         adj_eq_f2 = matlabFunction(adj_eq2);
        resu = @(t) uh_interp(t) - dudt(t);
        adj_w = @(t) adj_eq_f(t).*resu(t);
%         adj_w2 = @(t) adj_eq_f2(t).*resu(t);
%         if i==Ks
%             err(i) = integral(adj_w,ta(1),ta(end));
%         else
%             err(i) = integral(adj_w2,ta(1),ta(end));
%         end
%         integral(@(t) (exp(1)./exp(t) - 1).*(pu(1).*t + pu(2) - pu(1)),tu(1),tu(end))
        err(i) = integral(adj_w,tu(1),tu(end));
        
%         
        if i==1
            err(i) = err(i) + u(1) - 1;
%         else
%             err(i) = err(i) + u(1) - uh{i-1}(end);
        end
%         res{i} = resu(linspace(ta(1),ta(end),20));
    end
    return
end