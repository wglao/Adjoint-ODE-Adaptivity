% setup problem
clc; clear all; close all;
addpath("utils")
Globals1D;

tspan=[0 1];
y0 = 1;

ode_fn = @(t,y) y;
% adj_fn = @(t,y) -y;
adj_fn = @(t,y) -y -1;
% obj_fn = @(y) y(end);

% v = VideoWriter('dg.avi');
% v.FrameRate = 5;
% open(v);
% figure;x
% ax = gca();
xplot = linspace(0,1);

Ks = 2;
n = 2;
Ns = n.*ones(Ks,1);
times = linspace(tspan(1),tspan(2),Ks+1);

error = 1;
tol = 1e-5;
it = 0;
maxit = 30;
global y1 t1 y1f t1f blim;
while  abs(error) > tol && it <= maxit
    disp('Ju')
    fprintf('%.10e\n',integral(@(x) exp(x), times(end-1), times(end)))
    [t1,y1] = dg_march(Ns,Ks,times,y0);
    [t1f,y1f] = dg_march(Ns+1,Ks,times,y0);
    [t2,y2,err_con1] = adj_march(Ns+1,Ks,times);
    [t3,y3,err_con2] = adj_rec(Ns,Ks,times);
    
    % [t,y] = fwd_eul_march(y0,times,ode_fn);
    opts = odeset('RelTol',1e-6,'AbsTol',1e-7);
%     [ta,ya] = ode45(adj_fn,[tspan(2) tspan(1)],0,opts);
    syms a(t)
    a_ode = diff(a,t) == -a -1;
%     a_ode = diff(a,t) == -a -heaviside(t-times(end-1));
    a_ode2 = diff(a,t) == -a;
%     a_ode = diff(a,t) == -a -t.^2;
    adj_eq(t) = dsolve(a_ode,a(1)==0);
    adj_eq_f = matlabFunction(adj_eq);
    adj_eq2(t) = dsolve(a_ode2,a(times(end-1))==adj_eq_f(times(end-1)));
    adj_eq_f2 = matlabFunction(adj_eq2);

    [err_con3, res] = err_contribution(Ks,Ns,y1,y2,t1,t2);
    err_bars = [(abs(err_con1))'
                (abs(err_con2))'
                (abs(err_con3))'];

    fig = figure;
    fig.Position = [200 200 1200 900];
    hold on
    xplot = linspace(0,1,500);
    yyaxis right
    p1 = plot(xplot,y0*exp(xplot),'b-','LineWidth',1);
    xplot = linspace(times(end-1),1,ceil(500*(1-times(end-1))));
    p2 = plot(xplot,adj_eq_f(xplot),'r-','LineWidth',1);
    xplot = linspace(0,times(end-1),ceil(500*(times(end-1))));
    p3 = plot(xplot,adj_eq_f2(xplot),'r-','LineWidth',1);
    l = legend('Primal','Adjoint','',Location='best');
    l.AutoUpdate = "off";
    for i=1:Ks
        yyaxis left
        tk = [times(i),times(i+1)];
        [bars, idx] = sort(abs(err_bars(:,i)),'descend');
        for j=1:length(bars)
            ba = bar(mean(tk),bars(j),diff(tk),'FaceColor','flat');
            if idx(j) == 1
                ba.CData = [0.4660 0.6740 0.1880];
            elseif idx(j) == 2
                ba.CData = [0.9290 0.6940 0.1250];
            else
                ba.CData = [0.4940 0.1840 0.5560];
            end

        end
        
%         bar(mean(tk),abs(err_con1(i)),diff(tk),'g')
       ylabel(gca,'Error Contribution')
        if it == 0
            blim = gca().YLim;
        else
            ylim(blim);
        end
        

        yyaxis right
        plot(t1{i},y1{i},'b--','LineWidth',1)
        plot(t2{i},y2{i},'r--*','LineWidth',1)
        plot(t3{i},y3{i},'r--s','LineWidth',1)
        ylabel('Solution')
        
    end
    ax = gca;
    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = 'k';
%     legend(labels,'Location','best');
    xlim(tspan)
    frame = getframe(gcf);
    
    %     adapt
    
    ref_i = find(abs(err_con2)==max(abs(err_con2)));
    times(ref_i+2:end+1) = times(ref_i+1:end);
    times(ref_i+1) = mean(times([ref_i,ref_i+2]));
    Ks = Ks + 1;
    Ns(end+1) = n;

    if it == 0
        imwrite(frame.cdata, 'init_rec.png')
    elseif it == 29
        imwrite(frame.cdata, 'refine30_rec.png')
    end

%     writeVideo(v,frame)

%     close(fig)
%     dt = diff(times(end-1:end));
%     pu = polyfit(t1{end},y1{end},length(y1{end})-1);
%     u1_interp = @(t) polyval(pu,t);

%     puf = polyfit(t1f{end},y1f{end},length(y1f{end})-1);
%     uf_interp = @(t) polyval(puf,t);
%     Juh = integral(uf_interp,times(1),times(end));
%     JuH = integral(u1_interp,times(1),times(2));
%     disp(integral(@(x) exp(x), times(1), times(2)) - JuH)
    error = sum(err_con1);
%     disp(error);
    it = it + 1;
%     close(fig)

end
% close(v)
% close("all")
