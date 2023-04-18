% test Jacobian
Globals1D;
N = 1;
Np = N+1;
tspan = [0,1];
% R(U) = A*U + M_tilde(U) + F
% dR/dU = A*I + dMt/dU = A*I + dx/dr .* Phi' * diag(w.*df/du|_r) * Phi
fem_setup(N,1,tspan,4*N)
hk = x(end) - x(1);

% Loop through imaginary steps to test convergence
err = zeros(30,13);
h = 10.^(-1:-1:-13);
for j = 1:13
    for k=1:30
        % draw random U and d
        U = rand(Np,1);
        d = rand(Np,1); d = d./norm(d);
    
        % Construct Forward A Matrix and Jacobian                
        % polyfit for interpolation for A(f(u))
        pu = polyfit(x,U,N);
        x_interp = x(1) + (1+r).*hk./2;
        ur_k = polyval(pu, x_interp);
                    
        wfu = w.*(sin(ur_k));
        wdf = diag(w.*cos(ur_k));
        M_tilde = hk/2 .* Phi'*wfu;
        dMtdU = hk/2 .* Phi'*wdf*Phi;
        S = (V*V')\Dr;
        B = zeros(Np,Np); B(end) = -1;
        F = zeros(Np,1); F(1) = 1;
        
        A = S'+B;
        dRdU = A+dMtdU;
        RU = A + M_tilde + F;

        U_new = U + (0 + 1i).*h(j).*d;

        pu = polyfit(x,U_new,N);
        x_interp = x(1) + (1+r).*hk./2;
        ur_k = polyval(pu, x_interp);
                    
        wfu = w.*(sin(ur_k));
        wdf = diag(w.*cos(ur_k));
        M_tilde = hk/2 .* Phi'*wfu;
        dMtdU = hk/2 .* Phi'*wdf*Phi;
        
        A = S'+B;
        dRdU = A+dMtdU;
        
        RU_new = A*U_new + M_tilde + F;
        
        dR_imstep = imag(RU_new)./h(j);
        err(k,j) = norm(dR_imstep-dRdU*d)/norm(dRdU*d);
    end
end

mean_err = mean(err);
std_err = std(err);
% plot err over h
figure;
% scatter(h,err);
errorbar(h,mean_err,std_err,'LineWidth',1.5)
set(gca, 'xscale', 'log')
set(gca, 'yscale', 'log')
set(gca, 'xlim', [1e-14,1e0])
grid on
xlabel('h')
ylabel('Error in Jacobian')
title('Jacobian convergence for imaginary step')


