writematrix(Dr, 'Dr.txt')
writematrix(EToE, 'EToE.txt')
writematrix(EToF, 'EToF.txt')
writematrix(Fmask, 'Fmask.txt')
writematrix(Fscale, 'Fscale.txt')
writematrix(Fx, 'Fx.txt')
writematrix(invV, 'invV.txt')
writematrix(J, 'J.txt')
writematrix(K, 'K.txt')
writematrix(LIFT, 'LIFT.txt')
writematrix(mapB, 'mapB.txt')
writematrix(mapI, 'mapI.txt')
writematrix(mapO, 'mapO.txt')
writematrix(N, 'N.txt')
writematrix(Nfaces, 'Nfaces.txt')
writematrix(Nfp, 'Nfp.txt')
writematrix(NODETOL, 'NODETOL.txt')
writematrix(Np, 'Np.txt')
writematrix(nx, 'nx.txt')
writematrix(r, 'r.txt')
writematrix(rk4a, 'rk4a.txt')
writematrix(rk4b, 'rk4b.txt')
writematrix(rk4c, 'rk4c.txt')
writematrix(rx, 'rx.txt')
writematrix(V, 'V.txt')
writematrix(vmapB, 'vmapB.txt')
writematrix(vmapI, 'vmapI.txt')
writematrix(vmapM, 'vmapM.txt')
writematrix(vmapO, 'vmapO.txt')
writematrix(vmapP, 'vmapP.txt')
writematrix(VX, 'VX.txt')
writematrix(x, 'x.txt')
writematrix(dt, 'dt.txt')
writematrix(invV, 'invV.txt')


% % Create a table with the data and variable names
% T = table(vmapI, 'VariableNames', { 'vmapI'} )
% % Write data to text file
% writetable(T, 'vmapI.txt')

% %
% T = table(vmapO, 'VariableNames', { 'vmapO'} )
% % Write data to text file
% writetable(T, 'vmapO.txt')

% % Create a table with the data and variable names
% T = table(mapI, 'VariableNames', { 'mapI'} )
% % Write data to text file
% writetable(T, 'mapI.txt')

% T = table(mapO, 'VariableNames', { 'mapO'} )
% % Write data to text file
% writetable(T, 'mapO.txt')

% % Create a table with the data and variable names
% T = table(vmapM, 'VariableNames', { 'vmapM'} )
% % Write data to text file
% writetable(T, 'vmapM.txt')

% T = table(vmapP, 'VariableNames', { 'vmapP'} )
% % Write data to text file
% writetable(T, 'vmapP.txt')

% T = table(Fscale, 'VariableNames', { 'Fscale'} )
% % Write data to text file
% writetable(T, 'Fscale.txt')

% T = table(LIFT, 'VariableNames', { 'LIFT'} )
% % Write data to text file
% writetable(T, 'LIFT.txt')

% T = table(rx, 'VariableNames', { 'rx'} )
% % Write data to text file
% writetable(T, 'rx.txt')

% T = table(Dr, 'VariableNames', { 'Dr'} )
% % Write data to text file
% writetable(T, 'Dr.txt')

% T = table(rk4a, 'VariableNames', { 'rk4a'} )
% % Write data to text file
% writetable(T, 'rk4a.txt')

% T = table(rk4b, 'VariableNames', { 'rk4b'} )
% % Write data to text file
% writetable(T, 'rk4b.txt')

% T = table(rk4c, 'VariableNames', { 'rk4c'} )
% % Write data to text file
% writetable(T, 'rk4c.txt')

% T = table(x, 'VariableNames', { 'x'} )
% % Write data to text file
% writetable(T, 'x.txt')

% T = table(N, 'VariableNames', { 'N'} )
% % Write data to text file
% writetable(T, 'N.txt')

% T = table(K, 'VariableNames', { 'K'} )
% % Write data to text file
% writetable(T, 'K.txt')

% T = table(nx, 'VariableNames', { 'nx'} )
% % Write data to text file
% writetable(T, 'nx.txt')

