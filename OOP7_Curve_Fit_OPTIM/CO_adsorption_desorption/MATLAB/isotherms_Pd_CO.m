clc, clear, close all

k_CO_fwd = 8.2832E+07;
k_CO_rev = 8.9897E-02; % avg : 0.112109
K_CO = k_CO_fwd/k_CO_rev;

total_sites = 18432;

P_eqn = logspace(-14,0,1000);
theta_eqn = (K_CO.*P_eqn)./(1+(K_CO.*P_eqn));

data_CO_lat=importdata('t_avg_CO.txt');
P_lat = data_CO_lat(:,1);
n_lat = data_CO_lat(:,2);
theta_lat = n_lat./total_sites;

data_CO_nolat=importdata('t_avg_CO_no_lateral.txt');
P_nolat = data_CO_nolat(:,1);
n_nolat = data_CO_nolat(:,2);
theta_nolat = n_nolat./total_sites;

figure(1)
semilogx(P_eqn,theta_eqn,'r',P_lat,theta_lat,'b*',P_nolat,theta_nolat,'ms')
xlabel('Pressure CO [bar]')
ylabel('theta CO*')
title('CO only')
legend('Isotherm Equation','With lateral interactions','No lateral interactions')

figure(2)
loglog(P_eqn,theta_eqn,'r',P_lat,theta_lat,'b*',P_nolat,theta_nolat,'ms')
xlabel('Pressure CO [bar]')
ylabel('theta CO*')
title('CO only')
legend('Isotherm Equation','With lateral interactions','No lateral interactions')





