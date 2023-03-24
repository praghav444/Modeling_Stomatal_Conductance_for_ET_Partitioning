function [Es,Ei,Tr,rsc,rss]=ETestimation_emp(Zh,Ta_Avg,e_Avg,LAI,Prss_Avg,ustar,wnd_spd,As,A,theta,theta_2,VPD,Km,z0s,cd,dl,m,gc_ref,alpha,beta,theta_w,theta_s)

rss=exp(alpha-beta*theta./theta_s);% Soil resistance
g_c = surface_conductance_emp(VPD, m, gc_ref, theta_2, theta_w, theta_s); % mol m-2 s-1
R = 8.3143;  % Gas Constant; J/K/mol
T_K = Ta_Avg + 273.15; % K
Pa = Prss_Avg.*1e3; % Pa
g_c = g_c.*R.*T_K./Pa;  % PV = nRT --> V (m3m-2s-2) = nRT/P; n = mol of H2O vapor
g_c = g_c.*LAI;  % m/s
rsc=1./g_c; % s/m
rsc(rsc<=0)=0;
rsc(rsc>=10000)=10000;
[Es,Ei,Tr]=ETsw(e_Avg,Prss_Avg,Ta_Avg,rsc,LAI,Zh,rss,ustar,wnd_spd,As,A,VPD,z0s,cd,Km,dl);
if Tr<0
    Tr=0;
end
end














