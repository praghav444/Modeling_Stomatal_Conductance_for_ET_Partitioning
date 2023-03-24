function gc = surface_conductance_emp(VPD, m, gc_ref, theta_rz, theta_w, theta_s)
% This function calculates stomatal conductance (gc) to CO2, mol/m2/s
% In the empirical representation: gc = gc_star*(1-mlnVPD)
% gc_star = gc_ref*max(0,min((theta-theta_w)/(theta_s-theta_w),1))

% Input
% VPD: vapor pressure deficit (kPa)
% m: tunable empirical parameter [0,1], ln(kPa)-1
% gc_ref = maximum gs_star [0,1], mol m-2 s-1
% theta_rz = root-zone soil moisture (m3 m-3)
% theta_w = SM at wilting point (i.e, when stomata fully closes)
% theta_s = saturation SM

% Output:
% gc: stomatal conductance to CO2 (mol m-2 s-1)
gc_star = gc_ref.*max(0,min((theta_rz-theta_w)./(theta_s-theta_w),1));
gc = gc_star.*(1-m.*real(log(VPD)));
gc(gc < 0.001) = 0.001;
end