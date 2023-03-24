function [Es, Ei,T]=ETsw(e_Avg,Prss_Avg,Ta_Avg,rsc,LAI,Zh,rss,ustar,wnd_spd,As,A,VPD,z0s,cd,Km,dl)

[Cp,Psy,rou,~,delta]= metalib(e_Avg,Prss_Avg,Ta_Avg);% get metro data
%%
% Calculate reference height of measurement (d0) and the roughness
% lengths(z0) governing the transfer of momentum [m]
z0=zeros(1,1);
X=(cd).*LAI;
d0=1.1*Zh.*log(1+X.^0.25);
z0(X<0.2)=(z0s)+0.3*Zh(X<0.2).*(X(X<0.2).^0.5);
z0(X>0.2)=0.3*Zh(X>0.2).*(1-d0(X>0.2)./Zh(X>0.2));  % Bug here (1-d0)/Zh
z0 = z0';
%%
% the eddy diffusion coefficient at the top of the canopy (Kh)
Kh=0.4*ustar.*(Zh-d0);

%%
%canopy boundary layer resistance (rac) and  the mean boundary layer
%resistance(rb), calcualted from the wind speed at the top of canopy
%(uh)and the characteristic leaf dimension (dl)
Uh=wnd_spd./(1+log(MyConstants.Zu_m-Zh+1));
rb=(100./(Km)).*((dl)./Uh).^0.5./(1-exp(-(Km)/2));  % Fixed the bug here
rac=rb./(2*LAI);

%%
%aerodynamic resistances ras and raa are calculated by integrating the eddy
%diffusion coefficients from the soil surface to the level of the
%preferred sink of momentum in the canopy [s/m]
raa=(1./(0.4.*ustar)).*log(((MyConstants.Zu_m)-d0)./(Zh-d0))+(Zh./((Km).*Kh)).*(exp((Km).*(1-(d0+z0)./Zh))-1);
ras=(Zh.*(exp(Km))./((Km).*Kh)).*(exp((-(Km).*(z0s))./Zh)-exp(-(Km).*((d0+z0)./Zh)));

%%
%Two source SW model calculation
Rc=(delta+Psy).*rac+Psy.*rsc;
Rs=(delta+Psy).*ras+Psy.*rss;
Ra=(delta+Psy).*raa;
wc=1./(1+Rc.*Ra./(Rs.*(Rc+Ra)));
ws=1./(1+Rs.*Ra./(Rc.*(Rs+Ra)));
PMc=(delta.*A+(rou.*Cp.*VPD-delta.*rac.*As)./(raa+rac))./(delta+Psy.*(1+rsc./(raa+rac)));
PMs=(delta.*A+(rou.*Cp.*VPD-delta.*ras.*(A-As))./(raa+ras))./(delta+Psy.*(1+rss./(raa+ras)));  % Fixed the bug here
%--Interception Losses (Addeded by Raghav)----
RH = e_Avg./(VPD+e_Avg);
relative_surface_wetness = RH.^4;
relative_surface_wetness(Ta_Avg<=0)=0;
PRIESTLEY_TAYLOR_ALPHA = 1.26;
epsilon = delta./(delta + Psy);
interception_evaporation = relative_surface_wetness .* PRIESTLEY_TAYLOR_ALPHA .* epsilon .* (A-As);
interception_evaporation(interception_evaporation < 0) = 0;
%-----------------------------------------------
T=wc.*PMc;
Es=ws.*PMs;
Ei=interception_evaporation;
end