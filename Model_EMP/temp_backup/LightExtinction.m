function [VegKr]=LightExtinction(DOY, lat, x)
B = (DOY-81)*2*pi/365;
ET = 9.87*sin(2*B)-7.53*cos(B)-1.5*sin(B);
DA = 23.45*sin((284+DOY)*2*pi/365);   % Deviation angle (declination)
LST = mod(DOY*24*60,24*60);
AST = LST+ET;
h = (AST-12*60)/4; % hour angle
alpha = asin((sin(pi/180*lat).*sin(pi/180.*DA)+cos(pi/180*lat).*cos(pi/180.*DA).*cos(pi/180*h)))*180/pi; % solar altitude
zenith_angle = 90-alpha; % zenith_angle <= 90
zenith_angle(zenith_angle>=80) = 80;
VegKr = sqrt(x.^2+tan(zenith_angle/180*pi).^2)/(x+1.774*(1+1.182).^(-0.733)); % Campbell and Norman 1998
end
