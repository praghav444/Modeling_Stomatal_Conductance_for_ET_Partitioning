clc;
clear all;
%-----------------   Load Data matrix     -----------------%
site = "site_name";
data = readmatrix(strjoin(["../../Input_Data/FluxData/",site,"_complete_data.csv"],''));
data(data==-9999) = nan;
temp = readtable(strjoin(["../../Input_Data/FluxData/",site,"_complete_data.csv"],''));
data = array2table(data, 'VariableNames',temp.Properties.VariableNames);
data.DateTime = temp.DateTime;
%----Extract data based on ML training----
train_dat = readmatrix(strjoin(["../../Input_Data/Train_Val_Data/X_train_temp_",site,".csv"],''));
train_dat(train_dat==-9999) = nan;
temp = readtable(strjoin(["../../Input_Data/Train_Val_Data/X_train_temp_",site,".csv"],''));
train_dat = array2table(train_dat, 'VariableNames',temp.Properties.VariableNames);
train_dat = [train_dat.Rn,train_dat.USTAR,train_dat.SRad,train_dat.H,train_dat.TA,train_dat.LE,train_dat.WS,train_dat.G,train_dat.CO2,train_dat.fPAR];
train_dat = array2table(train_dat, 'VariableNames',{'NETRAD','USTAR','SW_IN_F_MDS','H_F_MDS','TA_F_MDS','LE_F_MDS','WS','G_F_MDS','CO2_F_MDS','fPAR'});
row_index = find(ismember(data.NETRAD,train_dat.NETRAD,'rows') & ismember(data.USTAR,train_dat.USTAR,'rows') &...
    ismember(data.SW_IN_F_MDS,train_dat.SW_IN_F_MDS,'rows') & ismember(data.H_F_MDS,train_dat.H_F_MDS,'rows')...
    & ismember(data.TA_F_MDS,train_dat.TA_F_MDS,'rows')& ismember(data.LE_F_MDS,train_dat.LE_F_MDS,'rows')...
    & ismember(data.WS,train_dat.WS,'rows')& ismember(data.G_F_MDS,train_dat.G_F_MDS,'rows')...
    & ismember(data.CO2_F_MDS,train_dat.CO2_F_MDS,'rows')& ismember(data.fPAR,train_dat.fPAR,'rows'));
data = data(row_index,:);
%-----------------   Load Energy data     -----------------% 
H             = data.H_F_MDS; %Sensible heat, used as validation                        [W/m2]
H(H < -100 | H > 1200) = nan;
LE            = data.LE_F_MDS; %Latent heat, used as validation                         [W/m2]
LE(LE < -100 | LE > 1200) = nan;
Gs_Avg        = data.G_F_MDS; %Ground heat, used as validation                          [W/m2]
Rn_Avg        = data.NETRAD; %Net radiation, used as model input                        [W/m2]
Rs_inc_Avg    = data.SW_IN_F_MDS; %Income solar radiation, used as model input          [W/m2]
%-----------------   Load Metro data     -----------------% 
Rh_Avg        = data.RH./100; %Relative humidity, used as model input                   [-]
Ta_Avg        = data.TA_F_MDS - 273.15; %Air temperature, used as model input           [C]
U             = data.WS; %wind speed , used as model input                              [m/s]
Prss_Avg      = data.PA./1000; %Air pressure , used as model input                            [Kpa]
VPD_Avg       = data.VPD_F_MDS./10; %VPD data , used as model input                     [Kpa]
CO2           = data.CO2_F_MDS.*44.01.*0.0409; %CO2 concentrition                       [mg/m3]
                %if no data available, set to NaN below
ObukhovLength = zeros(length(LE),1);   %ObukhovLength length,used as model input        [m]
ObukhovLength(:) = NaN;
ustar         = data.USTAR; %Friction velocity, used as model input                     [m/s]
Rain          = data.P; %Precipitation amount, used as Diagnostic parameters            [mm]
%-----------------   Load Surface and ground data     -----------------% 
LAI           = data.LAI; %Leaf area index, used as model input                         [m2/m2]
theta         = data.SWC_F_MDS_1./100; %Soil moisture at top layer,  used as model input        [-]
Zh            = data.hc; %Vegetation Height, used as model input                  [m]
                %if no data available, set theta_2=theta
% Let's find all the SWC columns
mask_SWC = startsWith( data.Properties.VariableNames, 'SWC') & ~endsWith(data.Properties.VariableNames, 'QC');
temp = data(:,mask_SWC);
temp = table2array(temp);
temp = nanmean(temp,2);  % Mean across all the sensors
theta_2       = temp./100; %Soil moisture at root layer, used as model input        [-] 
theta(isnan(theta)) = theta_2(isnan(theta));
%------- GPP Data (Raghav)--------------%
GPP = data.GPP_NT_VUT_REF;
%-----------------Additional Info---------------------------------------%
DOY           = data.DoY;  % Day of the Year
lat           = data.lat;  % Latitude of the site

QCflag=ones(length(DOY),1);
%%
%-----------------  convert     -----------------% 
theta(theta>1)         = theta(theta>1) /100;
Rh_Avg(Rh_Avg>1)       = Rh_Avg(Rh_Avg>1)/100;
Ta_Avg(Ta_Avg>50)      = Ta_Avg(Ta_Avg>50)-273.15;

%%
%calculate Metro varaible
e_Avg=0.6108*exp(17.27*Ta_Avg./(Ta_Avg + 237.3)).*Rh_Avg; %Kpa
esat_air=0.6108*exp(17.27*Ta_Avg./(Ta_Avg + 237.3));%Kpa
VPD_Avg(isnan(VPD_Avg)==1)=esat_air(isnan(VPD_Avg)==1)-e_Avg(isnan(VPD_Avg)==1);
VPD_Avg(VPD_Avg<0) = 0;
%%
%----------------- Calculate ground heat flux if no observations available -----------------% 
Gs_Avg(isnan(Gs_Avg)) = 0.4.*Rn_Avg(isnan(Gs_Avg)).*exp(-0.5.*LAI(isnan(Gs_Avg))); % Choudhoury (1988)
%----------------- Calculate ground heat and water storage heat  -----------------% 
if MyConstants.Flooding==1
    cw=4184;%water specific heat;[Jkg-1K-1]
    rouw=1000;% water density     [kgm-3]
    Gw=ones(length(DOY),1);
    Gw(1)=0.0;
    for i=2:length(DOY)
        Gw(i)=cw*rouw*(WaterDepth(i)).*(WaterTemp(i)-WaterTemp(i-1))/ MyConstants.dt;
    end
    G=Gs_Avg+Gw; %ground heat + water storage heat
else
    G=Gs_Avg;
end
clear Gs_Avg GW cw rouw i
%%
%%
%Check the flux imblance
imbalance=abs(1-(Rn_Avg-G)./(H+LE));
QCflag(imbalance>0.2)=0;%Mark as unrelaible data

%Force flux balance
if MyConstants.Flooding==1
    G(imbalance>0.2)=Rn_Avg(imbalance>0.2)*0.2; %if flooding,G is more unreliable, force it =0.2Rn
end 
[LE_c,H_c]=flux_c(G,H,LE,Rn_Avg);
LE_c(LE_c < -100 | LE_c > 1200) = nan;
w_T=H_c./1.23/1004;
ObukhovLength(isnan(ObukhovLength)==1)=-((ustar(isnan(ObukhovLength)==1)).^3./(0.4*(9.8./Ta_Avg(isnan(ObukhovLength)==1)).*(w_T(isnan(ObukhovLength)==1))));
% Fixed bug in line above


%QCcheck
QCflag(LE_c>1200|LE_c<-100)=0;
QCflag(H_c>1200|H_c<-100)=0;
QCflag(Rn_Avg>1200|Rn_Avg<-200)=0;
QCflag(Rn_Avg>1200|Rn_Avg<-200)=0;
QCflag(Rh_Avg>=1)=0;
QCflag(ustar<0.06)=0;
if MyConstants.Flooding==1
   QCflag(WaterDepth>0)=0;
end

if Rain(1)>0
   QCflag(1)=0;
end
for i=2:length(DOY)-1
  if Rain(i)>0
    for j=-1:1:1
        QCflag(i+j)=0;
    end
  end
end

%%
%---------------------------------
% Calculate Vegetation Light Extinction Coefficient (Kr)
leaf_angle_distr = 1;
Kr = LightExtinction(DOY, lat, leaf_angle_distr);
Kr(:) = 0.6;
%%
%###############################
A=Rn_Avg-G;                              %Total available energy               [w/m2]
Rns=Rn_Avg.*exp(-Kr.*LAI); %Radiation reaching the soil surface  [w/m2]
As=Rns-G;                                %Available energy for the soil surface[w/m2] 
%-----------------------------------------------
%initialization
Es=zeros(length(Zh),1);                  %Soil evaporation                     [w/m2] 
Ei=zeros(length(Zh),1);                  %Interception evaporation                     [w/m2] 
Tr=zeros(length(Zh),1);                  %Canopy evaporation                   [w/m2] 
rsc=zeros(length(Zh),1);                 %Canopy resistance                    [s/m] 
rss=zeros(length(Zh),1);                 %Soil resistance                      [s/m] 
Tem_c=zeros(length(Zh),1);               %Canopy temperature                   [C]
Tc_temp=Ta_Avg+2.0;%Assume canopy temperature is 2 degree higher than air temperature
k=length(Zh);
%%
%% Rain-Mask (for interception)
Rain_Mask=zeros(length(Zh),1);  
for it = 1:k
    if Rain(it) > 0
        Rain_Mask(it:it+8) = 1; % Rain hour plus following 4 hours
    end
end

%------Soil Properties----
df_soil_theta_s = readtable('../../Input_Data/Soil_Data/theta_s_all_sites.csv');   
df_soil_theta_s = df_soil_theta_s(strcmp(df_soil_theta_s.Site,site),:);
theta_s = nanmean([df_soil_theta_s.theta_s1, df_soil_theta_s.theta_s2,...
    df_soil_theta_s.theta_s3,df_soil_theta_s.theta_s4]);
if isnan(theta_s)
theta_s = 0.36;
end
df_soil_theta_w = readtable('../../Input_Data/Soil_Data/theta_r_all_sites.csv');   
df_soil_theta_w = df_soil_theta_w(strcmp(df_soil_theta_w.Site,site),:);
theta_w = nanmean([df_soil_theta_w.theta_r1, df_soil_theta_w.theta_r2,...
    df_soil_theta_w.theta_r3,df_soil_theta_w.theta_r4]);
if isnan(theta_w)
theta_w = 0.18;
end

%load parameters
Km = load('Parameters/MCMC_out_calib_Km_m_gcref.mat', 'Km');
Km = Km.Km;
z0s = 0.01;
cd = 0.001;
dl = 0.068;
m = load('Parameters/MCMC_out_calib_Km_m_gcref.mat', 'm');
m = m.m;
gc_ref = load('Parameters/MCMC_out_calib_Km_m_gcref.mat', 'gc_ref');
gc_ref = gc_ref.gc_ref;
alpha = 8.206;
beta = 4.225;
[Es,Ei,Tr,rsc,rss]=ETestimation_emp(Zh,Ta_Avg,e_Avg,...
    LAI,...
    Prss_Avg,ustar,U...
    ,As,A,theta,theta_2,VPD_Avg,Km,z0s,cd,dl,m,gc_ref,alpha,beta,theta_w,theta_s);
Ei(Rain_Mask==0) = 0;
Ei(Ei<0)=0;
Ei = 0;    % Let's not consider interception for now
Tr(Tr<0)=0;
LES=Es+Ei+Tr;
LES(imag(LES)~=0) = nan;

%
data.LE_c = LE_c;
data.QCflag = QCflag;
r = corrcoef(LES(LES<800&LE<800&LE>-100),LE(LES<800&LE<800&LE>-100), 'rows','complete');
r = r(2);
rmse = sqrt(mean((LE(LES<800&LE<800&LE>-100) - LES(LES<800&LE<800&LE>-100)).^2));

data.(sprintf('Tr_SW')) = Tr;
data.(sprintf('LE_SW')) = LES;
data.(sprintf('rsc')) = rsc;
data = table(data.DateTime,LE,LE_c,VPD_Avg,theta,theta_2,Ta_Avg,Rain,LAI,...
 QCflag,LES,Tr,rsc,Rn_Avg,G,H,'VariableNames', {'DateTime','LE','LE_c','VPD_Avg','theta','theta_2','Ta_Avg','Rain','LAI',...
'QCflag','LE_SW','Tr_SW','rsc','Rn','G','H'});
writetable(data, strjoin(["Output/Out_SW_",site,"_emp_train.csv"],''))