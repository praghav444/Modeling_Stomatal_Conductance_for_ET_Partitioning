classdef MyConstants  
    properties (Constant = true)  
        dt        =   30*60;     %Time step, if hourly it should be 60*60                           [second]
        
        Isotope   =   0;         %  =1 Isotope is calculation
        lwc_measurement= 1 ;     %  =1 leaf water content is measured; =0 not measured (For isotope only)
        Flooding  =   0;         %  =1 Wetland cases (Mangrove, Paddy); =0 other cases

        Zu_m      =   tower_height;         %Wind speed measurement height                                     [m]
        Zt_m      =   tower_height;      %Temperature  measurement height                                   [m]  
        
        C3        =   1;         %  =1 C3 type; =0 C4 type vegetation
       %% Model calibaration parameters
      %  D0        =   0.25 ;      %
      %  rss_v     =   1 ;
       %% depends on field condition
       % Km        =   2.5  ;    % the effective roughness length of the soil substrate 
       % z0s       =   0.01;     %0.001 in?
       % cd        =   0.1;      %Maruyama(2008,2010); =0.1 Meyers and Paw et al. (1987)
       % dl        =   0.068;    %leaf wide
       % Kr        =   0.6;      %
       % g_min_c   =   0.4;      %

    end  
end  