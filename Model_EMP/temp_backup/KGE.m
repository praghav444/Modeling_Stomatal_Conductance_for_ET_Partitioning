function [KGE]=KGE(sim, obs, method, s)
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% `KGE`: Kling-Gupta Efficiency (The optimal value of KGE is 1)
% Author : Pushpendra Raghav, The University of Alabama, USA
% Email: ppushpendra@crimson.ua.edu
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% References:
% 1. Hoshin V. Gupta, Harald Kling, Koray K. Yilmaz, Guillermo F. Martinez, 
% Decomposition of the mean squared error and NSE performance criteria: 
% Implications for improving hydrological modelling, Journal of Hydrology, 
% Volume 377, Issues 1-2, 20 October 2009, Pages 80-91, DOI: 10.1016/j.jhydrol.2009.08.003. ISSN 0022-1694, 

% 2. Kling, H., M. Fuchs, and M. Paulin (2012), Runoff conditions in the upper
% Danube basin under an ensemble of climate change scenarios,  Journal of Hydrology, 
% Volumes 424-425, 6 March 2012, Pages 264-277, DOI:10.1016/j.jhydrol.2012.01.011.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% Arguments: 
% sim: vector of simulated values
% obs: vector of observed values
% method: method to be used, either `2009` or `2012`
% s: scaling factors {Default: s=[1, 1, 1]
% Note: vectors mod and obs should be of same length
%% Output:
% KGE: Kling-Gupta Efficiency between 'sim' and 'obs'
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%% Formula
% if(method=='2009')
% KGE = 1 - sqrt( (s[1]*(r-1))^2 + (s[2]*(alpha-1))^2 + (s[3]*(beta-1))^2 )
% end
% if(method=='2012')
% KGE = 1 - sqrt( (s[1]*(r-1))^2 + (s[2]*(gamma-1))^2 + (s[3]*(beta-1))^2 )
% end
% r = Pearson correlation coefficient between mod and obs
% alpha = measure of relative variability between `sim` and `obs` values =
% sigma_sim/sigma_obs
% beta = mean_sim/mean_obs
% gamma = CV_sim/CV_obs with CV_sim = sigma_sim/mean_sim and CV_obs =
% sigma_obs/mean_obs
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% E.g., sim=[1 2 5 8 4 3 8 11 NaN 155 1 5 8 23 45 28];
% obs=[5 4 6 nan 15 24 36 14 28 15 18 19 NaN 50 24 36];
% KGE = KGE(sim,obs)
% KGE = KGE(sim,obs,2012)
% KGE = KGE(sim,obs,2009,[0.5,0.2,0.3])
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
if nargin == 2
    method = 2009;
    s = [1 1 1];
end
if nargin == 3
    s = [1 1 1];
end
%---Delete rows with nan
T = table(obs,sim);
T = rmmissing(T);
sim = T.sim;
obs = T.obs;
count1 = nnz(~isnan(sim));
count2 = nnz(~isnan(obs));
if count1 >=1 && count2 >= 1
    % Pearson Correlation Coefficient
    r = corrcoef(sim,obs, 'rows','complete');
    r = r(2);
    mean_sim = mean(sim,'omitnan');
    sigma_sim = std(sim,'omitnan');
    mean_obs = mean(obs,'omitnan');
    sigma_obs = std(obs,'omitnan');
    CV_sim = sigma_sim/mean_sim;
    CV_obs = sigma_obs/mean_obs;
    
    alpha = sigma_sim/sigma_obs;
    beta = mean_sim/mean_obs;
    gamma = CV_sim/CV_obs;
    if method==2009
        KGE = 1 - sqrt( (s(1)*(r-1))^2 + (s(2)*(alpha-1))^2 + (s(3)*(beta-1))^2 );
    end
    if method==2012
        KGE = 1 - sqrt( (s(1)*(r-1))^2 + (s(2)*(gamma-1))^2 + (s(3)*(beta-1))^2 );
    end
else
    KGE = NaN;
end 
end
