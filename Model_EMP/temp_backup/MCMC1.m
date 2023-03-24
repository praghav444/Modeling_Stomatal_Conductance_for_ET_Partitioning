%calculating prior cost function value
J_old = 1 - KGE(mod,obs);

J_old1=J_old;

N_params = 3;
Parameters_keep= [0 0 0]';
nsimu=20000;
upgrade=1;
allow=5;
par_old=par;
diff=Max-Min;
% part 1: MCMC run to calculate parameter covariances
for simu=1:nsimu
    while (true)
        %propose new sets of parameters
        par_new = par_old+(rand(N_params,1)-0.5).*diff/allow;
        %check if paramaters are within their minima and maxima
        if (par_new(1)>Min(1)&& par_new(1)<Max(1)...
                && par_new(2)>Min(2)&& par_new(2)<Max(2)...
                && par_new(3)>Min(3)&& par_new(3)<Max(3))
            break;
        end
    end
    Km = par_new(1);
    m = par_new(2);
    gc_ref = par_new(3);
    
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

    modl=LES;

    J_new = 1 - KGE(modl,obs);
    
    delta_J = J_new-J_old;
    
    if min(1,exp(-delta_J))>rand    
        Parameters_keep(:,upgrade)=par_new;
        J_keep(upgrade)=J_new;
        upgrade=upgrade+1;
        par_old=par_new;
        J_old=J_new;     
    end
    dispX = ['Simulation: ', num2str(simu), ' upgrade: ',num2str(upgrade)];
    disp(dispX)
    
    Parameters_rec(:,simu)=par_old;
    J_rec(:,simu)=J_old;
    
end
J_keep1=J_keep;
covars=cov(Parameters_rec(:,(simu/2):simu)');
upgrade=1;
simu=1;
nsimu=150000;
J_old=J_old1;
sd=2.4/((N_params.^(0.499)));

% part 2: MCMC run with updating covariances
for simu=1:nsimu
    while (true)
        
        par_new = mvnrnd(par_old,covars)';
        
        if (par_new(1)>Min(1)&& par_new(1)<Max(1)...
                && par_new(2)>Min(2)&& par_new(2)<Max(2)...
                && par_new(3)>Min(3)&& par_new(3)<Max(3))
            break;
        end
    end
    Km = par_new(1);
    m = par_new(2);
    gc_ref = par_new(3);
        
    [Es,Ei,Tr,rsc,rss]=ETestimation_emp(Zh,Ta_Avg,e_Avg,...
        LAI,...
        Prss_Avg,ustar,U...
        ,As,A,theta,theta_2,VPD_Avg,Km,z0s,cd,dl,m,gc_ref,alpha,beta,theta_w,theta_s);
    Ei(Rain_Mask==0) = 0;
    Ei(Rain_Mask==0) = 0;
    Ei(Ei<0)=0;
    Ei = 0;    % Let's not consider interception for now
    Tr(Tr<0)=0;
    LES=Es+Ei+Tr;
    LES(imag(LES)~=0) = nan;
    
    modl=LES;
    
    J_new = 1 - KGE(modl,obs);
    
    delta_J = J_new-J_old;
    
    if min(1,exp(-delta_J))>rand
        
        Parameters_keep(:,upgrade)=par_new;
        J_keep(upgrade)=J_new;
        par_old=par_new;
        J_old=J_new;
        coef=upgrade/simu;
        
        upgrade=upgrade+1;
    end
    
    dispX = ['Simulation: ', num2str(simu), ' upgrade: ',num2str(upgrade)];
    disp(dispX)
    
    Parameters_rec1(:,simu)=par_old;
    J_rec(:,simu)=J_old;
    if (simu>4000)
        covars=sd*(cov(Parameters_rec1(:,2000:simu)'));
    end
    
end