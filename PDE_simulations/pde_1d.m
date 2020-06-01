
function out = pde_1d(Tmax,drift,delta,GF)

mex mex_PME.c

delta1 = delta(1); delta2 = delta(2);

out.delta1 = delta1;
out.delta2 = delta2;

L = 5;                     % computational domain [-L, L]
N = 1000 + 2;              % total number of cells (2 ghost nodes)
dt = 1.0e-6;               % time step

t = 0.0;

dx = 2*L/(N-2); % spatial grid size
x = linspace(-L-dx,L+dx,N+1);          % xgrid (add two extra points at ends, so that they are the ghost nodes are are fixed at ends)
xmid = (x(1:end-1)+x(2:end))/2;        % cell center

tout = linspace(0, Tmax,41)'; % times out

% uniform initial data 
rho = ones(size(xmid));

% potential b
b_fun = get_drift(drift,L);

bmid = b_fun(xmid);
b = b_fun(x);

xi = zeros(1,N);          
u1 = zeros(1,N+1);              
u2 = zeros(1,N+1);           

% normalize the mass (skip 2 end points that are outside domain)
mass= sum(rho(2:end-1))*dx;
rho = rho/mass;

Edata = zeros(length(tout),1);
rhodata = zeros(length(tout), length(xmid)-2); % get rid of ghost nodes
tout_real = 0*tout;
ind_next = 1;
tout(end+1) = 100*Tmax; % to make sure we don't enter the if statement after Tmax

if GF
    % AGF
    Tend = 2*Tmax;
else
    % Exact GF
    Tend = 1.00001*Tmax;
end

while (t<Tend)
    if (t>= tout(ind_next))
        % output data
        tout_real(ind_next) = t;
        rhodata(ind_next,:) = rho(2:end-1);
        Edata(ind_next) = compute_entropy(rho(2:end-1));
        ind_next = ind_next + 1;
        disp([num2str(t) ' out of ' num2str(Tmax)]);
    end
    
    % 3rd order Strong Stability Preserving Runge-Kutta method
    % Calculate the convolution at cell interfaces, then interpolated to the cell center
    
    [flux1,newdt] = mex_PME(N,delta1,delta2,dt,x,xmid,rho,b,bmid,xi,u1,u2,1, GF);
    temprho1 = rho ...
        - newdt*(flux1(2:end)-flux1(1:end-1))./(x(2:end)-x(1:end-1));
    
    [flux2,tempdt] = mex_PME(N,delta1,delta2,dt,x,xmid,temprho1,b,bmid,xi,u1,u2,0, GF);
    temprho2 = 0.75*rho+0.25*temprho1 ...
        - 0.25*newdt*(flux2(2:end)-flux2(1:end-1))./(x(2:end)-x(1:end-1));
    
    [flux3,tempdt] = mex_PME(N,delta1,delta2,dt,x,xmid,temprho2,b,bmid,xi,u1,u2,0, GF);
    temprho3 = rho/3+2*temprho2/3 ...
        - 2*newdt/3*(flux3(2:end)-flux3(1:end-1))./(x(2:end)-x(1:end-1));
    t = t + newdt;
    dnorm = norm(temprho3-rho);
    rho = temprho3;
end

out.dnorm = dnorm;
out.f1 = flux1;
out.f2 = flux2;
out.f3 = flux3;

if GF
    % AGF computes entropy by long time simulation
    Einf = compute_entropy(rho(2:end-1));
    rhoinf = rho(2:end-1); % rho infinity
else
    outeq = pde1d_eq(drift,delta,L,N-2);
    Einf = outeq.Eeq;
    rhoinf = outeq.rhoeq;
end


% don't output ghost nodes
out.x = xmid(2:end-1);
out.t = tout_real;
out.rho = rhodata;
out.Erel = Edata - Einf;
out.Ebreg = Edata - Einf - compute_bregman(rhodata,rhoinf);

out.rhoinf = rhoinf;
out.Einf = Einf; % E infinity
out.b = bmid(2:end-1);


    function E = compute_entropy(rho)
        
        rhologrho = rho.*log(rho);
        rhologrho(rho<=0) = 0;
        E = sum((rhologrho + delta1*rho.^2/2 + delta2*rho.*bmid(2:end-1)).*dx);
        
    end

    function E1 = compute_bregman(rho, rhoinf)
        logrhoinf = log(rhoinf);
        logrhoinf(rhoinf<=0) = 0;
        E1 = sum((logrhoinf + delta1*rhoinf + delta2*bmid(2:end-1)).*(rho-rhoinf).*dx,2);
    end


end
