% compute equilibrium solution for the GF

function out = pde1d_eq(drift, delta,L,N)

delta1 = delta(1); delta2 = delta(2);
out.delta1 = delta1;
out.delta2 = delta2;

dx = 2*L/(N); % spatial grid size
x = linspace(-L,L,N+1)';                % xgrid (add two extra points at ends, so that they are the ghost nodes are are fixed at ends)
xmid = (x(1:end-1)+x(2:end))/2;        % cell center

nv=numel(xmid);

%% get drift (b density)

b_fun = get_drift(drift,L);
bmid = b_fun(xmid);

chi = 0;

% initial condition
rho = 1 + delta2*(1-bmid); mass = sum(rho)*dx;
rho = rho/mass;

sol = [rho;chi];

% Parameters for the Newton scheme
maxiter = 500;
tau = 1;
maxerr = 1e-9;

% trapezoidal rule
quadop = dx*ones(size(rho))';


F = @(rho, chi) [log(rho) + delta1*rho + delta2*bmid - chi * ones(size(rho)); sum(rho)*dx - 1];
JF = @(rho, chi) [[diag(1./rho + delta1*ones(size(rho))) -ones(size(rho))]; [quadop 0]];

iter = 1;
err = 1;

while ((iter < maxiter) && (err > maxerr))
    
    Fval = F(rho,chi);
    JFval = JF(rho,chi);
    update = JFval\Fval;
    
    sol = sol - tau * update;
    
    rho = sol(1:nv);
    chi = sol(end);
    err = norm(update);
    iter = iter+1;
    display(['Iter = ' num2str(iter) '. Error = ' num2str(err) '. Error in rho = ' num2str(norm(update(1:end-1)))]);
end

display(['Newton solver converged after ' num2str(iter) ' iterations with an error of ' num2str(err)]);

display(['Masses: sum(rho) = ' num2str(dx*sum(rho))]);

out.x = xmid';
out.rhoeq = rho';
out.b = bmid';
out.Eeq = compute_entropy(rho);

    function E = compute_entropy(rho)
        
        rhologrho = rho.*log(rho);
        rhologrho(rho<=0) = 0;
        E = sum((rhologrho + delta1*rho.^2/2 + delta2*rho.*bmid).*dx);
        
    end


end
