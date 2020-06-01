%% paper PDE simulations

d = 2; % problem dimension
Nr = 100; Nb = 200; % number of particles
eps_r = 0.01; eps_b = 0.03; % particle diamters
Tmax = 0.2;  % final time

% Example 1
drift = 1; % convex example
GF = 0; % full GF
% GF = 1; % AGF

% % Example 2
% drift = 2; % nonconvex example
% GF = 0; % full GF
% % GF = 1; % AGF

% Compute deltas (varepsilons in paper)

eps_br = (eps_r+eps_b)/2;
    
delta(1) = (Nr-1)*2*(d-1)*pi/d*eps_r^d;
delta(2) = Nb*2*pi/d*eps_br^d;
delta(3) = Nb*2*(d-1)*pi/d*eps_br^d;

% solve time-dep PDE with finite-volume scheme
out = pde_1d(Tmax, drift, delta, GF);