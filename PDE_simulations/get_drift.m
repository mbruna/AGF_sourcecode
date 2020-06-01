function V = get_drift(drift,L)

switch drift
    case 1
        disp('Convex example')
        V = @(x) x.^2 + 0.75;
    case 2
        disp('Nonconvex example')
        V = @(x) (1+0.1*sin(20*x)).*(x.^2 + 0.75);
end

mass = integral(V, -L, L);
if mass>0
    V = @(x) V(x)/mass;
end

