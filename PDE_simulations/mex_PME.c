#include "mex.h"
#include <math.h>

/*
 * The 2nd order numerical flux for the finite volume scheme for the equation
 *      rho_t = ( rho * (1 - d2*b) * [ log(rho) + d1*rho + d2*b ]_x + d2*b*rho * [d1*rho + d2*b ]_x )_x
 * at the cell interfaces, i.e  rho * (1 - d2*b) * [ log(rho) + d1*rho + d2*b ]_x + d2*b*rho * [d1*rho + d2*b ]_x  at the interfaces by calling
 *      [flux,newdt] = mex_PME(N,delta1,delta2,dt,x,xmid,rho,b,bmid,xi,u, stage);
 *  
 * Input: 
 *   N:     the number of cells  (grid: a=x[0]<x[1]< ... < x[N-1] < x[N]=b
 *   delta1, delta2:  coefficients of the nonlinear diffusion and cross diffusion
 *   dt:    default time step (could be reduced by CLF condition)
 *   x:     end points of the cells ( size N+1 )
 *   xmid:  center of the cells ( size N)
 *   rho:   the solution at the mid point in each cell
 *   b, bmid:     porosity distribution, at edges and centre of cells
 *   xi1:   log(rho) + d1*rho + d2*b at the center of the cell
 *   u1,u2:    the velocity corresponding to xi1,xi2 at the cell interface (or end points)
 *   stage: there are three calculations of the flux in the 
            Strong Stability Preserving RK3 (SSPRK(3,3)), the actual time step is 
            determined in the first evaluation. The optimal SSP coefficient is 
            the same as the forward Euler
 *   GF: whether to solve the full or asymptotic GF. GF = 0 for AGF, GF = 1 for GF pair 1
 *            
 * Output: 
 *   flux:  the flux at the cell interface (or end points)
 *   newdt:  the actual estimated time step (could be different from dt, if the CLF is 
 *           not satisfied with dt)
 *
 */


/* the maximum of a and b, used to select the upwind density */
double max(double a, double b)
{
    return a>b?a:b;
}

/* the minimum of a and b, used to select the upwind density */
double min(double a, double b)
{
    return a>b?b:a;
}

/* The minmod slope limiter in recontruction of function from values at cell center */
double minmod(double a, double b, double c) {
    if (a>0 && b>0 && c>0) {        
        return min(a,min(b,c));                
    } else if (a<0 && b<0 && c<0) {
        return max(a,max(b,c));        
    } else {
        return 0;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *xi, *rho, delta1, delta2, *x, *xmid, *flux, dt, newdt, *u1, *u2, *b, *bmid;
    int N, stage, GF;
    double u1max, u1min, u2max, u2min, tempu, lambda, rhoEast,rhoWest, rhodev;        
    const double theta = 2.0;
    int i;
    
    /* Get the input data*/
    N = mxGetScalar(prhs[0]);
    delta1 = mxGetScalar(prhs[1]);
    delta2 = mxGetScalar(prhs[2]);    
    dt = mxGetScalar(prhs[3]);
    x = mxGetPr(prhs[4]);
    xmid = mxGetPr(prhs[5]);
    rho = mxGetPr(prhs[6]);
    b = mxGetPr(prhs[7]);
    bmid = mxGetPr(prhs[8]);
    xi = mxGetPr(prhs[9]);    
    u1 = mxGetPr(prhs[10]);
    u2 = mxGetPr(prhs[11]);
    stage = mxGetScalar(prhs[12]);
    GF = mxGetScalar(prhs[13]);
    
    /* Create the first output, the numerical flux */
    plhs[0] = mxCreateDoubleMatrix(1,N+1,mxREAL);
    flux = mxGetPr(plhs[0]);
 
    
    //////* ENTROPY PART */////
    /* Find xi1 = log(rho) + delta1*rho + delta2*b at the cell center */
    for (i=0; i<N; i++) {
        if (rho[i]>0)
            xi[i] = log(rho[i]) + delta1*rho[i] + delta2*bmid[i];
        else
            xi[i] = log(1.e-10) + delta2*bmid[i];
    }
    
    /* Find u1 at the cell interface */
    u1[0] = 0.0;
    u1[N] = 0.0;
    for (i=1; i<N; i++) {
            u1[i] = -(xi[i]-xi[i-1])/(xmid[i]-xmid[i-1]);
    }
    
          //////* EXTRA PART */////
    /* Find xi2 = d1*rho + d2*b at the cell center */
    for (i=0; i<N; i++) {
        if (rho[i]>0)
            xi[i] = delta1*rho[i] + delta2*bmid[i];
        else
            xi[i] = delta2*bmid[i];
    }
    
    /* Find u2 at the cell interface */
    u2[0] = 0.0;
    u2[N] = 0.0;
    for (i=1; i<N; i++) {
            u2[i] = -(xi[i]-xi[i-1])/(xmid[i]-xmid[i-1]);
    }
    
    /* Find the flux: ( rho*(1-delta*b) * u1 + rho*delta2*b * u2 ) */
    flux[0] = 0.0;
    flux[1] = 0.0;
    flux[N-1] = 0.0;
    flux[N] = 0.0;   
    for (i=2; i<=N-2; i++) {
        u1max = max(u1[i],0.0);
        u1min = min(u1[i],0.0);
        
        u2max = max(u2[i],0.0);
        u2min = min(u2[i],0.0);
        
        /* Key step for second order, reconstruct the solution use values 
         at the cell center */
        /* Find rho at the right (East) side of the cell i-1 
         *             (same as the left (West) side of cell i)*/   
        rhodev = (rho[i]-rho[i-2])/(xmid[i]-xmid[i-2]); /* cadidate slope */
        /* using minmod limiter if negative density appears in cell */
        if ( (rho[i-1]-rhodev*(xmid[i-1]-x[i-1])<0) || 
             (rho[i-1]+rhodev*(x[i]-xmid[i-1])<0)     )
            rhodev = minmod( theta*(rho[i]-rho[i-1])/(xmid[i]-xmid[i-1]),
                    theta*(rho[i-1]-rho[i-2])/(xmid[i-1]-xmid[i-2]),
                    (rho[i]-rho[i-2])/(xmid[i]-xmid[i-2]) );
        rhoEast = rho[i-1] + rhodev*(x[i]-xmid[i-1]);        
        /* Find rho at the left (West) side of the cell i 
         *             (same as the right (East) side of cell i-1)*/   
        rhodev = (rho[i+1]-rho[i-1])/(xmid[i+1]-xmid[i-1]);
        /* using minmod limiter if negative density appears in cell */        
        if ( (rho[i]-rhodev*(xmid[i]-x[i])<0) || 
             (rho[i]+rhodev*(x[i+1]-xmid[i])<0) ) 
            rhodev = minmod( theta*(rho[i+1]-rho[i])/(xmid[i+1]-xmid[i]),
                    theta*(rho[i]-rho[i-1])/(xmid[i]-xmid[i-1]),
                    (rho[i+1]-rho[i-1])/(xmid[i+1]-xmid[i-1]) );
        rhoWest = rho[i]-rhodev*(xmid[i]-x[i]);                            
        flux[i] = (1 - delta2*b[i])*(u1max*rhoEast+u1min*rhoWest) + (1-GF)*delta2*b[i]*(u2max*rhoEast+u2min*rhoWest);
    }
    
    

    
    /* Estimate the CFL condition (over-estimate by a factor of two, IGNORING second term since higher order)*/
    if (stage==1) {
        newdt = dt;
        for (i=1; i<N; i++) {
            tempu = 4.0*max(u1[i+1],0.0)/(x[i+1]-x[i]);
            if ((1-tempu*newdt)<0) 
                newdt = 1/tempu;   /* Choose the largest possible dt */
            tempu = -4.0*min(u1[i],0.0)/(x[i]-x[i-1]);
            if ((1-tempu*newdt)<0) 
                newdt = 1/tempu;   /* Choose the largest possible dt */     
        }
    }
    
    /* Output the new time  and thenew time step */    
    plhs[1] = mxCreateDoubleScalar(newdt);   
}
