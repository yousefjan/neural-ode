#pragma once

#include "dynmlp.h"
#include "ode_solver.h"
#include "utils.h"

typedef struct {
    DynMLP net;
    int nparams;
    double trace_eps; /* finite-difference epsilon for trace computation */
} CNF;

typedef struct {
    double *z1;        
    double delta_logp;
    int nfe;
} CNFSampleResult;

typedef struct {
  double *z0;
  double delta_logp; 
    int nfe;
} CNFLogProbResult;

typedef struct {
    double *dL_dz0;      
    double *dL_dtheta;   
    int nfe;
} CNFBackwardResult;

void cnf_init(CNF *cnf, int D, int H, double *theta, RNG *r);

CNFSampleResult cnf_sample(const CNF *cnf, const double *theta,
                            const double *z0, double t0, double t1,
                            double atol, double rtol);

CNFLogProbResult cnf_log_prob(const CNF *cnf, const double *theta,
                               const double *z1, double t0, double t1,
                               double atol, double rtol);

CNFBackwardResult cnf_backward(const CNF *cnf, const double *theta,
                                const double *z1,
                                double dL_dlogp, const double *dL_dz1_in,
                                double t0, double t1,
                                double atol, double rtol);
