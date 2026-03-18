#pragma once

#include "utils.h"
#include "dynmlp.h"
#include "ode_solver.h"

typedef struct {
    DynMLP net;
    const double *theta;
    int state_dim;
    int nparams;
    Workspace *ws;
} AdjointCtx;

typedef struct {
    double *z1;
    double *dL_dz0;
    double *dL_dtheta;
    int nfe_forward;
    int nfe_backward;
} NeuralODEOutput;

typedef struct {
    double *z_traj;
    double *dL_dz0;
    double *dL_dtheta;
    int nfe_forward;
    int nfe_backward;
} MultiObsNeuralODEOutput;

void neural_ode_rhs(const double *state, double t, const double *params,
                    int dim, double *out, void *ctx);

NeuralODEOutput neural_ode_forward_backward(
    const DynMLP *net, const double *theta,
    const double *z0, double t0, double t1,
    const double *target, double atol, double rtol,
    int num_checkpoints);

MultiObsNeuralODEOutput neural_ode_forward_backward_multi(
    const DynMLP *net, const double *theta,
    const double *z0, const double *times,
    const double *targets, int ntimes,
    double atol, double rtol);
