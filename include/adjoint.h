#pragma once

#include "dynnet.h"
#include "ode_solver.h"

typedef struct {
    DynNet *net;
    const double *theta;
    double *ws;    /* size net->total_workspace */
    double *neg_a; /* size net->D */
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
    DynNet *net, const double *theta,
    const double *z0, double t0, double t1,
    const double *target, double atol, double rtol,
    int num_checkpoints);

MultiObsNeuralODEOutput neural_ode_forward_backward_multi(
    DynNet *net, const double *theta,
    const double *z0, const double *times,
    const double *targets, int ntimes,
    double atol, double rtol);
