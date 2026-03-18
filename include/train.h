#pragma once

#include "dynnet.h"
#include "adam.h"

typedef struct {
    double loss;
    int nfe_fwd;
    int nfe_bwd;
} TrainStepResult;

TrainStepResult train_step(DynNet *net, double *theta,
                           const double **z0s, const double **targets,
                           double t0, double t1, int batch_size,
                           Adam *adam, double atol, double rtol,
                           int num_checkpoints);
