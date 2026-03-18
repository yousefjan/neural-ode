#pragma once

#include "dynnet.h"

typedef struct {
    double **z0;
    double **target;
    int num_samples;
} Dataset;

Dataset generate_spiral_dataset(int num_samples, double t0, double t1,
                                double noise_std, RNG *r);
void    dataset_free(Dataset *ds);
double  evaluate(DynNet *net, const double *theta,
                 const Dataset *ds, double t0, double t1,
                 double atol, double rtol);
