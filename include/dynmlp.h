#pragma once

#include "utils.h"

typedef struct {
    int D;
    int H;
    int nparams;
} DynMLP;

#define DYNMLP_W1(D, H)  (0)
#define DYNMLP_b1(D, H)  ((D + 1) * (H))
#define DYNMLP_W2(D, H)  ((D + 1) * (H) + (H))
#define DYNMLP_b2(D, H)  ((D + 1) * (H) + (H) + (H) * (D))

int  dynmlp_nparams(int D, int H);
void dynmlp_init(DynMLP *net, int D, int H, double *theta, RNG *r);
void dynmlp_forward(const DynMLP *net, const double *theta,
                    const double *z, double t, double *out,
                    Workspace *ws);
void dynmlp_vjp(const DynMLP *net, const double *theta,
                const double *z, double t, const double *v,
                double *vjp_z, double *vjp_theta,
                Workspace *ws);
