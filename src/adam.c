#include "adam.h"
#include "utils.h"

#include <math.h>
#include <stdlib.h>

Adam adam_init(int nparams, double lr, double beta1, double beta2, double eps) {
    Adam a;
    a.m = vec_zeros(nparams);
    a.v = vec_zeros(nparams);
    a.nparams = nparams;
    a.lr = lr;
    a.beta1 = beta1;
    a.beta2 = beta2;
    a.eps = eps;
    a.t = 0;
    return a;
}

void adam_update(Adam *a, double *theta, const double *grad) {
    a->t++;
    double bc1 = 1.0 - pow(a->beta1, (double)a->t);
    double bc2 = 1.0 - pow(a->beta2, (double)a->t);
    double alpha = a->lr * sqrt(bc2) / bc1;
    for (int i = 0; i < a->nparams; i++) {
        a->m[i] = a->beta1 * a->m[i] + (1.0 - a->beta1) * grad[i];
        a->v[i] = a->beta2 * a->v[i] + (1.0 - a->beta2) * grad[i] * grad[i];
        theta[i] -= alpha * a->m[i] / (sqrt(a->v[i]) + a->eps);
    }
}

void adam_free(Adam *a) {
    free(a->m);
    free(a->v);
}
