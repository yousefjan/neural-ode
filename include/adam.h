#pragma once

typedef struct {
    double *m;
    double *v;
    int nparams;
    double lr;
    double beta1;
    double beta2;
    double eps;
    int t;
} Adam;

Adam adam_init(int nparams, double lr, double beta1, double beta2, double eps);
void adam_update(Adam *a, double *theta, const double *grad);
void adam_free(Adam *a);
