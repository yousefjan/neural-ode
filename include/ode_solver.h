#pragma once

typedef void (*ode_rhs_fn)(const double *state, double t, const double *params,
                           int dim, double *out, void *ctx);

typedef struct {
    double *y;
    int nfe;
} ODEResult;

ODEResult ode_solve(ode_rhs_fn f, const double *y0, double t0, double t1,
                    const double *params, int dim, double atol, double rtol,
                    void *ctx);

ODEResult ode_solve_times(ode_rhs_fn f, const double *y0, const double *times,
                          int ntimes, const double *params, int dim,
                          double atol, double rtol, void *ctx);
