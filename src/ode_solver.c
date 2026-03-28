#include "ode_solver.h"
#include "utils.h"

#include <math.h>
#include <stdlib.h>

static const double dp_c[7]  = { 0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0 };
static const double dp_a2[1] = { 1.0/5.0 };
static const double dp_a3[2] = { 3.0/40.0, 9.0/40.0 };
static const double dp_a4[3] = { 44.0/45.0, -56.0/15.0, 32.0/9.0 };
static const double dp_a5[4] = { 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0 };
static const double dp_a6[5] = { 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0 };
static const double dp_b[7]  = { 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0 };
static const double dp_e[7]  = {
     35.0/384.0   - 5179.0/57600.0,
     0.0,
     500.0/1113.0 - 7571.0/16695.0,
     125.0/192.0  - 393.0/640.0,
     -2187.0/6784.0 + 92097.0/339200.0,
     11.0/84.0    - 187.0/2100.0,
     -1.0/40.0
};

ODEResult ode_solve(ode_rhs_fn f, const double *y0, double t0, double t1,
                    const double *params, int dim, double atol, double rtol,
                    void *ctx) {
    double *buf = vec_alloc(11 * dim);
    double *k[7];
    for (int i = 0; i < 7; i++) k[i] = buf + i * dim;
    double *y   = buf + 7 * dim;
    double *y5  = buf + 8 * dim;
    double *err = buf + 9 * dim;
    double *stg = buf + 10 * dim;

    ODEResult res = { vec_alloc(dim), 0 };
    vec_copy(y0, y, dim);

    double t = t0;
    double h = 0.01 * (t1 - t0);
    int k1_fresh = 0;

    for (int step = 0; step < 1000000; step++) {
        if (t1 > t0) {
            if (t >= t1) break;
            if (t + h > t1) h = t1 - t;
        } else {
            if (t <= t1) break;
            if (t + h < t1) h = t1 - t;
        }

        if (!k1_fresh) { f(y, t, params, dim, k[0], ctx); res.nfe++; k1_fresh = 1; }

        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * dp_a2[0]*k[0][i];
        f(stg, t + dp_c[1]*h, params, dim, k[1], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a3[0]*k[0][i] + dp_a3[1]*k[1][i]);
        f(stg, t + dp_c[2]*h, params, dim, k[2], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a4[0]*k[0][i] + dp_a4[1]*k[1][i] + dp_a4[2]*k[2][i]);
        f(stg, t + dp_c[3]*h, params, dim, k[3], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a5[0]*k[0][i] + dp_a5[1]*k[1][i]
                                + dp_a5[2]*k[2][i] + dp_a5[3]*k[3][i]);
        f(stg, t + dp_c[4]*h, params, dim, k[4], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a6[0]*k[0][i] + dp_a6[1]*k[1][i]
                                + dp_a6[2]*k[2][i] + dp_a6[3]*k[3][i] + dp_a6[4]*k[4][i]);
        f(stg, t + dp_c[5]*h, params, dim, k[5], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            y5[i] = y[i] + h * (dp_b[0]*k[0][i] + dp_b[2]*k[2][i]
                               + dp_b[3]*k[3][i] + dp_b[4]*k[4][i] + dp_b[5]*k[5][i]);
        f(y5, t + h, params, dim, k[6], ctx); res.nfe++;

        for (int i = 0; i < dim; i++)
            err[i] = h * (dp_e[0]*k[0][i] + dp_e[2]*k[2][i] + dp_e[3]*k[3][i]
                        + dp_e[4]*k[4][i] + dp_e[5]*k[5][i] + dp_e[6]*k[6][i]);

        double err_sq = 0.0;
        for (int i = 0; i < dim; i++) {
            double sc = atol + rtol * fmax(fabs(y[i]), fabs(y5[i]));
            double e  = err[i] / sc;
            err_sq   += e * e;
        }
        double err_norm = sqrt(err_sq / (double)dim);

        double factor;
        if (err_norm == 0.0) {
            factor = 5.0;
        } else {
            factor = 0.9 * pow(err_norm, -0.2);
            if (factor < 0.2) factor = 0.2;
            if (factor > 5.0) factor = 5.0;
        }

        if (err_norm <= 1.0) {
            vec_copy(y5, y, dim);
            t += h;
            double *tmp = k[0]; k[0] = k[6]; k[6] = tmp;
            h *= factor;
        } else {
            if (factor > 1.0) factor = 1.0;
            h *= factor;
        }
    }

    vec_copy(y, res.y, dim);
    free(buf);

    return res;
}

ODEResult ode_solve_times(ode_rhs_fn f, const double *y0, const double *times,
                          int ntimes, const double *params, int dim,
                          double atol, double rtol, void *ctx) {
    ODEResult res = { vec_alloc(dim * ntimes), 0 };
    vec_copy(y0, res.y, dim);
    for (int i = 1; i < ntimes; i++) {
        ODEResult seg = ode_solve(f, res.y + (i-1)*dim, times[i-1], times[i],
                                  params, dim, atol, rtol, ctx);
        vec_copy(seg.y, res.y + i * dim, dim);
        res.nfe += seg.nfe;

        free(seg.y);
    }
    return res;
}
