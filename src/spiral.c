#include "spiral.h"
#include "adjoint.h"
#include "ode_solver.h"

#include <math.h>
#include <stdlib.h>

static void spiral_rhs(const double *state, double t, const double *params,
                       int dim, double *out, void *ctx) {
    (void)t; (void)dim; (void)ctx;
    double alpha = params[0];
    out[0] = alpha * state[1];
    out[1] = -alpha * state[0];
}

Dataset generate_spiral_dataset(int num_samples, double t0, double t1,
                                double noise_std, RNG *r) {
    Dataset ds;
    ds.num_samples = num_samples;
    ds.z0     = (double **)xmalloc(num_samples * sizeof(double *));
    ds.target = (double **)xmalloc(num_samples * sizeof(double *));

    for (int i = 0; i < num_samples; i++) {
        double alpha = 1.0;
        double angle  = 2.0 * M_PI * rng_uniform(r);
        double radius = 0.5 + 1.0 * rng_uniform(r);

        ds.z0[i]     = vec_alloc(2);
        ds.target[i] = vec_alloc(2);

        ds.z0[i][0] = radius * cos(angle);
        ds.z0[i][1] = radius * sin(angle);

        ODEResult res = ode_solve(spiral_rhs, ds.z0[i], t0, t1,
                                  &alpha, 2, 1e-8, 1e-8, NULL);
        vec_copy(res.y, ds.target[i], 2);
        free(res.y);

        /* add noise to the initial observation */
        ds.z0[i][0] += noise_std * rng_normal(r);
        ds.z0[i][1] += noise_std * rng_normal(r);
    }
    return ds;
}

void dataset_free(Dataset *ds) {
    for (int i = 0; i < ds->num_samples; i++) {
        free(ds->z0[i]);
        free(ds->target[i]);
    }
    free(ds->z0);
    free(ds->target);
}

double evaluate(DynNet *net, const double *theta,
                const Dataset *ds, double t0, double t1,
                double atol, double rtol) {
    int D = net->D;
    double *ws    = vec_alloc(net->total_workspace);
    double *neg_a = vec_alloc(D);
    AdjointCtx ac = { net, theta, ws, neg_a };
    double total_loss = 0.0;
    for (int i = 0; i < ds->num_samples; i++) {
        ODEResult fwd = ode_solve(neural_ode_rhs, ds->z0[i], t0, t1,
                                  NULL, D, atol, rtol, &ac);
        for (int j = 0; j < D; j++) {
            double d = fwd.y[j] - ds->target[i][j];
            total_loss += 0.5 * d * d;
        }
        free(fwd.y);
    }
    free(ws);
    free(neg_a);
    return total_loss / (double)ds->num_samples;
}
