#include "tests.h"
#include "dynmlp.h"
#include "ode_solver.h"
#include "adjoint.h"
#include "adam.h"
#include "train.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void rhs_decay(const double *y, double t, const double *p, int d, double *out, void *ctx) {
    (void)t; (void)p; (void)d; (void)ctx;
    out[0] = -y[0];
}

static void rhs_rotation(const double *y, double t, const double *p, int d, double *out, void *ctx) {
    (void)t; (void)p; (void)d; (void)ctx;
    out[0] = -y[1]; out[1] = y[0];
}

void test_ode_solver(void) {
    const double atol = 1e-8, rtol = 1e-8, tol = 1e-6;

    { double y0 = 1.0;
      ODEResult r = ode_solve(rhs_decay, &y0, 0.0, 1.0, NULL, 1, atol, rtol, NULL);
      double err = fabs(r.y[0] - exp(-1.0));
      printf("ODE test 1 (decay):    err=%.2e  nfe=%d  %s\n", err, r.nfe, err < tol ? "PASS" : "FAIL");
      free(r.y); }

    { double y0[2] = {1.0, 0.0};
      ODEResult r = ode_solve(rhs_rotation, y0, 0.0, 2.0 * M_PI, NULL, 2, atol, rtol, NULL);
      double err = sqrt((r.y[0]-1.0)*(r.y[0]-1.0) + r.y[1]*r.y[1]);
      printf("ODE test 2 (rotation): err=%.2e  nfe=%d  %s\n", err, r.nfe, err < tol ? "PASS" : "FAIL");
      free(r.y); }

    { double y0 = exp(-1.0);
      ODEResult r = ode_solve(rhs_decay, &y0, 1.0, 0.0, NULL, 1, atol, rtol, NULL);
      double err = fabs(r.y[0] - 1.0);
      printf("ODE test 3 (backward): err=%.2e  nfe=%d  %s\n", err, r.nfe, err < tol ? "PASS" : "FAIL");
      free(r.y); }
}

void test_dynmlp_gradients(RNG *r) {
    const int D = 3, H = 8;
    const double EPS = 1e-7, TOL = 1e-5;

    int np = dynmlp_nparams(D, H);
    double *theta = vec_alloc(np);
    double *z = vec_alloc(D);
    double *v = vec_alloc(D);
    double *out_p = vec_alloc(D);
    double *out_m = vec_alloc(D);

    DynMLP net;
    dynmlp_init(&net, D, H, theta, r);
    for (int i = 0; i < D; i++) z[i] = rng_normal(r);
    for (int i = 0; i < D; i++) v[i] = rng_normal(r);
    double t = rng_normal(r);

    Workspace ws = workspace_alloc(D, H, np);

    double *vjp_z     = vec_zeros(D);
    double *vjp_theta = vec_zeros(np);
    dynmlp_vjp(&net, theta, z, t, v, vjp_z, vjp_theta, &ws);

    double *num_vjp_z = vec_alloc(D);
    for (int i = 0; i < D; i++) {
        double zi = z[i];
        z[i] = zi + EPS; dynmlp_forward(&net, theta, z, t, out_p, &ws);
        z[i] = zi - EPS; dynmlp_forward(&net, theta, z, t, out_m, &ws);
        z[i] = zi;
        num_vjp_z[i] = (vec_dot(v, out_p, D) - vec_dot(v, out_m, D)) / (2.0 * EPS);
    }
    double max_err_z = 0.0;
    for (int i = 0; i < D; i++) {
        double e = fabs(vjp_z[i] - num_vjp_z[i]);
        if (e > max_err_z) max_err_z = e;
    }

    double *num_vjp_theta = vec_alloc(np);
    for (int k = 0; k < np; k++) {
        double tk = theta[k];
        theta[k] = tk + EPS; dynmlp_forward(&net, theta, z, t, out_p, &ws);
        theta[k] = tk - EPS; dynmlp_forward(&net, theta, z, t, out_m, &ws);
        theta[k] = tk;
        num_vjp_theta[k] = (vec_dot(v, out_p, D) - vec_dot(v, out_m, D)) / (2.0 * EPS);
    }
    double max_err_theta = 0.0;
    for (int k = 0; k < np; k++) {
        double e = fabs(vjp_theta[k] - num_vjp_theta[k]);
        if (e > max_err_theta) max_err_theta = e;
    }

    printf("dL/dz:     max_err=%.2e  %s\n", max_err_z, max_err_z < TOL ? "PASS" : "FAIL");
    printf("dL/dtheta: max_err=%.2e  %s\n", max_err_theta, max_err_theta < TOL ? "PASS" : "FAIL");

    workspace_free(&ws);
    free(theta); free(z); free(v); free(out_p); free(out_m);
    free(vjp_z); free(vjp_theta); free(num_vjp_z); free(num_vjp_theta);
}

void test_adjoint_gradients(RNG *r) {
    const int D = 2, H = 8;
    const double EPS = 1e-5, atol = 1e-7, rtol = 1e-7;
    const double t0 = 0.0, t1 = 1.0;

    int np = dynmlp_nparams(D, H);
    double *theta  = vec_alloc(np);
    double *z0 = vec_alloc(D);
    double *target = vec_alloc(D);

    DynMLP net;
    dynmlp_init(&net, D, H, theta, r);
    for (int i = 0; i < D; i++) z0[i] = rng_normal(r);
    for (int i = 0; i < D; i++) target[i] = rng_normal(r);

    NeuralODEOutput out = neural_ode_forward_backward(&net, theta, z0, t0, t1,
                                                      target, atol, rtol, 10);

    Workspace ws = workspace_alloc(D, H, np);

#define FWD_LOSS(z0_, theta_) ({ \
    AdjointCtx ac_ = { net, (theta_), D, np, &ws }; \
    ODEResult r_ = ode_solve(neural_ode_rhs, (z0_), t0, t1, NULL, D, atol, rtol, &ac_); \
    double l_ = 0.0; \
    for (int _i = 0; _i < D; _i++) { double _d = r_.y[_i] - target[_i]; l_ += 0.5*_d*_d; } \
    free(r_.y); l_; \
})

    double *num_dL_dtheta = vec_alloc(np);
    for (int k = 0; k < np; k++) {
        double tk = theta[k];
        theta[k] = tk + EPS; double lp = FWD_LOSS(z0, theta);
        theta[k] = tk - EPS; double lm = FWD_LOSS(z0, theta);
        theta[k] = tk;
        num_dL_dtheta[k] = (lp - lm) / (2.0 * EPS);
    }
    double max_num_theta = 0.0;
    for (int k = 0; k < np; k++)
        if (fabs(num_dL_dtheta[k]) > max_num_theta) max_num_theta = fabs(num_dL_dtheta[k]);
    double max_err_theta = 0.0;
    for (int k = 0; k < np; k++) {
        double e = fabs(out.dL_dtheta[k] - num_dL_dtheta[k]);
        if (e > max_err_theta) max_err_theta = e;
    }
    double rel_theta = max_err_theta / (max_num_theta + 1e-8);
    printf("adjoint dL/dtheta: max_rel_err=%.2e  nfe_fwd=%d  nfe_bwd=%d  %s\n",
           rel_theta, out.nfe_forward, out.nfe_backward, rel_theta < 1e-3 ? "PASS" : "FAIL");

    double *num_dL_dz0 = vec_alloc(D);
    for (int i = 0; i < D; i++) {
        double zi = z0[i];
        z0[i] = zi + EPS; double lp = FWD_LOSS(z0, theta);
        z0[i] = zi - EPS; double lm = FWD_LOSS(z0, theta);
        z0[i] = zi;
        num_dL_dz0[i] = (lp - lm) / (2.0 * EPS);
    }
    double max_num_z0 = 0.0;
    for (int i = 0; i < D; i++)
        if (fabs(num_dL_dz0[i]) > max_num_z0) max_num_z0 = fabs(num_dL_dz0[i]);
    double max_err_z0 = 0.0;
    for (int i = 0; i < D; i++) {
        double e = fabs(out.dL_dz0[i] - num_dL_dz0[i]);
        if (e > max_err_z0) max_err_z0 = e;
    }
    double rel_z0 = max_err_z0 / (max_num_z0 + 1e-8);
    printf("adjoint dL/dz0:    max_rel_err=%.2e  %s\n", rel_z0, rel_z0 < 1e-3 ? "PASS" : "FAIL");

#undef FWD_LOSS

    workspace_free(&ws);
    free(out.z1); free(out.dL_dz0); free(out.dL_dtheta);
    free(num_dL_dtheta); free(num_dL_dz0);
    free(theta); free(z0); free(target);
}

void test_multi_obs_adjoint(RNG *r) {
    const int D = 2, H = 8;
    const double EPS = 1e-5, atol = 1e-7, rtol = 1e-7;
    const int ntimes = 5;

    double times[5] = { 0.0, 0.5, 1.0, 1.5, 2.0 };

    int np = dynmlp_nparams(D, H);
    double *theta = vec_alloc(np);
    double *z0 = vec_alloc(D);
    double *targets = vec_alloc(ntimes * D);

    DynMLP net;
    dynmlp_init(&net, D, H, theta, r);
    for (int i = 0; i < D; i++) z0[i] = rng_normal(r);
    for (int i = 0; i < ntimes * D; i++) targets[i] = rng_normal(r);

    MultiObsNeuralODEOutput out = neural_ode_forward_backward_multi(
        &net, theta, z0, times, targets, ntimes, atol, rtol);

    Workspace ws = workspace_alloc(D, H, np);
    AdjointCtx ac = { net, theta, D, np, &ws };

    /* Numerical dL/dtheta */
    double *num_dL_dtheta = vec_alloc(np);
    for (int k = 0; k < np; k++) {
        double tk = theta[k];

        theta[k] = tk + EPS;
        ODEResult rp = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                       NULL, D, atol, rtol, &ac);
        double lp = 0.0;
        for (int i = 0; i < ntimes * D; i++) {
            double d = rp.y[i] - targets[i]; lp += 0.5 * d * d;
        }
        free(rp.y);

        theta[k] = tk - EPS;
        ODEResult rm = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                       NULL, D, atol, rtol, &ac);
        double lm = 0.0;
        for (int i = 0; i < ntimes * D; i++) {
            double d = rm.y[i] - targets[i]; lm += 0.5 * d * d;
        }
        free(rm.y);

        theta[k] = tk;
        num_dL_dtheta[k] = (lp - lm) / (2.0 * EPS);
    }

    double max_num_theta = 0.0;
    for (int k = 0; k < np; k++)
        if (fabs(num_dL_dtheta[k]) > max_num_theta) max_num_theta = fabs(num_dL_dtheta[k]);
    double max_err_theta = 0.0;
    for (int k = 0; k < np; k++) {
        double e = fabs(out.dL_dtheta[k] - num_dL_dtheta[k]);
        if (e > max_err_theta) max_err_theta = e;
    }
    double rel_theta = max_err_theta / (max_num_theta + 1e-8);
    printf("multi-obs adjoint dL/dtheta: max_rel_err=%.2e  nfe_fwd=%d  nfe_bwd=%d  %s\n",
           rel_theta, out.nfe_forward, out.nfe_backward, rel_theta < 1e-3 ? "PASS" : "FAIL");

    /* Numerical dL/dz0 */
    double *num_dL_dz0 = vec_alloc(D);
    for (int i = 0; i < D; i++) {
        double zi = z0[i];

        z0[i] = zi + EPS;
        ODEResult rp = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                       NULL, D, atol, rtol, &ac);
        double lp = 0.0;
        for (int j = 0; j < ntimes * D; j++) {
            double d = rp.y[j] - targets[j]; lp += 0.5 * d * d;
        }
        free(rp.y);

        z0[i] = zi - EPS;
        ODEResult rm = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                       NULL, D, atol, rtol, &ac);
        double lm = 0.0;
        for (int j = 0; j < ntimes * D; j++) {
            double d = rm.y[j] - targets[j]; lm += 0.5 * d * d;
        }
        free(rm.y);

        z0[i] = zi;
        num_dL_dz0[i] = (lp - lm) / (2.0 * EPS);
    }

    double max_num_z0 = 0.0;
    for (int i = 0; i < D; i++)
        if (fabs(num_dL_dz0[i]) > max_num_z0) max_num_z0 = fabs(num_dL_dz0[i]);
    double max_err_z0 = 0.0;
    for (int i = 0; i < D; i++) {
        double e = fabs(out.dL_dz0[i] - num_dL_dz0[i]);
        if (e > max_err_z0) max_err_z0 = e;
    }
    double rel_z0 = max_err_z0 / (max_num_z0 + 1e-8);
    printf("multi-obs adjoint dL/dz0:    max_rel_err=%.2e  %s\n",
           rel_z0, rel_z0 < 1e-3 ? "PASS" : "FAIL");

    workspace_free(&ws);
    free(out.z_traj); free(out.dL_dz0); free(out.dL_dtheta);
    free(num_dL_dtheta); free(num_dL_dz0);
    free(theta); free(z0); free(targets);
}

void test_training(RNG *r) {
    const int D = 2, H = 16;
    const int N = 50, BATCH = 10, ITERS = 300;
    const double t0 = 0.0, t1 = 1.0;
    const double atol = 1e-4, rtol = 1e-4;

    DynMLP net;
    int nparams = dynmlp_nparams(D, H);
    double *theta = vec_alloc(nparams);
    dynmlp_init(&net, D, H, theta, r);
    Adam adam = adam_init(nparams, 1e-3, 0.9, 0.999, 1e-8);

    double **z0s = (double **)xmalloc(N * sizeof(double *));
    double **targets = (double **)xmalloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        double angle = 2.0 * M_PI * rng_uniform(r);
        z0s[i] = vec_alloc(D);
        targets[i] = vec_alloc(D);
        z0s[i][0] = cos(angle);
        z0s[i][1] = sin(angle);
        targets[i][0] = -z0s[i][1];
        targets[i][1] =  z0s[i][0];
    }

    const double **batch_z0  = (const double **)xmalloc(BATCH * sizeof(double *));
    const double **batch_tgt = (const double **)xmalloc(BATCH * sizeof(double *));

    printf("\n--- Training test (D=2, H=16, 90-deg rotation) ---\n");
    for (int iter = 0; iter < ITERS; iter++) {
        for (int b = 0; b < BATCH; b++) {
            int idx = (int)(rng_next(r) % (uint64_t)N);
            batch_z0[b]  = z0s[idx];
            batch_tgt[b] = targets[idx];
        }
        TrainStepResult res = train_step(&net, theta, batch_z0, batch_tgt,
                                         t0, t1, BATCH, &adam, atol, rtol, 10);
        if ((iter + 1) % 50 == 0)
            printf("iter %3d  loss=%.4f  nfe_fwd=%d\n", iter + 1, res.loss, res.nfe_fwd);
    }

    Workspace ws = workspace_alloc(D, H, nparams);
    AdjointCtx ac = { net, theta, D, nparams, &ws };
    double final_loss = 0.0;
    for (int i = 0; i < N; i++) {
        ODEResult fwd = ode_solve(neural_ode_rhs, z0s[i], t0, t1, NULL, D, atol, rtol, &ac);
        for (int j = 0; j < D; j++) {
            double d = fwd.y[j] - targets[i][j];
            final_loss += 0.5 * d * d;
        }
        free(fwd.y);
    }
    final_loss /= (double)N;
    printf("Loss: %.4f\n", final_loss);

    workspace_free(&ws);
    adam_free(&adam);
    free(batch_z0); free(batch_tgt);
    for (int i = 0; i < N; i++) { free(z0s[i]); free(targets[i]); }
    free(z0s); free(targets);
    free(theta);
}
