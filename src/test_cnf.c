#include "test_cnf.h"
#include "cnf.h"
#include "adjoint.h"
#include "adam.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- Test 1: Trace via FD agrees with exact trace ---- */
void test_cnf_trace(RNG *r) {
    const int D = 3, H = 8;
    const double atol = 1e-8, rtol = 1e-8;
    const double EPS_FD = 1e-5, TOL = 1e-4;

    CNF cnf;
    int nparams = 0;
    {
        DynNet *tmp = dynnet_create(D);
        dynnet_add_time_concat(tmp);
        dynnet_add_linear(tmp, H);
        dynnet_add_tanh(tmp);
        dynnet_add_linear(tmp, D);
        dynnet_finalize(tmp);
        nparams = tmp->total_params;
        dynnet_free(tmp);
    }
    double *theta = vec_alloc(nparams);
    cnf_init(&cnf, D, H, theta, r);

    double z[3];
    for (int i = 0; i < D; i++) z[i] = rng_normal(r);
    double t = 0.5;

    double *ws  = vec_alloc(cnf.net->total_workspace);
    double *z_p = vec_alloc(D), *f_p = vec_alloc(D), *f_m = vec_alloc(D);

    double tr_fd = 0.0;
    for (int i = 0; i < D; i++) {
        vec_copy(z, z_p, D);
        z_p[i] += EPS_FD;
        dynnet_forward(cnf.net, theta, z_p, t, f_p, ws);
        z_p[i] = z[i] - EPS_FD;
        dynnet_forward(cnf.net, theta, z_p, t, f_m, ws);
        tr_fd += (f_p[i] - f_m[i]) / (2.0 * EPS_FD);
    }

    double *out0 = vec_alloc(D);
    dynnet_forward(cnf.net, theta, z, t, out0, ws);
    double tr_jac = 0.0;
    for (int i = 0; i < D; i++) {
        double *col_p = vec_alloc(D), *col_m = vec_alloc(D);
        vec_copy(z, z_p, D);
        z_p[i] += EPS_FD;
        dynnet_forward(cnf.net, theta, z_p, t, col_p, ws);
        z_p[i] = z[i] - EPS_FD;
        dynnet_forward(cnf.net, theta, z_p, t, col_m, ws);
        tr_jac += (col_p[i] - col_m[i]) / (2.0 * EPS_FD);
        free(col_p); free(col_m);
    }

    double err = fabs(tr_fd - tr_jac);
    printf("CNF trace vs full Jacobian trace: err=%.2e  tr_fd=%.6f  tr_jac=%.6f  %s\n",
           err, tr_fd, tr_jac, err < TOL ? "PASS" : "FAIL");

    double dt = 1e-4;
    CNFSampleResult sr = cnf_sample(&cnf, theta, z, 0.0, dt, atol, rtol);
    double delta_approx = -tr_fd * dt;
    double err2 = fabs(sr.delta_logp - delta_approx);
    printf("CNF delta_logp Euler approx:      err=%.2e  %s\n",
           err2, err2 < 1e-2 ? "PASS" : "FAIL");
    free(sr.z1);

    free(ws); free(z_p); free(f_p); free(f_m); free(out0);
    cnf_free(&cnf);
    free(theta);
}

/* ---- Test 2: cnf_sample followed by cnf_log_prob recovers z0 ---- */
void test_cnf_invertibility(RNG *r) {
    const int D = 2, H = 16;
    const double atol = 1e-8, rtol = 1e-8, TOL = 1e-4;
    const double t0 = 0.0, t1 = 1.0;

    int nparams = 0;
    {
        DynNet *tmp = dynnet_create(D);
        dynnet_add_time_concat(tmp);
        dynnet_add_linear(tmp, H);
        dynnet_add_tanh(tmp);
        dynnet_add_linear(tmp, D);
        dynnet_finalize(tmp);
        nparams = tmp->total_params;
        dynnet_free(tmp);
    }
    double *theta = vec_alloc(nparams);
    CNF cnf;
    cnf_init(&cnf, D, H, theta, r);

    double z0[2];
    for (int i = 0; i < D; i++) z0[i] = rng_normal(r);

    CNFSampleResult sr = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
    CNFLogProbResult lr = cnf_log_prob(&cnf, theta, sr.z1, t0, t1, atol, rtol);

    double err = 0.0;
    for (int i = 0; i < D; i++) {
        double d = lr.z0[i] - z0[i];
        err += d * d;
    }
    err = sqrt(err);
    double logp_err = fabs(sr.delta_logp - lr.delta_logp);

    printf("CNF invertibility: z0_err=%.2e  logp_err=%.2e  nfe_fwd=%d  nfe_bwd=%d  %s\n",
           err, logp_err, sr.nfe, lr.nfe,
           (err < TOL && logp_err < TOL) ? "PASS" : "FAIL");

    free(theta); free(sr.z1); free(lr.z0);
    cnf_free(&cnf);
}

/* ---- Test 3: adjoint gradients via finite differences ---- */
void test_cnf_gradients(RNG *r) {
    const int D = 2, H = 8;
    const double EPS = 1e-5, atol = 1e-7, rtol = 1e-7;
    const double t0 = 0.0, t1 = 0.5;

    int nparams = 0;
    {
        DynNet *tmp = dynnet_create(D);
        dynnet_add_time_concat(tmp);
        dynnet_add_linear(tmp, H);
        dynnet_add_tanh(tmp);
        dynnet_add_linear(tmp, D);
        dynnet_finalize(tmp);
        nparams = tmp->total_params;
        dynnet_free(tmp);
    }
    double *theta = vec_alloc(nparams);
    CNF cnf;
    cnf_init(&cnf, D, H, theta, r);

    double z0[2];
    for (int i = 0; i < D; i++) z0[i] = rng_normal(r);

    CNFSampleResult sr = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
    double *z1 = sr.z1;

    double dL_dlogp  = -1.0;
    double *dL_dz1   = vec_alloc(D);
    for (int i = 0; i < D; i++) dL_dz1[i] = z1[i];

    CNFBackwardResult br = cnf_backward(&cnf, theta, z1, dL_dlogp, dL_dz1,
                                         t0, t1, atol, rtol);

    double *num_grad = vec_alloc(nparams);
    for (int k = 0; k < nparams; k++) {
        double tk = theta[k];

        theta[k] = tk + EPS;
        CNFSampleResult sp = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
        double lp = 0.0;
        for (int i = 0; i < D; i++) lp += 0.5 * sp.z1[i] * sp.z1[i];
        lp -= sp.delta_logp;
        free(sp.z1);

        theta[k] = tk - EPS;
        CNFSampleResult sm = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
        double lm = 0.0;
        for (int i = 0; i < D; i++) lm += 0.5 * sm.z1[i] * sm.z1[i];
        lm -= sm.delta_logp;
        free(sm.z1);

        theta[k] = tk;
        num_grad[k] = (lp - lm) / (2.0 * EPS);
    }

    double max_num = 0.0, max_err = 0.0;
    for (int k = 0; k < nparams; k++) {
        if (fabs(num_grad[k]) > max_num) max_num = fabs(num_grad[k]);
        double e = fabs(br.dL_dtheta[k] - num_grad[k]);
        if (e > max_err) max_err = e;
    }
    double rel = max_err / (max_num + 1e-8);
    printf("CNF adjoint dL/dtheta: max_rel_err=%.2e  nfe_bwd=%d  %s\n",
           rel, br.nfe, rel < 1e-2 ? "PASS" : "FAIL");

    free(theta); free(z1); free(dL_dz1);
    free(br.dL_dz0); free(br.dL_dtheta);
    free(num_grad);
    cnf_free(&cnf);
}

/* ---- Test 4: training loss decreases on a simple target ---- */
void test_cnf_training(RNG *r) {
    const int D = 2, H = 16;
    const int ITERS = 30;
    const double t0 = 0.0, t1 = 1.0;
    const double atol = 1e-4, rtol = 1e-4;

    int nparams = 0;
    {
        DynNet *tmp = dynnet_create(D);
        dynnet_add_time_concat(tmp);
        dynnet_add_linear(tmp, H);
        dynnet_add_tanh(tmp);
        dynnet_add_linear(tmp, D);
        dynnet_finalize(tmp);
        nparams = tmp->total_params;
        dynnet_free(tmp);
    }
    double *theta = vec_alloc(nparams);
    CNF cnf;
    cnf_init(&cnf, D, H, theta, r);
    Adam adam = adam_init(nparams, 1e-3, 0.9, 0.999, 1e-8);

    const double mu[2][2] = {{-1.5, 0.0}, {1.5, 0.0}};
    const double sigma2   = 0.25;

    double first_loss = 0.0, last_loss = 0.0;

    for (int iter = 0; iter < ITERS; iter++) {
        const int BATCH = 8;
        double *dL_dtheta_acc = vec_zeros(nparams);
        double loss = 0.0;

        for (int b = 0; b < BATCH; b++) {
            double z0[2];
            for (int i = 0; i < D; i++) z0[i] = rng_normal(r);

            CNFSampleResult sr = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
            double *z1 = sr.z1;

            double lc[2];
            for (int k = 0; k < 2; k++) {
                double dx = z1[0] - mu[k][0], dy = z1[1] - mu[k][1];
                lc[k] = -0.5 * (dx * dx + dy * dy) / sigma2;
            }
            double mx = lc[0] > lc[1] ? lc[0] : lc[1];
            double log_p_target = mx + log(0.5 * (exp(lc[0] - mx) + exp(lc[1] - mx)));

            double log_p_base = -0.5 * (z0[0]*z0[0] + z0[1]*z0[1])
                                - (double)D * 0.5 * log(2.0 * M_PI);

            loss += -log_p_target + log_p_base + sr.delta_logp;

            double dL_dz1[2];
            double w0 = exp(lc[0] - mx), w1 = exp(lc[1] - mx);
            double wsum = w0 + w1;
            dL_dz1[0] = -(-w0 * (z1[0] - mu[0][0]) / sigma2
                          - w1 * (z1[0] - mu[1][0]) / sigma2) / wsum;
            dL_dz1[1] = -(-w0 * (z1[1] - mu[0][1]) / sigma2
                          - w1 * (z1[1] - mu[1][1]) / sigma2) / wsum;

            CNFBackwardResult br = cnf_backward(&cnf, theta, z1,
                                                 1.0, dL_dz1,
                                                 t0, t1, atol, rtol);
            vec_add_scaled(dL_dtheta_acc, 1.0 / BATCH, br.dL_dtheta, nparams);
            free(br.dL_dz0); free(br.dL_dtheta);
            free(z1);
        }

        if (iter == 0)         first_loss = loss / BATCH;
        if (iter == ITERS - 1) last_loss  = loss / BATCH;

        adam_update(&adam, theta, dL_dtheta_acc);
        free(dL_dtheta_acc);
    }

    printf("CNF training loss: first=%.4f  last=%.4f  %s\n",
           first_loss, last_loss, last_loss < first_loss ? "PASS" : "FAIL");

    adam_free(&adam);
    cnf_free(&cnf);
    free(theta);
}
