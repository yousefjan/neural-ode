#include "cnf_train.h"
#include "cnf.h"
#include "adam.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const double TARGET_MU[4][2] = {
    { 1.5,  0.0}, {-1.5,  0.0},
    { 0.0,  1.5}, { 0.0, -1.5}
};
static const double TARGET_SIGMA2 = 0.2;
static const int    TARGET_K      = 4;

static double log_p_target(const double *z) {
    double lc[4], mx = -1e300;
    for (int k = 0; k < TARGET_K; k++) {
        double dx = z[0] - TARGET_MU[k][0], dy = z[1] - TARGET_MU[k][1];
        lc[k] = -0.5 * (dx * dx + dy * dy) / TARGET_SIGMA2;
        if (lc[k] > mx) mx = lc[k];
    }
    double s = 0.0;
    for (int k = 0; k < TARGET_K; k++) s += exp(lc[k] - mx);
    return mx + log(s / (double)TARGET_K)
             - log(2.0 * M_PI * TARGET_SIGMA2);
}

static void grad_log_p_target(const double *z, double *g) {
    double lc[4], mx = -1e300;
    for (int k = 0; k < TARGET_K; k++) {
        double dx = z[0] - TARGET_MU[k][0], dy = z[1] - TARGET_MU[k][1];
        lc[k] = -0.5 * (dx * dx + dy * dy) / TARGET_SIGMA2;
        if (lc[k] > mx) mx = lc[k];
    }
    double w[4], wsum = 0.0;
    for (int k = 0; k < TARGET_K; k++) { w[k] = exp(lc[k] - mx); wsum += w[k]; }
    g[0] = g[1] = 0.0;
    for (int k = 0; k < TARGET_K; k++) {
        double wk = w[k] / wsum;
        g[0] += wk * (-(z[0] - TARGET_MU[k][0]) / TARGET_SIGMA2);
        g[1] += wk * (-(z[1] - TARGET_MU[k][1]) / TARGET_SIGMA2);
    }
}

/* ---- Training loop ---- */
void cnf_train_demo(RNG *r) {
    const int D = 2, H = 32;
    const int ITERS = 200, BATCH = 16, LOG_EVERY = 20;
    const double t0 = 0.0, t1 = 1.0;
    const double atol = 1e-4, rtol = 1e-4;
    const double LR = 1e-3;

    int nparams = dynmlp_nparams(D, H);
    double *theta = vec_alloc(nparams);
    CNF cnf;
    cnf_init(&cnf, D, H, theta, r);
    Adam adam = adam_init(nparams, LR, 0.9, 0.999, 1e-8);

    printf("\n--- CNF density matching (4-Gaussian target, D=2, H=%d) ---\n\n", H);
    printf("%-6s  %-14s  %-10s  %-10s\n", "Iter", "Loss", "FwdNFE", "BwdNFE");
    printf("--------------------------------------------------\n");

    for (int iter = 1; iter <= ITERS; iter++) {
        double *dL_dtheta = vec_zeros(nparams);
        double loss = 0.0;
        int total_nfe_fwd = 0, total_nfe_bwd = 0;

        for (int b = 0; b < BATCH; b++) {
            double z0[2];
            for (int i = 0; i < D; i++) z0[i] = rng_normal(r);

            CNFSampleResult sr = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
            double *z1 = sr.z1;
            total_nfe_fwd += sr.nfe;

            double log_pb = -0.5 * (z0[0]*z0[0] + z0[1]*z0[1])
                            - (double)D * 0.5 * log(2.0 * M_PI);

            double log_pm = log_pb + sr.delta_logp;
            double log_pt = log_p_target(z1);

             *   = log_pb + delta_logp - log_pt
             * (KL(p_model || p_target) estimator) */
            loss += log_pm - log_pt;

            double dL_dz1[2];
            grad_log_p_target(z1, dL_dz1);
            dL_dz1[0] = -dL_dz1[0];
            dL_dz1[1] = -dL_dz1[1];

            CNFBackwardResult br = cnf_backward(&cnf, theta, z1,
                                                 1.0, dL_dz1,
                                                 t0, t1, atol, rtol);
            total_nfe_bwd += br.nfe;
            vec_add_scaled(dL_dtheta, 1.0 / BATCH, br.dL_dtheta, nparams);
            free(br.dL_dz0); free(br.dL_dtheta);
            free(z1);
        }

        adam_update(&adam, theta, dL_dtheta);
        free(dL_dtheta);

        if (iter % LOG_EVERY == 0) {
            printf("%-6d  %-14.4f  %-10d  %-10d\n",
                   iter, loss / BATCH,
                   total_nfe_fwd / BATCH, total_nfe_bwd / BATCH);
            fflush(stdout);
        }
    }

    printf("\n--------------------------------------------------\n");

    const int EVAL_N = 200;
    double eval_loss = 0.0;
    for (int i = 0; i < EVAL_N; i++) {
        double z0[2];
        for (int j = 0; j < D; j++) z0[j] = rng_normal(r);
        CNFSampleResult sr = cnf_sample(&cnf, theta, z0, t0, t1, atol, rtol);
        eval_loss += log_p_target(sr.z1);
        free(sr.z1);
    }
    printf("Eval mean log p_target of samples: %.4f\n", eval_loss / EVAL_N);
    printf("Total parameters: %d\n", nparams);

    adam_free(&adam);
    free(theta);
}
