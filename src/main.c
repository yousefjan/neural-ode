#include "utils.h"
#include "dynmlp.h"
#include "ode_solver.h"
#include "adjoint.h"
#include "adam.h"
#include "train.h"
#include "spiral.h"
#include "tests.h"
#include "test_cnf.h"
#include "cnf_train.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    RNG r = rng_init(42);

    /* --- sanity checks --- */
    test_ode_solver();
    test_dynmlp_gradients(&r);
    test_adjoint_gradients(&r);
    test_multi_obs_adjoint(&r);
    test_training(&r);

    printf("\n--- CNF tests ---\n");
    test_cnf_trace(&r);
    test_cnf_invertibility(&r);
    test_cnf_gradients(&r);
    test_cnf_training(&r);

    cnf_train_demo(&r);

    printf("\n--- Training demo (spiral) ---\n\n");

    r = rng_init((uint64_t)time(NULL));

    const double t0 = 0.0, t1 = 1.5;
    const double noise_std = 0.1;
    const double atol_train = 1e-3, rtol_train = 1e-3;
    const double atol_eval  = 1e-5, rtol_eval  = 1e-5;
    const int TRAIN_N = 200, TEST_N = 50;
    const int ITERS = 500, BATCH = 16, LOG_EVERY = 25;

    Dataset train_ds = generate_spiral_dataset(TRAIN_N, t0, t1, noise_std, &r);
    Dataset test_ds  = generate_spiral_dataset(TEST_N,  t0, t1, noise_std, &r);

    const int D = 2, H = 32;
    DynMLP net;
    int nparams = dynmlp_nparams(D, H);
    double *theta = vec_alloc(nparams);
    dynmlp_init(&net, D, H, theta, &r);

    Adam adam = adam_init(nparams, 1e-3, 0.9, 0.999, 1e-8);

    printf("%-6s  %-12s  %-12s  %-10s  %-10s",
           "Iter", "Train Loss", "Test Loss", "Fwd NFE", "Bwd NFE");
    printf("\n--------------------------------------------------\n");

    const double **batch_z0  = (const double **)xmalloc(BATCH * sizeof(double *));
    const double **batch_tgt = (const double **)xmalloc(BATCH * sizeof(double *));

    for (int iter = 1; iter <= ITERS; iter++) {
        for (int b = 0; b < BATCH; b++) {
            int idx = (int)(rng_next(&r) % (uint64_t)TRAIN_N);
            batch_z0[b]  = train_ds.z0[idx];
            batch_tgt[b] = train_ds.target[idx];
        }

        TrainStepResult res = train_step(&net, theta, batch_z0, batch_tgt,
                                         t0, t1, BATCH, &adam,
                                         atol_train, rtol_train, 10);

        if (iter % LOG_EVERY == 0) {
            double test_loss = evaluate(&net, theta, &test_ds,
                                        t0, t1, atol_eval, rtol_eval);
            printf("%-6d  %-12.6f  %-12.6f  %-10d  %-10d\n",
                   iter, res.loss, test_loss,
                   res.nfe_fwd, res.nfe_bwd);
            fflush(stdout);
        }
    }

    free(batch_z0);
    free(batch_tgt);

    printf("\n--------------------------------------------------\n");
    double final_test_loss = evaluate(&net, theta, &test_ds,
                                      t0, t1, atol_eval, rtol_eval);
    printf("Final test loss : %.6f\n", final_test_loss);
    printf("Total parameters: %d\n", nparams);

    printf("\nSample predictions:\n");
    Workspace ws = workspace_alloc(D, H, nparams);
    AdjointCtx ac = { net, theta, D, nparams, &ws };
    for (int s = 0; s < 5; s++) {
        int idx = (int)(rng_next(&r) % (uint64_t)TEST_N);
        ODEResult fwd = ode_solve(neural_ode_rhs,
                                  test_ds.z0[idx], t0, t1,
                                  NULL, D, atol_eval, rtol_eval, &ac);
        printf("  z0=(%.4f, %.4f)  predicted=(%.4f, %.4f)  target=(%.4f, %.4f)\n",
               test_ds.z0[idx][0], test_ds.z0[idx][1],
               fwd.y[0], fwd.y[1],
               test_ds.target[idx][0], test_ds.target[idx][1]);
        free(fwd.y);
    }
    workspace_free(&ws);

    free(theta);
    adam_free(&adam);
    dataset_free(&train_ds);
    dataset_free(&test_ds);

    return 0;
}
