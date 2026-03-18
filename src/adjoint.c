#include "adjoint.h"

#include <stdlib.h>

typedef struct {
    double *dL_dz0;
    double *dL_dtheta;
    int nfe;
} AdjointResult;

typedef struct {
    int     num_checkpoints;
    double *times;
    double **states;
    double *z1;
    int     nfe;
} ForwardResult;

typedef struct {
    double *dL_dz0;
    double *dL_dtheta;
    int nfe;
} MultiObsAdjointResult;

void neural_ode_rhs(const double *state, double t, const double *params,
                    int dim, double *out, void *ctx) {
    (void)params;
    (void)dim;

    AdjointCtx *ac = (AdjointCtx *)ctx;
    dynmlp_forward(&ac->net, ac->theta, state, t, out, ac->ws);
}

static void adjoint_dynamics(const double *aug_state, double t, const double *params,
                             int aug_dim, double *aug_out, void *ctx) {
    (void)params;
    (void)aug_dim;

    AdjointCtx *ac = (AdjointCtx *)ctx;
    int D = ac->state_dim;
    int nparams = ac->nparams;
    const double *z = aug_state;
    const double *a = aug_state + D;

    dynmlp_forward(&ac->net, ac->theta, z, t, aug_out, ac->ws);

    double *neg_a     = ac->ws->neg_a;
    double *vjp_z     = ac->ws->vjp_z;
    double *vjp_theta = ac->ws->vjp_theta;
    vec_zero(vjp_theta, nparams);
    for (int i = 0; i < D; i++) neg_a[i] = -a[i];

    dynmlp_vjp(&ac->net, ac->theta, z, t, neg_a, vjp_z, vjp_theta, ac->ws);

    vec_copy(vjp_z, aug_out + D, D);
    vec_copy(vjp_theta, aug_out + 2 * D, nparams);
}

static void forward_result_free(ForwardResult *fr) {
    free(fr->times);
    for (int i = 0; i <= fr->num_checkpoints; i++)
        free(fr->states[i]);
    free(fr->states);
    free(fr->z1);
}

static ForwardResult forward_solve(const DynMLP *net, const double *theta,
                                   const double *z0, double t0, double t1,
                                   double atol, double rtol, int num_checkpoints) {
    int D = net->D;
    Workspace ws = workspace_alloc(D, net->H, net->nparams);
    AdjointCtx ac = { *net, theta, D, net->nparams, &ws };

    ForwardResult fr;
    fr.num_checkpoints = num_checkpoints;
    fr.nfe = 0;

    fr.times  = vec_alloc(num_checkpoints + 1);
    fr.states = (double **)xmalloc((size_t)(num_checkpoints + 1) * sizeof(double *));
    for (int i = 0; i <= num_checkpoints; i++)
        fr.states[i] = vec_alloc(D);

    for (int i = 0; i <= num_checkpoints; i++)
        fr.times[i] = t0 + (t1 - t0) * (double)i / (double)num_checkpoints;

    vec_copy(z0, fr.states[0], D);

    for (int i = 0; i < num_checkpoints; i++) {
        ODEResult seg = ode_solve(neural_ode_rhs, fr.states[i],
                                  fr.times[i], fr.times[i + 1],
                                  NULL, D, atol, rtol, &ac);
        vec_copy(seg.y, fr.states[i + 1], D);
        fr.nfe += seg.nfe;
        free(seg.y);
    }

    fr.z1 = vec_alloc(D);
    vec_copy(fr.states[num_checkpoints], fr.z1, D);

    workspace_free(&ws);
    return fr;
}

static AdjointResult adjoint_solve(const DynMLP *net, const double *theta,
                                   const ForwardResult *fr, const double *dL_dz1,
                                   double atol, double rtol) {
    int D = net->D;
    int nparams = net->nparams;
    int aug_dim = 2 * D + nparams;

    double *aug = vec_zeros(aug_dim);
    vec_copy(fr->z1, aug, D);
    vec_copy(dL_dz1, aug + D, D);

    Workspace ws = workspace_alloc(D, net->H, nparams);
    AdjointCtx ac = { *net, theta, D, nparams, &ws };
    int total_nfe = 0;

    for (int k = fr->num_checkpoints; k >= 1; k--) {
        /* Replace z with stored checkpoint to prevent numerical drift */
        vec_copy(fr->states[k], aug, D);

        ODEResult seg = ode_solve(adjoint_dynamics, aug,
                                  fr->times[k], fr->times[k - 1],
                                  NULL, aug_dim, atol, rtol, &ac);
        vec_copy(seg.y, aug, aug_dim);
        total_nfe += seg.nfe;
        free(seg.y);
    }

    AdjointResult ar;
    ar.dL_dz0    = vec_alloc(D);
    ar.dL_dtheta = vec_alloc(nparams);
    ar.nfe       = total_nfe;
    vec_copy(aug + D, ar.dL_dz0, D);
    vec_copy(aug + 2 * D, ar.dL_dtheta, nparams);

    workspace_free(&ws);
    free(aug);
    return ar;
}

NeuralODEOutput neural_ode_forward_backward(const DynMLP *net, const double *theta,
                                            const double *z0, double t0, double t1,
                                            const double *target, double atol, double rtol,
                                            int num_checkpoints) {
    int D = net->D;

    ForwardResult fr = forward_solve(net, theta, z0, t0, t1, atol, rtol, num_checkpoints);

    double *dL_dz1 = vec_alloc(D);
    for (int i = 0; i < D; i++) dL_dz1[i] = fr.z1[i] - target[i];

    AdjointResult ar = adjoint_solve(net, theta, &fr, dL_dz1, atol, rtol);

    NeuralODEOutput out;
    out.z1           = vec_alloc(D);
    vec_copy(fr.z1, out.z1, D);
    out.dL_dz0       = ar.dL_dz0;
    out.dL_dtheta    = ar.dL_dtheta;
    out.nfe_forward  = fr.nfe;
    out.nfe_backward = ar.nfe;

    forward_result_free(&fr);
    free(dL_dz1);
    return out;
}

static MultiObsAdjointResult adjoint_solve_multi(
    const DynMLP *net,
    const double *theta,
    const double *z_traj,
    const double *times,
    const double *dL_dz_each,
    int ntimes,
    double atol,
    double rtol)
{
    int D       = net->D;
    int nparams = net->nparams;
    int aug_dim = 2 * D + nparams;

    Workspace ws = workspace_alloc(D, net->H, nparams);
    AdjointCtx ac = { *net, theta, D, nparams, &ws };

    double *a = vec_alloc(D);
    double *dtheta = vec_zeros(nparams);
    double *aug = vec_alloc(aug_dim);
    int total_nfe = 0;

    vec_copy(dL_dz_each + (ntimes - 1) * D, a, D);

    for (int i = ntimes - 1; i >= 1; i--) {
        vec_copy(z_traj + i * D, aug, D);
        vec_copy(a,      aug + D,      D);
        vec_copy(dtheta, aug + 2 * D,  nparams);

        ODEResult seg = ode_solve(adjoint_dynamics, aug,
                                  times[i], times[i - 1],
                                  NULL, aug_dim, atol, rtol, &ac);
        vec_copy(seg.y, aug, aug_dim);
        total_nfe += seg.nfe;
        free(seg.y);

        vec_copy(aug + D, a, D);
        vec_copy(aug + 2 * D, dtheta, nparams);

        /* Kick: add per-observation loss gradient at time t_{i-1} */
        vec_add_scaled(a, 1.0, dL_dz_each + (i - 1) * D, D);
    }

    MultiObsAdjointResult result;
    result.dL_dz0    = a;
    result.dL_dtheta = dtheta;
    result.nfe       = total_nfe;

    workspace_free(&ws);
    free(aug);
    return result;
}

MultiObsNeuralODEOutput neural_ode_forward_backward_multi(
    const DynMLP *net,
    const double *theta,
    const double *z0,
    const double *times,
    const double *targets,
    int ntimes,
    double atol,
    double rtol)
{
    int D = net->D;
    Workspace ws = workspace_alloc(D, net->H, net->nparams);
    AdjointCtx ac = { *net, theta, D, net->nparams, &ws };

    ODEResult fwd = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                    NULL, D, atol, rtol, &ac);
    workspace_free(&ws);

    double *dL_dz_each = vec_alloc(ntimes * D);
    for (int i = 0; i < ntimes * D; i++)
        dL_dz_each[i] = fwd.y[i] - targets[i];

    MultiObsAdjointResult ar = adjoint_solve_multi(net, theta, fwd.y, times,
                                                   dL_dz_each, ntimes, atol, rtol);
    free(dL_dz_each);

    MultiObsNeuralODEOutput out;
    out.z_traj       = fwd.y;
    out.dL_dz0       = ar.dL_dz0;
    out.dL_dtheta    = ar.dL_dtheta;
    out.nfe_forward  = fwd.nfe;
    out.nfe_backward = ar.nfe;
    return out;
}
