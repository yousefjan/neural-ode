#include "train.h"
#include "adjoint.h"
#include "utils.h"

#include <stdlib.h>

static double train_one(DynNet *net, const double *theta,
                        const double *z0, double t0, double t1,
                        const double *target, double *grad_accum,
                        double atol, double rtol, int num_checkpoints,
                        int *nfe_fwd, int *nfe_bwd) {
    NeuralODEOutput out = neural_ode_forward_backward(net, theta, z0, t0, t1,
                                                      target, atol, rtol, num_checkpoints);
    double loss = 0.0;
    int D = net->D;
    for (int i = 0; i < D; i++) {
        double d = out.z1[i] - target[i];
        loss += 0.5 * d * d;
    }
    for (int i = 0; i < net->total_params; i++) grad_accum[i] += out.dL_dtheta[i];
    *nfe_fwd += out.nfe_forward;
    *nfe_bwd += out.nfe_backward;
    free(out.z1);
    free(out.dL_dz0);
    free(out.dL_dtheta);
    return loss;
}

TrainStepResult train_step(DynNet *net, double *theta,
                           const double **z0s, const double **targets,
                           double t0, double t1, int batch_size,
                           Adam *adam, double atol, double rtol, int num_checkpoints) {
    int nparams = net->total_params;
    double *grad_accum = vec_zeros(nparams);
    TrainStepResult res = { 0.0, 0, 0 };

    for (int b = 0; b < batch_size; b++) {
        res.loss += train_one(net, theta, z0s[b], t0, t1, targets[b],
                              grad_accum, atol, rtol, num_checkpoints,
                              &res.nfe_fwd, &res.nfe_bwd);
    }
    res.loss /= (double)batch_size;
    for (int i = 0; i < nparams; i++) grad_accum[i] /= (double)batch_size;
    adam_update(adam, theta, grad_accum);
    free(grad_accum);
    return res;
}
