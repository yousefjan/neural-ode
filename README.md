# Neural ODE

A implementation of [Neural Ordinary Differential Equations by Chen et al. (2018)](https://arxiv.org/abs/1806.07366) in C.

## Build
```bash
make
```

## Run
```bash
make run
```

## Usage

### DynNet

A composable network used as the ODE right-hand side: `dz/dt = f_θ(z, t)`. Build a network by stacking layers, then call `dynnet_finalize`.

```c
#include "dynnet.h"

DynNet *dynnet_create(int D);
void dynnet_add_linear(DynNet *net, int out_dim);
void dynnet_add_tanh(DynNet *net);
void dynnet_add_softplus(DynNet *net);
void dynnet_add_swish(DynNet *net);
void dynnet_add_layernorm(DynNet *net);
void dynnet_add_residual_begin(DynNet *net);
void dynnet_add_residual_end(DynNet *net);
void dynnet_add_time_concat(DynNet *net);
void dynnet_finalize(DynNet *net);
void dynnet_init_params(DynNet *net, double *theta, RNG *r);
void dynnet_free(DynNet *net);

void dynnet_forward(DynNet *net, const double *theta,
                    const double *z, double t, double *out, double *workspace);
void dynnet_vjp(DynNet *net, const double *theta,
                const double *z, double t, const double *v,
                double *vjp_z, double *vjp_theta, double *workspace);
```

- `D`: state dimension (input/output).
- `theta` must point to at least `net->total_params` doubles (caller-owned).
- `dynnet_init_params` fills `theta` with random Xavier weights.
- `workspace` must be at least `net->total_workspace` doubles; allocate once and reuse.
- `time_concat` appends `t` to the current feature vector (increases dim by 1).
- Residual blocks: wrap layers between `residual_begin` / `residual_end` (dims must match).

### ODE solver

Adaptive-step Dormand–Prince (RK45) integrator.

```c
#include "ode_solver.h"

// Integrate from t0 to t1, return state at t1.
ODEResult ode_solve(ode_rhs_fn f, const double *y0, double t0, double t1,
                    const double *params, int dim, double atol, double rtol,
                    void *ctx);

// Integrate and record state at each entry of times[0..ntimes-1].
ODEResult ode_solve_times(ode_rhs_fn f, const double *y0, const double *times,
                          int ntimes, const double *params, int dim,
                          double atol, double rtol, void *ctx);
```

`ODEResult.y` is a heap-allocated array of `dim` doubles (single solve) or
`ntimes * dim` doubles (multi-time solve). The caller must `free(result.y)`.
`ODEResult.nfe` is the number of function evaluations used.

### Neural ODE forward + adjoint backward

```c
#include "adjoint.h"

// Single target: integrate z0 → z1, compute MSE loss, backprop via adjoint.
NeuralODEOutput neural_ode_forward_backward(
    DynNet *net, const double *theta,
    const double *z0, double t0, double t1,
    const double *target, double atol, double rtol,
    int num_checkpoints);

// Multiple observation times: integrate z0 → z(times[0..ntimes-1]).
MultiObsNeuralODEOutput neural_ode_forward_backward_multi(
    DynNet *net, const double *theta,
    const double *z0, const double *times,
    const double *targets, int ntimes,
    double atol, double rtol);
```

Both outputs carry heap-allocated `dL_dz0` (`D` doubles) and `dL_dtheta`
(`nparams` doubles). The caller must `free` each field.

For inference only, set up an `AdjointCtx` and call `ode_solve` directly:

```c
double *ws = vec_alloc(net->total_workspace);
double *neg_a = vec_alloc(D);
AdjointCtx ctx = { net, theta, ws, neg_a };
ODEResult fwd = ode_solve(neural_ode_rhs, z0, t0, t1, NULL, D, atol, rtol, &ctx);
// fwd.y holds the predicted state
free(fwd.y);
free(ws);
free(neg_a);
```

### Adam optimizer

```c
#include "adam.h"

Adam adam_init(int nparams, double lr, double beta1, double beta2, double eps);
void adam_update(Adam *a, double *theta, const double *grad);
void adam_free(Adam *a);
```

### Training step

```c
#include "train.h"

TrainStepResult train_step(DynNet *net, double *theta,
                           const double **z0s, const double **targets,
                           double t0, double t1, int batch_size,
                           Adam *adam, double atol, double rtol,
                           int num_checkpoints);
```

Runs a forward+backward pass over a mini-batch, averages gradients, and calls
`adam_update`. Returns the mean MSE loss and forward/backward NFE counts.

### Continuous Normalizing Flow (CNF)

```c
#include "cnf.h"

void cnf_init(CNF *cnf, int D, int H, double *theta, RNG *r);

// Sample noise z0 forward to data space z1.
CNFSampleResult  cnf_sample  (const CNF *cnf, const double *theta,
                               const double *z0, double t0, double t1,
                               double atol, double rtol);

// Evaluate log-probability of a data point z1.
CNFLogProbResult cnf_log_prob(const CNF *cnf, const double *theta,
                               const double *z1, double t0, double t1,
                               double atol, double rtol);

// Backpropagate through the CNF.
CNFBackwardResult cnf_backward(const CNF *cnf, const double *theta,
                                const double *z1,
                                double dL_dlogp, const double *dL_dz1_in,
                                double t0, double t1,
                                double atol, double rtol);
```

`CNFSampleResult.z1` and `CNFLogProbResult.z0` are heap-allocated; the caller
must `free` them. `CNFBackwardResult.dL_dz0` and `.dL_dtheta` must also be freed.

### Utilities

```c
#include "utils.h"

RNG rng_init(uint64_t seed);   // seed the PRNG
double rng_normal(RNG *r);     // standard normal sample

double *vec_alloc(int n);      // malloc n doubles
double *vec_zeros(int n);      // calloc n doubles
```

### End-to-end example

```c
#include "utils.h"
#include "dynnet.h"
#include "adjoint.h"
#include "adam.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int D = 2, H = 32;
    const double t0 = 0.0, t1 = 1.5;

    RNG r = rng_init(42);

    DynNet *net = dynnet_create(D);
    dynnet_add_time_concat(net);
    dynnet_add_linear(net, H);
    dynnet_add_tanh(net);
    dynnet_add_linear(net, D);
    dynnet_finalize(net);

    int nparams = net->total_params;
    double *theta = vec_alloc(nparams);
    dynnet_init_params(net, theta, &r);

    Adam adam = adam_init(nparams, 1e-3, 0.9, 0.999, 1e-8);

    // training data
    double z0[2] = {1.0, 0.0};
    double target[2] = {0.0, 1.0};
    const double *z0s[]  = {z0};
    const double *tgts[] = {target};

    // training loop
    for (int iter = 1; iter <= 500; iter++) {
        TrainStepResult res = train_step(net, theta, z0s, tgts,
                                         t0, t1, 1, &adam,
                                         1e-3, 1e-3, 10);
        if (iter % 100 == 0)
            printf("iter %d  loss=%.4f\n", iter, res.loss);
    }

    // inference
    double *ws    = vec_alloc(net->total_workspace);
    double *neg_a = vec_alloc(D);
    AdjointCtx ctx = { net, theta, ws, neg_a };
    ODEResult fwd = ode_solve(neural_ode_rhs, z0, t0, t1,
                              NULL, D, 1e-5, 1e-5, &ctx);
    free(fwd.y);
    free(ws);
    free(neg_a);

    free(theta);
    adam_free(&adam);
    dynnet_free(net);
    return 0;
}
```


## TODOs

- [ ] Data and training visualization (matplotlib)
