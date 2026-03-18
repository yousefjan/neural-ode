# Neural ODE

A implementation of [Neural Ordinary Differential Equations by Chen et al. (2018)](https://arxiv.org/abs/1806.07366) in C.

## Build
```bash
cc -O2 -Wall -Wextra -Iinclude src/utils.c src/dynmlp.c src/ode_solver.c src/adjoint.c src/adam.c src/train.c src/spiral.c src/tests.c src/cnf.c src/cnf_train.c src/test_cnf.c src/main.c -lm -o neural_ode
```

## Run
```bash
./neural_ode
```

## Usage

### DynMLP

A two-layer dynamic multi layer perceptron used as the ODE right-hand side: `dz/dt = f_θ(z, t)`.

```c
#include "dynmlp.h"

int     dynmlp_nparams(int D, int H);
void    dynmlp_init(DynMLP *net, int D, int H, double *theta, RNG *r);
void    dynmlp_forward(const DynMLP *net, const double *theta,
                       const double *z, double t, double *out, Workspace *ws);
void    dynmlp_vjp(const DynMLP *net, const double *theta,
                   const double *z, double t, const double *v,
                   double *vjp_z, double *vjp_theta, Workspace *ws);
```

- `D`: state dimension, `H`: hidden size.
- `theta` must point to at least `dynmlp_nparams(D, H)` doubles (caller-owned).
- `dynmlp_init` fills `theta` with random Xavier weights.
- `Workspace` is a scratch buffer; allocate once with `workspace_alloc(D, H, nparams)` and free with `workspace_free`.

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
    const DynMLP *net, const double *theta,
    const double *z0, double t0, double t1,
    const double *target, double atol, double rtol,
    int num_checkpoints);

// Multiple observation times: integrate z0 → z(times[0..ntimes-1]).
MultiObsNeuralODEOutput neural_ode_forward_backward_multi(
    const DynMLP *net, const double *theta,
    const double *z0, const double *times,
    const double *targets, int ntimes,
    double atol, double rtol);
```

Both outputs carry heap-allocated `dL_dz0` (`D` doubles) and `dL_dtheta`
(`nparams` doubles). The caller must `free` each field.

For inference only, set up an `AdjointCtx` and call `ode_solve` directly:

```c
Workspace ws = workspace_alloc(D, H, nparams);
AdjointCtx ctx = { net, theta, D, nparams, &ws };
ODEResult fwd = ode_solve(neural_ode_rhs, z0, t0, t1, NULL, D, atol, rtol, &ctx);
// fwd.y holds the predicted state
free(fwd.y);
workspace_free(&ws);
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

TrainStepResult train_step(const DynMLP *net, double *theta,
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

Workspace workspace_alloc(int D, int H, int nparams);
void      workspace_free(Workspace *ws);
```

### End-to-end example

```c
#include "utils.h"
#include "dynmlp.h"
#include "adjoint.h"
#include "adam.h"
#include "train.h"
#include <stdlib.h>

int main(void) {
    const int D = 2, H = 32;
    const double t0 = 0.0, t1 = 1.5;

    RNG r = rng_init(42);
    int nparams = dynmlp_nparams(D, H);
    double *theta = vec_alloc(nparams);

    DynMLP net;
    dynmlp_init(&net, D, H, theta, &r);

    Adam adam = adam_init(nparams, 1e-3, 0.9, 0.999, 1e-8);

    // single training step
    double z0[2]     = {1.0, 0.0};
    double target[2] = {0.0, 1.0};
    const double *z0s[]  = {z0};
    const double *tgts[] = {target};

    TrainStepResult res = train_step(&net, theta, z0s, tgts,
                                     t0, t1, 1, &adam,
                                     1e-3, 1e-3, 10);

    // inference
    Workspace ws = workspace_alloc(D, H, nparams);
    AdjointCtx ctx = { net, theta, D, nparams, &ws };
    ODEResult fwd = ode_solve(neural_ode_rhs, z0, t0, t1,
                              NULL, D, 1e-5, 1e-5, &ctx);
    free(fwd.y);
    workspace_free(&ws);

    free(theta);
    adam_free(&adam);
    return 0;
}
```


## TODOs

- [ ] Deeper Networks
- [ ] Data and training visualization (matplotlib)




