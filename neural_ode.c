#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
   § Utilities: RNG, memory, vector/matrix ops
   ============================================================ */

typedef struct { uint64_t state; } RNG;

static RNG rng_init(uint64_t seed) {
    RNG r;
    r.state = seed ? seed : 1;
    return r;
}

static uint64_t rng_next(RNG *r) {
    uint64_t x = r->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->state = x;
    return x;
}

static double rng_uniform(RNG *r) {
    return (double)(rng_next(r) >> 11) / (double)(UINT64_C(1) << 53);
}

static double rng_normal(RNG *r) {
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "fatal: malloc(%zu) failed\n", n); abort(); }
    return p;
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) { fprintf(stderr, "fatal: calloc(%zu, %zu) failed\n", count, size); abort(); }
    return p;
}

static double *vec_alloc(int n)  { return (double *)xmalloc((size_t)n * sizeof(double)); }
static double *vec_zeros(int n)  { return (double *)xcalloc((size_t)n, sizeof(double)); }
static void    vec_zero(double *v, int n) { memset(v, 0, (size_t)n * sizeof(double)); }
static void    vec_copy(const double *src, double *dst, int n) { memcpy(dst, src, (size_t)n * sizeof(double)); }

static void vec_add_scaled(double *dst, double alpha, const double *v, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * v[i];
}

static double vec_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void mat_vec(const double *M, const double *x, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int j = 0; j < cols; j++) s += M[i * cols + j] * x[j];
        dst[i] = s;
    }
}

static void mat_vec_T(const double *M, const double *v, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j] += M[i * cols + j] * v[i];
}

static void mat_outer_add(double *M, double alpha,
                          const double *a, const double *b, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[i * cols + j] += alpha * a[i] * b[j];
}

/* ============================================================
   § DynMLP: f(z, t, θ): R^(D+1) -> R^D
   ============================================================ */

typedef struct {
    int D;        
    int H;       
    int nparams;   
} DynMLP;

#define DYNMLP_W1(D, H)  (0)
#define DYNMLP_b1(D, H)  ((D + 1) * (H))
#define DYNMLP_W2(D, H)  ((D + 1) * (H) + (H))
#define DYNMLP_b2(D, H)  ((D + 1) * (H) + (H) + (H) * (D))

static int dynmlp_nparams(int D, int H) {
    return (D + 1) * H + H + H * D + D;
}

static void xavier_init(double *w, int fan_in, int fan_out, RNG *r) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    int n = fan_in * fan_out;
    for (int i = 0; i < n; i++)
        w[i] = (2.0 * rng_uniform(r) - 1.0) * limit;
}

static void dynmlp_init(DynMLP *net, int D, int H, double *theta, RNG *r) {
    net->D = D;
    net->H = H;
    net->nparams = dynmlp_nparams(D, H);
    xavier_init(theta + DYNMLP_W1(D, H), D + 1, H, r);
    vec_zero(theta + DYNMLP_b1(D, H), H);
    xavier_init(theta + DYNMLP_W2(D, H), H, D, r);
    vec_zero(theta + DYNMLP_b2(D, H), D);
}

static void dynmlp_forward(const DynMLP *net, const double *theta,
                           const double *z, double t, double *out) {
    int D = net->D, H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);
    const double *b2 = theta + DYNMLP_b2(D, H);

    double *x = vec_alloc(D + 1);
    double *h_pre = vec_alloc(H);
    double *h = vec_alloc(H);

    vec_copy(z, x, D);
    x[D] = t;

    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);

    for (int i = 0; i < H; i++) h[i] = tanh(h_pre[i]);

    mat_vec(W2, h, out, D, H);
    vec_add_scaled(out, 1.0, b2, D);

    free(x); free(h_pre); free(h);
}

/* Vector-Jacobian product: vjp_z = v^T (∂f/∂z),  vjp_theta += v^T (∂f/∂θ)
   Note: vjp_theta is accumulated into, not overwritten. */
static void dynmlp_vjp(const DynMLP *net, const double *theta,
                       const double *z, double t, const double *v,
                       double *vjp_z, double *vjp_theta) {
    int D = net->D;
    int H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);
    double *dW1 = vjp_theta + DYNMLP_W1(D, H);
    double *db1 = vjp_theta + DYNMLP_b1(D, H);
    double *dW2 = vjp_theta + DYNMLP_W2(D, H);
    double *db2 = vjp_theta + DYNMLP_b2(D, H);

    double *x     = vec_alloc(D + 1);
    double *h_pre = vec_alloc(H);
    double *h     = vec_alloc(H);
    vec_copy(z, x, D);
    x[D] = t;
    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);
    
    for (int i = 0; i < H; i++)
      h[i] = tanh(h_pre[i]);

    double *dh     = vec_zeros(H);
    double *dh_pre = vec_alloc(H);
    double *dx     = vec_zeros(D + 1);

    mat_vec_T(W2, v, dh, D, H);
    mat_outer_add(dW2, 1.0, v, h, D, H);
    vec_add_scaled(db2, 1.0, v, D);

    for (int i = 0; i < H; i++) 
      dh_pre[i] = (1.0 - h[i] * h[i]) * dh[i];

    mat_vec_T(W1, dh_pre, dx, H, D + 1);
    mat_outer_add(dW1, 1.0, dh_pre, x, H, D + 1);
    vec_add_scaled(db1, 1.0, dh_pre, H);

    vec_copy(dx, vjp_z, D);

    free(x); free(h_pre); free(h); free(dh); free(dh_pre); free(dx);
}

/* ============================================================
   § RK45 solver 
   ============================================================ */

typedef void (*ode_rhs_fn)(const double *state, double t, const double *params,
                           int dim, double *out, void *ctx);

typedef struct {
    double *y;
    int nfe; // number of fn evaluations
} ODEResult;

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
    double **k = (double **)xmalloc(7 * sizeof(double *));
    for (int i = 0; i < 7; i++) k[i] = vec_alloc(dim);
    double *y   = vec_alloc(dim);
    double *y5  = vec_alloc(dim);
    double *err = vec_alloc(dim);
    double *stg = vec_alloc(dim);

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
    for (int i = 0; i < 7; i++)  free(k[i]);
    
    free(k);
    free(y);
    free(y5);
    free(err);
    free(stg);

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

/* ============================================================
   § Adjoint sensitivity method (Algorithm 1)
   ============================================================ */

typedef struct {
    DynMLP net;
    const double *theta;
    int state_dim;
    int nparams;
} AdjointCtx;

static void neural_ode_rhs(const double *state, double t, const double *params,
                           int dim, double *out, void *ctx) {
    (void)params; 
    (void)dim;
    
    AdjointCtx *ac = (AdjointCtx *)ctx;
    
    dynmlp_forward(&ac->net, ac->theta, state, t, out);
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

    dynmlp_forward(&ac->net, ac->theta, z, t, aug_out);  

    double *neg_a = vec_alloc(D);
    double *vjp_z = vec_alloc(D);
    double *vjp_theta = vec_zeros(nparams); 
    for (int i = 0; i < D; i++) neg_a[i] = -a[i];

    dynmlp_vjp(&ac->net, ac->theta, z, t, neg_a, vjp_z, vjp_theta);

    vec_copy(vjp_z, aug_out + D, D);
    vec_copy(vjp_theta, aug_out + 2 * D, nparams); 

    free(neg_a); 
    free(vjp_z);
    free(vjp_theta);
}

typedef struct {
    double *dL_dz0;    
    double *dL_dtheta;
    int nfe;
} AdjointResult;

typedef struct {
    int     num_checkpoints;  
    double *times;  /* times[0..num_checkpoints], length num_checkpoints+1 */
    double **states;  /* states[0..num_checkpoints], state at each checkpoint time */
    double *z1;  /* separate copy of states[num_checkpoints] = z(t1) */
    int     nfe;
} ForwardResult;

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
    AdjointCtx ac = { *net, theta, D, net->nparams };

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

    AdjointCtx ac = { *net, theta, D, nparams };
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

    free(aug);
    return ar;
}

typedef struct {
    double *z1;
    double *dL_dz0;
    double *dL_dtheta;
    int nfe_forward;
    int nfe_backward;
} NeuralODEOutput;

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

/* ============================================================
   § Multi-observation adjoint
   ============================================================ */

typedef struct {
    double *dL_dz0;  
    double *dL_dtheta; 
    int nfe;   
} MultiObsAdjointResult;

/* adjoint_solve_multi: backward pass with `kicks' at each observation time.
   z_traj[i*D .. i*D+D] = z(times[i]) from the forward pass.
   dL_dz_each[i*D .. i*D+D] = dL_i/dz(times[i]) for each observation. */
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

    AdjointCtx ac = { *net, theta, D, nparams };

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

    free(aug);
    return result;
}

typedef struct {
    double *z_traj; 
    double *dL_dz0;
    double *dL_dtheta;
    int nfe_forward;
    int nfe_backward;
} MultiObsNeuralODEOutput;

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
    AdjointCtx ac = { *net, theta, D, net->nparams };

    ODEResult fwd = ode_solve_times(neural_ode_rhs, z0, times, ntimes,
                                    NULL, D, atol, rtol, &ac);

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

/* ============================================================
   § Training loop
   ============================================================ */


typedef struct {
    double *m;   
    double *v;   
    int nparams;
    double lr;
    double beta1;
    double beta2;
    double eps;
    int t;   
} Adam;

static Adam adam_init(int nparams, double lr, double beta1, double beta2, double eps) {
    Adam a;
    a.m = vec_zeros(nparams);
    a.v = vec_zeros(nparams);
    a.nparams = nparams;
    a.lr = lr;
    a.beta1 = beta1;
    a.beta2 = beta2;
    a.eps = eps;
    a.t = 0;
    return a;
}

static void adam_update(Adam *a, double *theta, const double *grad) {
    a->t++;
    double bc1 = 1.0 - pow(a->beta1, (double)a->t);
    double bc2 = 1.0 - pow(a->beta2, (double)a->t);
    double alpha = a->lr * sqrt(bc2) / bc1;
    for (int i = 0; i < a->nparams; i++) {
        a->m[i] = a->beta1 * a->m[i] + (1.0 - a->beta1) * grad[i];
        a->v[i] = a->beta2 * a->v[i] + (1.0 - a->beta2) * grad[i] * grad[i];
        theta[i] -= alpha * a->m[i] / (sqrt(a->v[i]) + a->eps);
    }
}

static void adam_free(Adam *a) {
    free(a->m);
    free(a->v);
}

static double train_one(const DynMLP *net, const double *theta,
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
    for (int i = 0; i < net->nparams; i++) grad_accum[i] += out.dL_dtheta[i];
    *nfe_fwd += out.nfe_forward;
    *nfe_bwd += out.nfe_backward;
    free(out.z1);
    free(out.dL_dz0);
    free(out.dL_dtheta);
    return loss;
}

typedef struct {
    double loss;
    int nfe_fwd;
    int nfe_bwd;
} TrainStepResult;

static TrainStepResult train_step(const DynMLP *net, double *theta,
                                  const double **z0s, const double **targets,
                                  double t0, double t1, int batch_size,
                                  Adam *adam, double atol, double rtol, int num_checkpoints) {
    int nparams = net->nparams;
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

/* ============================================================
   § Spiral dataset
   ============================================================ */

static void spiral_rhs(const double *state, double t, const double *params,
                       int dim, double *out, void *ctx) {
    (void)t; (void)dim; (void)ctx;
    double alpha = params[0];
    out[0] = alpha * state[1];
    out[1] = -alpha * state[0];
}

typedef struct {
    double **z0;
    double **target;
    int num_samples;
} Dataset;

static Dataset generate_spiral_dataset(int num_samples, double t0, double t1,
                                       double noise_std, RNG *r) {
    Dataset ds;
    ds.num_samples = num_samples;
    ds.z0     = (double **)xmalloc(num_samples * sizeof(double *));
    ds.target = (double **)xmalloc(num_samples * sizeof(double *));

    for (int i = 0; i < num_samples; i++) {
        double alpha = 1.0;
        double angle  = 2.0 * M_PI * rng_uniform(r);
        double radius = 0.5 + 1.0 * rng_uniform(r);   /* uniform in [0.5, 1.5] */

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

static void dataset_free(Dataset *ds) {
    for (int i = 0; i < ds->num_samples; i++) {
        free(ds->z0[i]);
        free(ds->target[i]);
    }
    free(ds->z0);
    free(ds->target);
}

static double evaluate(const DynMLP *net, const double *theta,
                       const Dataset *ds, double t0, double t1,
                       double atol, double rtol) {
    int D = net->D;
    AdjointCtx ac = { *net, theta, D, net->nparams };
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
    return total_loss / (double)ds->num_samples;
}

/* ============================================================
   § Tests
   ============================================================ */

static void rhs_decay(const double *y, double t, const double *p, int d, double *out, void *ctx) {
    (void)t; (void)p; (void)d; (void)ctx;
    out[0] = -y[0];
}

static void rhs_rotation(const double *y, double t, const double *p, int d, double *out, void *ctx) {
    (void)t; (void)p; (void)d; (void)ctx;
    out[0] = -y[1]; out[1] = y[0];
}

static void test_ode_solver(void) {
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

static void test_dynmlp_gradients(RNG *r) {
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

    double *vjp_z     = vec_zeros(D);
    double *vjp_theta = vec_zeros(np);
    dynmlp_vjp(&net, theta, z, t, v, vjp_z, vjp_theta);

    double *num_vjp_z = vec_alloc(D);
    for (int i = 0; i < D; i++) {
        double zi = z[i];
        z[i] = zi + EPS; dynmlp_forward(&net, theta, z, t, out_p);
        z[i] = zi - EPS; dynmlp_forward(&net, theta, z, t, out_m);
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
        theta[k] = tk + EPS; dynmlp_forward(&net, theta, z, t, out_p);
        theta[k] = tk - EPS; dynmlp_forward(&net, theta, z, t, out_m);
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

    free(theta); free(z); free(v); free(out_p); free(out_m);
    free(vjp_z); free(vjp_theta); free(num_vjp_z); free(num_vjp_theta);
}

static void test_adjoint_gradients(RNG *r) {
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

#define FWD_LOSS(z0_, theta_) ({ \
    AdjointCtx ac_ = { net, (theta_), D, np }; \
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

    free(out.z1); free(out.dL_dz0); free(out.dL_dtheta);
    free(num_dL_dtheta); free(num_dL_dz0);
    free(theta); free(z0); free(target);
}

static void test_multi_obs_adjoint(RNG *r) {
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

    AdjointCtx ac = { net, theta, D, np };

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

    free(out.z_traj); free(out.dL_dz0); free(out.dL_dtheta);
    free(num_dL_dtheta); free(num_dL_dz0);
    free(theta); free(z0); free(targets);
}

static void test_training(RNG *r) {
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

    AdjointCtx ac = { net, theta, D, nparams };
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

    adam_free(&adam);
    free(batch_z0); free(batch_tgt);
    for (int i = 0; i < N; i++) { free(z0s[i]); free(targets[i]); }
    free(z0s); free(targets);
    free(theta);
}



int main(void) {
    RNG r = rng_init(42);

    /* --- sanity checks --- */
    test_ode_solver();
    test_dynmlp_gradients(&r);
    test_adjoint_gradients(&r);
    test_multi_obs_adjoint(&r);
    test_training(&r);

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
    AdjointCtx ac = { net, theta, D, nparams };
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

    free(theta);
    adam_free(&adam);
    dataset_free(&train_ds);
    dataset_free(&test_ds);

    return 0;
}
