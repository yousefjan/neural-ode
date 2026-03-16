#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

typedef struct {
    uint64_t state;
} RNG;

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
    if (!p) {
        fprintf(stderr, "fatal: malloc(%zu) failed\n", n);
        abort();
    }
    return p;
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) {
        fprintf(stderr, "fatal: calloc(%zu, %zu) failed\n", count, size);
        abort();
    }
    return p;
}

static double *vec_alloc(int n) {
    return (double *)xmalloc((size_t)n * sizeof(double));
}

static double *vec_zeros(int n) {
    return (double *)xcalloc((size_t)n, sizeof(double));
}

static void vec_copy(const double *src, double *dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(double));
}

static void vec_zero(double *v, int n) {
    memset(v, 0, (size_t)n * sizeof(double));
}

static void vec_axpy(const double *a, double alpha, const double *b,
                     double *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = a[i] + alpha * b[i];
}

static void vec_add_scaled(double *dst, double alpha, const double *v, int n) {
    for (int i = 0; i < n; i++)
        dst[i] += alpha * v[i];
}

static void vec_scale(const double *v, double alpha, double *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = alpha * v[i];
}

static double vec_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}

static double vec_norm(const double *v, int n) {
    return sqrt(vec_dot(v, v, n));
}

static void mat_vec(const double *M, const double *x, double *dst,
                    int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int j = 0; j < cols; j++)
            s += M[i * cols + j] * x[j];
        dst[i] = s;
    }
}

static void mat_vec_T(const double *M, const double *v, double *dst,
                      int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            dst[j] += M[i * cols + j] * v[i];
    }
}

static void mat_outer_add(double *M, double alpha,
                          const double *a, const double *b,
                          int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            M[i * cols + j] += alpha * a[i] * b[j];
    }
}

static void xavier_init(double *w, int fan_in, int fan_out, RNG *r) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    int n = fan_in * fan_out;
    for (int i = 0; i < n; i++)
        w[i] = (2.0 * rng_uniform(r) - 1.0) * limit;
}

static void bias_init(double *b, int n) {
    vec_zero(b, n);
}

static void act_tanh(const double *x, double *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = tanh(x[i]);
}

static void act_dtanh(const double *y, double *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = 1.0 - y[i] * y[i];
}


/* --- DynMLP: f(z, t, θ): R^(D+1) -> R^D via tanh hidden layer --- */

typedef struct {
    int D;       /* state dimension  */
    int H;       /* hidden dimension */
    int nparams; /* total number of parameters */
} DynMLP;

#define DYNMLP_W1(D, H)  (0)
#define DYNMLP_b1(D, H)  ((D + 1) * (H))
#define DYNMLP_W2(D, H)  ((D + 1) * (H) + (H))
#define DYNMLP_b2(D, H)  ((D + 1) * (H) + (H) + (H) * (D))

static int dynmlp_nparams(int D, int H) {
    return (D + 1) * H   /* W1 */
         + H             /* b1 */
         + H * D         /* W2 */
         + D;            /* b2 */
}

static void dynmlp_init(DynMLP *net, int D, int H, double *theta, RNG *r) {
    net->D       = D;
    net->H       = H;
    net->nparams = dynmlp_nparams(D, H);

    double *W1 = theta + DYNMLP_W1(D, H);
    double *b1 = theta + DYNMLP_b1(D, H);
    double *W2 = theta + DYNMLP_W2(D, H);
    double *b2 = theta + DYNMLP_b2(D, H);

    xavier_init(W1, D + 1, H, r);
    bias_init(b1, H);
    xavier_init(W2, H, D, r);
    bias_init(b2, D);
}

static void dynmlp_forward(const DynMLP *net, const double *theta,
                           const double *z, double t, double *out) {
    int D = net->D, H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);
    const double *b2 = theta + DYNMLP_b2(D, H);

    double *x     = vec_alloc(D + 1);
    double *h_pre = vec_alloc(H);
    double *h     = vec_alloc(H);

    /* x = [z; t], length D+1 */
    vec_copy(z, x, D);
    x[D] = t;

    /* h_pre = W1 * x + b1, length H */
    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);

    /* h = tanh(h_pre), length H */
    act_tanh(h_pre, h, H);

    /* out = W2 * h + b2, length D */
    mat_vec(W2, h, out, D, H);
    vec_add_scaled(out, 1.0, b2, D);

    free(x);
    free(h_pre);
    free(h);
}

static void dynmlp_vjp(const DynMLP *net, const double *theta,
                       const double *z, double t, const double *v,
                       double *vjp_z, double *vjp_theta) {
    int D = net->D, H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);

    double *dW1 = vjp_theta + DYNMLP_W1(D, H);
    double *db1 = vjp_theta + DYNMLP_b1(D, H);
    double *dW2 = vjp_theta + DYNMLP_W2(D, H);
    double *db2 = vjp_theta + DYNMLP_b2(D, H);

    /* --- recover intermediates --- */
    double *x     = vec_alloc(D + 1);
    double *h_pre = vec_alloc(H);
    double *h     = vec_alloc(H);

    vec_copy(z, x, D);
    x[D] = t;
    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);
    act_tanh(h_pre, h, H);

    /* --- backward --- */
    double *dh     = vec_zeros(H);
    double *dh_pre = vec_alloc(H);
    double *dx     = vec_zeros(D + 1);

    mat_vec_T(W2, v, dh, D, H);      /* W2: D×H */

    /* dW2 += outer(v, h), db2 += v */
    mat_outer_add(dW2, 1.0, v, h, D, H);
    vec_add_scaled(db2, 1.0, v, D);

    /* dh_pre = dh * (1 - h*h) element-wise */
    act_dtanh(h, dh_pre, H);
    for (int i = 0; i < H; i++)
        dh_pre[i] *= dh[i];

    /* dx = W1^T * dh_pre */
    mat_vec_T(W1, dh_pre, dx, H, D + 1);  /* W1: H×(D+1) */

    /* dW1 += outer(dh_pre, x), db1 += dh_pre */
    mat_outer_add(dW1, 1.0, dh_pre, x, H, D + 1);
    vec_add_scaled(db1, 1.0, dh_pre, H);

    /* vjp_z = dx[0..D-1] */
    vec_copy(dx, vjp_z, D);

    free(x);
    free(h_pre);
    free(h);
    free(dh);
    free(dh_pre);
    free(dx);
}

/* --- ODE Solver: Dormand-Prince RK45 --- */

typedef void (*ode_rhs_fn)(const double *state, double t, const double *params,
                           int dim, double *out, void *ctx);

typedef struct {
    double *y;  /* final state, length dim — caller must free */
    int nfe;    /* total number of f evaluations */
} ODEResult;

/* Butcher tableau nodes */
static const double dp_c[7] = {
    0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
};

/* a coefficients, row by row (lower triangular) */
static const double dp_a2[1] = { 1.0/5.0 };
static const double dp_a3[2] = { 3.0/40.0, 9.0/40.0 };
static const double dp_a4[3] = { 44.0/45.0, -56.0/15.0, 32.0/9.0 };
static const double dp_a5[4] = { 19372.0/6561.0, -25360.0/2187.0,
                                  64448.0/6561.0,   -212.0/729.0 };
static const double dp_a6[5] = { 9017.0/3168.0, -355.0/33.0,
                                  46732.0/5247.0,   49.0/176.0, -5103.0/18656.0 };

/* 5th-order weights (b); b7=0 so FSAL: stage-7 input == y5 */
static const double dp_b[7] = {
    35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0
};

/* 4th-order weights (b*) — stored for reference */
static const double dp_bs[7] = {
    5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
    -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
};

/* Error coefficients: e = b - b* */
static const double dp_e[7] = {
    35.0/384.0     - 5179.0/57600.0,
    0.0,
    500.0/1113.0   - 7571.0/16695.0,
    125.0/192.0    - 393.0/640.0,
    -2187.0/6784.0 + 92097.0/339200.0,
    11.0/84.0      - 187.0/2100.0,
    -1.0/40.0
};

ODEResult ode_solve(ode_rhs_fn f, const double *y0, double t0, double t1,
                    const double *params, int dim, double atol, double rtol,
                    void *ctx) {
    (void)dp_bs; /* stored for reference; dp_e encodes b-b* */

    double **k = (double **)xmalloc(7 * sizeof(double *));
    for (int i = 0; i < 7; i++) k[i] = vec_alloc(dim);
    double *y   = vec_alloc(dim);
    double *y5  = vec_alloc(dim);
    double *err = vec_alloc(dim);
    double *stg = vec_alloc(dim);

    ODEResult res;
    res.y   = vec_alloc(dim);
    res.nfe = 0;

    vec_copy(y0, y, dim);
    double t  = t0;
    double h  = 0.01 * (t1 - t0);  /* sign encodes direction */
    int k1_fresh = 0;

    const double safety = 0.9;
    const int max_steps = 1000000;

    for (int step = 0; step < max_steps; step++) {
        /* Check termination */
        if (t1 > t0) {
            if (t >= t1) break;
            if (t + h > t1) h = t1 - t;
        } else {
            if (t <= t1) break;
            if (t + h < t1) h = t1 - t;
        }

        /* Stage 1 (skip on FSAL reuse) */
        if (!k1_fresh) {
            f(y, t, params, dim, k[0], ctx);
            res.nfe++;
            k1_fresh = 1;
        }

        /* Stage 2 */
        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a2[0]*k[0][i]);
        f(stg, t + dp_c[1]*h, params, dim, k[1], ctx);
        res.nfe++;

        /* Stage 3 */
        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a3[0]*k[0][i] + dp_a3[1]*k[1][i]);
        f(stg, t + dp_c[2]*h, params, dim, k[2], ctx);
        res.nfe++;

        /* Stage 4 */
        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a4[0]*k[0][i] + dp_a4[1]*k[1][i]
                                + dp_a4[2]*k[2][i]);
        f(stg, t + dp_c[3]*h, params, dim, k[3], ctx);
        res.nfe++;

        /* Stage 5 */
        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a5[0]*k[0][i] + dp_a5[1]*k[1][i]
                                + dp_a5[2]*k[2][i] + dp_a5[3]*k[3][i]);
        f(stg, t + dp_c[4]*h, params, dim, k[4], ctx);
        res.nfe++;

        /* Stage 6 */
        for (int i = 0; i < dim; i++)
            stg[i] = y[i] + h * (dp_a6[0]*k[0][i] + dp_a6[1]*k[1][i]
                                + dp_a6[2]*k[2][i] + dp_a6[3]*k[3][i]
                                + dp_a6[4]*k[4][i]);
        f(stg, t + dp_c[5]*h, params, dim, k[5], ctx);
        res.nfe++;

        /* 5th-order solution y5 (b2=0, b7=0) */
        for (int i = 0; i < dim; i++)
            y5[i] = y[i] + h * (dp_b[0]*k[0][i] + dp_b[2]*k[2][i]
                               + dp_b[3]*k[3][i] + dp_b[4]*k[4][i]
                               + dp_b[5]*k[5][i]);

        /* Stage 7 / FSAL: f at (t+h, y5) */
        f(y5, t + h, params, dim, k[6], ctx);
        res.nfe++;

        /* Error estimate: e2=0, k[1] not used */
        for (int i = 0; i < dim; i++)
            err[i] = h * (dp_e[0]*k[0][i] + dp_e[2]*k[2][i]
                        + dp_e[3]*k[3][i] + dp_e[4]*k[4][i]
                        + dp_e[5]*k[5][i] + dp_e[6]*k[6][i]);

        /* RMS error norm with mixed tolerance scaling */
        double err_sq = 0.0;
        for (int i = 0; i < dim; i++) {
            double sc = atol + rtol * fmax(fabs(y[i]), fabs(y5[i]));
            double e  = err[i] / sc;
            err_sq   += e * e;
        }
        double err_norm = sqrt(err_sq / (double)dim);

        /* Compute new step size factor */
        double factor;
        if (err_norm == 0.0) {
            factor = 5.0;
        } else {
            factor = safety * pow(err_norm, -0.2);
            if (factor < 0.2) factor = 0.2;
            if (factor > 5.0) factor = 5.0;
        }

        if (err_norm <= 1.0) {
            /* Accept: advance state, FSAL swap k[0] <-> k[6] */
            vec_copy(y5, y, dim);
            t += h;
            double *tmp = k[0]; k[0] = k[6]; k[6] = tmp;
            h *= factor;
        } else {
            /* Reject: shrink only */
            if (factor > 1.0) factor = 1.0;
            h *= factor;
            /* k1_fresh stays 1 — y and t unchanged */
        }
    }

    vec_copy(y, res.y, dim);

    for (int i = 0; i < 7; i++) free(k[i]);
    free(k);
    free(y);
    free(y5);
    free(err);
    free(stg);

    return res;
}

/* Solve and record state at each time in times[0..ntimes-1].
   times[0] is the start; result->y is dim*ntimes doubles. */
ODEResult ode_solve_times(ode_rhs_fn f, const double *y0, const double *times,
                          int ntimes, const double *params, int dim,
                          double atol, double rtol, void *ctx) {
    ODEResult res;
    res.y   = vec_alloc(dim * ntimes);
    res.nfe = 0;

    vec_copy(y0, res.y, dim);  /* state at times[0] */

    for (int i = 1; i < ntimes; i++) {
        const double *cur = res.y + (i - 1) * dim;
        ODEResult seg = ode_solve(f, cur, times[i-1], times[i],
                                  params, dim, atol, rtol, ctx);
        vec_copy(seg.y, res.y + i * dim, dim);
        res.nfe += seg.nfe;
        free(seg.y);
    }

    return res;
}

/* --- ODE solver tests --- */

static void rhs_decay(const double *state, double t, const double *params,
                      int dim, double *out, void *ctx) {
    (void)t; (void)params; (void)dim; (void)ctx;
    out[0] = -state[0];
}

static void rhs_rotation(const double *state, double t, const double *params,
                         int dim, double *out, void *ctx) {
    (void)t; (void)params; (void)dim; (void)ctx;
    out[0] = -state[1];
    out[1] =  state[0];
}

static void test_ode_solver(void) {
    const double atol = 1e-8, rtol = 1e-8;
    const double check_tol = 1e-6;

    /* Test 1: scalar decay dy/dt = -y, y(0)=1 -> y(1) = e^{-1} */
    {
        double y0 = 1.0;
        ODEResult r = ode_solve(rhs_decay, &y0, 0.0, 1.0, NULL, 1, atol, rtol, NULL);
        double exact = exp(-1.0);
        double err   = fabs(r.y[0] - exact);
        printf("ODE test 1 (decay):    err=%.2e  nfe=%d  %s\n",
               err, r.nfe, err < check_tol ? "PASS" : "FAIL");
        free(r.y);
    }

    /* Test 2: 2D rotation, one full period -> back to [1, 0] */
    {
        double y0[2] = {1.0, 0.0};
        ODEResult r = ode_solve(rhs_rotation, y0, 0.0, 2.0 * M_PI,
                                NULL, 2, atol, rtol, NULL);
        double err = sqrt((r.y[0]-1.0)*(r.y[0]-1.0) + r.y[1]*r.y[1]);
        printf("ODE test 2 (rotation): err=%.2e  nfe=%d  %s\n",
               err, r.nfe, err < check_tol ? "PASS" : "FAIL");
        free(r.y);
    }

    /* Test 3: backward integration, decay from t=1 to t=0 */
    {
        double y0 = exp(-1.0);
        ODEResult r = ode_solve(rhs_decay, &y0, 1.0, 0.0, NULL, 1, atol, rtol, NULL);
        double err = fabs(r.y[0] - 1.0);
        printf("ODE test 3 (backward): err=%.2e  nfe=%d  %s\n",
               err, r.nfe, err < check_tol ? "PASS" : "FAIL");
        free(r.y);
    }
}

static void test_dynmlp_gradients(RNG *r) {
    const int D = 3;
    const int H = 8;
    const double EPS = 1e-7;
    const double TOL = 1e-5;

    int np = dynmlp_nparams(D, H);
    double *theta  = vec_alloc(np);
    double *z      = vec_alloc(D);
    double *v      = vec_alloc(D);
    double *out_p  = vec_alloc(D);
    double *out_m  = vec_alloc(D);

    DynMLP net;
    dynmlp_init(&net, D, H, theta, r);
    for (int i = 0; i < D;  i++) z[i] = rng_normal(r);
    for (int i = 0; i < D;  i++) v[i] = rng_normal(r);
    double t = rng_normal(r);

    /* --- analytical VJP --- */
    double *vjp_z     = vec_zeros(D);
    double *vjp_theta = vec_zeros(np);
    dynmlp_vjp(&net, theta, z, t, v, vjp_z, vjp_theta);

    /* --- numerical VJP w.r.t. z --- */
    double *num_vjp_z = vec_alloc(D);
    for (int i = 0; i < D; i++) {
        double zi = z[i];

        z[i] = zi + EPS;
        dynmlp_forward(&net, theta, z, t, out_p);

        z[i] = zi - EPS;
        dynmlp_forward(&net, theta, z, t, out_m);

        z[i] = zi;
        num_vjp_z[i] = vec_dot(v, out_p, D) - vec_dot(v, out_m, D);
        num_vjp_z[i] /= 2.0 * EPS;
    }

    double max_err_z = 0.0;
    for (int i = 0; i < D; i++) {
        double e = fabs(vjp_z[i] - num_vjp_z[i]);
        if (e > max_err_z) max_err_z = e;
    }

    /* --- numerical VJP w.r.t. theta --- */
    double *num_vjp_theta = vec_alloc(np);
    for (int k = 0; k < np; k++) {
        double tk = theta[k];

        theta[k] = tk + EPS;
        dynmlp_forward(&net, theta, z, t, out_p);

        theta[k] = tk - EPS;
        dynmlp_forward(&net, theta, z, t, out_m);

        theta[k] = tk;
        num_vjp_theta[k] = vec_dot(v, out_p, D) - vec_dot(v, out_m, D);
        num_vjp_theta[k] /= 2.0 * EPS;
    }

    double max_err_theta = 0.0;
    for (int k = 0; k < np; k++) {
        double e = fabs(vjp_theta[k] - num_vjp_theta[k]);
        if (e > max_err_theta) max_err_theta = e;
    }

    printf("max grad error z: %.2e %s\n",
           max_err_z, max_err_z < TOL ? "PASS":"FAIL");
    printf("max grad error theta: %.2e %s\n",
           max_err_theta, max_err_theta < TOL ? "PASS" : "FAIL");

    free(theta); 
    free(z); 
    free(v);
    free(out_p);
    free(out_m);
    free(vjp_z);
    free(vjp_theta);
    free(num_vjp_z);
    free(num_vjp_theta);
}



int main(void) {
    RNG r = rng_init(42);
    test_dynmlp_gradients(&r);
    test_ode_solver();
    return 0;
}
