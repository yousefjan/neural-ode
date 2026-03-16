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
    return 0;
}
