#include <stdio.h>
#include <stdlib.h>
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


