#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

RNG rng_init(uint64_t seed) {
    RNG r;
    r.state = seed ? seed : 1;
    return r;
}

uint64_t rng_next(RNG *r) {
    uint64_t x = r->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->state = x;
    return x;
}

double rng_uniform(RNG *r) {
    return (double)(rng_next(r) >> 11) / (double)(UINT64_C(1) << 53);
}

double rng_normal(RNG *r) {
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "fatal: malloc(%zu) failed\n", n); abort(); }
    return p;
}

void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    if (!p) { fprintf(stderr, "fatal: calloc(%zu, %zu) failed\n", count, size); abort(); }
    return p;
}

double *vec_alloc(int n)  { return (double *)xmalloc((size_t)n * sizeof(double)); }
double *vec_zeros(int n)  { return (double *)xcalloc((size_t)n, sizeof(double)); }
void    vec_zero(double *v, int n) { memset(v, 0, (size_t)n * sizeof(double)); }
void    vec_copy(const double *src, double *dst, int n) { memcpy(dst, src, (size_t)n * sizeof(double)); }

void vec_add_scaled(double *dst, double alpha, const double *v, int n) {
    for (int i = 0; i < n; i++) dst[i] += alpha * v[i];
}

double vec_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void mat_vec(const double *M, const double *x, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int j = 0; j < cols; j++) s += M[i * cols + j] * x[j];
        dst[i] = s;
    }
}

void mat_vec_T(const double *M, const double *v, double *dst, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[j] += M[i * cols + j] * v[i];
}

void mat_outer_add(double *M, double alpha,
                   const double *a, const double *b, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M[i * cols + j] += alpha * a[i] * b[j];
}

Workspace workspace_alloc(int D, int H, int nparams) {
    Workspace ws;
    ws.x = vec_alloc(D + 1);
    ws.h_pre = vec_alloc(H);
    ws.h = vec_alloc(H);
    ws.dh = vec_alloc(H);
    ws.dh_pre = vec_alloc(H);
    ws.dx = vec_alloc(D + 1);
    ws.neg_a = vec_alloc(D);
    ws.vjp_z = vec_alloc(D);
    ws.vjp_theta = vec_alloc(nparams);
    return ws;
}

void workspace_free(Workspace *ws) {
    free(ws->x);
    free(ws->h_pre);
    free(ws->h);
    free(ws->dh);
    free(ws->dh_pre);
    free(ws->dx);
    free(ws->neg_a);
    free(ws->vjp_z);
    free(ws->vjp_theta);
}
