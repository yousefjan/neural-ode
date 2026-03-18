#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct { uint64_t state; } RNG;

typedef struct {
    double *x;
    double *h_pre;
    double *h;
    double *dh;
    double *dh_pre;
    double *dx;
    double *neg_a;
    double *vjp_z;
    double *vjp_theta;
} Workspace;

RNG      rng_init(uint64_t seed);
uint64_t rng_next(RNG *r);
double   rng_uniform(RNG *r);
double   rng_normal(RNG *r);

void    *xmalloc(size_t n);
void    *xcalloc(size_t count, size_t size);

double  *vec_alloc(int n);
double  *vec_zeros(int n);
void     vec_zero(double *v, int n);
void     vec_copy(const double *src, double *dst, int n);
void     vec_add_scaled(double *dst, double alpha, const double *v, int n);
double   vec_dot(const double *a, const double *b, int n);
void     mat_vec(const double *M, const double *x, double *dst, int rows, int cols);
void     mat_vec_T(const double *M, const double *v, double *dst, int rows, int cols);
void     mat_outer_add(double *M, double alpha, const double *a, const double *b, int rows, int cols);

Workspace workspace_alloc(int D, int H, int nparams);
void      workspace_free(Workspace *ws);
