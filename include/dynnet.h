#pragma once

#include "utils.h"

typedef struct {
    void (*forward)(const void *cfg, const double *theta,
                    const double *in, double *out, double *workspace);
    void (*vjp)(const void *cfg, const double *theta,
                const double *in, const double *v_out,
                double *v_in, double *v_theta, double *workspace);
    int (*nparams)(const void *cfg);
    int (*workspace_size)(const void *cfg);
} LayerOps;

typedef struct DynNet {
    int num_layers;
    const LayerOps **ops;
    void **layer_configs;
    int *param_offsets;
    int *ws_offsets;
    int *in_dims;
    int total_params;
    int total_workspace;
    int v_buf_offset;
    int max_dim;
    int D;
    double current_t;
    /* build state (non-null before finalize) */
    void *_pre;
    int _pre_n, _pre_cap, _cur_dim;
} DynNet;

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
