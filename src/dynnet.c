#include "dynnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum {
    PRE_LINEAR, PRE_TANH, PRE_SOFTPLUS, PRE_SWISH, PRE_LAYERNORM,
    PRE_TIME_CONCAT, PRE_RESIDUAL_BEGIN, PRE_RESIDUAL_END
} PreType;

typedef struct { int type; int param; } PreLayer;


typedef struct { int in_dim; int out_dim; } LinearCfg;
typedef struct { int dim; } ActCfg;
typedef struct { int in_dim; double *t_ptr; } TimeConcatCfg;
typedef struct { DynNet *subnet; int dim; double *t_ptr; } ResidualCfg;


static const LayerOps linear_ops;
static const LayerOps tanh_ops;
static const LayerOps softplus_ops;
static const LayerOps swish_ops;
static const LayerOps layernorm_ops;
static const LayerOps time_concat_ops;
static const LayerOps residual_ops;

static void xavier_init(double *w, int fan_in, int fan_out, RNG *r) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    int n = fan_in * fan_out;
    for (int i = 0; i < n; i++)
        w[i] = (2.0 * rng_uniform(r) - 1.0) * limit;
}


static void linear_forward(const void *cfg, const double *theta,
                            const double *in, double *out, double *workspace) {
    (void)workspace;
    const LinearCfg *c = (const LinearCfg *)cfg;
    const double *W = theta;
    const double *b = theta + c->in_dim * c->out_dim;
    mat_vec(W, in, out, c->out_dim, c->in_dim);
    vec_add_scaled(out, 1.0, b, c->out_dim);
}

static void linear_vjp(const void *cfg, const double *theta,
                        const double *in, const double *v_out,
                        double *v_in, double *v_theta, double *workspace) {
    (void)workspace;
    const LinearCfg *c = (const LinearCfg *)cfg;
    const double *W = theta;
    double *dW = v_theta;
    double *db = v_theta + c->in_dim * c->out_dim;
    /* Compute dW and db before zeroing v_in (safe if v_in aliases in) */
    mat_outer_add(dW, 1.0, v_out, in, c->out_dim, c->in_dim);
    vec_add_scaled(db, 1.0, v_out, c->out_dim);
    vec_zero(v_in, c->in_dim);
    mat_vec_T(W, v_out, v_in, c->out_dim, c->in_dim);
}

static int linear_nparams(const void *cfg) {
    const LinearCfg *c = (const LinearCfg *)cfg;
    return c->in_dim * c->out_dim + c->out_dim;
}

static int linear_ws(const void *cfg) { (void)cfg; return 0; }


static void tanh_forward(const void *cfg, const double *theta,
                          const double *in, double *out, double *workspace) {
    (void)theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++) out[i] = tanh(in[i]);
}

static void tanh_vjp(const void *cfg, const double *theta,
                      const double *in, const double *v_out,
                      double *v_in, double *v_theta, double *workspace) {
    (void)theta; (void)v_theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++) {
        double h = tanh(in[i]);
        v_in[i] = (1.0 - h * h) * v_out[i];
    }
}

static int act_nparams(const void *cfg) { (void)cfg; return 0; }
static int act_ws(const void *cfg) { (void)cfg; return 0; }


static inline double softplus(double x) {
    return x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x));
}

static inline double sigmoid(double x) {
    return x > 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}

static void softplus_forward(const void *cfg, const double *theta,
                              const double *in, double *out, double *workspace) {
    (void)theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++) out[i] = softplus(in[i]);
}

static void softplus_vjp(const void *cfg, const double *theta,
                          const double *in, const double *v_out,
                          double *v_in, double *v_theta, double *workspace) {
    (void)theta; (void)v_theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++)
        v_in[i] = sigmoid(in[i]) * v_out[i];
}


static void swish_forward(const void *cfg, const double *theta,
                           const double *in, double *out, double *workspace) {
    (void)theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++) out[i] = in[i] * sigmoid(in[i]);
}

static void swish_vjp(const void *cfg, const double *theta,
                       const double *in, const double *v_out,
                       double *v_in, double *v_theta, double *workspace) {
    (void)theta; (void)v_theta; (void)workspace;
    const ActCfg *c = (const ActCfg *)cfg;
    for (int i = 0; i < c->dim; i++) {
        double s = sigmoid(in[i]);
        v_in[i] = (s + in[i] * s * (1.0 - s)) * v_out[i];
    }
}


static int layernorm_ws(const void *cfg) {
    const ActCfg *c = (const ActCfg *)cfg;
    return 2 + 2 * c->dim;
}

static int layernorm_nparams(const void *cfg) {
    const ActCfg *c = (const ActCfg *)cfg;
    return 2 * c->dim;
}

static void layernorm_forward(const void *cfg, const double *theta,
                               const double *in, double *out, double *workspace) {
    const ActCfg *c = (const ActCfg *)cfg;
    int dim = c->dim;
    const double eps = 1e-5;
    const double *gamma = theta;
    const double *beta  = theta + dim;

    double mean = 0.0;
    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= (double)dim;

    double var = 0.0;
    for (int i = 0; i < dim; i++) { double d = in[i] - mean; var += d * d; }
    var /= (double)dim;

    workspace[0] = mean;
    workspace[1] = var;
    double *x_hat = workspace + 2;
    double std = sqrt(var + eps);

    for (int i = 0; i < dim; i++) {
        x_hat[i] = (in[i] - mean) / std;
        out[i] = gamma[i] * x_hat[i] + beta[i];
    }
}

static void layernorm_vjp(const void *cfg, const double *theta,
                           const double *in, const double *v_out,
                           double *v_in, double *v_theta, double *workspace) {
    (void)in;
    const ActCfg *c = (const ActCfg *)cfg;
    int dim = c->dim;
    const double eps = 1e-5;
    const double *gamma = theta;
    double *dg  = v_theta;
    double *db  = v_theta + dim;

    double mean  = workspace[0]; (void)mean;
    double var   = workspace[1];
    double std   = sqrt(var + eps);
    double *x_hat  = workspace + 2;
    double *dx_hat = workspace + 2 + dim;

    for (int i = 0; i < dim; i++) {
        dg[i] += v_out[i] * x_hat[i];
        db[i] += v_out[i];
        dx_hat[i] = v_out[i] * gamma[i];
    }

    double dvar = 0.0, sum_dx = 0.0;
    for (int i = 0; i < dim; i++) {
        dvar += dx_hat[i] * x_hat[i];
        sum_dx += dx_hat[i];
    }
    dvar *= -0.5 / (std * std);
    double dmean = -sum_dx / std;

    for (int i = 0; i < dim; i++) {
        v_in[i] = dx_hat[i] / std
                  + dvar * 2.0 * x_hat[i] * std / (double)dim
                  + dmean / (double)dim;
    }
}


static void time_concat_forward(const void *cfg, const double *theta,
                                 const double *in, double *out, double *workspace) {
    (void)theta; (void)workspace;
    const TimeConcatCfg *c = (const TimeConcatCfg *)cfg;
    vec_copy(in, out, c->in_dim);
    out[c->in_dim] = *c->t_ptr;
}

static void time_concat_vjp(const void *cfg, const double *theta,
                              const double *in, const double *v_out,
                              double *v_in, double *v_theta, double *workspace) {
    (void)theta; (void)in; (void)v_theta; (void)workspace;
    const TimeConcatCfg *c = (const TimeConcatCfg *)cfg;
    vec_copy(v_out, v_in, c->in_dim);
}

static int time_concat_nparams(const void *cfg) { (void)cfg; return 0; }
static int time_concat_ws(const void *cfg) { (void)cfg; return 0; }


static void residual_forward(const void *cfg, const double *theta,
                              const double *in, double *out, double *workspace) {
    const ResidualCfg *c = (const ResidualCfg *)cfg;
    dynnet_forward(c->subnet, theta, in, *c->t_ptr, out, workspace);
    for (int i = 0; i < c->dim; i++) out[i] += in[i];
}

static void residual_vjp(const void *cfg, const double *theta,
                          const double *in, const double *v_out,
                          double *v_in, double *v_theta, double *workspace) {
    const ResidualCfg *c = (const ResidualCfg *)cfg;
    dynnet_vjp(c->subnet, theta, in, *c->t_ptr, v_out, v_in, v_theta, workspace);
    for (int i = 0; i < c->dim; i++) v_in[i] += v_out[i];
}

static int residual_nparams(const void *cfg) {
    const ResidualCfg *c = (const ResidualCfg *)cfg;
    return c->subnet->total_params;
}

static int residual_ws(const void *cfg) {
    const ResidualCfg *c = (const ResidualCfg *)cfg;
    return c->subnet->total_workspace;
}


static const LayerOps linear_ops     = { linear_forward,      linear_vjp,      linear_nparams,      linear_ws      };
static const LayerOps tanh_ops       = { tanh_forward,        tanh_vjp,        act_nparams,         act_ws         };
static const LayerOps softplus_ops   = { softplus_forward,    softplus_vjp,    act_nparams,         act_ws         };
static const LayerOps swish_ops      = { swish_forward,       swish_vjp,       act_nparams,         act_ws         };
static const LayerOps layernorm_ops  = { layernorm_forward,   layernorm_vjp,   layernorm_nparams,   layernorm_ws   };
static const LayerOps time_concat_ops = { time_concat_forward, time_concat_vjp, time_concat_nparams, time_concat_ws };
static const LayerOps residual_ops   = { residual_forward,    residual_vjp,    residual_nparams,    residual_ws    };


DynNet *dynnet_create(int D) {
    DynNet *net = (DynNet *)xcalloc(1, sizeof(DynNet));
    net->D = D;
    net->_cur_dim = D;
    return net;
}

static void pre_append(DynNet *net, int type, int param) {
    if (net->_pre_n == net->_pre_cap) {
        int new_cap = net->_pre_cap ? net->_pre_cap * 2 : 8;
        PreLayer *old = (PreLayer *)net->_pre;
        PreLayer *new_buf = (PreLayer *)xmalloc((size_t)new_cap * sizeof(PreLayer));
        if (old) {
            memcpy(new_buf, old, (size_t)net->_pre_n * sizeof(PreLayer));
            free(old);
        }
        net->_pre = new_buf;
        net->_pre_cap = new_cap;
    }
    ((PreLayer *)net->_pre)[net->_pre_n++] = (PreLayer){ type, param };
}

void dynnet_add_linear(DynNet *net, int out_dim) {
    pre_append(net, PRE_LINEAR, out_dim);
    net->_cur_dim = out_dim;
}

void dynnet_add_tanh(DynNet *net) {
    pre_append(net, PRE_TANH, 0);
}

void dynnet_add_softplus(DynNet *net) {
    pre_append(net, PRE_SOFTPLUS, 0);
}

void dynnet_add_swish(DynNet *net) {
    pre_append(net, PRE_SWISH, 0);
}

void dynnet_add_layernorm(DynNet *net) {
    pre_append(net, PRE_LAYERNORM, 0);
}

void dynnet_add_residual_begin(DynNet *net) {
    pre_append(net, PRE_RESIDUAL_BEGIN, 0);
}

void dynnet_add_residual_end(DynNet *net) {
    pre_append(net, PRE_RESIDUAL_END, 0);
}

void dynnet_add_time_concat(DynNet *net) {
    pre_append(net, PRE_TIME_CONCAT, 0);
    net->_cur_dim++;
}


typedef struct {
    const LayerOps *ops;
    void *cfg;
    int in_dim;
    int out_dim;
} LInfo;

static int find_matching_end(PreLayer *pre, int n, int start) {
    int depth = 0;
    for (int i = start; i < n; i++) {
        if (pre[i].type == PRE_RESIDUAL_BEGIN) depth++;
        else if (pre[i].type == PRE_RESIDUAL_END) {
            if (--depth == 0) return i;
        }
    }
    return -1;
}

static LInfo *build_linfo(PreLayer *pre, int n, int D_in,
                           double *t_ptr, int *out_n, int *out_D) {
    int cap = n > 0 ? n : 1;
    LInfo *linfo = (LInfo *)xmalloc((size_t)cap * sizeof(LInfo));
    int cnt = 0;
    int cur = D_in;

    for (int i = 0; i < n; ) {
        if (cnt == cap) {
            cap *= 2;
            LInfo *tmp = (LInfo *)xmalloc((size_t)cap * sizeof(LInfo));
            memcpy(tmp, linfo, (size_t)cnt * sizeof(LInfo));
            free(linfo);
            linfo = tmp;
        }
        PreLayer p = pre[i];

        if (p.type == PRE_LINEAR) {
            LinearCfg *cfg = (LinearCfg *)xmalloc(sizeof(LinearCfg));
            cfg->in_dim = cur; cfg->out_dim = p.param;
            linfo[cnt++] = (LInfo){ &linear_ops, cfg, cur, p.param };
            cur = p.param; i++;

        } else if (p.type == PRE_TANH) {
            ActCfg *cfg = (ActCfg *)xmalloc(sizeof(ActCfg));
            cfg->dim = cur;
            linfo[cnt++] = (LInfo){ &tanh_ops, cfg, cur, cur };
            i++;

        } else if (p.type == PRE_SOFTPLUS) {
            ActCfg *cfg = (ActCfg *)xmalloc(sizeof(ActCfg));
            cfg->dim = cur;
            linfo[cnt++] = (LInfo){ &softplus_ops, cfg, cur, cur };
            i++;

        } else if (p.type == PRE_SWISH) {
            ActCfg *cfg = (ActCfg *)xmalloc(sizeof(ActCfg));
            cfg->dim = cur;
            linfo[cnt++] = (LInfo){ &swish_ops, cfg, cur, cur };
            i++;

        } else if (p.type == PRE_LAYERNORM) {
            ActCfg *cfg = (ActCfg *)xmalloc(sizeof(ActCfg));
            cfg->dim = cur;
            linfo[cnt++] = (LInfo){ &layernorm_ops, cfg, cur, cur };
            i++;

        } else if (p.type == PRE_TIME_CONCAT) {
            TimeConcatCfg *cfg = (TimeConcatCfg *)xmalloc(sizeof(TimeConcatCfg));
            cfg->in_dim = cur; cfg->t_ptr = t_ptr;
            linfo[cnt++] = (LInfo){ &time_concat_ops, cfg, cur, cur + 1 };
            cur++; i++;

        } else if (p.type == PRE_RESIDUAL_BEGIN) {
            int end = find_matching_end(pre, n, i);
            if (end < 0) {
                fprintf(stderr, "dynnet: unmatched residual_begin\n"); abort();
            }
            DynNet *subnet = (DynNet *)xcalloc(1, sizeof(DynNet));
            subnet->D = cur;

            int sub_n, sub_D;
            LInfo *sub_linfo = build_linfo(pre + i + 1, end - i - 1,
                                           cur, &subnet->current_t, &sub_n, &sub_D);
            if (sub_D != cur) {
                fprintf(stderr, "dynnet: residual in_dim=%d out_dim=%d mismatch\n", cur, sub_D);
                abort();
            }

            subnet->num_layers = sub_n;
            subnet->ops          = (const LayerOps **)xmalloc((size_t)sub_n * sizeof(LayerOps *));
            subnet->layer_configs = (void **)xmalloc((size_t)sub_n * sizeof(void *));
            subnet->param_offsets = (int *)xmalloc((size_t)(sub_n + 1) * sizeof(int));
            subnet->ws_offsets    = (int *)xmalloc((size_t)(sub_n + 1) * sizeof(int));
            subnet->in_dims       = (int *)xmalloc((size_t)sub_n * sizeof(int));

            int p_off = 0, w_off = 0, max_d = cur;
            for (int k = 0; k < sub_n; k++) {
                subnet->ops[k]          = sub_linfo[k].ops;
                subnet->layer_configs[k] = sub_linfo[k].cfg;
                subnet->in_dims[k]       = sub_linfo[k].in_dim;
                subnet->param_offsets[k] = p_off;
                subnet->ws_offsets[k]    = w_off;
                p_off += sub_linfo[k].ops->nparams(sub_linfo[k].cfg);
                w_off += sub_linfo[k].in_dim + sub_linfo[k].ops->workspace_size(sub_linfo[k].cfg);
                if (sub_linfo[k].in_dim  > max_d) max_d = sub_linfo[k].in_dim;
                if (sub_linfo[k].out_dim > max_d) max_d = sub_linfo[k].out_dim;
            }
            subnet->total_params   = p_off;
            subnet->v_buf_offset   = w_off;
            subnet->max_dim        = max_d;
            subnet->total_workspace = w_off + 2 * max_d;
            free(sub_linfo);

            ResidualCfg *cfg = (ResidualCfg *)xmalloc(sizeof(ResidualCfg));
            cfg->subnet = subnet; cfg->dim = cur; cfg->t_ptr = t_ptr;
            linfo[cnt++] = (LInfo){ &residual_ops, cfg, cur, cur };
            i = end + 1;

        } else if (p.type == PRE_RESIDUAL_END) {
            i++; /* handled by parent */

        } else {
            fprintf(stderr, "dynnet: unknown pre-layer type %d\n", p.type); abort();
        }
    }

    *out_n = cnt;
    *out_D = cur;
    return linfo;
}

void dynnet_finalize(DynNet *net) {
    int out_n, out_D;
    LInfo *linfo = build_linfo((PreLayer *)net->_pre, net->_pre_n,
                               net->D, &net->current_t, &out_n, &out_D);
    if (out_D != net->D) {
        fprintf(stderr, "dynnet: final dim %d != D %d\n", out_D, net->D); abort();
    }

    net->num_layers   = out_n;
    net->ops          = (const LayerOps **)xmalloc((size_t)out_n * sizeof(LayerOps *));
    net->layer_configs = (void **)xmalloc((size_t)out_n * sizeof(void *));
    net->param_offsets = (int *)xmalloc((size_t)(out_n + 1) * sizeof(int));
    net->ws_offsets    = (int *)xmalloc((size_t)(out_n + 1) * sizeof(int));
    net->in_dims       = (int *)xmalloc((size_t)out_n * sizeof(int));

    int p_off = 0, w_off = 0, max_d = net->D;
    for (int i = 0; i < out_n; i++) {
        net->ops[i]           = linfo[i].ops;
        net->layer_configs[i]  = linfo[i].cfg;
        net->in_dims[i]        = linfo[i].in_dim;
        net->param_offsets[i]  = p_off;
        net->ws_offsets[i]     = w_off;
        p_off += linfo[i].ops->nparams(linfo[i].cfg);
        w_off += linfo[i].in_dim + linfo[i].ops->workspace_size(linfo[i].cfg);
        if (linfo[i].in_dim  > max_d) max_d = linfo[i].in_dim;
        if (linfo[i].out_dim > max_d) max_d = linfo[i].out_dim;
    }
    net->total_params    = p_off;
    net->v_buf_offset    = w_off;
    net->max_dim         = max_d;
    net->total_workspace = w_off + 2 * max_d;

    free(linfo);
    free(net->_pre);
    net->_pre = NULL;
    net->_pre_n = net->_pre_cap = 0;
}


void dynnet_init_params(DynNet *net, double *theta, RNG *r) {
    for (int i = 0; i < net->num_layers; i++) {
        double *th = theta + net->param_offsets[i];
        if (net->ops[i] == &linear_ops) {
            LinearCfg *c = (LinearCfg *)net->layer_configs[i];
            xavier_init(th, c->in_dim, c->out_dim, r);
            vec_zero(th + c->in_dim * c->out_dim, c->out_dim);
        } else if (net->ops[i] == &layernorm_ops) {
            ActCfg *c = (ActCfg *)net->layer_configs[i];
            for (int j = 0; j < c->dim; j++) th[j] = 1.0;
            vec_zero(th + c->dim, c->dim);
        } else if (net->ops[i] == &residual_ops) {
            ResidualCfg *c = (ResidualCfg *)net->layer_configs[i];
            dynnet_init_params(c->subnet, th, r);
        }
    }
}


void dynnet_free(DynNet *net) {
    for (int i = 0; i < net->num_layers; i++) {
        if (net->ops[i] == &residual_ops) {
            ResidualCfg *c = (ResidualCfg *)net->layer_configs[i];
            dynnet_free(c->subnet);
        }
        free(net->layer_configs[i]);
    }
    free((void *)net->ops);
    free(net->layer_configs);
    free(net->param_offsets);
    free(net->ws_offsets);
    free(net->in_dims);
    if (net->_pre) free(net->_pre);
    free(net);
}


void dynnet_forward(DynNet *net, const double *theta,
                    const double *z, double t, double *out, double *workspace) {
    net->current_t = t;
    int N = net->num_layers;

    vec_copy(z, workspace + net->ws_offsets[0], net->in_dims[0]);

    for (int i = 0; i < N; i++) {
        double *in_i    = workspace + net->ws_offsets[i];
        double *out_i   = (i < N - 1) ? workspace + net->ws_offsets[i + 1] : out;
        double *scratch = in_i + net->in_dims[i];
        net->ops[i]->forward(net->layer_configs[i],
                              theta + net->param_offsets[i],
                              in_i, out_i, scratch);
    }
}


void dynnet_vjp(DynNet *net, const double *theta,
                const double *z, double t, const double *v,
                double *vjp_z, double *vjp_theta, double *workspace) {
    net->current_t = t;
    int N = net->num_layers;

    vec_copy(z, workspace + net->ws_offsets[0], net->in_dims[0]);
    double *throwaway = workspace + net->v_buf_offset;
    for (int i = 0; i < N; i++) {
        double *in_i    = workspace + net->ws_offsets[i];
        double *out_i   = (i < N - 1) ? workspace + net->ws_offsets[i + 1] : throwaway;
        double *scratch = in_i + net->in_dims[i];
        net->ops[i]->forward(net->layer_configs[i],
                              theta + net->param_offsets[i],
                              in_i, out_i, scratch);
    }

    double *v_bufs[2] = {
        workspace + net->v_buf_offset,
        workspace + net->v_buf_offset + net->max_dim
    };
    int cur = 0;
    vec_copy(v, v_bufs[cur], net->D);

    for (int i = N - 1; i >= 0; i--) {
        double *in_i    = workspace + net->ws_offsets[i];
        double *scratch = in_i + net->in_dims[i];
        double *v_in_i  = (i == 0) ? vjp_z : v_bufs[1 - cur];

        vec_zero(v_in_i, net->in_dims[i]);
        net->ops[i]->vjp(net->layer_configs[i],
                          theta + net->param_offsets[i],
                          in_i, v_bufs[cur], v_in_i,
                          vjp_theta + net->param_offsets[i],
                          scratch);
        if (i > 0) cur ^= 1;
    }
}
