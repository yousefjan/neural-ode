#include "cnf.h"

#include <stdlib.h>

/* Compute tr(df/dz) via central finite differences.
 * Uses z_p (D) as scratch; writes intermediate outputs to f_p and f_m (D). */
static double compute_trace_fd(DynNet *net, const double *theta,
                                const double *z, double t, double *ws,
                                double eps, double *z_p, double *f_p, double *f_m) {
    int D = net->D;
    double tr = 0.0;
    for (int i = 0; i < D; i++) {
        vec_copy(z, z_p, D);
        z_p[i] += eps;
        dynnet_forward(net, theta, z_p, t, f_p, ws);
        z_p[i] = z[i] - eps;
        dynnet_forward(net, theta, z_p, t, f_m, ws);
        tr += (f_p[i] - f_m[i]) / (2.0 * eps);
    }
    return tr;
}

typedef struct {
    DynNet      *net;
    const double *theta;
    int          D;
    double      *ws;
    double       eps;
    double      *z_p, *f_p, *f_m;
} CNFFwdCtx;

static void cnf_fwd_rhs(const double *aug, double t, const double *params,
                         int aug_dim, double *out, void *ctx) {
    (void)params; (void)aug_dim;
    CNFFwdCtx *c = (CNFFwdCtx *)ctx;
    int D = c->D;

    dynnet_forward(c->net, c->theta, aug, t, out, c->ws);
    out[D] = -compute_trace_fd(c->net, c->theta, aug, t, c->ws,
                                c->eps, c->z_p, c->f_p, c->f_m);
}

typedef struct {
    DynNet      *net;
    const double *theta;
    int          D;
    int          nparams;
    double      *ws;
    double       eps;
    double       a_logp;
    double *z_oj;
    double *z_p2, *f_p2, *f_m2;
    double *neg_a_z;
    double *dg_tmp;
    double *da_z_tmp;
    double *v_ei;
    double *vjp_z_tmp;
    double *vjp_th_p;
    double *vjp_th_m;
} CNFAdjCtx;

static void cnf_adj_rhs(const double *aug, double t, const double *params,
                         int aug_dim, double *out, void *ctx) {
    (void)params; (void)aug_dim;
    CNFAdjCtx *cc = (CNFAdjCtx *)ctx;
    int D       = cc->D;
    int nparams = cc->nparams;
    const double *z   = aug;
    const double *a_z = aug + D;

    dynnet_forward(cc->net, cc->theta, z, t, out, cc->ws);

    for (int i = 0; i < D; i++) cc->neg_a_z[i] = -a_z[i];
    vec_zero(cc->dg_tmp, nparams);
    dynnet_vjp(cc->net, cc->theta, z, t, cc->neg_a_z,
               out + D, cc->dg_tmp, cc->ws);
    vec_copy(cc->dg_tmp, out + 2 * D, nparams);

    if (cc->a_logp == 0.0) return;

    for (int j = 0; j < D; j++) {
        vec_copy(z, cc->z_oj, D);
        cc->z_oj[j] += cc->eps;
        double tr_p = compute_trace_fd(cc->net, cc->theta, cc->z_oj, t, cc->ws,
                                       cc->eps, cc->z_p2, cc->f_p2, cc->f_m2);
        cc->z_oj[j] = z[j] - cc->eps;
        double tr_m = compute_trace_fd(cc->net, cc->theta, cc->z_oj, t, cc->ws,
                                       cc->eps, cc->z_p2, cc->f_p2, cc->f_m2);
        out[D + j] += cc->a_logp * (tr_p - tr_m) / (2.0 * cc->eps);
    }

    vec_zero(cc->v_ei, D);
    for (int i = 0; i < D; i++) {
        cc->v_ei[i] = 1.0;

        vec_copy(z, cc->z_oj, D);
        cc->z_oj[i] += cc->eps;
        vec_zero(cc->vjp_th_p, nparams);
        dynnet_vjp(cc->net, cc->theta, cc->z_oj, t, cc->v_ei,
                   cc->vjp_z_tmp, cc->vjp_th_p, cc->ws);

        cc->z_oj[i] = z[i] - cc->eps;
        vec_zero(cc->vjp_th_m, nparams);
        dynnet_vjp(cc->net, cc->theta, cc->z_oj, t, cc->v_ei,
                   cc->vjp_z_tmp, cc->vjp_th_m, cc->ws);

        cc->v_ei[i] = 0.0;

        double scale = cc->a_logp / (2.0 * cc->eps);
        for (int k = 0; k < nparams; k++)
            out[2 * D + k] += scale * (cc->vjp_th_p[k] - cc->vjp_th_m[k]);
    }
}

/* ---- Public API ---- */

void cnf_init(CNF *cnf, int D, int H, double *theta, RNG *r) {
    DynNet *net = dynnet_create(D);
    dynnet_add_time_concat(net);
    dynnet_add_linear(net, H);
    dynnet_add_tanh(net);
    dynnet_add_linear(net, D);
    dynnet_finalize(net);
    dynnet_init_params(net, theta, r);
    cnf->net       = net;
    cnf->nparams   = net->total_params;
    cnf->trace_eps = 1e-5;
}

void cnf_free(CNF *cnf) {
    dynnet_free(cnf->net);
    cnf->net = NULL;
}

CNFSampleResult cnf_sample(CNF *cnf, const double *theta,
                            const double *z0, double t0, double t1,
                            double atol, double rtol) {
    int D = cnf->net->D;
    double *ws  = vec_alloc(cnf->net->total_workspace);
    double *z_p = vec_alloc(D), *f_p = vec_alloc(D), *f_m = vec_alloc(D);
    CNFFwdCtx ctx = { cnf->net, theta, D, ws, cnf->trace_eps, z_p, f_p, f_m };

    double *aug0 = vec_zeros(D + 1);
    vec_copy(z0, aug0, D);

    ODEResult res = ode_solve(cnf_fwd_rhs, aug0, t0, t1,
                               NULL, D + 1, atol, rtol, &ctx);
    free(aug0);
    free(z_p); free(f_p); free(f_m);
    free(ws);

    CNFSampleResult out;
    out.z1 = vec_alloc(D);
    vec_copy(res.y, out.z1, D);
    out.delta_logp = res.y[D];
    out.nfe = res.nfe;
    free(res.y);
    return out;
}

CNFLogProbResult cnf_log_prob(CNF *cnf, const double *theta,
                               const double *z1, double t0, double t1,
                               double atol, double rtol) {
    int D = cnf->net->D;
    double *ws  = vec_alloc(cnf->net->total_workspace);
    double *z_p = vec_alloc(D), *f_p = vec_alloc(D), *f_m = vec_alloc(D);
    CNFFwdCtx ctx = { cnf->net, theta, D, ws, cnf->trace_eps, z_p, f_p, f_m };

    double *aug1 = vec_zeros(D + 1);
    vec_copy(z1, aug1, D);
    ODEResult res = ode_solve(cnf_fwd_rhs, aug1, t1, t0,
                               NULL, D + 1, atol, rtol, &ctx);
    free(aug1);
    free(z_p); free(f_p); free(f_m);
    free(ws);

    CNFLogProbResult out;
    out.z0 = vec_alloc(D);
    vec_copy(res.y, out.z0, D);
    out.delta_logp = -res.y[D];
    out.nfe = res.nfe;
    free(res.y);
    return out;
}

CNFBackwardResult cnf_backward(CNF *cnf, const double *theta,
                                const double *z1,
                                double dL_dlogp, const double *dL_dz1_in,
                                double t0, double t1,
                                double atol, double rtol) {
    int D       = cnf->net->D;
    int nparams = cnf->nparams;
    int aug_dim = 2 * D + nparams;

    double *ws = vec_alloc(cnf->net->total_workspace);

    CNFAdjCtx ctx;
    ctx.net     = cnf->net;
    ctx.theta   = theta;
    ctx.D       = D;
    ctx.nparams = nparams;
    ctx.ws      = ws;
    ctx.eps     = cnf->trace_eps;
    ctx.a_logp  = dL_dlogp;
    ctx.z_oj    = vec_alloc(D);
    ctx.z_p2    = vec_alloc(D);
    ctx.f_p2    = vec_alloc(D);
    ctx.f_m2    = vec_alloc(D);
    ctx.neg_a_z = vec_alloc(D);
    ctx.dg_tmp  = vec_alloc(nparams);
    ctx.da_z_tmp = vec_alloc(D);
    ctx.v_ei    = vec_zeros(D);
    ctx.vjp_z_tmp = vec_alloc(D);
    ctx.vjp_th_p  = vec_alloc(nparams);
    ctx.vjp_th_m  = vec_alloc(nparams);

    double *aug = vec_zeros(aug_dim);
    vec_copy(z1, aug, D);
    vec_copy(dL_dz1_in, aug + D, D);

    ODEResult res = ode_solve(cnf_adj_rhs, aug, t1, t0,
                               NULL, aug_dim, atol, rtol, &ctx);
    free(aug);

    CNFBackwardResult out;
    out.dL_dz0    = vec_alloc(D);
    out.dL_dtheta = vec_alloc(nparams);
    vec_copy(res.y + D,     out.dL_dz0,    D);
    vec_copy(res.y + 2 * D, out.dL_dtheta, nparams);
    out.nfe = res.nfe;
    free(res.y);

    free(ctx.z_oj);  free(ctx.z_p2); free(ctx.f_p2); free(ctx.f_m2);
    free(ctx.neg_a_z); free(ctx.dg_tmp); free(ctx.da_z_tmp);
    free(ctx.v_ei); free(ctx.vjp_z_tmp); free(ctx.vjp_th_p); free(ctx.vjp_th_m);
    free(ws);
    return out;
}
