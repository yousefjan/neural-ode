#include "dynmlp.h"

#include <math.h>

static void xavier_init(double *w, int fan_in, int fan_out, RNG *r) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    int n = fan_in * fan_out;
    for (int i = 0; i < n; i++)
        w[i] = (2.0 * rng_uniform(r) - 1.0) * limit;
}

int dynmlp_nparams(int D, int H) {
    return (D + 1) * H + H + H * D + D;
}

void dynmlp_init(DynMLP *net, int D, int H, double *theta, RNG *r) {
    net->D = D;
    net->H = H;
    net->nparams = dynmlp_nparams(D, H);
    xavier_init(theta + DYNMLP_W1(D, H), D + 1, H, r);
    vec_zero(theta + DYNMLP_b1(D, H), H);
    xavier_init(theta + DYNMLP_W2(D, H), H, D, r);
    vec_zero(theta + DYNMLP_b2(D, H), D);
}

void dynmlp_forward(const DynMLP *net, const double *theta,
                    const double *z, double t, double *out,
                    Workspace *ws) {
    int D = net->D, H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);
    const double *b2 = theta + DYNMLP_b2(D, H);

    double *x     = ws->x;
    double *h_pre = ws->h_pre;
    double *h     = ws->h;

    vec_copy(z, x, D);
    x[D] = t;

    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);

    for (int i = 0; i < H; i++) h[i] = tanh(h_pre[i]);

    mat_vec(W2, h, out, D, H);
    vec_add_scaled(out, 1.0, b2, D);
}

void dynmlp_vjp(const DynMLP *net, const double *theta,
                const double *z, double t, const double *v,
                double *vjp_z, double *vjp_theta,
                Workspace *ws) {
    int D = net->D;
    int H = net->H;
    const double *W1 = theta + DYNMLP_W1(D, H);
    const double *b1 = theta + DYNMLP_b1(D, H);
    const double *W2 = theta + DYNMLP_W2(D, H);
    double *dW1 = vjp_theta + DYNMLP_W1(D, H);
    double *db1 = vjp_theta + DYNMLP_b1(D, H);
    double *dW2 = vjp_theta + DYNMLP_W2(D, H);
    double *db2 = vjp_theta + DYNMLP_b2(D, H);

    double *x     = ws->x;
    double *h_pre = ws->h_pre;
    double *h     = ws->h;
    vec_copy(z, x, D);
    x[D] = t;
    mat_vec(W1, x, h_pre, H, D + 1);
    vec_add_scaled(h_pre, 1.0, b1, H);

    for (int i = 0; i < H; i++)
        h[i] = tanh(h_pre[i]);

    double *dh     = ws->dh;
    double *dh_pre = ws->dh_pre;
    double *dx     = ws->dx;
    vec_zero(dh, H);
    vec_zero(dx, D + 1);

    mat_vec_T(W2, v, dh, D, H);
    mat_outer_add(dW2, 1.0, v, h, D, H);
    vec_add_scaled(db2, 1.0, v, D);

    for (int i = 0; i < H; i++)
        dh_pre[i] = (1.0 - h[i] * h[i]) * dh[i];

    mat_vec_T(W1, dh_pre, dx, H, D + 1);
    mat_outer_add(dW1, 1.0, dh_pre, x, H, D + 1);
    vec_add_scaled(db1, 1.0, dh_pre, H);

    vec_copy(dx, vjp_z, D);
}
