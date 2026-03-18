#pragma once

#include "utils.h"

void test_ode_solver(void);
void test_dynnet_gradients(RNG *r);
void test_adjoint_gradients(RNG *r);
void test_multi_obs_adjoint(RNG *r);
void test_training(RNG *r);
void test_dynnet_deep_gradients(RNG *r);
void test_dynnet_residual_gradients(RNG *r);
void test_dynnet_layernorm_gradients(RNG *r);
void test_dynnet_time_inject_gradients(RNG *r);
void test_dynnet_swish_gradients(RNG *r);
void test_dynnet_softplus_gradients(RNG *r);
void test_adjoint_deep(RNG *r);
