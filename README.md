# Neural ODE

A implementation of [Neural Ordinary Differential Equations by Chen et al. (2018)](https://arxiv.org/abs/1806.07366) in C.

## Build
```bash
cc -O2 -Wall -Wextra -Iinclude src/utils.c src/dynmlp.c src/ode_solver.c src/adjoint.c src/adam.c src/train.c src/spiral.c src/tests.c src/main.c -lm -o neural_ode
```

## Run
```bash
./neural_ode
```







